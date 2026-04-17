"""
口袋中心（Pocket-centric）3D评估模块

用于等排体替换模型的深度评估，包含：
1. 结合亲和力评估（Vina Score）- 需要AutoDock Vina
2. 物理合理性评估（Steric Clash, Strain Energy）
3. 相互作用保持率（PLIP分析）- 需要PLIP
4. 几何与拓扑质量（键长/键角分布JSD）

借鉴：TargetDiff, DecompDiff, DiffSBDD的评估标准
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import subprocess
import tempfile
import json

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolTransforms
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, MMFFGetMoleculeForceField

# =============================================================================
# 配置
# =============================================================================
@dataclass
class EvalConfig:
    # 空间碰撞阈值（Å）- 范德华半径之和的0.75倍
    clash_threshold: float = 2.0  # 保守阈值：任何距离<2Å视为严重碰撞
    clash_ratio_threshold: float = 0.75  # 标准阈值：vdW半径的0.75倍

    # 构象能量评估
    strain_energy_threshold: float = 20.0  # kcal/mol，超过此值视为不合理构象

    # 键长分布参考（Å）
    reference_bond_lengths: Dict = None  # 从CrossDocked数据集统计

    # 外部工具路径
    vina_path: str = "vina"  # AutoDock Vina路径
    plip_path: str = "plip"  # PLIP路径

    # 输出目录
    output_dir: str = "./eval_results"


# =============================================================================
# 原子范德华半径（Å）- 借鉴DiffSBDD constants.py
# =============================================================================
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'B': 1.80, 'Si': 2.10, 'As': 1.85, 'Se': 1.90, 'default': 1.70
}

ATOM_SYMBOL_TO_VDW = defaultdict(lambda: VDW_RADII['default'])
ATOM_SYMBOL_TO_VDW.update(VDW_RADII)


# =============================================================================
# 核心评估函数
# =============================================================================

def check_steric_clash(ligand_pos: np.ndarray,
                       ligand_atoms: List[str],
                       protein_pos: np.ndarray,
                       protein_atoms: List[str],
                       threshold_ratio: float = 0.75) -> Dict:
    """
    检查配体与蛋白质之间的空间碰撞

    Args:
        ligand_pos: 配体原子坐标 (N_lig, 3)
        ligand_atoms: 配体原子符号列表 ['C', 'C', 'N', ...]
        protein_pos: 蛋白原子坐标 (N_prot, 3)
        protein_atoms: 蛋白原子符号列表
        threshold_ratio: 碰撞阈值比例（相对于vdW半径之和）

    Returns:
        clash_stats: {
            'has_clash': bool,
            'clash_count': int,
            'clash_ratio': float,
            'max_overlap': float,
            'clash_pairs': List[Tuple]  # 碰撞的原子对详情
        }
    """
    if ligand_pos is None or protein_pos is None:
        return {'has_clash': False, 'clash_count': 0, 'clash_ratio': 0.0,
                'max_overlap': 0.0, 'clash_pairs': []}

    ligand_pos = np.asarray(ligand_pos)
    protein_pos = np.asarray(protein_pos)

    n_lig = ligand_pos.shape[0]
    n_prot = protein_pos.shape[0]

    # 计算所有配体-蛋白原子对的距离
    # 使用广播避免循环
    dist_matrix = np.linalg.norm(
        ligand_pos[:, np.newaxis, :] - protein_pos[np.newaxis, :, :],
        axis=2
    )  # (N_lig, N_prot)

    # 计算每对的vdW半径之和
    vdw_lig = np.array([ATOM_SYMBOL_TO_VDW.get(a, 1.7) for a in ligand_atoms])
    vdw_prot = np.array([ATOM_SYMBOL_TO_VDW.get(a, 1.7) for a in protein_atoms])
    vdw_sum_matrix = vdw_lig[:, np.newaxis] + vdw_prot[np.newaxis, :]  # (N_lig, N_prot)

    # 碰撞阈值 = threshold_ratio * vdW_sum
    clash_thresholds = threshold_ratio * vdw_sum_matrix

    # 检测碰撞
    clash_mask = dist_matrix < clash_thresholds
    clash_count = clash_mask.sum()

    # 计算重叠量（负值表示穿透）
    overlaps = clash_thresholds - dist_matrix
    overlaps[~clash_mask] = 0  # 非碰撞区域重叠为0

    # 收集碰撞详情
    clash_pairs = []
    if clash_count > 0:
        clash_indices = np.argwhere(clash_mask)
        for idx in clash_indices[:10]:  # 只记录前10个最严重的碰撞
            i, j = idx
            clash_pairs.append({
                'ligand_idx': int(i),
                'protein_idx': int(j),
                'ligand_atom': ligand_atoms[i],
                'protein_atom': protein_atoms[j],
                'distance': float(dist_matrix[i, j]),
                'vdw_sum': float(vdw_sum_matrix[i, j]),
                'overlap': float(overlaps[i, j])
            })

    return {
        'has_clash': clash_count > 0,
        'clash_count': int(clash_count),
        'clash_ratio': float(clash_count / (n_lig * n_prot)),
        'max_overlap': float(overlaps.max()) if clash_count > 0 else 0.0,
        'clash_pairs': clash_pairs
    }


def compute_strain_energy(mol: Chem.Mol, conf_idx: int = -1) -> Dict:
    """
    计算构象的应变能（Strain Energy）

    借鉴TargetDiff评估方法：
    - 使用MMFF94力场计算当前构象能量
    - 使用UFF或MMFF优化后的最低能量作为参考
    - 能量差即为应变能

    Args:
        mol: RDKit分子对象（需要有3D构象）
        conf_idx: 构象索引，-1表示当前构象

    Returns:
        strain_stats: {
            'mmff_energy': float,
            'uff_energy': float,
            'mmff_optimized_energy': float,
            'strain_mmff': float,
            'strain_uff': float,
            'is_reasonable': bool
        }
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {'mmff_energy': 0.0, 'uff_energy': 0.0,
                'strain_mmff': float('inf'), 'strain_uff': float('inf'),
                'is_reasonable': False}

    try:
        # MMFF94能量计算
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            # MMFF不支持某些原子类型，回退到UFF
            ff = UFFGetMoleculeForceField(mol, confId=conf_idx)
            if ff is None:
                return {'mmff_energy': 0.0, 'uff_energy': 0.0,
                        'strain_mmff': float('inf'), 'strain_uff': float('inf'),
                        'is_reasonable': False}
            current_energy = ff.CalcEnergy()
        else:
            ff = MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_idx)
            current_energy = ff.CalcEnergy()

        # 优化后的最低能量（使用简单的局部优化）
        mol_optimized = Chem.Mol(mol)
        try:
            AllChem.MMFFOptimizeMolecule(mol_optimized, maxIters=200)
            mmff_props_opt = AllChem.MMFFGetMoleculeProperties(mol_optimized)
            if mmff_props_opt is not None:
                ff_opt = MMFFGetMoleculeForceField(mol_optimized, mmff_props_opt)
                optimized_energy = ff_opt.CalcEnergy()
            else:
                optimized_energy = current_energy
        except:
            optimized_energy = current_energy

        strain_energy = current_energy - optimized_energy

        # 判断是否合理（应变能<20 kcal/mol）
        is_reasonable = strain_energy < 20.0

        return {
            'mmff_energy': float(current_energy),
            'mmff_optimized_energy': float(optimized_energy),
            'strain_mmff': float(strain_energy),
            'is_reasonable': is_reasonable
        }

    except Exception as e:
        return {'mmff_energy': 0.0, 'uff_energy': 0.0,
                'strain_mmff': float('inf'), 'strain_uff': float('inf'),
                'is_reasonable': False, 'error': str(e)}


def compute_bond_length_distribution(mol: Chem.Mol) -> Dict:
    """
    计算分子键长分布，用于与参考数据集对比

    Returns:
        bond_stats: {
            'bond_lengths': Dict[Tuple, List[float]],  # 按原子对类型分组
            'bond_types': Dict[int, List[float]],      # 按键类型分组
            'mean_lengths': Dict,
            'std_lengths': Dict,
        }
    """
    if mol is None:
        return {}

    conf = mol.GetConformer()
    bond_lengths = defaultdict(list)
    bond_types_lengths = defaultdict(list)  # SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        atom_i = bond.GetBeginAtom().GetSymbol()
        atom_j = bond.GetEndAtom().GetSymbol()

        # 键长计算
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        length = np.linalg.norm(np.array([pos_i.x, pos_i.y, pos_i.z]) -
                                np.array([pos_j.x, pos_j.y, pos_j.z]))

        # 按原子对分组（按字母序）
        pair = tuple(sorted([atom_i, atom_j]))
        bond_lengths[pair].append(length)

        # 按键类型分组
        bond_type = bond.GetBondType()
        type_idx = int(bond_type)  # Chem.BondType.SINGLE=1, etc.
        bond_types_lengths[type_idx].append(length)

    # 统计均值和标准差
    mean_lengths = {pair: np.mean(vals) for pair, vals in bond_lengths.items()}
    std_lengths = {pair: np.std(vals) for pair, vals in bond_lengths.items()}

    return {
        'bond_lengths': dict(bond_lengths),
        'bond_types_lengths': dict(bond_types_lengths),
        'mean_lengths': mean_lengths,
        'std_lengths': std_lengths,
        'total_bonds': mol.GetNumBonds()
    }


def compute_jsd_bond_lengths(gen_bond_stats: Dict,
                              ref_bond_stats: Dict) -> float:
    """
    计算生成分子与参考数据集之间键长分布的Jensen-Shannon Divergence

    Args:
        gen_bond_stats: 生成分子的键长统计
        ref_bond_stats: 参考数据集的键长统计（预先计算）

    Returns:
        avg_jsd: 平均JSD值（越小越好，0表示完全一致）
    """
    from scipy.stats import entropy

    jsd_values = []

    for pair in gen_bond_stats.get('mean_lengths', {}).keys():
        if pair not in ref_bond_stats.get('mean_lengths', {}).keys():
            continue

        gen_lengths = gen_bond_stats['bond_lengths'].get(pair, [])
        ref_lengths = ref_bond_stats['bond_lengths'].get(pair, [])

        if len(gen_lengths) < 5 or len(ref_lengths) < 5:
            continue

        # 构建直方图分布
        bins = np.linspace(1.0, 2.5, 30)  # 键长范围1.0-2.5Å
        gen_hist, _ = np.histogram(gen_lengths, bins=bins, density=True)
        ref_hist, _ = np.histogram(ref_lengths, bins=bins, density=True)

        # 归一化
        gen_hist = gen_hist / gen_hist.sum()
        ref_hist = ref_hist / ref_hist.sum()

        # JSD = (KL(P||M) + KL(Q||M)) / 2, where M = (P+Q)/2
        m = (gen_hist + ref_hist) / 2
        jsd = (entropy(gen_hist + 1e-10, m + 1e-10) +
               entropy(ref_hist + 1e-10, m + 1e-10)) / 2
        jsd_values.append(jsd)

    return np.mean(jsd_values) if jsd_values else 0.0


# =============================================================================
# Vina和PLIP外部工具调用（需要安装）
# =============================================================================

def run_vina_docking(ligand_sdf: str, protein_pdb: str,
                     center: Tuple[float, float, float],
                     size: Tuple[int, int, int] = (20, 20, 20),
                     vina_path: str = "vina") -> Dict:
    """
    使用AutoDock Vina进行分子对接，计算结合亲和力

    Args:
        ligand_sdf: 配体SDF文件路径
        protein_pdb: 蛋白PDB文件路径
        center: 对接盒子中心坐标 (x, y, z)
        size: 对接盒子大小 (Å)
        vina_path: vina可执行文件路径

    Returns:
        docking_result: {
            'vina_score': float,  # 最佳构象的结合能
            'rmsd_lb': float,     # RMSD下界
            'rmsd_ub': float,     # RMSD上界
            'all_scores': List[float],  # 所有构象的得分
            'success': bool
        }
    """
    # 检查vina是否可用
    try:
        subprocess.run([vina_path, "--version"], capture_output=True, check=True)
    except:
        return {'vina_score': 0.0, 'success': False,
                'error': 'Vina not available. Install with: apt-get install autodock-vina'}

    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 准备输入文件
            ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
            protein_pdbqt = os.path.join(tmpdir, "protein.pdbqt")
            output_pdbqt = os.path.join(tmpdir, "out.pdbqt")

            # 转换SDF到PDBQT（需要obabel或prepare_ligand4.py）
            # 这里假设使用smina（Vina的改进版，支持SDF直接输入）
            cmd = [
                vina_path,
                "--ligand", ligand_sdf,
                "--receptor", protein_pdb,
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(size[0]),
                "--size_y", str(size[1]),
                "--size_z", str(size[2]),
                "--out", output_pdbqt,
                "--score_only"  # 只计算得分，不进行完整对接
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # 解析输出
            output = result.stdout
            # Vina输出格式: "Affinity: -7.3 (kcal/mol)"
            import re
            match = re.search(r'Affinity:\s*([-.\d]+)', output)
            if match:
                vina_score = float(match.group(1))
                return {'vina_score': vina_score, 'success': True,
                        'raw_output': output}
            else:
                return {'vina_score': 0.0, 'success': False,
                        'error': 'Failed to parse Vina output',
                        'raw_output': output}

    except Exception as e:
        return {'vina_score': 0.0, 'success': False, 'error': str(e)}


def run_plip_analysis(complex_pdb: str) -> Dict:
    """
    使用PLIP分析蛋白-配体相互作用

    Args:
        complex_pdb: 复合物PDB文件路径

    Returns:
        interactions: {
            'hydrogen_bonds': List[Dict],
            'hydrophobic': List[Dict],
            'pi_stack': List[Dict],
            'salt_bridge': List[Dict],
            'water_bridges': List[Dict],
            'total_count': int
        }
    """
    try:
        # 检查PLIP是否可用
        subprocess.run(["plip", "--help"], capture_output=True, check=False)

        # 运行PLIP
        cmd = ["plip", "-f", complex_pdb, "-o", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 解析JSON输出
        # PLIP输出格式较复杂，这里简化处理
        interactions = {
            'hydrogen_bonds': [],
            'hydrophobic': [],
            'pi_stack': [],
            'salt_bridge': [],
            'water_bridges': [],
            'total_count': 0,
            'success': False,
            'error': 'PLIP parsing not fully implemented'
        }

        return interactions

    except FileNotFoundError:
        return {'success': False,
                'error': 'PLIP not available. Install with: pip install plip'}


# =============================================================================
# 综合评估类
# =============================================================================

class PocketCentricEvaluator:
    """
    口袋中心评估器

    整合所有评估维度，生成综合报告
    """

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.reference_bond_stats = {}  # 预加载参考数据集统计

    def evaluate_single(self,
                        ligand_mol: Chem.Mol,
                        protein_pos: Optional[np.ndarray] = None,
                        protein_atoms: Optional[List[str]] = None,
                        original_mol: Optional[Chem.Mol] = None) -> Dict:
        """
        评估单个生成的分子

        Args:
            ligand_mol: 生成的配体分子（需要有3D构象）
            protein_pos: 蛋白口袋坐标（可选）
            protein_atoms: 蛋白原子符号（可选）
            original_mol: 原始分子（用于对比）

        Returns:
            evaluation: 综合评估结果
        """
        results = {}

        # 1. 基本有效性检查
        try:
            Chem.SanitizeMol(ligand_mol)
            results['valid'] = True
        except:
            results['valid'] = False
            results['sanitize_error'] = True
            return results

        # 2. 连通性检查
        fragments = Chem.GetMolFrags(ligand_mol)
        results['num_fragments'] = len(fragments)
        results['connected'] = len(fragments) == 1

        # 3. 构象应变能
        results['strain_energy'] = compute_strain_energy(ligand_mol)

        # 4. 键长分布
        results['bond_length_stats'] = compute_bond_length_distribution(ligand_mol)

        # 5. 与参考数据集的JSD（如果有参考）
        if self.reference_bond_stats:
            results['bond_length_jsd'] = compute_jsd_bond_lengths(
                results['bond_length_stats'],
                self.reference_bond_stats
            )

        # 6. 空间碰撞检查（如果有蛋白数据）
        if protein_pos is not None:
            ligand_conf = ligand_mol.GetConformer()
            ligand_pos = np.array([[ligand_conf.GetAtomPosition(i).x,
                                    ligand_conf.GetAtomPosition(i).y,
                                    ligand_conf.GetAtomPosition(i).z]
                                   for i in range(ligand_mol.GetNumAtoms())])
            ligand_atoms = [ligand_mol.GetAtomWithIdx(i).GetSymbol()
                           for i in range(ligand_mol.GetNumAtoms())]

            results['steric_clash'] = check_steric_clash(
                ligand_pos, ligand_atoms,
                np.asarray(protein_pos), protein_atoms,
                self.config.clash_ratio_threshold
            )

        # 7. 与原始分子的对比（如果有）
        if original_mol is not None:
            # 2D指纹相似度
            from rdkit import DataStructs
            from rdkit.Chem import AllChem

            fp_gen = AllChem.GetMorganGenerator(radius=2)
            fp_orig = fp_gen.GetFingerprint(original_mol)
            fp_new = fp_gen.GetFingerprint(ligand_mol)
            results['fingerprint_similarity'] = DataStructs.TanimotoSimilarity(fp_orig, fp_new)

            # USR形状相似度（如果有3D构象）
            if original_mol.GetNumConformers() > 0 and ligand_mol.GetNumConformers() > 0:
                from rdkit.Chem import rdShapeHelpers
                try:
                    shape_sim = rdShapeHelpers.ComputeUSRShapeSimilarity(
                        original_mol.GetConformer(), ligand_mol.GetConformer()
                    )
                    results['shape_similarity'] = shape_sim
                except:
                    results['shape_similarity'] = 0.0

        return results

    def evaluate_batch(self,
                       ligand_mols: List[Chem.Mol],
                       protein_pos: Optional[np.ndarray] = None,
                       protein_atoms: Optional[List[str]] = None,
                       original_mols: Optional[List[Chem.Mol]] = None) -> Dict:
        """
        批量评估并生成统计报告
        """
        all_results = []

        for i, mol in enumerate(ligand_mols):
            orig_mol = original_mols[i] if original_mols else None
            result = self.evaluate_single(mol, protein_pos, protein_atoms, orig_mol)
            all_results.append(result)

        # 统计汇总
        summary = {
            'total_count': len(all_results),
            'valid_count': sum(1 for r in all_results if r.get('valid', False)),
            'connected_count': sum(1 for r in all_results if r.get('connected', False)),
            'validity_rate': sum(1 for r in all_results if r.get('valid', False)) / len(all_results),

            # 空间碰撞统计
            'clash_free_count': sum(1 for r in all_results
                                    if r.get('steric_clash', {}).get('has_clash', True) is False),
            'clash_free_rate': None,  # 有蛋白数据时计算

            # 构象应变能统计
            'reasonable_strain_count': sum(1 for r in all_results
                                           if r.get('strain_energy', {}).get('is_reasonable', True)),
            'avg_strain_energy': np.mean([r.get('strain_energy', {}).get('strain_mmff', 0)
                                          for r in all_results if r.get('strain_energy')]),

            # 键长JSD
            'avg_bond_jsd': np.mean([r.get('bond_length_jsd', 0)
                                     for r in all_results if 'bond_length_jsd' in r]),

            # 与原始分子对比
            'avg_fp_similarity': np.mean([r.get('fingerprint_similarity', 0)
                                          for r in all_results if 'fingerprint_similarity' in r]),
            'avg_shape_similarity': np.mean([r.get('shape_similarity', 0)
                                            for r in all_results if 'shape_similarity' in r]),
        }

        if protein_pos is not None:
            summary['clash_free_rate'] = summary['clash_free_count'] / len(all_results)

        return {
            'individual_results': all_results,
            'summary': summary
        }

    def generate_report(self, results: Dict, output_path: str = None) -> str:
        """
        生成评估报告
        """
        report_lines = [
            "=" * 60,
            "口袋中心3D评估报告",
            "=" * 60,
            "",
            "【总体统计】",
            f"  总数: {results['summary']['total_count']}",
            f"  合法分子: {results['summary']['valid_count']} ({results['summary']['validity_rate']:.1%})",
            f"  连通分子: {results['summary']['connected_count']}",
            "",
            "【物理合理性】",
            f"  合理构象（应变能<20 kcal/mol）: {results['summary']['reasonable_strain_count']}",
            f"  平均应变能: {results['summary']['avg_strain_energy']:.2f} kcal/mol",
            "",
            "【键长分布】",
            f"  平均JSD (vs 参考数据集): {results['summary']['avg_bond_jsd']:.4f}",
            "",
        ]

        if results['summary']['clash_free_rate'] is not None:
            report_lines.extend([
                "【空间碰撞】",
                f"  无碰撞分子: {results['summary']['clash_free_count']} ({results['summary']['clash_free_rate']:.1%})",
                "",
            ])

        if results['summary']['avg_fp_similarity'] > 0:
            report_lines.extend([
                "【与原始分子对比】",
                f"  平均指纹相似度: {results['summary']['avg_fp_similarity']:.3f}",
                f"  平均形状相似度: {results['summary']['avg_shape_similarity']:.3f}",
                "",
            ])

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


# =============================================================================
# 命令行接口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="口袋中心3D评估")
    parser.add_argument("--ligand", required=True, help="配体SDF文件")
    parser.add_argument("--protein", help="蛋白口袋PDB文件（可选）")
    parser.add_argument("--reference", help="原始分子SDF文件（可选）")
    parser.add_argument("--output", default="./eval_report.txt", help="输出报告路径")

    args = parser.parse_args()

    evaluator = PocketCentricEvaluator()

    # 加载分子
    suppl = Chem.SDMolSupplier(args.ligand)
    ligand_mols = [m for m in suppl if m is not None]

    # 加载蛋白（如果有）
    protein_pos = None
    protein_atoms = None
    if args.protein:
        # 使用BioPython或简单解析PDB
        # 这里简化处理，实际需要完整实现
        print(f"注意: 蛋白质解析需要BioPython或自定义解析器")

    # 加载原始分子（如果有）
    original_mols = None
    if args.reference:
        ref_suppl = Chem.SDMolSupplier(args.reference)
        original_mols = [m for m in ref_suppl if m is not None]

    # 执行评估
    results = evaluator.evaluate_batch(ligand_mols, protein_pos, protein_atoms, original_mols)

    # 生成报告
    report = evaluator.generate_report(results, args.output)
    print(report)