"""
covalent_utils.py

共价药物设计工具箱
包含：
1. 共价弹头库 (Warhead Library)
2. 几何锚点寻找器 (Anchor Finder)
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Part 1: 共价弹头库 (Warhead Library)
# ============================================================================

WARHEAD_LIBRARY = {
    "Cys": [
        {
            "name": "Acrylamide",
            "smarts": "[*:1]NC(=O)C=C",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NC(=O)C=C",
            "attachment_atom": "N",
            "mechanism": "Michael Addition",
            "reactivity": "moderate"
        },
        {
            "name": "Methacrylamide",
            "smarts": "[*:1]NC(=O)C(C)=C",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NC(=O)C(C)=C",
            "attachment_atom": "N",
            "mechanism": "Michael Addition",
            "reactivity": "low"
        },
        {
            "name": "Propiolamide",
            "smarts": "[*:1]NC(=O)C#C",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NC(=O)C#C",
            "attachment_atom": "N",
            "mechanism": "Michael Addition",
            "reactivity": "high"
        },
        {
            "name": "Chloroacetamide",
            "smarts": "[*:1]NC(=O)CCl",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NC(=O)CCl",
            "attachment_atom": "N",
            "mechanism": "SN2",
            "reactivity": "high"
        },
        {
            "name": "Vinyl Sulfonamide",
            "smarts": "[*:1]NS(=O)(=O)C=C",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NS(=O)(=O)C=C",
            "attachment_atom": "N",
            "mechanism": "Michael Addition",
            "reactivity": "moderate"
        },
        {
            "name": "Cyanoacrylamide",
            "smarts": "[*:1]NC(=O)C(C#N)=C",
            "rxn_smarts": "[N:2]([H])>>[N:2][*:1]",
            "smiles_fragment": "NC(=O)C(C#N)=C",
            "attachment_atom": "N",
            "mechanism": "Michael Addition",
            "reactivity": "high"
        },
    ],
    "Lys": [
        {
            "name": "Sulfonyl Fluoride",
            "smarts": "[*:1]S(=O)(=O)F",
            "rxn_smarts": "[C:2]([H])>>[C:2][*:1]",
            "smiles_fragment": "S(=O)(=O)F",
            "attachment_atom": "S",
            "mechanism": "SuFEx",
            "reactivity": "moderate"
        },
        {
            "name": "Aldehyde",
            "smarts": "[*:1]C=O",
            "rxn_smarts": "[C:2]([H])>>[C:2][*:1]",
            "smiles_fragment": "C=O",
            "attachment_atom": "C",
            "mechanism": "Schiff Base",
            "reactivity": "moderate"
        },
        {
            "name": "NHS Ester",
            "smarts": "[*:1]C(=O)ON1C(=O)CCC1=O",
            "rxn_smarts": "[C:2]([H])>>[C:2][*:1]",
            "smiles_fragment": "C(=O)ON1C(=O)CCC1=O",
            "attachment_atom": "C",
            "mechanism": "Aminolysis",
            "reactivity": "high"
        },
    ],
    "Ser": [
        {
            "name": "Boronic Acid",
            "smarts": "[*:1]B(O)O",
            "rxn_smarts": "[C:2]([H])>>[C:2][*:1]",
            "smiles_fragment": "B(O)O",
            "attachment_atom": "B",
            "mechanism": "Reversible Covalent",
            "reactivity": "moderate"
        },
    ]
}


def get_warhead_fragment(name: str, aa_type: str) -> Optional[Chem.Mol]:
    """
    获取带虚原子的 RDKit 分子对象，用于拼接
    """
    for wh in WARHEAD_LIBRARY.get(aa_type, []):
        if wh["name"] == name:
            # 优先使用 SMARTS，因为它包含连接点信息
            mol = Chem.MolFromSmarts(wh["smarts"])
            if mol:
                # 确保虚原子标记为同位素0，方便后续识别
                # 注意：MolFromSmarts 生成的分子可能带有 Query 属性，需要转换
                # 这里我们简单地将其转换为 RWMol 并清理
                rw_mol = Chem.RWMol(mol)
                for atom in rw_mol.GetAtoms():
                    if atom.GetSymbol() == "*":
                        atom.SetIsotope(0)
                    # 确保原子不是 Query 原子
                    if atom.GetAtomicNum() > 0:
                        atom.SetNoImplicit(False)

                try:
                    Chem.SanitizeMol(rw_mol)
                except:
                    rw_mol.UpdatePropertyCache(strict=False)

                return rw_mol.GetMol()
    return None


# ============================================================================
# Part 2: 锚点寻找器 (Anchor Finder)
# ============================================================================

class AnchorType(Enum):
    CARBON_SP3 = "C_sp3"
    CARBON_SP2 = "C_sp2"
    CARBON_AROMATIC = "C_aromatic"
    NITROGEN = "N"
    OTHER = "other"


@dataclass
class AnchorPoint:
    """锚点数据类"""
    atom_idx: int  # 骨架上的原子索引
    atom_symbol: str  # 元素符号
    anchor_type: AnchorType  # 类型
    distance: float  # 锚点原子到目标残基的距离 (Å)
    alignment_score: float  # 几何对齐得分 (0-1, 1表示C-H键完美指向目标)
    total_score: float  # 综合评分
    vector_to_target: np.ndarray  # 指向目标的单位向量

    def to_dict(self) -> Dict:
        return {
            "atom_idx": self.atom_idx,
            "type": self.anchor_type.value,
            "distance": round(self.distance, 2),
            "alignment": round(self.alignment_score, 2),
            "score": round(self.total_score, 3)
        }


class AnchorFinder:
    def __init__(self,
                 distance_range: Tuple[float, float] = (3.0, 6.0),
                 allowed_elements: List[str] = ['C', 'N']):
        self.min_dist, self.max_dist = distance_range
        self.allowed_elements = set(allowed_elements)

        # 评分权重
        self.w_dist = 1.0
        self.w_align = 2.0  # 对齐非常重要，权重调高
        self.w_chem = 0.5

    def find_anchors(self,
                     mol: Chem.Mol,
                     target_coords: Union[Tuple[float, float, float], np.ndarray]
                     ) -> List[AnchorPoint]:
        """
        寻找最佳锚点
        :param mol: 具有3D构象的RDKit分子
        :param target_coords: 目标残基关键原子坐标 (如 Cys-S)
        """
        if mol.GetNumConformers() == 0:
            return []

        target_vec = np.array(target_coords)
        conf = mol.GetConformer()
        positions = conf.GetPositions()  # (N, 3) numpy array

        anchors = []

        # 1. 向量化计算所有原子到目标的距离
        dist_vecs = target_vec - positions
        distances = np.linalg.norm(dist_vecs, axis=1)

        # 2. 遍历原子进行筛选
        for idx, atom in enumerate(mol.GetAtoms()):
            # --- 基础过滤 ---
            if atom.GetSymbol() not in self.allowed_elements: continue
            if atom.GetTotalNumHs() == 0: continue  # 必须有氢可取代

            dist = distances[idx]
            if not (self.min_dist <= dist <= self.max_dist): continue

            # --- 化学规则检查 ---
            chem_score, anchor_type = self._check_chemistry(atom)
            if chem_score <= 0: continue

            # --- 几何对齐分析 ---
            dir_to_target = dist_vecs[idx] / (dist + 1e-6)
            bond_dir = self._estimate_bond_direction(mol, conf, idx)
            alignment = np.dot(bond_dir, dir_to_target)

            if alignment < 0.2: continue  # 几何方向不匹配

            # --- 综合评分 ---
            optimal_dist = (self.min_dist + self.max_dist) / 2
            dist_score = np.exp(-((dist - optimal_dist) ** 2) / 2)

            total_score = (self.w_dist * dist_score +
                           self.w_align * alignment +
                           self.w_chem * chem_score)

            anchors.append(AnchorPoint(
                atom_idx=idx,
                atom_symbol=atom.GetSymbol(),
                anchor_type=anchor_type,
                distance=dist,
                alignment_score=alignment,
                total_score=total_score,
                vector_to_target=dir_to_target
            ))

        # 按分数排序
        anchors.sort(key=lambda x: x.total_score, reverse=True)
        return anchors

    def _check_chemistry(self, atom: Chem.Atom) -> Tuple[float, AnchorType]:
        """评估化学环境并打分"""
        symbol = atom.GetSymbol()
        hyb = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()

        score = 0.0
        a_type = AnchorType.OTHER

        if symbol == 'C':
            if hyb == Chem.HybridizationType.SP3:
                score = 1.0  # sp3 碳最理想
                a_type = AnchorType.CARBON_SP3
            elif is_aromatic:
                score = 0.6  # 芳香碳次之
                a_type = AnchorType.CARBON_AROMATIC
            elif hyb == Chem.HybridizationType.SP2:
                score = 0.4  # 普通双键碳较难
                a_type = AnchorType.CARBON_SP2
        elif symbol == 'N':
            if not is_aromatic and hyb == Chem.HybridizationType.SP3:
                score = 0.9
                a_type = AnchorType.NITROGEN
            else:
                score = 0.3

        return score, a_type

    def _estimate_bond_direction(self, mol: Chem.Mol, conf: Chem.Conformer, atom_idx: int) -> np.ndarray:
        """估算取代基（氢）的空间指向"""
        center_pos = np.array(conf.GetAtomPosition(atom_idx))
        atom = mol.GetAtomWithIdx(atom_idx)

        neighbor_vecs = []
        for n in atom.GetNeighbors():
            if n.GetAtomicNum() > 1:  # 只考虑重原子邻居
                n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
                v = n_pos - center_pos
                v = v / (np.linalg.norm(v) + 1e-6)
                neighbor_vecs.append(v)

        if not neighbor_vecs:
            return np.array([1.0, 0.0, 0.0])

        sum_vec = np.sum(neighbor_vecs, axis=0)
        bond_dir = -sum_vec  # 取反方向
        norm = np.linalg.norm(bond_dir)

        if norm < 1e-3:
            return np.array([0.0, 0.0, 1.0])

        return bond_dir / norm

# ============================================================================
# Part 3: 弹头附着器 (Warhead Attacher)
# ============================================================================

import copy

class WarheadAttacher:
    """弹头附着器：将弹头拼接到骨架锚点"""

    def __init__(self):
        self.warhead_library = WARHEAD_LIBRARY

    def attach(
        self,
        scaffold: Chem.Mol,
        warhead_info: Dict,
        anchor_idx: int
    ) -> Tuple[Optional[Chem.Mol], Optional[Dict]]:
        """
        将弹头附着到骨架的指定锚点

        Returns:
            (产物分子, 原子映射) 或 (None, None)
        """
        try:
            if scaffold is None or anchor_idx >= scaffold.GetNumAtoms():
                return None, None

            anchor_atom = scaffold.GetAtomWithIdx(anchor_idx)
            if anchor_atom.GetTotalNumHs() < 1:
                return None, None

            # 准备弹头
            warhead_smarts = warhead_info.get("smarts", "")
            warhead_mol = Chem.MolFromSmarts(warhead_smarts)
            if warhead_mol is None:
                return None, None

            # 找到弹头连接点
            attach_idx, connect_idx = self._find_attachment_point(warhead_mol)
            if attach_idx is None:
                return None, None

            # 执行拼接
            product, atom_mapping = self._stitch(
                scaffold, warhead_mol, anchor_idx, attach_idx, connect_idx
            )

            return product, atom_mapping

        except Exception as e:
            return None, None

    def _find_attachment_point(self, mol: Chem.Mol) -> Tuple[Optional[int], Optional[int]]:
        """找到弹头的连接点（虚原子及其邻居）"""
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_idx = atom.GetIdx()
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 0:
                        return attach_idx, neighbor.GetIdx()
        return None, None

    def _stitch(
        self,
        scaffold: Chem.Mol,
        warhead: Chem.Mol,
        anchor_idx: int,
        attach_idx: int,
        connect_idx: int
    ) -> Tuple[Optional[Chem.Mol], Optional[Dict]]:
        """执行分子拼接"""
        try:
            # 记录骨架原子数
            scaffold_atom_count = scaffold.GetNumAtoms()

            # 添加显式氢到骨架
            scaffold_h = Chem.AddHs(scaffold)
            scaffold_rw = Chem.RWMol(scaffold_h)

            # 找到锚点上的氢
            anchor_atom = scaffold_rw.GetAtomWithIdx(anchor_idx)
            h_to_remove = None
            for neighbor in anchor_atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    h_to_remove = neighbor.GetIdx()
                    break

            if h_to_remove is None:
                return None, None

            # 准备弹头（移除虚原子）
            warhead_rw = Chem.RWMol(copy.deepcopy(warhead))
            warhead_rw.RemoveAtom(attach_idx)
            adjusted_connect = connect_idx - 1 if attach_idx < connect_idx else connect_idx
            warhead_clean = warhead_rw.GetMol()

            # 合并分子
            combined = Chem.CombineMols(scaffold_rw.GetMol(), warhead_clean)
            combined_rw = Chem.RWMol(combined)

            # 计算新索引
            scaffold_with_h_count = scaffold_rw.GetNumAtoms()
            warhead_connect_in_combined = scaffold_with_h_count + adjusted_connect

            # 移除氢并添加键
            combined_rw.RemoveAtom(h_to_remove)
            if h_to_remove < warhead_connect_in_combined:
                warhead_connect_in_combined -= 1

            combined_rw.AddBond(anchor_idx, warhead_connect_in_combined, Chem.BondType.SINGLE)

            # 清理
            product = combined_rw.GetMol()
            product = Chem.RemoveHs(product)
            Chem.SanitizeMol(product)

            # 构建原子映射
            atom_mapping = {
                "scaffold_atom_count": scaffold_atom_count,
                "anchor_idx": anchor_idx
            }

            return product, atom_mapping

        except Exception as e:
            return None, None


# ============================================================================
# Part 4: 构象生成器 (Conformer Generator with coordMap)
# ============================================================================

from rdkit.Geometry import Point3D

class ConformerGenerator:
    """构象生成器：保持骨架坐标，只优化弹头部分"""

    def __init__(self, force_field: str = "MMFF", max_iters: int = 500):
        self.force_field = force_field
        self.max_iters = max_iters

    def generate(
        self,
        product: Chem.Mol,
        scaffold: Chem.Mol,
        atom_mapping: Dict
    ) -> Tuple[Optional[Chem.Mol], float]:
        """
        生成产物的3D构象，保持骨架坐标不变

        Returns:
            (优化后的分子, 能量)
        """
        try:
            if scaffold.GetNumConformers() == 0:
                return None, float('inf')

            scaffold_atom_count = atom_mapping.get("scaffold_atom_count", 0)

            # 构建 coordMap
            coord_map = {}
            scaffold_conf = scaffold.GetConformer()

            for i in range(min(scaffold_atom_count, scaffold.GetNumAtoms())):
                pos = scaffold_conf.GetAtomPosition(i)
                coord_map[i] = Point3D(pos.x, pos.y, pos.z)

            # 嵌入构象
            product_h = Chem.AddHs(product)

            result = AllChem.EmbedMolecule(
                product_h,
                coordMap=coord_map,
                useRandomCoords=True,
                randomSeed=42
            )

            if result == -1:
                result = AllChem.EmbedMolecule(product_h, randomSeed=42)
                if result == -1:
                    return None, float('inf')

            # 约束优化
            energy = self._constrained_optimize(
                product_h,
                list(range(scaffold_atom_count))
            )

            product_final = Chem.RemoveHs(product_h)
            return product_final, energy

        except Exception as e:
            return None, float('inf')

    def _constrained_optimize(self, mol: Chem.Mol, fixed_atoms: List[int]) -> float:
        """约束能量最小化"""
        try:
            if self.force_field == "MMFF":
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is None:
                    return self._uff_optimize(mol, fixed_atoms)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)

            if ff is None:
                return float('inf')

            # 固定骨架原子
            for idx in fixed_atoms:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)

            ff.Minimize(maxIts=self.max_iters)
            return ff.CalcEnergy()

        except:
            return float('inf')

    def _uff_optimize(self, mol: Chem.Mol, fixed_atoms: List[int]) -> float:
        """UFF优化备用"""
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is None:
                return float('inf')
            for idx in fixed_atoms:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
            ff.Minimize(maxIts=self.max_iters)
            return ff.CalcEnergy()
        except:
            return float('inf')


# ============================================================================
# Part 5: 方向检测器 (Direction Checker)
# ============================================================================

class DirectionQuality(Enum):
    """方向质量等级"""
    EXCELLENT = "excellent"  # cos > 0.8
    GOOD = "good"            # cos > 0.5
    ACCEPTABLE = "acceptable"  # cos > 0.0
    POOR = "poor"            # cos < 0.0


class DirectionChecker:
    """弹头方向检测器"""

    def __init__(
        self,
        min_cosine: float = 0.0,
        good_cosine: float = 0.5,
        excellent_cosine: float = 0.8
    ):
        self.min_cosine = min_cosine
        self.good_cosine = good_cosine
        self.excellent_cosine = excellent_cosine

    def check(
        self,
        mol: Chem.Mol,
        anchor_idx: int,
        target_coords: Tuple[float, float, float],
        scaffold_atom_count: int
    ) -> Tuple[float, DirectionQuality, bool]:
        """
        检测弹头方向

        Returns:
            (余弦相似度, 质量等级, 是否通过)
        """
        try:
            if mol is None or mol.GetNumConformers() == 0:
                return 0.0, DirectionQuality.POOR, False

            conf = mol.GetConformer()

            # 锚点坐标
            anchor_pos = conf.GetAtomPosition(anchor_idx)
            anchor = np.array([anchor_pos.x, anchor_pos.y, anchor_pos.z])

            # 目标坐标
            target = np.array(target_coords)

            # 找弹头核心原子
            warhead_idx = self._find_warhead_atom(mol, anchor_idx, scaffold_atom_count)
            if warhead_idx is None:
                return 0.0, DirectionQuality.POOR, False

            warhead_pos = conf.GetAtomPosition(warhead_idx)
            warhead = np.array([warhead_pos.x, warhead_pos.y, warhead_pos.z])

            # 计算向量
            growth_vec = warhead - anchor
            target_vec = target - anchor

            norm_growth = np.linalg.norm(growth_vec)
            norm_target = np.linalg.norm(target_vec)

            if norm_growth < 1e-6 or norm_target < 1e-6:
                return 0.0, DirectionQuality.POOR, False

            cosine = np.dot(growth_vec, target_vec) / (norm_growth * norm_target)
            cosine = float(np.clip(cosine, -1.0, 1.0))

            # 判断质量
            if cosine >= self.excellent_cosine:
                quality = DirectionQuality.EXCELLENT
            elif cosine >= self.good_cosine:
                quality = DirectionQuality.GOOD
            elif cosine >= self.min_cosine:
                quality = DirectionQuality.ACCEPTABLE
            else:
                quality = DirectionQuality.POOR

            is_valid = cosine >= self.min_cosine

            return cosine, quality, is_valid

        except Exception as e:
            return 0.0, DirectionQuality.POOR, False

    def _find_warhead_atom(
        self, mol: Chem.Mol, anchor_idx: int, scaffold_count: int
    ) -> Optional[int]:
        """找到弹头核心原子（与锚点相连的第一个非骨架原子）"""
        anchor_atom = mol.GetAtomWithIdx(anchor_idx)
        for neighbor in anchor_atom.GetNeighbors():
            if neighbor.GetIdx() >= scaffold_count:
                return neighbor.GetIdx()
        # 备用：返回第一个非骨架原子
        for i in range(scaffold_count, mol.GetNumAtoms()):
            return i
        return None


# ============================================================================
# Part 6: 整合类 (Covalent Designer)
# ============================================================================

@dataclass
class CovalentResult:
    """共价生成结果"""
    success: bool
    molecule: Optional[Chem.Mol]
    smiles: Optional[str]
    warhead_name: str
    anchor_idx: int
    energy: float
    direction_score: float
    direction_quality: DirectionQuality
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "smiles": self.smiles,
            "warhead_name": self.warhead_name,
            "anchor_idx": self.anchor_idx,
            "energy": round(self.energy, 2) if self.energy != float('inf') else None,
            "direction_score": round(self.direction_score, 3),
            "direction_quality": self.direction_quality.value,
            "error": self.error_message
        }


class CovalentDesigner:
    """
    共价药物设计器
    整合：锚点发现 → 弹头附着 → 构象生成 → 方向检测
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        self.anchor_finder = AnchorFinder(
            distance_range=config.get("distance_range", (3.0, 6.0)),
            allowed_elements=config.get("allowed_elements", ['C', 'N'])
        )
        self.warhead_attacher = WarheadAttacher()
        self.conformer_generator = ConformerGenerator(
            force_field=config.get("force_field", "MMFF")
        )
        self.direction_checker = DirectionChecker(
            min_cosine=config.get("min_cosine", 0.0)
        )

        self.reject_poor_direction = config.get("reject_poor_direction", True)
        self.max_anchors = config.get("max_anchors", 5)
        self.energy_threshold = config.get("energy_threshold", 500.0)

    def design(
        self,
        scaffold: Chem.Mol,
        target_residue_type: str,
        target_coords: Tuple[float, float, float]
    ) -> List[CovalentResult]:
        """
        执行共价药物设计

        Parameters:
            scaffold: 骨架分子（需有3D构象）
            target_residue_type: 目标残基类型 ("Cys", "Lys", "Ser")
            target_coords: 目标残基关键原子坐标

        Returns:
            按能量排序的共价产物列表
        """
        results = []

        # 确保骨架有3D构象
        if scaffold.GetNumConformers() == 0:
            scaffold = self._ensure_3d(scaffold)
            if scaffold is None:
                return results

        # 获取弹头
        warheads = WARHEAD_LIBRARY.get(target_residue_type, [])
        if not warheads:
            return results

        # 寻找锚点
        anchors = self.anchor_finder.find_anchors(scaffold, target_coords)
        if not anchors:
            return results

        anchors = anchors[:self.max_anchors]
        scaffold_atom_count = scaffold.GetNumAtoms()

        # 遍历锚点和弹头
        for anchor in anchors:
            for warhead in warheads:
                result = self._process_single(
                    scaffold, warhead, anchor,
                    target_coords, scaffold_atom_count
                )
                if result.success:
                    results.append(result)

        # 按能量排序
        results.sort(key=lambda x: x.energy)

        return results

    def _ensure_3d(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """确保分子有3D构象"""
        try:
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            return Chem.RemoveHs(mol)
        except:
            return None

    def _process_single(
        self,
        scaffold: Chem.Mol,
        warhead: Dict,
        anchor: AnchorPoint,
        target_coords: Tuple[float, float, float],
        scaffold_atom_count: int
    ) -> CovalentResult:
        """处理单个锚点-弹头组合"""

        # 1. 附着弹头
        product, atom_mapping = self.warhead_attacher.attach(
            scaffold, warhead, anchor.atom_idx
        )

        if product is None:
            return CovalentResult(
                success=False, molecule=None, smiles=None,
                warhead_name=warhead["name"], anchor_idx=anchor.atom_idx,
                energy=float('inf'), direction_score=0.0,
                direction_quality=DirectionQuality.POOR,
                error_message="弹头附着失败"
            )

        # 2. 生成构象
        product_3d, energy = self.conformer_generator.generate(
            product, scaffold, atom_mapping
        )

        if product_3d is None or energy > self.energy_threshold:
            return CovalentResult(
                success=False, molecule=None,
                smiles=Chem.MolToSmiles(product) if product else None,
                warhead_name=warhead["name"], anchor_idx=anchor.atom_idx,
                energy=energy, direction_score=0.0,
                direction_quality=DirectionQuality.POOR,
                error_message="构象生成失败或能量过高"
            )

        # 3. 方向检测
        cosine, quality, is_valid = self.direction_checker.check(
            product_3d, anchor.atom_idx, target_coords, scaffold_atom_count
        )

        if self.reject_poor_direction and not is_valid:
            return CovalentResult(
                success=False, molecule=product_3d,
                smiles=Chem.MolToSmiles(product_3d),
                warhead_name=warhead["name"], anchor_idx=anchor.atom_idx,
                energy=energy, direction_score=cosine,
                direction_quality=quality,
                error_message="弹头方向不合理"
            )

        # 成功
        return CovalentResult(
            success=True, molecule=product_3d,
            smiles=Chem.MolToSmiles(product_3d),
            warhead_name=warhead["name"], anchor_idx=anchor.atom_idx,
            energy=energy, direction_score=cosine,
            direction_quality=quality
        )