# filename: bioisosteric_v2_5_isostere_diffusion.py
"""
[V5改进] 等排体片段扩散模型 - 两阶段课程学习

核心改进（参考问题总结1.txt）：
1. 真正的等排体配对挖掘：骨架相似 + 片段不同
   - scaffold Tanimoto > 0.5（同类分子）
   - fragment Tanimoto < 0.4（发生了替换）
   - 原子数差异 ≤ 3（大小相近）

2. 两阶段课程学习：
   - Phase 1: 自重建（10 epochs）- 学习基本生成能力
   - Phase 2: 等排体生成（40 epochs）- 学习真正的替换

3. 生成模式支持：
   - 'self_reconstruct': 强引导，生成相似片段
   - 'isostere': 弱引导，自由探索不同片段
   - 'guided_isostere': 位置引导，不约束原子类型

4. 等排体质量评估（新指标）：
   - 分子量相近（±50 Da）
   - logP相近（±1.5）
   - 片段多样性（与原片段不同才是等排体）

数据量建议：关闭test_mode使用完整ChEMBL数据集，以支持等排体配对挖掘

"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing")
warnings.filterwarnings("ignore", category=UserWarning)

import math
import pickle
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected

from rdkit import Chem, RDLogger, RDConfig, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, BRICS

# Flexible Prior系统导入
from flexible_prior import (
    FlexiblePriorManager,
    create_prior,
    remove_mean_with_mask,
    sum_except_batch,
)
import sys
import io

# 强制 stdout/stderr 使用 UTF-8，修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# =============================================================================
# 1. 全局设置
# =============================================================================
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger("BioIsosteric_V2_5_Isostere")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler("bioisosteric_v2_5_isostere.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原子类型（与v2_3一致）
ATOM_TYPES = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
atom_type_map = {z: i for i, z in enumerate(ATOM_TYPES)}
atom_type_map_rev = {i: z for z, i in atom_type_map.items()}

# [修复V3] 原子类型类别权重重新平衡
# 索引顺序: [H, B, C, N, O, F, P, S, Cl, Br, I, dummy]
# 原则：稀有原子上调，但不要超过常见原子的10x，避免过度偏向杂原子
ATOM_TYPE_WEIGHTS = torch.tensor([
    2.0,   # H (氢) - 较稀有
    10.0,  # B (硼) - 极稀有
    0.5,   # C (碳) - 最常见，轻微降权防塌缩即可（原来0.05太极端，导致碳被惩罚）
    5.0,   # N (氮) - 较常见（原来30太高，强迫生成氮原子）
    1.0,   # O (氧) - 常见
    8.0,   # F (氟) - 较稀有
    10.0,  # P (磷) - 稀有
    5.0,   # S (硫) - 较稀有
    8.0,   # Cl (氯) - 较稀有
    10.0,  # Br (溴) - 稀有
    15.0,  # I (碘) - 极稀有
    0.1,   # dummy - 降权，避免生成通配符
])

# 键类型
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
BOND_TYPE_MAP_REV = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE,
    3: Chem.rdchem.BondType.AROMATIC,
}

# [修复V3] 键类型类别权重（调整以改善芳香键预测）
# 索引顺序: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
# [改进] 大幅增加AROMATIC权重，降低SINGLE权重，促进芳香键正确预测
BOND_TYPE_WEIGHTS = torch.tensor([
    0.2,   # SINGLE - 降低权重（预测单键惩罚更低，但真实芳香键时惩罚更高）
    3.0,   # DOUBLE - 适度惩罚
    10.0,  # TRIPLE - 稀有，高权重
    3.0,   # AROMATIC - [调整] 大幅提高权重1.5→3.0，确保芳香键正确预测
])

# [新增V3] 键长参考值（Angstrom）- 用于键长-键类型联合约束
# 索引顺序与BOND_TYPE_MAP一致: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
# 基础版本：仅按键类型（适用于未知原子类型）
BOND_LENGTHS = {
    0: 1.54,   # SINGLE - 单键标准长度
    1: 1.34,   # DOUBLE - 双键标准长度
    2: 1.20,   # TRIPLE - 三键标准长度
    3: 1.40,   # AROMATIC - 芳香键长度（介于单键和双键之间）
}
BOND_LENGTH_TOLERANCE = 0.15  # 键长容差（±0.15Å）

# [改进] Atom-pair specific键长表（借鉴DiffSBDD constants.py）
# 单位：Angstrom（原DiffSBDD使用pm，这里已转换 pm/100）
# 用于更精确的键长约束
BOND_LENGTHS_PAIR = {
    # 单键 (bond_type=0)
    'single': {
        ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43, ('C', 'S'): 1.82,
        ('C', 'F'): 1.35, ('C', 'Cl'): 1.77, ('C', 'Br'): 1.94, ('C', 'I'): 2.14,
        ('N', 'N'): 1.45, ('N', 'O'): 1.40, ('N', 'S'): 1.68,
        ('O', 'O'): 1.48, ('O', 'S'): 1.51,
        ('S', 'S'): 2.04,
        ('H', 'C'): 1.09, ('H', 'N'): 1.01, ('H', 'O'): 0.96, ('H', 'S'): 1.34,
    },
    # 双键 (bond_type=1)
    'double': {
        ('C', 'C'): 1.34, ('C', 'N'): 1.29, ('C', 'O'): 1.20, ('C', 'S'): 1.60,
        ('N', 'N'): 1.25, ('N', 'O'): 1.21,
        ('O', 'O'): 1.21, ('O', 'P'): 1.50,
    },
    # 三键 (bond_type=2)
    'triple': {
        ('C', 'C'): 1.20, ('C', 'N'): 1.16, ('C', 'O'): 1.13,
        ('N', 'N'): 1.10,
    },
    # 芳香键 (bond_type=3) - 通常介于单键和双键之间
    'aromatic': {
        ('C', 'C'): 1.40, ('C', 'N'): 1.37, ('C', 'O'): 1.38,
        ('N', 'N'): 1.35, ('N', 'O'): 1.34,
    }
}

# 原子类型映射（用于键长查询）
ATOM_SYMBOL_MAP = {
    0: 'H', 1: 'B', 2: 'C', 3: 'N', 4: 'O', 5: 'F',
    6: 'P', 7: 'S', 8: 'Cl', 9: 'Br', 10: 'I', 11: 'dummy'
}

PHARM_DIM = 8
ED_DIM = 15

# =============================================================================
# 2. 配置 - v2_5: 等排体生成改进版（保持v2_4工程风格）
# =============================================================================
# ★ [V5说明] 保持frozen=False以支持运行时切换curriculum_phase
# ★ 建议：调试时使用test_mode=True（默认），正式训练时改为False使用完整数据集
@dataclass
class Config:
    # 数据路径：使用绝对路径指向数据目录
    # 可选：'E:/zuhui/chembl_data_sample' (测试样本) 或 'E:/zuhui/chembl_data' (完整数据)
    # ★ [V5建议] 等排体训练需要大量数据，建议关闭test_mode使用完整数据集
    data_dir: str = "E:/zuhui/chembl_data_sample"
    test_mode: bool = True  # ★ 默认开启测试模式，方便调试；正式训练改为False
    test_mode_limit: int = 5000  # 测试模式限制数据量（调试时5000足够）
    max_atoms_per_mol: int = 100

    # 片段配置
    min_fragment_atoms: int = 2
    max_fragment_atoms: int = 15
    min_scaffold_atoms: int = 5
    max_scaffold_atoms: int = 50  # 新增：限制骨架大小，避免过大骨架导致重建困难

    # ACCFG集成
    use_accfg_decomposition: bool = False  # 使用ACCFG分解（很慢，建议关闭）
    use_accfg_fg_detection: bool = False  # 使用ACCFG官能团识别（很慢，建议关闭，用SMARTS替代）

    # Stage 1: 骨架编码器预训练
    stage1_epochs: int = 20
    stage1_batch_size: int = 1024

    # Stage 2: 片段配对挖掘
    stage2_epochs: int = 20
    stage2_batch_size: int = 1024
    stage2_fp_threshold: float = 0.40  # 降低阈值获得更多配对
    stage2_sample_size: int = 1000  # 配对采样大小
    stage2_max_pairs: int = 1000000
    stage3_max_pairs: int = 1000000

    # Stage 3: 片段扩散训练
    stage3_epochs: int = 50
    stage3_batch_size: int = 128
    stage3_timesteps: int = 100
    stage3_lr: float = 2e-4

    # 注意力配置
    use_attention: bool = True
    attention_heads: int = 4
    attention_r_feat_dim: int = 20

    # 离散扩散
    use_discrete_diffusion: bool = True
    num_bond_types: int = 4

    # [新增] Prior分布模式：'uniform' | 'data_stats' | 'learnable' | 'conditional'
    # - 'uniform': 均匀分布（默认，会导致塌缩）
    # - 'data_stats': 数据集统计分布（推荐，快速解决塌缩）
    # - 'learnable': 可学习prior（最灵活，但需要更多数据）
    # - 'conditional': 条件相关prior（根据骨架动态调整）
    prior_mode: str = 'data_stats'

    # [新增V3] 位置坐标Prior配置
    # - 'custom': 手动设定prior参数（isotropic/anisotropic gaussian）
    # - 'learned': 端到端学习prior参数
    # - 'conditional': 条件注入动态prior
    position_prior_mode: str = 'learned'  # 'custom' | 'learned' | 'conditional'
    position_prior_type: str = 'isotropic_gaussian'  # 'isotropic_gaussian' | 'anisotropic_gaussian'
    position_prior_sigma: float = 1.0  # Custom模式的sigma参数

    # [新增V3] 键长-键类型联合约束配置
    bond_distance_weight: float = 5.0  # 键长一致性损失权重 [调整] 2.0 → 5.0
    connectivity_weight: float = 1.0  # 连通性损失权重
    connectivity_threshold: float = 0.1  # 连通性阈值 [调整] 0.5 → 0.1
    use_cfg_inference: bool = True  # 是否使用Classifier-Free Guidance推理
    cfg_scale: float = 2.0  # CFG guidance scale
    use_atom_pair_bond_lengths: bool = True  # 使用atom-pair特定键长表

    # ★ [V5改进] 课程学习配置 - 真正的两阶段训练
    # Phase 1: 自重建模式 - 学习完美重建原片段
    # Phase 2: 等排体生成模式 - 学习生成骨架相似+片段不同的替换
    # ★ [调整] 增加 Phase 1 epochs，确保充分学习重建能力
    self_reconstruction_mode: bool = True
    self_reconstruction_epochs: int = 20  # Phase 1 训练轮数（从10→20，确保重建能力充分学习）

    # ★ [V5新增] 等排体专用配置
    curriculum_phase: int = 1  # 1=Phase1自重建, 2=Phase2等排体（运行时切换）
    isostere_epochs: int = 30  # Phase 2 训练轮数（20+30=50，与v2_4总epochs一致）
    isostere_scaffold_threshold: float = 0.5  # 骨架相似阈值（必须相似）
    isostere_fragment_threshold: float = 0.4  # 片段差异阈值（必须不同）
    isostere_atom_count_diff: int = 3  # 片段原子数差异上限

    # ★ [V5新增] 测试阶段配置（方便分阶段测试）
    # 用于调试：可以只运行特定阶段，跳过其他阶段
    test_stage_only: int = 0  # 0=全部运行, 1=只Stage1, 2=只Stage2, 3=只Stage3Phase1, 4=只Stage3Phase2
    skip_stage1: bool = False  # 跳过Stage1（使用预训练权重）
    skip_stage2: bool = False  # 跳过Stage2
    skip_phase1: bool = False  # 跳过Phase1（直接训练Phase2）
    load_phase1_weights: str = ""  # 加载Phase1预训练权重路径（跳过Phase1时使用）

    # ★ [V5新增] 不同阶段的损失权重
    x0_loss_weight_phase1: float = 0.05  # Phase 1: 强约束到原片段
    x0_loss_weight_phase2: float = 0.01  # Phase 2: 弱约束，允许探索
    bond_distance_weight_phase1: float = 5.0
    bond_distance_weight_phase2: float = 2.0  # Phase 2: 降低键长约束

    # [新增V4] x0预测损失（片段重构损失）
    use_x0_prediction_loss: bool = True  # 是否启用x0预测损失
    x0_loss_weight: float = 0.05  # 默认值，运行时会根据phase调整

CFG = Config()


# =============================================================================
# 2.4 Prior分布定义（解决原子类型塌缩）
# =============================================================================
# 数据集统计的原子类型分布（借鉴DecompDiff utils/transforms.py:142-143）
# 索引顺序: [H, B, C, N, O, F, P, S, Cl, Br, I, dummy] (12类)
# 这里使用药物分子中常见片段的估计分布（需要根据实际数据集调整）
DATA_STATS_ATOM_PRIOR = np.array([
    0.02,   # H (氢) - 片段中较少
    0.001,  # B (硼) - 极稀有
    0.45,   # C (碳) - 主导
    0.15,   # N (氮) - 较常见
    0.20,   # O (氧) - 较常见
    0.05,   # F (氟) - 药物中常见
    0.02,   # P (磷) - 较稀有
    0.05,   # S (硫) - 药物中常见
    0.04,   # Cl (氯) - 药物中常见
    0.02,   # Br (溴) - 较稀有
    0.01,   # I (碘) - 极稀有
    0.04,   # dummy - 占位
])
DATA_STATS_ATOM_PRIOR = DATA_STATS_ATOM_PRIOR / DATA_STATS_ATOM_PRIOR.sum()  # 归一化

# 键类型分布（借鉴DecompDiff）
# 索引顺序: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
DATA_STATS_BOND_PRIOR = np.array([
    0.65,   # SINGLE - 最常见
    0.15,   # DOUBLE - 较常见
    0.02,   # TRIPLE - 稀有
    0.18,   # AROMATIC - 较常见
])
DATA_STATS_BOND_PRIOR = DATA_STATS_BOND_PRIOR / DATA_STATS_BOND_PRIOR.sum()  # 归一化


# =============================================================================
# 2.5 价键修复工具函数（借鉴DecompDiff）
# =============================================================================
import re
from copy import deepcopy

def calc_valence(rdatom):
    """计算RDKit原子的显式价键（可以在sanitize前调用）"""
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def get_max_valence(atom, mol=None):
    """
    获取原子的最大价键数（借鉴TargetDiff reconstruct.py第105-119行）

    特殊处理：
    - S原子有2个O邻居时（sulfone），允许6价
    - 使用RDKit和OpenBabel的最小值（更严格）

    Args:
        atom: RDKit原子对象
        mol: 可选的分子对象，用于检测sulfone

    Returns:
        最大价键数
    """
    pt = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()

    # 基础最大价键
    max_val = pt.GetDefaultValence(atomic_num)

    # Sulfone特殊处理：S原子有>=2个O邻居时允许6价
    if atomic_num == 16 and mol is not None:  # S
        o_neighbor_count = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:  # O
                o_neighbor_count += 1
        if o_neighbor_count >= 2:
            max_val = 6

    return max_val


def fix_valence_n4(mol):
    """
    修复N原子4价和卤素超价问题（借鉴DecompDiff reconstruct.py fix_valence函数）

    对于显式价键超过允许值的原子：
    - N原子4价：设置+1电荷使其合理化（NH4+形式）
    - 卤素原子（F, Cl, Br, I）超价：移除多余的键

    Returns:
        (mol, fixed): 修复后的分子和是否成功修复
    """
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    pt = Chem.GetPeriodicTable()

    # 卤素原子列表（只能有1价）
    halogens = {9: 1, 17: 1, 35: 1, 53: 1}  # F, Cl, Br, I

    while True:
        try:
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False  # 其他错误，无法修复

        cnt_loop += 1
        if cnt_loop > 100:
            break

        err_str = str(err)

        # 正则匹配N原子4价错误
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err_str)
        if len(index) > 0:
            # 设置+1电荷使N4价合理
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
            continue

        # 正则匹配S原子超价错误
        S7_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) S, ([0-9]{1,}), is greater than permitted")
        s_match = S7_valence.findall(err_str)
        if len(s_match) > 0:
            idx, val = int(s_match[0][0]), int(s_match[0][1])
            atom = mol.GetAtomWithIdx(idx)
            # 检查是否是sulfone（有O邻居），如果是则保持，否则降级
            o_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
            if o_count < 2:
                # 不是sulfone，需要修复：移除多余的键或降级
                # ★ [V5修复] Bond没有GetLength方法，需要从Conformer计算距离
                try:
                    conf = mol.GetConformer()
                    bonds = []
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            i = b.GetBeginAtomIdx()
                            j = b.GetEndAtomIdx()
                            dist = np.linalg.norm(conf.GetAtomPosition(i) - conf.GetAtomPosition(j))
                            bonds.append((dist, b))
                    if bonds:
                        bonds.sort(reverse=True, key=lambda x: x[0])
                        bond = bonds[0][1]
                        if bond.GetBondType() == Chem.BondType.TRIPLE:
                            bond.SetBondType(Chem.BondType.DOUBLE)
                        elif bond.GetBondType() == Chem.BondType.DOUBLE:
                            bond.SetBondType(Chem.BondType.SINGLE)
                except:
                    # 如果没有构象，直接降级第一个非单键
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            if b.GetBondType() == Chem.BondType.TRIPLE:
                                b.SetBondType(Chem.BondType.DOUBLE)
                            elif b.GetBondType() == Chem.BondType.DOUBLE:
                                b.SetBondType(Chem.BondType.SINGLE)
                            break
            continue

        # 正则匹配通用超价错误（捕获原子索引、元素符号和价键数）
        # 格式：Explicit valence for atom # X <element>, Y, is greater than permitted
        general_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) ([A-Z]{1,2}), ([0-9]{1,}), is greater than permitted")
        match = general_valence.findall(err_str)
        if len(match) > 0:
            idx = int(match[0][0])
            element = match[0][1]
            val = int(match[0][2])
            atom = mol.GetAtomWithIdx(idx)

            # 获取原子序数
            atomic_num = atom.GetAtomicNum()

            # 卤素超价处理（F=9, Cl=17, Br=35, I=53）
            if atomic_num in halogens and val > 1:
                # 卤素只能有1价，需要移除多余的键
                # 找到所有键并移除多余的（保留最短的键）
                # ★ [V5修复] Bond没有GetLength方法，需要从Conformer计算距离
                try:
                    conf = mol.GetConformer()
                    bonds = []
                    for b in atom.GetBonds():
                        i = b.GetBeginAtomIdx()
                        j = b.GetEndAtomIdx()
                        dist = np.linalg.norm(conf.GetAtomPosition(i) - conf.GetAtomPosition(j))
                        bonds.append((dist, b))
                    bonds.sort(reverse=True, key=lambda x: x[0])  # 最长的键先处理

                    # 需要移除 (val - 1) 个键
                    num_to_remove = val - 1
                    for i in range(min(num_to_remove, len(bonds) - 1)):  # 保留至少1个键
                        bond_to_remove = bonds[i][1]
                        try:
                            mol.RemoveBond(bond_to_remove.GetBeginAtomIdx(), bond_to_remove.GetEndAtomIdx())
                        except:
                            pass
                except:
                    # 如果没有构象，直接移除多余的键（从后往前）
                    bonds_list = list(atom.GetBonds())
                    num_to_remove = val - 1
                    for i in range(min(num_to_remove, len(bonds_list) - 1)):
                        bond_to_remove = bonds_list[i]
                        try:
                            mol.RemoveBond(bond_to_remove.GetBeginAtomIdx(), bond_to_remove.GetEndAtomIdx())
                        except:
                            pass
                continue

            # O原子超价处理（O只能有2价）
            if atomic_num == 8 and val > 2:
                # ★ [V5修复] Bond没有GetLength方法，需要从Conformer计算距离
                try:
                    conf = mol.GetConformer()
                    bonds = []
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            i = b.GetBeginAtomIdx()
                            j = b.GetEndAtomIdx()
                            dist = np.linalg.norm(conf.GetAtomPosition(i) - conf.GetAtomPosition(j))
                            bonds.append((dist, b))
                    if bonds:
                        bonds.sort(reverse=True, key=lambda x: x[0])
                        bond = bonds[0][1]
                        bond.SetBondType(Chem.BondType.SINGLE)
                except:
                    # 如果没有构象，直接降级第一个非单键
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            b.SetBondType(Chem.BondType.SINGLE)
                            break
                continue

            # P原子超价处理（P可以5价，但特殊情况需要处理）
            if atomic_num == 15 and val > 5:
                # ★ [V5修复] Bond没有GetLength方法，需要从Conformer计算距离
                try:
                    conf = mol.GetConformer()
                    bonds = []
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            i = b.GetBeginAtomIdx()
                            j = b.GetEndAtomIdx()
                            dist = np.linalg.norm(conf.GetAtomPosition(i) - conf.GetAtomPosition(j))
                            bonds.append((dist, b))
                    if bonds:
                        bonds.sort(reverse=True, key=lambda x: x[0])
                        bond = bonds[0][1]
                        if bond.GetBondType() == Chem.BondType.TRIPLE:
                            bond.SetBondType(Chem.BondType.DOUBLE)
                        elif bond.GetBondType() == Chem.BondType.DOUBLE:
                            bond.SetBondType(Chem.BondType.SINGLE)
                except:
                    # 如果没有构象，直接降级第一个非单键
                    for b in atom.GetBonds():
                        if b.GetBondType() != Chem.BondType.SINGLE:
                            if b.GetBondType() == Chem.BondType.TRIPLE:
                                b.SetBondType(Chem.BondType.DOUBLE)
                            elif b.GetBondType() == Chem.BondType.DOUBLE:
                                b.SetBondType(Chem.BondType.SINGLE)
                            break
                continue

    return mol, fixed


def fix_hypervalent_atoms(mol):
    """
    修复超价原子（借鉴DecompDiff reconstruct.py convert_ob_mol_to_rd_mol函数 + TargetDiff sulfone处理）

    对于DOUBLE/TRIPLE键导致的超价原子，降低键类型
    - TRIPLE -> DOUBLE
    - DOUBLE -> SINGLE

    特殊处理sulfone：S原子有2个O邻居时允许6价

    Returns:
        修复后的分子
    """
    mol = deepcopy(mol)

    # 收集所有非SINGLE键，按距离排序（距离越远越可能被降级）
    positions = mol.GetConformer().GetPositions()
    nonsingles = []

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i] - positions[j])
            nonsingles.append((dist, bond))

    # 按距离降序排序（最远的键最先处理）
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d, bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        val1 = calc_valence(a1)
        val2 = calc_valence(a2)
        # 使用新的get_max_valence函数（包含sulfone特殊处理）
        max1 = get_max_valence(a1, mol)
        max2 = get_max_valence(a2, mol)

        if val1 > max1 or val2 > max2:
            # 降低键类型
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.SINGLE)

    return mol


def fix_3_membered_ring_heteroatoms(mol):
    """
    修复3元环杂原子问题（借鉴DecompDiff reconstruct.py postprocess_rd_mol_2函数）

    对于3元环中的杂原子键：
    - 两个杂原子之间：移除键
    - 两个O原子之间：移除键并添加H

    Returns:
        修复后的分子
    """
    mol_edit = Chem.RWMol(mol)

    ring_info = mol.GetRingInfo()
    rings = [set(r) for r in ring_info.AtomRings()]

    for ring_a in rings:
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}

            for atom_idx in ring_a:
                symb = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)

            # 两个杂原子在3元环中：移除它们之间的键
            if len(non_carbon) == 2:
                try:
                    mol_edit.RemoveBond(non_carbon[0], non_carbon[1])
                except:
                    pass

            # 两个O原子在3元环中：移除键并添加H
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                try:
                    mol_edit.RemoveBond(atom_by_symb['O'][0], atom_by_symb['O'][1])
                    mol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                        mol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                    )
                    mol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                        mol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                    )
                except:
                    pass

    mol = mol_edit.GetMol()

    # 移除所有正电荷（避免后续sanitize问题）
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return mol


def fix_molecule_valence(mol):
    """
    综合价键修复函数，依次应用所有修复策略

    修复顺序（借鉴DecompDiff reconstruct_from_generated_with_bond函数）：
    1. 修复3元环杂原子键
    2. 修复超价原子（降低键类型）
    3. 修复N原子4价问题
    4. 尝试sanitize

    Returns:
        (mol, success): 修复后的分子和是否成功
    """
    if mol is None:
        return None, False

    # Step 1: 修复3元环杂原子
    mol = fix_3_membered_ring_heteroatoms(mol)

    # Step 2: 修复超价原子（降低键类型）
    mol = fix_hypervalent_atoms(mol)

    # Step 3: 修复N原子4价（设置电荷）
    mol, fixed = fix_valence_n4(mol)

    if fixed:
        try:
            Chem.SanitizeMol(mol)
            return mol, True
        except:
            return mol, False

    return mol, False


# =============================================================================
# 3. 离散扩散工具函数（从v2_3迁移）
# =============================================================================
def index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """将离散索引转换为log空间的one-hot表示"""
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes).float()
    log_x = torch.log(x_onehot.clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    """从log空间的one-hot转换回索引"""
    return log_x.argmax(dim=-1)


def log_sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """使用Gumbel-Max采样从log概率分布中采样离散类别"""
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """计算 log(exp(a) + exp(b))，数值稳定"""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_1_min_a(a: np.ndarray) -> np.ndarray:
    """计算 log(1 - exp(a))，数值稳定"""
    return np.log(1 - np.exp(a) + 1e-40)


def categorical_kl(log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
    """计算两个离散分布的KL散度"""
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


class DiscreteTransition(nn.Module):
    """
    离散扩散转移矩阵 - 支持多种Prior模式
    借鉴: DecompDiff/models/transitions.py 和 D3PM
    用于原子类型和键类型的离散扩散生成

    Prior模式:
    - 'uniform': 均匀分布（默认，会导致塌缩）
    - 'data_stats': 数据集统计分布（推荐）
    - 'learnable': 可学习prior（最灵活）
    """
    def __init__(self, num_timesteps: int, num_classes: int, s: float = 0.008,
                 prior_probs: Optional[np.ndarray] = None,
                 prior_mode: str = 'data_stats',
                 learnable_prior_init: Optional[np.ndarray] = None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.prior_mode = prior_mode

        # Cosine schedule（在log空间操作）
        steps = num_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
        alphas = np.clip(alphas, a_min=0.001, a_max=1.)
        alphas = np.sqrt(alphas)  # 使用sqrt，与DecompDiff一致

        # 所有操作在log空间
        log_alphas_v = np.log(alphas)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)

        self.log_alphas_v = nn.Parameter(
            torch.from_numpy(log_alphas_v).float(), requires_grad=False)
        self.log_one_minus_alphas_v = nn.Parameter(
            torch.from_numpy(log_1_min_a(log_alphas_v)).float(), requires_grad=False)
        self.log_alphas_cumprod_v = nn.Parameter(
            torch.from_numpy(log_alphas_cumprod_v).float(), requires_grad=False)
        self.log_one_minus_alphas_cumprod_v = nn.Parameter(
            torch.from_numpy(log_1_min_a(log_alphas_cumprod_v)).float(), requires_grad=False)

        # Prior分布 - 根据模式设置
        if prior_probs is not None:
            # 如果直接提供了prior，使用它
            log_probs = np.log(prior_probs.clip(min=1e-30))
            self.prior_probs = nn.Parameter(
                torch.from_numpy(log_probs).float().unsqueeze(0), requires_grad=False)
            self.prior_mode = 'provided'  # 标记为外部提供
        elif prior_mode == 'uniform':
            # 均匀分布
            uniform_probs = np.full(num_classes, -np.log(num_classes))
            self.prior_probs = nn.Parameter(
                torch.from_numpy(uniform_probs).float().unsqueeze(0), requires_grad=False)
        elif prior_mode == 'learnable':
            # 可学习prior - 初始化为数据统计或均匀分布
            if learnable_prior_init is not None:
                init_probs = learnable_prior_init.clip(min=1e-30)
            else:
                # 默认用数据统计初始化
                init_probs = np.full(num_classes, 1.0 / num_classes)
            log_init = np.log(init_probs)
            self.prior_logits = nn.Parameter(
                torch.from_numpy(log_init).float().unsqueeze(0), requires_grad=True)
            # prior_probs会在forward时动态计算
            self.prior_probs = None  # 不使用固定prior
        else:
            # 默认：data_stats模式，但如果没有提供prior_probs，使用均匀分布
            uniform_probs = np.full(num_classes, -np.log(num_classes))
            self.prior_probs = nn.Parameter(
                torch.from_numpy(uniform_probs).float().unsqueeze(0), requires_grad=False)

    def get_prior_probs(self, device: torch.device) -> torch.Tensor:
        """获取当前prior分布（支持动态计算）"""
        if self.prior_mode == 'learnable':
            # 可学习prior：从logits计算
            return F.log_softmax(self.prior_logits, dim=-1).to(device)
        elif self.prior_probs is not None:
            return self.prior_probs.to(device)
        else:
            # 兜底：均匀分布
            uniform = torch.full((1, self.num_classes), -np.log(self.num_classes), device=device)
            return uniform

    def _extract(self, a: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """从预计算数组中提取对应时间步的值，按batch扩展"""
        a = a.to(t.device)
        return a[t[batch]].unsqueeze(-1)

    def q_v_pred(self, log_v0: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        计算前向扩散概率: q(vt | v0)
        公式: log_prob = log_add_exp(log_alpha_bar * v0, log(1-alpha_bar) * prior)
        """
        log_cumprod_alpha_t = self._extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = self._extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        prior_probs = self.get_prior_probs(log_v0.device)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha + prior_probs
        )
        return log_probs

    def q_v_sample(self, log_v0: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样 vt ~ q(vt | v0)
        返回: (采样索引, log空间的one-hot)
        """
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_pred_one_timestep(self, log_vt_1: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        计算单步转移概率: q(vt | vt-1)
        公式: log_prob = alpha_t * vt-1 + (1-alpha_t) * prior
        """
        log_alpha_t = self._extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = self._extract(self.log_one_minus_alphas_v, t, batch)

        prior_probs = self.get_prior_probs(log_vt_1.device)
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t + prior_probs
        )
        return log_probs

    def q_v_posterior(self, log_v0: torch.Tensor, log_vt: torch.Tensor,
                      t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        计算后验概率: q(vt-1 | vt, v0)
        用于逆向采样过程
        """
        t_minus_1 = t - 1
        # 处理边界情况 t=0
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        log_qvt_vt_1 = self.q_v_pred_one_timestep(log_vt, t, batch)

        # 贝叶斯公式
        unnormed_logprobs = log_qvt1_v0 + log_qvt_vt_1
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0


# =============================================================================
# 3.5 可学习噪声调度（借鉴DiffSBDD en_diffusion.py）
# =============================================================================
class PositiveLinear(nn.Module):
    """
    强制权重为正值的线性层（借鉴DiffSBDD第1031-1061行）

    使用softplus激活确保权重始终为正，用于GammaNetwork单调递增约束
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: float = -2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(nn.Module):
    """
    可学习的单调递增噪声调度网络（借鉴DiffSBDD第1064-1102行）

    Gamma函数定义了噪声水平随时间的变化：
    - gamma(t) = log(alpha^2 / sigma^2) = log(SNR)
    - 单调递增：更多噪声随着时间增加
    - 边界约束：gamma(0)和gamma(1)可学习但受限

    使用场景：
    - 训练时：根据数据自适应调整噪声调度
    - 采样时：使用学习到的最优调度
    """

    def __init__(self, gamma_0: float = -5.0, gamma_1: float = 10.0):
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        # 可学习的边界值
        self.gamma_0 = nn.Parameter(torch.tensor([gamma_0]))
        self.gamma_1 = nn.Parameter(torch.tensor([gamma_1]))

    def gamma_tilde(self, t):
        """中间单调递增函数"""
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        """
        计算gamma(t)，归一化到[gamma_0, gamma_1]范围

        Args:
            t: 时间值，范围[0, 1]

        Returns:
            gamma值，单调递增
        """
        # 计算边界值以归一化
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # 归一化到[0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0 + 1e-8)

        # 重缩放到[gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

    def show_schedule(self, num_steps=50):
        """显示当前的噪声调度曲线"""
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        return gamma.detach().cpu().numpy().reshape(num_steps)


class LearnedDiscreteTransition(nn.Module):
    """
    使用可学习噪声调度的离散扩散转移（GammaNetwork替代固定cosine schedule）

    适用场景：
    - 当数据分布与cosine schedule不匹配时
    - 需要自适应优化噪声水平时
    """

    def __init__(self, num_timesteps: int, num_classes: int,
                 prior_probs: Optional[np.ndarray] = None,
                 use_learned_schedule: bool = False):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.use_learned_schedule = use_learned_schedule

        if use_learned_schedule:
            # 可学习噪声调度
            self.gamma_net = GammaNetwork()
            # gamma转换为alpha: alpha = sqrt(sigmoid(-gamma))
            # log_alpha = 0.5 * logsigmoid(-gamma)
        else:
            # 固定cosine schedule（原版）
            self.gamma_net = None

            s = 0.008
            steps = num_timesteps + 1
            x = np.linspace(0, steps, steps)
            alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
            alphas = np.clip(alphas, a_min=0.001, a_max=1.)
            alphas = np.sqrt(alphas)

            log_alphas_v = np.log(alphas)
            log_alphas_cumprod_v = np.cumsum(log_alphas_v)

            self.log_alphas_v = nn.Parameter(
                torch.from_numpy(log_alphas_v).float(), requires_grad=False)
            self.log_one_minus_alphas_v = nn.Parameter(
                torch.from_numpy(log_1_min_a(log_alphas_v)).float(), requires_grad=False)
            self.log_alphas_cumprod_v = nn.Parameter(
                torch.from_numpy(log_alphas_cumprod_v).float(), requires_grad=False)
            self.log_one_minus_alphas_cumprod_v = nn.Parameter(
                torch.from_numpy(log_1_min_a(log_alphas_cumprod_v)).float(), requires_grad=False)

        # Prior分布 - 支持多种模式
        if prior_probs is not None:
            # 外部提供
            log_probs = np.log(prior_probs.clip(min=1e-30))
            self.prior_probs = nn.Parameter(
                torch.from_numpy(log_probs).float().unsqueeze(0), requires_grad=False)
            self._prior_is_learnable = False
        else:
            # 默认使用均匀分布（LearnedDiscreteTransition暂不支持可学习prior）
            uniform_probs = np.full(num_classes, -np.log(num_classes))
            self.prior_probs = nn.Parameter(
                torch.from_numpy(uniform_probs).float().unsqueeze(0), requires_grad=False)
            self._prior_is_learnable = False

    def get_prior_probs(self, device: torch.device) -> torch.Tensor:
        """获取prior分布"""
        return self.prior_probs.to(device)

    def get_gamma(self, t):
        """获取gamma(t)"""
        if self.use_learned_schedule:
            return self.gamma_net(t.float() / self.num_timesteps)
        else:
            # 固定schedule: gamma = -log(alpha_bar)
            t_int = torch.clamp(t, 0, self.num_timesteps - 1).long()
            return -self.log_alphas_cumprod_v[t_int]

    def _extract(self, a: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """从预计算数组中提取值（固定schedule时使用）"""
        a = a.to(t.device)
        return a[t[batch]].unsqueeze(-1)

    def q_v_pred(self, log_v0: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """前向扩散概率 q(vt | v0)"""
        if self.use_learned_schedule:
            gamma = self.get_gamma(t)
            gamma_batch = gamma[batch].unsqueeze(-1)
            # alpha_bar = sigmoid(-gamma)
            log_alpha_bar = F.logsigmoid(-gamma_batch)
            log_1_minus_alpha_bar = F.logsigmoid(gamma_batch)

            prior_probs = self.prior_probs.to(log_v0.device)
            log_probs = log_add_exp(
                log_v0 + log_alpha_bar,
                log_1_minus_alpha_bar + prior_probs
            )
            return log_probs
        else:
            log_cumprod_alpha_t = self._extract(self.log_alphas_cumprod_v, t, batch)
            log_1_min_cumprod_alpha = self._extract(self.log_one_minus_alphas_cumprod_v, t, batch)

            prior_probs = self.prior_probs.to(log_v0.device)
            log_probs = log_add_exp(
                log_v0 + log_cumprod_alpha_t,
                log_1_min_cumprod_alpha + prior_probs
            )
            return log_probs

    def q_v_sample(self, log_v0: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样 vt ~ q(vt | v0)"""
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_pred_one_timestep(self, log_vt_1: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """单步转移概率 q(vt | vt-1)"""
        if self.use_learned_schedule:
            # 对于可学习schedule，使用累积概率近似
            gamma_t = self.get_gamma(t)
            gamma_t_1 = self.get_gamma(t - 1) if t.min() > 0 else self.get_gamma(torch.zeros_like(t))
            # alpha_t = alpha_bar_t / alpha_bar_{t-1}
            log_alpha_bar_t = F.logsigmoid(-gamma_t[batch].unsqueeze(-1))
            log_alpha_bar_t_1 = F.logsigmoid(-gamma_t_1[batch].unsqueeze(-1))
            log_alpha_t = log_alpha_bar_t - log_alpha_bar_t_1
            log_1_minus_alpha_t = torch.log1p(-torch.exp(log_alpha_t))

            prior_probs = self.prior_probs.to(log_vt_1.device)
            log_probs = log_add_exp(
                log_vt_1 + log_alpha_t,
                log_1_minus_alpha_t + prior_probs
            )
            return log_probs
        else:
            log_alpha_t = self._extract(self.log_alphas_v, t, batch)
            log_1_min_alpha_t = self._extract(self.log_one_minus_alphas_v, t, batch)

            prior_probs = self.prior_probs.to(log_vt_1.device)
            log_probs = log_add_exp(
                log_vt_1 + log_alpha_t,
                log_1_min_alpha_t + prior_probs
            )
            return log_probs

    def q_v_posterior(self, log_v0: torch.Tensor, log_vt: torch.Tensor,
                      t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """后验概率 q(vt-1 | vt, v0)"""
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        log_qvt_vt_1 = self.q_v_pred_one_timestep(log_vt, t, batch)

        unnormed_logprobs = log_qvt1_v0 + log_qvt_vt_1
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0


# =============================================================================
# 4. 特征编码模块（从v2_3迁移）
# =============================================================================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = max(self.dim // 2, 1)
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=time.device, dtype=torch.float32) * -scale
        )
        embeddings = time.float()[:, None] * embeddings[None, :]
        emb = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class GaussianSmearing(nn.Module):
    """高斯核距离编码，用于将距离转换为特征向量"""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 20):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        # 均匀分布的高斯中心点
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (..., 1) 或 (...) 距离值
        Returns:
            (..., num_gaussians) 高斯编码特征
        """
        if dist.dim() > 0 and dist.shape[-1] == 1:
            dist = dist.squeeze(-1)
        dist_expanded = dist.unsqueeze(-1)
        offset_expanded = self.offset.view(1, -1)
        diff = dist_expanded - offset_expanded
        output = torch.exp(self.coeff * torch.pow(diff, 2))
        return output


class CrossAttentionCondition(nn.Module):
    """
    Cross-Attention条件注入模块 - Kimi残差注意力版

    核心改进（借鉴Kimi）：
    1. 输出 = 输入 + 注意力增量（而非替换）
    2. 注意力权重作为调制，保留原始特征
    3. 多头注意力 + 残差连接 + LayerNorm

    与传统Cross-Attention对比：
    - 传统：output = Attention(h, cond)  # 完全替换
    - Kimi：output = h + Attention(h, cond)  # 残差叠加
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # ===== Kimi残差注意力组件 =====
        # Query: 从节点特征h生成
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Key/Value: 从条件向量cond生成
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 残差LayerNorm（Pre-LN风格，更稳定）
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # ===== AdaGN风格的条件调制（辅助）=====
        self.cond_scale = nn.Linear(hidden_dim, hidden_dim)
        self.cond_shift = nn.Linear(hidden_dim, hidden_dim)

        # 投影（用于维度不匹配时）
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """
        Kimi残差注意力前向传播

        Args:
            h: 节点特征 (num_nodes, hidden_dim)
            cond: 条件向量 (batch_size, hidden_dim)
            batch_vec: batch索引 (num_nodes,)

        Returns:
            融合后的节点特征 (num_nodes, hidden_dim)

        流程：
        1. Pre-LN: h_normed = LayerNorm(h)
        2. 扩展条件到每个节点
        3. 多头注意力计算：Q from h, K/V from cond
        4. 残差连接：h_new = h + Attention(h_normed, cond)
        5. AdaGN调制：h_new = scale * h_new + shift
        """
        num_nodes = h.size(0)
        batch_size = cond.size(0)

        h = h.view(num_nodes, -1)
        cond = cond.view(batch_size, -1)

        # 维度对齐
        if h.size(-1) != self.hidden_dim:
            h = self.attn_proj(h)
        if cond.size(-1) != self.hidden_dim:
            cond = self.attn_proj(cond)

        # ===== Pre-LN（Kimi风格，更稳定）=====
        h_normed = self.norm_h(h)

        # 扩展条件到每个节点
        cond_expanded = cond[batch_vec]  # (num_nodes, hidden_dim)

        # ===== 多头注意力计算 =====
        # Query from nodes, Key/Value from condition
        q = self.q_proj(h_normed)  # (num_nodes, hidden_dim)
        k = self.k_proj(cond_expanded)  # (num_nodes, hidden_dim)
        v = self.v_proj(cond_expanded)  # (num_nodes, hidden_dim)

        # 重塑为多头
        q = q.view(num_nodes, self.n_heads, self.head_dim)  # (num_nodes, n_heads, head_dim)
        k = k.view(num_nodes, self.n_heads, self.head_dim)
        v = v.view(num_nodes, self.n_heads, self.head_dim)

        # 计算注意力分数 (scaled dot-product)
        # q·k / sqrt(head_dim)
        attn_score = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)  # (num_nodes, n_heads)

        # Softmax（每个节点独立）
        attn_weights = F.softmax(attn_score, dim=-1)  # (num_nodes, n_heads)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合Value
        attn_out = attn_weights.unsqueeze(-1) * v  # (num_nodes, n_heads, head_dim)
        attn_out = attn_out.view(num_nodes, self.hidden_dim)  # (num_nodes, hidden_dim)

        # 输出投影
        attn_out = self.out_proj(attn_out)

        # ===== Kimi残差连接 =====
        # 核心改进：h_new = h + attn_out，而非 h_new = attn_out
        h_residual = h + attn_out  # 残差叠加，保留原始特征

        # ===== AdaGN调制（辅助）=====
        # scale和shift从条件生成
        scale = torch.sigmoid(self.cond_scale(cond_expanded)) * 2  # 范围约0-2
        shift = self.cond_shift(cond_expanded)

        h_modulated = h_residual * scale + shift

        # Post-LN
        h_out = self.norm_out(h_modulated)

        return h_out


# =============================================================================
# [P2新增] 药效团特征提取与3D Cross-Attention条件注入
# =============================================================================

# 药效团特征类型定义
PHARMACOPHORE_TYPES = {
    'hbond_acceptor': 0,    # 氢键受体（N, O等）
    'hbond_donor': 1,       # 氢键供体（NH, OH等）
    'aromatic': 2,          # 芳香环中心
    'positive': 3,          # 正电荷中心
    'negative': 4,          # 负电荷中心
    'hydrophobic': 5,       # 疏水中心
}
NUM_PHARM_TYPES = 6


def extract_pharmacophore_features(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从分子中提取药效团特征（用于P2 3D条件注入）

    药效团类型：
    - 氢键受体：N(非胺), O(非醇酚), 卤素等
    - 氢键供体：NH, OH
    - 芳香环：苯环等芳香环中心
    - 正/负电荷：带电荷的原子

    Returns:
        pharm_types: 药效团类型向量 (num_pharms, NUM_PHARM_TYPES) - one-hot编码
        pharm_pos: 药效团3D坐标 (num_pharms, 3)
    """
    if mol is None or mol.GetNumConformers() == 0:
        return torch.zeros((0, NUM_PHARM_TYPES)), torch.zeros((0, 3))

    conf = mol.GetConformer()
    pharm_list = []

    # 遍历每个原子，识别药效团特征
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        is_aromatic = atom.GetIsAromatic()
        formal_charge = atom.GetFormalCharge()
        num_h = atom.GetTotalNumHs()

        # 获取坐标
        pos = conf.GetAtomPosition(idx)
        pos_array = np.array([pos.x, pos.y, pos.z])

        # 氢键受体：N(非胺), O(非醇)
        # N: 有孤对电子（非胺的N）
        # O: 非醇/酚的O（羰基、醚等）
        is_acceptor = False
        if atomic_num == 7:  # N
            # 氮原子，检查是否是胺（有H则不是好的受体）
            if num_h == 0:  # 无H的氮是好的受体
                is_acceptor = True
        elif atomic_num == 8:  # O
            # 氧原子，检查是否是醇/酚（有H）
            if num_h == 0:  # 无H的氧是受体（羰基、醚）
                is_acceptor = True

        # 氢键供体：NH, OH
        is_donor = False
        if atomic_num in [7, 8] and num_h > 0:  # N-H 或 O-H
            is_donor = True

        # 芳香原子
        is_aromatic_atom = is_aromatic

        # 正/负电荷
        is_positive = formal_charge > 0
        is_negative = formal_charge < 0

        # 疏水中心：碳原子（非芳香）
        is_hydrophobic = atomic_num == 6 and not is_aromatic

        # 如果有任何药效团特征，添加
        if is_acceptor or is_donor or is_aromatic_atom or is_positive or is_negative or is_hydrophobic:
            pharm_feat = np.zeros(NUM_PHARM_TYPES)
            if is_acceptor:
                pharm_feat[PHARMACOPHORE_TYPES['hbond_acceptor']] = 1.0
            if is_donor:
                pharm_feat[PHARMACOPHORE_TYPES['hbond_donor']] = 1.0
            if is_aromatic_atom:
                pharm_feat[PHARMACOPHORE_TYPES['aromatic']] = 1.0
            if is_positive:
                pharm_feat[PHARMACOPHORE_TYPES['positive']] = 1.0
            if is_negative:
                pharm_feat[PHARMACOPHORE_TYPES['negative']] = 1.0
            if is_hydrophobic:
                pharm_feat[PHARMACOPHORE_TYPES['hydrophobic']] = 1.0

            pharm_list.append((pharm_feat, pos_array))

    # 转换为tensor
    if len(pharm_list) > 0:
        pharm_types = torch.tensor(np.stack([p[0] for p in pharm_list]), dtype=torch.float32)
        pharm_pos = torch.tensor(np.stack([p[1] for p in pharm_list]), dtype=torch.float32)
    else:
        pharm_types = torch.zeros((0, NUM_PHARM_TYPES), dtype=torch.float32)
        pharm_pos = torch.zeros((0, 3), dtype=torch.float32)

    return pharm_types, pharm_pos


class CrossAttentionCondition3D(nn.Module):
    """
    [P2新增] 3D Cross-Attention条件注入模块

    核心改进：
    - Q: 片段噪声原子特征（正在生成的原子）
    - K/V: 骨架节点特征 + 药效团特征（3D环境）

    物理意义：
    每个正在生成的片段原子"看到"骨架的3D环境：
    - 骨架所有原子（了解骨架几何）
    - 药效团特征（知道哪里有氢键受体/供体）
    - 药效团坐标（知道应该往哪里生成）

    这使得生成的片段原子能够：
    - 根据骨架3D几何调整位置
    - 根据药效团位置选择合适的原子类型
    - 自动满足药效团匹配约束
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 pharm_dim: int = NUM_PHARM_TYPES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.pharm_dim = pharm_dim

        assert hidden_dim % n_heads == 0

        # Query投影（片段原子）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # Key/Value投影（骨架节点 + 药效团）
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 药效团特征投影
        self.pharm_feat_proj = nn.Linear(pharm_dim + 3, hidden_dim)  # 类型+坐标

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # 位置编码（用于距离感知）
        self.dist_encoder = GaussianSmearing(0.0, 5.0, 20)

    def forward(self,
                h_fragment: torch.Tensor,
                pos_fragment: torch.Tensor,
                h_scaffold: torch.Tensor,
                pos_scaffold: torch.Tensor,
                pharm_types: Optional[torch.Tensor] = None,
                pharm_pos: Optional[torch.Tensor] = None,
                batch_fragment: torch.Tensor = None) -> torch.Tensor:
        """
        3D Cross-Attention前向传播

        Args:
            h_fragment: 片段原子特征 (num_frag_atoms, hidden_dim)
            pos_fragment: 片段原子坐标 (num_frag_atoms, 3)
            h_scaffold: 骨架节点特征 (num_scaffold_atoms, hidden_dim)
            pos_scaffold: 骨架节点坐标 (num_scaffold_atoms, 3)
            pharm_types: 药效团类型 (num_pharms, NUM_PHARM_TYPES)
            pharm_pos: 药效团坐标 (num_pharms, 3)
            batch_fragment: 片段batch索引 (num_frag_atoms,)

        Returns:
            融合后的片段原子特征 (num_frag_atoms, hidden_dim)
        """
        num_frag = h_fragment.size(0)

        # Normalize Q
        q = self.norm_q(h_fragment)
        q = self.q_proj(q)  # (num_frag, hidden_dim)

        # 构建K/V：骨架节点 + 药效团
        # 骨架节点作为条件
        k_scaffold = self.k_proj(h_scaffold)  # (num_scaffold, hidden_dim)
        v_scaffold = self.v_proj(h_scaffold)

        # 药效团特征（如果有）
        if pharm_types is not None and pharm_pos is not None and pharm_types.size(0) > 0:
            # 药效团特征向量：类型 + 坐标
            pharm_feat = torch.cat([pharm_types, pharm_pos], dim=-1)  # (num_pharms, pharm_dim+3)
            pharm_embed = self.pharm_feat_proj(pharm_feat)  # (num_pharms, hidden_dim)

            k_pharm = self.k_proj(pharm_embed)
            v_pharm = self.v_proj(pharm_embed)

            # 合并骨架和药效团作为K/V
            k = torch.cat([k_scaffold, k_pharm], dim=0)  # (num_scaffold + num_pharms, hidden_dim)
            v = torch.cat([v_scaffold, v_pharm], dim=0)
        else:
            k = k_scaffold
            v = v_scaffold

        num_cond = k.size(0)  # 条件数量

        # 计算距离编码（增强3D感知）
        # 片段原子到每个骨架/药效团位置的距离
        if pos_fragment is not None and pos_scaffold is not None:
            # 计算距离矩阵
            pos_cond = pos_scaffold
            if pharm_pos is not None and pharm_pos.size(0) > 0:
                pos_cond = torch.cat([pos_scaffold, pharm_pos], dim=0)

            # 距离矩阵 (num_frag, num_cond)
            dist_matrix = torch.cdist(pos_fragment, pos_cond)

            # 距离编码
            dist_feat = self.dist_encoder(dist_matrix)  # (num_frag, num_cond, 20)

            # 将距离特征融入注意力计算
            # 在计算q*k时，加入距离权重（近距离更强关注）
            dist_weight = torch.exp(-dist_matrix / 2.0)  # 距离越近权重越高
        else:
            dist_weight = None

        # 多头注意力计算
        q = q.view(num_frag, self.n_heads, self.head_dim)
        k = k.view(num_cond, self.n_heads, self.head_dim)
        v = v.view(num_cond, self.n_heads, self.head_dim)

        # 注意力分数：Q · K^T
        # (num_frag, n_heads, head_dim) x (num_cond, n_heads, head_dim)^T
        # 结果：(num_frag, n_heads, num_cond)
        attn_score = torch.einsum('fhd,chd->fhc', q, k) / math.sqrt(self.head_dim)

        # 加入距离权重
        if dist_weight is not None:
            # (num_frag, num_cond) -> (num_frag, 1, num_cond) -> broadcast to (num_frag, n_heads, num_cond)
            dist_weight_expanded = dist_weight.unsqueeze(1)
            attn_score = attn_score + torch.log(dist_weight_expanded + 1e-8)  # 加上log距离权重

        # Softmax（每个片段原子对所有条件做softmax）
        attn_weights = F.softmax(attn_score, dim=-1)  # (num_frag, n_heads, num_cond)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合Value
        # (num_frag, n_heads, num_cond) x (num_cond, n_heads, head_dim)
        # 结果：(num_frag, n_heads, head_dim)
        attn_out = torch.einsum('fhc,chd->fhd', attn_weights, v)

        # 重塑
        attn_out = attn_out.reshape(num_frag, self.hidden_dim)

        # 输出投影
        attn_out = self.out_proj(attn_out)

        # 残差连接
        h_out = h_fragment + attn_out

        # Post-LN
        h_out = self.norm_out(h_out)

        return h_out

# SMARTS官能团模式定义
SMARTS_PATTERNS = {
    'alcohol': '[OX2H]',  # 醇羟基
    'carboxylic_acid': '[OX2H][CX3]=[OX1]',  # 羧酸
    'amine': '[NX3;H2,H1;!$(NC=[!#6]);!$(NC#[!#6])]',  # 胺基
    'amide': '[NX3][CX3]=[OX1]',  # 酰胺
    'ester': '[OX2][CX3]=[OX1]',  # 酯
    'ketone': '[CX3]=[OX1]',  # 酮（排除酯和酰胺）
    'ether': '[OX2]([#6])[#6]',  # 醚
    'phenol': '[OX2H][c]',  # 苯酚
    'thiol': '[SX2H]',  # 硫醇
    'sulfide': '[SX2]',  # 硫醚
    'nitro': '[NX3](=[OX1])[OX1]',  # 硝基
    'halide': '[F,Cl,Br,I]',  # 卤素
    'aldehyde': '[CX3H]=[OX1]',  # 醛
    'phosphate': '[PX4](=[OX1])([OX2])([OX2])',  # 磷酸
    'sulfone': '[SX4](=[OX1])(=[OX1])',  # 硫酮
    'imine': '[CX2]=[NX2]',  # 亚胺
    'nitrile': '[CX2]#[NX1]',  # 腈
    'aromatic_ring': 'c1ccccc1',  # 苯环
    'furan': 'o1cccc1',  # 呋喃
    'pyridine': 'n1cccc1',  # 吡啶
    'pyrrole': '[nH]1cccc1',  # 吡咯
    'imidazole': 'n1cnc2cc1n2',  # 咪唑
}

# 尝试加载ACCFG
try:
    from accfg import AccFG
    ACCFG_ANALYZER = AccFG(print_load_info=False)
except Exception:
    ACCFG_ANALYZER = None


def _functional_group_indices_smarts(mol: Chem.Mol) -> List[Tuple[str, Tuple[int, ...]]]:
    """使用SMARTS模式识别官能团"""
    results: List[Tuple[str, Tuple[int, ...]]] = []
    for name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        try:
            matches = mol.GetSubstructMatches(patt)
        except Exception:
            continue
        for m in matches:
            if not m:
                continue
            results.append((name, tuple(int(i) for i in m)))
    # 去重
    uniq: List[Tuple[str, Tuple[int, ...]]] = []
    seen = set()
    for name, idxs in results:
        key = (name, tuple(sorted(idxs)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, idxs))
    return uniq


def _functional_group_indices_accfg(smiles: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """使用ACCFG识别官能团"""
    if not CFG.use_accfg_fg_detection or ACCFG_ANALYZER is None:
        return []
    try:
        result = ACCFG_ANALYZER.run(smiles, show_atoms=True, show_graph=False, canonical=False)
        if result is None:
            return []
        fgs = result[0] if isinstance(result, tuple) and len(result) >= 1 else result
        if not fgs:
            return []

        out: List[Tuple[str, Tuple[int, ...]]] = []
        if isinstance(fgs, dict):
            for name, matches in fgs.items():
                if matches is None:
                    continue
                for idx_tuple in matches:
                    idxs = tuple(int(i) for i in idx_tuple)
                    if idxs:
                        out.append((str(name), idxs))
        elif isinstance(fgs, list):
            for item in fgs:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    name = str(item[0])
                    idxs = tuple(int(i) for i in item[1])
                    if idxs:
                        out.append((name, idxs))
        # 去重
        uniq: List[Tuple[str, Tuple[int, ...]]] = []
        seen = set()
        for name, idxs in out:
            key = (name, tuple(sorted(idxs)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((name, idxs))
        return uniq
    except Exception:
        return []


def get_functional_group_indices(mol: Chem.Mol) -> List[Tuple[str, Tuple[int, ...]]]:
    """识别官能团，优先使用ACCFG"""
    if CFG.use_accfg_fg_detection:
        try:
            smiles = Chem.MolToSmiles(mol)
            accfg_res = _functional_group_indices_accfg(smiles) if smiles else []
            if accfg_res:
                return accfg_res
        except Exception:
            pass
    return _functional_group_indices_smarts(mol)


# =============================================================================
# 6. 片段分割模块（核心新功能）
# =============================================================================
class FragmentSplitter:
    """
    使用BRICS规则分割分子为骨架和片段

    BRICS识别的可断键类型：
    - 非环单键连接两个环系统
    - 某些特定官能团连接键

    这些键断开后，分子被分为多个片段。
    我们选择一个片段作为"替换目标"，其余作为"骨架"。
    """

    @staticmethod
    def get_breakable_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
        """
        获取可断键的原子索引对

        根据CFG.use_accfg_decomposition开关选择分解方式：
        - True: 使用ACCFG官能团边界检测（化学合理但很慢）
        - False: 使用拓扑规则（快速，默认）
        """
        if CFG.use_accfg_decomposition:
            return FragmentSplitter._get_breakable_bonds_accfg(mol)
        else:
            return FragmentSplitter._get_breakable_bonds_topology(mol)

    @staticmethod
    def _get_breakable_bonds_topology(mol: Chem.Mol) -> List[Tuple[int, int]]:
        """
        拓扑规则分解（快速，默认方式）

        规则：
        - 非环单键
        - 连接两个至少有2个邻居的原子
        - 不连接氢原子
        """
        try:
            breakable = []

            for bond in mol.GetBonds():
                if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                    continue
                if bond.GetIsAromatic():
                    continue
                if bond.IsInRing():
                    continue

                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()

                atom_begin = mol.GetAtomWithIdx(begin)
                atom_end = mol.GetAtomWithIdx(end)

                if atom_begin.GetAtomicNum() == 1 or atom_end.GetAtomicNum() == 1:
                    continue

                if atom_begin.GetDegree() >= 2 and atom_end.GetDegree() >= 2:
                    breakable.append((begin, end))

            return breakable

        except Exception as e:
            logger.warning(f"Topology bond detection failed: {e}")
            return []

    @staticmethod
    def _get_breakable_bonds_accfg(mol: Chem.Mol) -> List[Tuple[int, int]]:
        """
        ACCFG官能团边界检测（化学合理但很慢）

        核心思想（借鉴v2_3）：
        - ACCFG识别官能团（等排体替换目标）
        - 可断键 = 官能团原子与Core原子之间的键
        - 骨架 = Core（非官能团部分）
        - 片段 = 官能团（等排体替换候选）
        """
        try:
            breakable = []

            # 使用ACCFG识别官能团
            fg_infos = get_functional_group_indices(mol)

            if not fg_infos:
                # ACCFG失败时，使用拓扑规则fallback
                return FragmentSplitter._get_breakable_bonds_topology(mol)

            all_atom_indices = set(range(mol.GetNumAtoms()))

            # 收集所有官能团原子
            fg_atoms_set: Set[int] = set()
            for fg_name, fg_indices in fg_infos:
                fg_atoms_set.update(i for i in fg_indices if 0 <= i < mol.GetNumAtoms())

            if not fg_atoms_set:
                return FragmentSplitter._get_breakable_bonds_topology(mol)

            # Core原子 = 非官能团原子
            core_atoms_set = all_atom_indices - fg_atoms_set

            if not core_atoms_set:
                # 整个分子都是官能团，无法分割
                return []

            # 找官能团与Core之间的键
            for bond in mol.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()

                # 只考虑单键（等排体替换通常涉及单键）
                if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                    continue

                # 检查是否连接官能团和Core
                if (begin in fg_atoms_set and end in core_atoms_set) or \
                   (end in fg_atoms_set and begin in core_atoms_set):
                    breakable.append((begin, end))

            return breakable

        except Exception as e:
            logger.warning(f"ACCFG bond detection failed: {e}, using topology fallback")
            return FragmentSplitter._get_breakable_bonds_topology(mol)

    @staticmethod
    def split_molecule(mol: Chem.Mol, break_bond: Tuple[int, int]) -> List[Set[int]]:
        """
        在指定键处分割分子，返回所有片段的原子集合

        处理多片段情况：选择合适的骨架和片段组合
        """
        # 创建编辑分子
        edit_mol = Chem.RWMol(mol)

        # 断开指定键
        bond = mol.GetBondBetweenAtoms(break_bond[0], break_bond[1])
        if bond is None:
            return None

        edit_mol.RemoveBond(break_bond[0], break_bond[1])
        edit_mol = edit_mol.GetMol()

        # 使用连通性分析获取所有片段
        try:
            frags = Chem.GetMolFrags(edit_mol)

            # 返回所有片段（原子集合列表）
            return [set(frag) for frag in frags]
        except Exception as e:
            return None

    @staticmethod
    def are_adjacent(mol: Chem.Mol, atoms1: Set[int], atoms2: Set[int]) -> bool:
        """
        检查两个原子集合是否相邻（有键直接连接）
        """
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if (begin in atoms1 and end in atoms2) or (end in atoms1 and begin in atoms2):
                return True
        return False

    @staticmethod
    def select_from_multiple_fragments(frags: List[Set[int]], mol: Chem.Mol,
                                        fg_atoms_set: Optional[Set[int]] = None) -> Tuple[Set[int], Set[int]]:
        """
        从多个片段中选择骨架和片段 - ACCFG版

        ACCFG分解逻辑：
        - 如果有fg_atoms_set（ACCFG识别的官能团），使用它确定骨架和片段
        - 骨架 = Core（非官能团部分，通常是主环系统）
        - 片段 = 官能团（等排体替换目标）

        策略：
        1. 如果知道官能团原子集合，直接用它作为片段候选
        2. 最大的非官能团片段作为骨架
        3. 选择一个大小合适的官能团片段作为替换目标
        """
        if len(frags) < 2:
            return None, None

        # 按大小排序
        frags_sorted = sorted(frags, key=len, reverse=True)

        # 如果有官能团原子集合信息（来自ACCFG）
        if fg_atoms_set is not None:
            # 筛选：哪些片段包含官能团原子
            fg_frags = [frag for frag in frags_sorted if frag & fg_atoms_set]
            core_frags = [frag for frag in frags_sorted if not (frag & fg_atoms_set)]

            if not fg_frags or not core_frags:
                # 没有官能团/核心片段，使用原来的逻辑
                return FragmentSplitter._select_from_multiple_fragments_legacy(frags_sorted, mol)

            # 骨架：选择最大的Core片段
            scaffold_candidate = core_frags[0]  # 已按大小排序
            if len(scaffold_candidate) < CFG.min_scaffold_atoms:
                return None, None
            if len(scaffold_candidate) > CFG.max_scaffold_atoms:
                return None, None

            # 片段：选择一个合适大小的官能团片段
            for fg_frag in fg_frags:
                frag_size = len(fg_frag)
                if CFG.min_fragment_atoms <= frag_size <= CFG.max_fragment_atoms:
                    # 检查骨架和片段是否相邻
                    if FragmentSplitter.are_adjacent(mol, scaffold_candidate, fg_frag):
                        return scaffold_candidate, fg_frag

            # 没找到合适大小的官能团片段，尝试任意大小的
            for fg_frag in fg_frags:
                if FragmentSplitter.are_adjacent(mol, scaffold_candidate, fg_frag):
                    return scaffold_candidate, fg_frag

            return None, None

        # 没有官能团信息，使用传统逻辑
        return FragmentSplitter._select_from_multiple_fragments_legacy(frags_sorted, mol)

    @staticmethod
    def _select_from_multiple_fragments_legacy(frags_sorted: List[Set[int]],
                                                 mol: Chem.Mol) -> Tuple[Set[int], Set[int]]:
        """
        传统片段选择逻辑（无ACCFG信息时）
        """
        # 尝试找一个合适大小的骨架
        scaffold_candidate = None
        for frag in frags_sorted:
            frag_size = len(frag)
            if CFG.min_scaffold_atoms <= frag_size <= CFG.max_scaffold_atoms:
                scaffold_candidate = frag
                break

        if scaffold_candidate is None:
            scaffold_candidate = frags_sorted[0]
            if len(scaffold_candidate) > CFG.max_scaffold_atoms:
                return None, None
            if len(scaffold_candidate) < CFG.min_scaffold_atoms:
                return None, None

        # 从剩余片段中找一个合适大小的作为片段
        for frag in frags_sorted:
            if frag == scaffold_candidate:
                continue
            frag_size = len(frag)
            if CFG.min_fragment_atoms <= frag_size <= CFG.max_fragment_atoms:
                if FragmentSplitter.are_adjacent(mol, scaffold_candidate, frag):
                    return scaffold_candidate, frag

        return None, None

    @staticmethod
    def select_fragment(frag1_atoms: Set[int], frag2_atoms: Set[int],
                        mol: Chem.Mol) -> Tuple[Set[int], Set[int]]:
        """
        选择哪个是骨架，哪个是片段

        策略：
        - 较大的作为骨架（保持不变）
        - 较小的作为片段（参与扩散）
        - 但片段不能太小（至少2个原子）
        """
        n1 = len(frag1_atoms)
        n2 = len(frag2_atoms)

        # 片段大小约束
        if n1 < CFG.min_fragment_atoms and n2 < CFG.min_fragment_atoms:
            return None, None

        if n1 < CFG.min_scaffold_atoms and n2 < CFG.min_scaffold_atoms:
            return None, None

        # 选择较大的作为骨架
        if n1 >= n2:
            scaffold_atoms = frag1_atoms
            fragment_atoms = frag2_atoms
        else:
            scaffold_atoms = frag2_atoms
            fragment_atoms = frag1_atoms

        # 验证大小约束
        if len(fragment_atoms) < CFG.min_fragment_atoms:
            return None, None
        if len(fragment_atoms) > CFG.max_fragment_atoms:
            return None, None
        if len(scaffold_atoms) < CFG.min_scaffold_atoms:
            return None, None

        return scaffold_atoms, fragment_atoms

    @staticmethod
    def get_attachment_points(mol: Chem.Mol, scaffold_atoms: Set[int],
                               fragment_atoms: Set[int], break_bond: Tuple[int, int]) -> Tuple[int, int]:
        """
        获取连接点：骨架和片段在断键处的原子

        返回：(scaffold_attachment_atom, fragment_attachment_atom)

        修复：不依赖break_bond，直接在原始分子中找骨架和片段之间的连接键
        """
        # 方法：遍历原始分子的所有键，找到连接骨架和片段的键
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            # 检查这个键是否连接骨架和片段
            if begin in scaffold_atoms and end in fragment_atoms:
                return begin, end
            elif end in scaffold_atoms and begin in fragment_atoms:
                return end, begin

        # 没找到连接键（不应该发生）
        logger.warning(f"No connecting bond found between scaffold and fragment")
        return None, None

    @staticmethod
    def extract_subgraph(mol: Chem.Mol, atom_indices: Set[int]) -> Data:
        """
        从分子中提取子图（骨架或片段），转为PyG Data

        保留原子特征、3D坐标、内部键
        """
        if not atom_indices:
            return None

        # 原子索引排序，建立映射
        sorted_atoms = sorted(atom_indices)
        atom_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_atoms)}

        # 提取原子特征
        features = []
        positions = []
        atom_types_list = []

        for old_idx in sorted_atoms:
            atom = mol.GetAtomWithIdx(old_idx)
            features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalValence(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                int(atom.GetHybridization()),
            ])

            # 3D坐标
            conf = mol.GetConformer()
            pos = conf.GetAtomPosition(old_idx)
            positions.append([pos.x, pos.y, pos.z])

            atom_types_list.append(atom_type_map.get(atom.GetAtomicNum(), 11))

        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        atom_types = torch.tensor(atom_types_list, dtype=torch.long)

        # 提取内部键
        edge_src = []
        edge_dst = []
        bond_types_list = []

        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            # 只保留两个原子都在集合内的键
            if begin in atom_indices and end in atom_indices:
                new_begin = atom_idx_map[begin]
                new_end = atom_idx_map[end]
                edge_src.append(new_begin)
                edge_dst.append(new_end)

                bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
                bond_types_list.append(bond_type)

        if edge_src:
            # 不使用to_undirected，保持单向边存储
            # 每条键只存储一次，避免重建时的重复问题
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            bond_types = torch.tensor(bond_types_list, dtype=torch.long)
            # 确保edge_index和bond_types数量一致
            assert edge_index.size(1) == bond_types.size(0), \
                f"Edge count mismatch: {edge_index.size(1)} edges, {bond_types.size(0)} bond types"
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            bond_types = torch.empty((0,), dtype=torch.long)

        # ========== 价键合理性检查（放宽版） ==========
        # 修复：骨架原子需要预留一个键位给片段连接，所以内部键数必须 <= max_valence - 1
        # 但这只对将要连接片段的原子（连接点）是必须的，其他骨架原子可以内部键数 == max_valence
        # 这里我们放宽检查：只确保内部键数不超过最大价键（不强制预留）
        # 连接点的价键检查在重建阶段处理
        max_valence_map = {1: 1, 5: 3, 6: 4, 7: 5, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1, 0: 4}

        # 计算每个原子在骨架内部的键数
        internal_bond_counts = torch.zeros(len(sorted_atoms), dtype=torch.long)
        for s, d in zip(edge_src, edge_dst):
            internal_bond_counts[s] += 1
            internal_bond_counts[d] += 1

        # 放宽检查：只检查明显超价的情况（内部键数 > 最大价键）
        # 不再要求内部键数 < 最大价键，因为某些原子可能不需要连接片段
        valid_valence = True
        problematic_atoms = []
        for i, old_idx in enumerate(sorted_atoms):
            atom = mol.GetAtomWithIdx(old_idx)
            atom_num = atom.GetAtomicNum()
            max_val = max_valence_map.get(atom_num, 4)
            internal_bonds = internal_bond_counts[i].item()

            # 放宽条件：内部键数可以等于最大价键（后续会选择合适的连接点）
            # 只拒绝明显超价的情况
            if internal_bonds > max_val:
                problematic_atoms.append((old_idx, atom_num, internal_bonds, max_val))
                valid_valence = False

        # 只有真正超价才跳过（内部键数严格大于最大价键）
        if not valid_valence and len(problematic_atoms) > 0:
            # 检查是否只是"等于最大价键"而非"超过"
            real_over_valence = [p for p in problematic_atoms if p[2] > p[3]]
            if len(real_over_valence) > 0:
                logger.warning(f"Scaffold valence check failed (over-valence): {real_over_valence[:5]}")
                return None  # 只在真正超价时跳过

        # 创建Data对象
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            bond_types=bond_types,
            atom_types=atom_types,
            mol=mol,  # 保留原始mol用于指纹计算
            original_indices=torch.tensor(sorted_atoms, dtype=torch.long),
        )

        return data

    @classmethod
    def decompose(cls, mol: Chem.Mol) -> Optional[Dict]:
        """
        完整分解：分子 -> (骨架, 片段, 连接点) - ACCFG版

        ACCFG分解流程：
        1. 使用ACCFG识别官能团（等排体替换目标）
        2. 检测官能团与Core之间的可断键
        3. 断开可断键，分割分子
        4. Core作为骨架，官能团作为片段

        返回：
        {
            'scaffold': Data,       # 骨架图（Core）
            'fragment': Data,       # 片段图（官能团）
            'attachment': Tuple[int, int],  # (骨架连接点原始索引, 片段连接点原始索引)
            'break_bond': Tuple[int, int],  # 断键位置
            'original_mol': Chem.Mol,
            'fg_info': Tuple[str, Tuple],  # 官能团信息（名称，原子索引）
        }
        """
        if mol is None or mol.GetNumConformers() == 0:
            return None

        # Step 1: 使用ACCFG识别官能团
        fg_infos = get_functional_group_indices(mol)

        if not fg_infos:
            # 没有识别到官能团，跳过
            logger.debug(f"No functional groups detected for molecule")
            return None

        # 收集所有官能团原子
        fg_atoms_set: Set[int] = set()
        for fg_name, fg_indices in fg_infos:
            fg_atoms_set.update(i for i in fg_indices if 0 <= i < mol.GetNumAtoms())

        if not fg_atoms_set:
            return None

        # Step 2: 获取可断键（官能团与Core之间的键）
        breakable_bonds = cls.get_breakable_bonds(mol)
        if not breakable_bonds:
            logger.debug(f"No breakable bonds between FG and Core")
            return None

        # Step 3: 尝试每个可断键，找一个有效的分割
        for break_bond in breakable_bonds:
            frags = cls.split_molecule(mol, break_bond)
            if frags is None or len(frags) < 2:
                continue

            # Step 4: 从多个片段中选择骨架和片段（传入官能团信息）
            scaffold_atoms, fragment_atoms = cls.select_from_multiple_fragments(
                frags, mol, fg_atoms_set
            )
            if scaffold_atoms is None:
                continue

            # Step 5: 获取连接点
            attachment = cls.get_attachment_points(mol, scaffold_atoms, fragment_atoms, break_bond)
            if attachment is None:
                continue

            # Step 6: 提取子图
            scaffold_data = cls.extract_subgraph(mol, scaffold_atoms)
            fragment_data = cls.extract_subgraph(mol, fragment_atoms)

            if scaffold_data is None or fragment_data is None:
                continue

            # Step 7: 找到片段对应的官能团名称
            fg_name_for_fragment = None
            for fg_name, fg_indices in fg_infos:
                fg_indices_set = set(fg_indices)
                if fg_indices_set == fragment_atoms or fg_indices_set.issubset(fragment_atoms):
                    fg_name_for_fragment = fg_name
                    break

            # 成功分解
            return {
                'scaffold': scaffold_data,
                'fragment': fragment_data,
                'attachment': attachment,  # (scaffold_attach_idx, fragment_attach_idx) 原始索引
                'break_bond': break_bond,
                'original_mol': mol,
                'fg_info': (fg_name_for_fragment, tuple(fragment_atoms)),
            }

        # 没有找到有效分割
        return None


# =============================================================================
# 4. 数据集
# =============================================================================
class FragmentDataset(Dataset):
    """
    片段数据集：从分子中提取骨架-片段配对

    与v2_3的ChEMBLDataset不同：
    - v2_3: 加载完整分子
    - v2_4: 加载分解后的骨架-片段配对
    - [新增] 稀有原子片段权重：过采样包含N、F、P、Cl、Br等稀有原子的片段
    """

    # 稀有原子类型（索引：N=3, F=5, P=6, Cl=8, Br=9, I=10）
    RARE_ATOM_INDICES = {3, 5, 6, 8, 9, 10}

    def __init__(self, data_dir: str, max_mols: Optional[int] = None):
        self.data_dir = data_dir
        self.decomposed_list: List[Dict] = []
        self.sample_weights: List[float] = []  # [新增] 样本权重列表

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        # [P0修改] 支持从单个大SDF文件读取（如chembl_36.sdf）
        if os.path.isfile(data_dir) and data_dir.endswith('.sdf'):
            # 单个大SDF文件模式
            files = [data_dir]
            logger.info(f"Loading from single large SDF file: {data_dir}")
        else:
            # 目录模式：遍历多个SDF文件
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.sdf')])
            if not files:
                logger.warning(f"No .sdf files in: {data_dir}")
                return

        total_mols = 0
        valid_decomps = 0
        fg_type_counts = defaultdict(int)  # 统计各类官能团数量

        limit = max_mols or (CFG.test_mode_limit if CFG.test_mode else None)

        # 创建分子级别进度条
        pbar = tqdm(desc="Loading molecules", unit="mol")

        for f in files:
            if limit and total_mols >= limit:
                break

            # [P0修复] 处理文件路径：
            # - 如果files列表中只有data_dir本身（单个大文件模式），直接使用data_dir
            # - 否则拼接目录路径
            if len(files) == 1 and files[0] == data_dir:
                sdf_path = data_dir
            elif os.path.isfile(f) and os.path.isabs(f):
                sdf_path = f
            else:
                sdf_path = os.path.join(data_dir, f)

            # [P0修改] 大文件时使用多线程供应商（如果可用）
            try:
                # 尝试使用ForwardSDMolSupplier以流式读取大文件
                suppl = Chem.ForwardSDMolSupplier(sdf_path, removeHs=False)
            except:
                suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)

            for mol in suppl:
                if limit and total_mols >= limit:
                    break

                if mol is None:
                    continue

                total_mols += 1

                # 尝试分解
                decomp = FragmentSplitter.decompose(mol)
                if decomp is not None:
                    # 计算骨架指纹（作为context）
                    scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(
                        decomp['scaffold'].mol, radius=2, nBits=2048
                    )
                    # ★ [V5修复] 同时保存原始指纹（用于TanimotoSimilarity）和tensor版本（用于模型）
                    decomp['scaffold_fp_raw'] = scaffold_fp  # ExplicitBitVect，用于相似度计算
                    fp_array = np.zeros((2048,))
                    Chem.DataStructs.ConvertToNumpyArray(scaffold_fp, fp_array)
                    decomp['scaffold_fp'] = torch.tensor(fp_array, dtype=torch.float32)  # tensor，用于模型

                    # 计算片段指纹（用于相似度匹配）
                    fragment_fp = AllChem.GetMorganFingerprintAsBitVect(
                        decomp['fragment'].mol, radius=2, nBits=2048
                    )
                    decomp['fragment_fp'] = fragment_fp  # ExplicitBitVect，用于相似度计算

                    # 统计官能团类型
                    if 'fg_info' in decomp and decomp['fg_info'][0] is not None:
                        fg_type_counts[decomp['fg_info'][0]] += 1

                    # [新增] 计算片段稀有原子权重 - 使用 atom_types 字段（已经是类型索引）
                    frag_atom_types = decomp['fragment'].atom_types.tolist()
                    rare_atom_count = sum(1 for at in frag_atom_types if at in self.RARE_ATOM_INDICES)
                    # 权重公式：基础权重1.0 + 稀有原子数量*5.0（每增加一个稀有原子，权重增加5倍）
                    weight = 1.0 + rare_atom_count * 5.0
                    self.sample_weights.append(weight)

                    self.decomposed_list.append(decomp)
                    valid_decomps += 1

                # 更新进度条显示
                success_rate = valid_decomps / total_mols * 100 if total_mols > 0 else 0
                pbar.set_postfix(valid=valid_decomps, rate=f"{success_rate:.1f}%")
                if limit:
                    pbar.total = limit
                pbar.update(1)

        pbar.close()

        logger.info(f"Total molecules: {total_mols}, Valid decompositions: {valid_decomps}")
        if fg_type_counts:
            logger.info(f"Functional group distribution: {dict(fg_type_counts)}")

    def __len__(self):
        return len(self.decomposed_list)

    def __getitem__(self, idx):
        return self.decomposed_list[idx]

    def get_sample_weight(self, idx):
        """[新增] 返回样本权重，用于过采样稀有原子片段"""
        return self.sample_weights[idx]


def compute_scaffold_similarity(fp1, fp2) -> float:
    """计算骨架指纹的Tanimoto相似度"""
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# =============================================================================
# 5. 片段配对挖掘（Stage 2）
# =============================================================================
def mine_fragment_pairs(dataset: FragmentDataset,
                        max_pairs: int = 100000,
                        fp_threshold: float = 0.45) -> List[Tuple[int, int]]:
    """
    挖掘相似骨架上的片段配对

    逻辑：
    - 找骨架相似的分子（Tanimoto > threshold）
    - 这些分子的片段可以互换，形成等排体替换
    """
    logger.info("Mining fragment pairs...")

    pairs = []
    n = len(dataset)

    # 首先收集所有骨架指纹
    scaffold_fps = []
    for i in range(n):
        decomp = dataset[i]
        scaffold_fps.append(decomp['fragment_fp'])  # 用片段指纹来匹配

    # 随机采样配对（避免O(n^2)复杂度）
    sample_size = min(n * n, CFG.stage2_max_pairs)
    checked = 0

    while len(pairs) < max_pairs and checked < sample_size:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        if i == j:
            continue

        checked += 1

        # 计算骨架相似度（实际上比较片段指纹）
        sim = compute_scaffold_similarity(scaffold_fps[i], scaffold_fps[j])

        if sim > fp_threshold:
            # 验证骨架确实相似（原子数相近）
            decomp_i = dataset[i]
            decomp_j = dataset[j]

            # 骨架原子数应该相近
            scaffold_size_diff = abs(
                decomp_i['scaffold'].x.size(0) - decomp_j['scaffold'].x.size(0)
            )

            if scaffold_size_diff <= 3:  # 允许最多相差3个原子
                pairs.append((i, j))

    logger.info(f"Mined {len(pairs)} fragment pairs from {checked} comparisons")
    return pairs[:max_pairs]


def mine_isostere_pairs(dataset: FragmentDataset,
                       max_pairs: int = 100000) -> List[Tuple[int, int]]:
    """
    ★ [V5新增] 挖掘真正的等排体配对

    等排体定义：骨架相似 + 片段不同
    - 这意味着两个分子属于同一类（骨架相似），但发生了片段替换

    标准：
    - scaffold Tanimoto > 0.5  （骨架相似，说明是同类分子）
    - fragment Tanimoto < 0.4  （片段不同，说明发生了替换）
    - fragment atom count 差异 <= 3 （大小相近，确保是合理替换）
    """
    logger.info("Mining TRUE isostere pairs (scaffold-similar, fragment-different)...")

    n = len(dataset)
    # ★ [V5修复] 使用 scaffold_fp_raw (ExplicitBitVect) 而非 scaffold_fp (tensor)
    scaffold_fps = [dataset[i]['scaffold_fp_raw'] for i in range(n)]
    fragment_fps = [dataset[i]['fragment_fp'] for i in range(n)]
    frag_sizes = [dataset[i]['fragment'].x.size(0) for i in range(n)]

    pairs = []
    checked = 0
    max_checks = max_pairs * 50

    while len(pairs) < max_pairs and checked < max_checks:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        checked += 1

        # 条件1：骨架相似（这是等排体的基础 - 同类分子）
        scaffold_sim = DataStructs.TanimotoSimilarity(scaffold_fps[i], scaffold_fps[j])
        if scaffold_sim < CFG.isostere_scaffold_threshold:  # 0.5
            continue

        # 条件2：片段不同（这是等排体的核心 - 发生了替换）
        frag_sim = DataStructs.TanimotoSimilarity(fragment_fps[i], fragment_fps[j])
        if frag_sim >= CFG.isostere_fragment_threshold:  # 0.4 - 太相似就不是等排体
            continue

        # 条件3：片段大小相近（确保是合理的替换）
        if abs(frag_sizes[i] - frag_sizes[j]) > CFG.isostere_atom_count_diff:  # 3
            continue

        pairs.append((i, j))

    logger.info(f"Found {len(pairs)} isostere pairs from {checked} checks")

    # 统计配对质量
    if pairs:
        sample_pairs = pairs[:min(100, len(pairs))]
        scaffold_sims = [DataStructs.TanimotoSimilarity(scaffold_fps[i], scaffold_fps[j])
                         for i, j in sample_pairs]
        frag_sims = [DataStructs.TanimotoSimilarity(fragment_fps[i], fragment_fps[j])
                     for i, j in sample_pairs]
        logger.info(f"  Scaffold similarity: avg={np.mean(scaffold_sims):.3f} (should be >0.5)")
        logger.info(f"  Fragment similarity: avg={np.mean(frag_sims):.3f} (lower=better for isosteres)")
        logger.info(f"  Fragment diversity: avg={1-np.mean(frag_sims):.3f}")

    return pairs[:max_pairs]


# =============================================================================
# 6. 扩散模型组件（简化版，只扩散片段）
# =============================================================================
class GaussianSmearing(nn.Module):
    """距离的高斯展开"""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (0.5 ** 2)  # 固定宽度
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SinusoidalPositionalEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(max_positions)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, t: torch.Tensor):
        emb = self.emb.unsqueeze(0) * t.unsqueeze(1).float()
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ScaffoldEncoder(nn.Module):
    """
    3D骨架编码器 - 使用EGNN编码骨架的3D结构

    替代简单的指纹编码，保留骨架的空间信息
    输出: 骨架的3D结构编码向量 (hidden_dim)

    [改进] 使用特征敏感的Attention Pooling + 方差正则化
    """

    def __init__(self, node_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3,
                 edge_dim: int = 20):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # 输入投影
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # 距离编码
        self.dist_encoder = GaussianSmearing(0.0, 5.0, edge_dim)

        # EGNN层
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(EGNNLayer(hidden_dim, edge_dim))

        # [改进] 特征敏感的Attention Pooling
        # 使用原子特征的多个维度计算attention，增强区分性
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU避免梯度消失
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
        )
        # 初始化：让最后一层有较大的bias范围，产生更明显的权重差异
        nn.init.xavier_uniform_(self.attention_pool[0].weight, gain=2.0)
        nn.init.xavier_uniform_(self.attention_pool[2].weight, gain=2.0)
        # 让最后一层有随机bias，不同原子会有不同初始权重
        self.attention_pool[4].bias.data.uniform_(-0.5, 0.5)

        # 输出层 - 将节点特征聚合为骨架编码
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 骨架原子特征 (num_atoms, node_dim)
            pos: 骨架原子坐标 (num_atoms, 3)
            edge_index: 边索引 (2, num_edges)
            batch: batch向量 (num_atoms,)

        Returns:
            骨架编码 (batch_size, hidden_dim)
        """
        num_atoms = x.size(0)

        # 边索引边界检查
        if edge_index.size(1) > 0:
            edge_index = edge_index.clamp(0, num_atoms - 1)

        # 初始节点特征
        h = self.node_proj(x)

        # EGNN层迭代
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, num_atoms)

        # 全局池化 - 使用Attention Pooling
        if batch is None:
            batch = torch.zeros(num_atoms, dtype=torch.long, device=x.device)

        # [改进] Attention-based pooling
        # 计算每个节点的注意力权重
        attn_logits = self.attention_pool(h)  # (num_atoms, 1)

        # 按batch分组计算softmax
        attn_weights = self._batch_softmax(attn_logits, batch)  # (num_atoms, 1)

        # 加权聚合
        weighted_h = h * attn_weights  # (num_atoms, hidden_dim)
        scaffold_encoding = global_mean_pool(weighted_h, batch)  # (batch_size, hidden_dim)

        # 为了保持权重总和为1的效果，需要除以实际分子数
        # 但global_mean_pool已经按batch聚合，这里直接使用结果

        scaffold_encoding = self.output_proj(scaffold_encoding)

        return scaffold_encoding

    def _batch_softmax(self, logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        按batch分组计算softmax

        Args:
            logits: (num_atoms, 1) 注意力logits
            batch: (num_atoms,) batch索引

        Returns:
            weights: (num_atoms, 1) softmax后的权重
        """
        # 简化实现：使用纯PyTorch按batch计算softmax
        weights = torch.zeros_like(logits)
        unique_batches = batch.unique()

        for b in unique_batches:
            mask = batch == b
            batch_logits = logits[mask]
            # softmax
            batch_softmax = torch.softmax(batch_logits, dim=0)
            weights[mask] = batch_softmax

        return weights


class FragmentEGNN(nn.Module):
    """
    完整版EGNN，用于片段扩散

    基于EGNN论文的标准架构：
    - 消息函数: m_ij = φ_e(h_i, h_j, dist^2, edge_attr)
    - 节点更新: h_i' = φ_h(h_i, Σ_j m_ij)
    - 坐标更新: x_i' = x_i + Σ_j (x_i - x_j) * φ_x(m_ij)
    """

    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int = 4,
                 edge_dim: int = 20):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        # 输入投影
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Context嵌入（骨架编码）
        self.context_embed = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 距离编码
        self.dist_encoder = GaussianSmearing(0.0, 5.0, edge_dim)

        # EGNN层 - 完整版
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(EGNNLayer(hidden_dim, edge_dim))

        # 输出头
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.atom_type_head = nn.Linear(hidden_dim, len(ATOM_TYPES))
        self.bond_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, CFG.num_bond_types),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x: 原子特征 (num_atoms, node_dim)
            pos: 原子坐标 (num_atoms, 3)
            edge_index: 边索引 (2, num_edges)
            t: 时间步 (batch_size,)
            context: 骨架指纹 (batch_size, 2048)
            batch: batch向量 (num_atoms,)
        """
        num_atoms = x.size(0)

        # 边索引边界检查
        if edge_index.size(1) > 0:
            edge_index = edge_index.clamp(0, num_atoms - 1)
            valid_mask = (edge_index[0] < num_atoms) & (edge_index[1] < num_atoms) & \
                         (edge_index[0] >= 0) & (edge_index[1] >= 0)
            edge_index = edge_index[:, valid_mask]

        # 初始节点特征
        h = self.node_proj(x)

        # 时间和context嵌入
        batch_size = t.size(0) if t is not None and t.dim() > 0 else 1
        t_safe = t.clamp(0, CFG.stage3_timesteps - 1) if t is not None and t.dim() > 0 else \
                 torch.zeros(batch_size, dtype=torch.long, device=x.device)

        if batch is None or batch.dim() == 0:
            batch_safe = torch.zeros(num_atoms, dtype=torch.long, device=x.device)
        else:
            batch_safe = batch.clamp(0, batch_size - 1)

        t_emb = self.time_embed(t_safe)
        t_emb_per_atom = t_emb[batch_safe]

        c_emb = self.context_embed(context)
        c_emb_per_atom = c_emb[batch_safe]

        # 融合时间和context
        h = h + t_emb_per_atom + c_emb_per_atom

        # EGNN层迭代
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, num_atoms)

        # 输出预测
        pos_pred = self.pos_head(h)
        atom_type_logits = self.atom_type_head(h)

        # 键类型预测 - 只预测unique边的键类型
        bond_type_logits = None
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]  # [修复] 直接获取src和dst
            src = src.clamp(max=num_atoms - 1)
            dst = dst.clamp(max=num_atoms - 1)
            # 只取unique边（src < dst）
            unique_mask = src < dst
            src_unique = src[unique_mask]
            dst_unique = dst[unique_mask]

            if src_unique.size(0) > 0:
                delta_pos = pos[dst_unique] - pos[src_unique]
                dist = torch.norm(delta_pos, dim=-1, keepdim=True).squeeze(-1)
                dist_feat = self.dist_encoder(dist)

                bond_input = torch.cat([h[src_unique], h[dst_unique], dist_feat], dim=-1)
                bond_type_logits = self.bond_type_head(bond_input)

        return pos_pred, atom_type_logits, bond_type_logits


class EGNNLayer(nn.Module):
    """
    单个EGNN层 - 完整消息传递

    注意：这是消息传递层，不应与注意力模块混淆
    Kimi残差注意力在CrossAttentionCondition中实现
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # 边消息函数 φ_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 坐标更新函数 φ_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 节点更新函数 φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 距离编码
        self.dist_encoder = GaussianSmearing(0.0, 5.0, edge_dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, num_atoms: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: 节点特征 (num_atoms, hidden_dim)
            pos: 坐标 (num_atoms, 3)
            edge_index: 边索引 (2, num_edges)
            num_atoms: 原子数

        Returns:
            h_new: 更新后的节点特征
            pos_new: 更新后的坐标
        """
        if edge_index.size(1) == 0:
            return h, pos

        src, dst = edge_index[0], edge_index[1]  # [修复] 直接获取src和dst
        src = src.clamp(max=num_atoms - 1)
        dst = dst.clamp(max=num_atoms - 1)

        # 边特征：节点特征 + 距离
        hi, hj = h[src], h[dst]
        delta_pos = pos[dst] - pos[src]
        dist_sq = (delta_pos ** 2).sum(dim=-1, keepdim=True)
        dist_feat = self.dist_encoder(dist_sq.squeeze(-1))

        # 消息 m_ij
        edge_input = torch.cat([hi, hj, dist_feat], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (num_edges, hidden_dim)

        # 节点更新：聚合所有入边消息
        agg_msg_src = torch.zeros_like(h)
        agg_msg_dst = torch.zeros_like(h)

        agg_msg_src.index_add_(0, src.long(), m_ij)
        agg_msg_dst.index_add_(0, dst.long(), m_ij)

        agg_msg = agg_msg_src + agg_msg_dst

        # 节点更新 h_i' = h + φ_h(h_i, Σ_j m_ij)
        node_input = torch.cat([h, agg_msg], dim=-1)
        h_new = h + self.node_mlp(node_input)  # 残差连接

        # 坐标更新
        coord_scale = self.coord_mlp(m_ij)
        coord_update = delta_pos * coord_scale

        pos_new = pos.clone()
        pos_new.index_add_(0, src.long(), coord_update)
        pos_new.index_add_(0, dst.long(), -coord_update)

        return h_new, pos_new


class FragmentDiffusion(nn.Module):
    """
    片段扩散模型 - 升级版（支持多种Prior模式）

    只扩散片段原子，骨架编码作为条件
    新增：
    - 离散扩散（DiscreteTransition）
    - Cross-Attention条件注入
    - 价键约束损失
    - 多种Prior模式切换（uniform/data_stats/learnable/conditional）
    """

    def __init__(self, hidden_dim: int = 128, num_timesteps: int = 100,
                 num_atom_types: int = len(ATOM_TYPES), use_discrete: bool = True,
                 prior_mode: str = CFG.prior_mode):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.num_atom_types = num_atom_types
        self.use_discrete = use_discrete
        self.prior_mode = prior_mode

        # EGNN - 使用完整版
        self.egnn = FragmentEGNN(
            node_dim=6,
            hidden_dim=hidden_dim,
            num_layers=4,
            edge_dim=20,
        )

        # Cross-Attention条件注入
        self.cross_attention = CrossAttentionCondition(hidden_dim, n_heads=4)
        self.context_proj = nn.Linear(2048, hidden_dim)  # 指纹投影
        self.scaffold_3d_proj = nn.Linear(hidden_dim, hidden_dim)  # 3D编码投影

        # [P2新增] Scaffold节点投影（用于3D Cross-Attention）
        self.scaffold_node_proj = nn.Linear(6, hidden_dim)  # 骨架原子特征投影

        # [P2新增] 3D Cross-Attention（骨架节点 + 药效团）
        self.cross_attention_3d = CrossAttentionCondition3D(hidden_dim, n_heads=4)

        # [新增] 条件相关Prior网络（用于prior_mode='conditional'）
        if prior_mode == 'conditional':
            self.scaffold_to_prior_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_atom_types),
            )
            logger.info(f"[Prior] 使用条件相关Prior模式 (conditional)")

        # 离散扩散转移矩阵 - 根据prior_mode初始化
        if use_discrete:
            # 获取prior分布
            atom_prior = self._get_atom_prior(prior_mode, num_atom_types)
            bond_prior = self._get_bond_prior(prior_mode, CFG.num_bond_types)

            self.atom_type_trans = DiscreteTransition(
                num_timesteps=num_timesteps,
                num_classes=num_atom_types,
                s=0.008,
                prior_probs=atom_prior,
                prior_mode=prior_mode,
            )
            self.bond_type_trans = DiscreteTransition(
                num_timesteps=num_timesteps,
                num_classes=CFG.num_bond_types,
                s=0.008,
                prior_probs=bond_prior,
                prior_mode=prior_mode,
            )

            # 记录Prior信息
            logger.info(f"[Prior] 原子类型Prior模式: {prior_mode}")
            if atom_prior is not None:
                logger.info(f"[Prior] 原子类型Prior分布: {atom_prior}")
            logger.info(f"[Prior] 键类型Prior模式: {prior_mode}")
            if bond_prior is not None:
                logger.info(f"[Prior] 键类型Prior分布: {bond_prior}")

        # 扩散参数（坐标使用连续扩散）
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # 后验方差
        posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas)
        self.register_buffer('posterior_variance', posterior_variance)

        # 原子类型预测头（离散扩散用）
        self.atom_type_head = nn.Linear(hidden_dim, num_atom_types)

        # 原子特征投影（用于Cross-Attention）
        self.x_feat_proj = nn.Linear(num_atom_types, hidden_dim)

        # [新增V3] 位置坐标Prior初始化
        self.position_prior = create_prior(
            n_dim=3,
            in_node_nf=num_atom_types,
            mode=CFG.position_prior_mode,
            prior_type=CFG.position_prior_type,
            custom_sigma_x=CFG.position_prior_sigma,
        )
        logger.info(f"[Prior] 位置坐标Prior模式: {CFG.position_prior_mode}, 类型: {CFG.position_prior_type}")

    def _get_atom_prior(self, prior_mode: str, num_classes: int) -> Optional[np.ndarray]:
        """根据模式获取原子类型prior分布"""
        if prior_mode == 'uniform':
            # 均匀分布（会导致塌缩）
            return None
        elif prior_mode == 'data_stats':
            # 数据集统计分布
            # 确保长度匹配
            prior = DATA_STATS_ATOM_PRIOR.copy()
            if len(prior) < num_classes:
                # 补充缺失的类别（用较小的概率）
                prior = np.concatenate([prior, np.full(num_classes - len(prior), 0.01)])
            elif len(prior) > num_classes:
                prior = prior[:num_classes]
            return prior / prior.sum()  # 归一化
        elif prior_mode == 'learnable':
            # 可学习prior，初始化为数据统计
            return self._get_atom_prior('data_stats', num_classes)
        elif prior_mode == 'conditional':
            # 条件相关prior，基础prior用数据统计
            return self._get_atom_prior('data_stats', num_classes)
        else:
            return None

    def _get_bond_prior(self, prior_mode: str, num_classes: int) -> Optional[np.ndarray]:
        """根据模式获取键类型prior分布"""
        if prior_mode == 'uniform':
            return None
        elif prior_mode == 'data_stats':
            prior = DATA_STATS_BOND_PRIOR.copy()
            if len(prior) < num_classes:
                prior = np.concatenate([prior, np.full(num_classes - len(prior), 0.01)])
            elif len(prior) > num_classes:
                prior = prior[:num_classes]
            return prior / prior.sum()
        elif prior_mode == 'learnable':
            return self._get_bond_prior('data_stats', num_classes)
        elif prior_mode == 'conditional':
            return self._get_bond_prior('data_stats', num_classes)
        else:
            return None

    def get_conditional_prior(self, scaffold_context: torch.Tensor,
                               batch: torch.Tensor) -> torch.Tensor:
        """
        [新增] 根据骨架context动态调整原子类型prior

        Args:
            scaffold_context: 骨架编码 (batch_size, hidden_dim)
            batch: 原子所属的batch索引 (num_atoms,)

        Returns:
            log_prior: 调整后的log prior分布 (num_atoms, num_atom_types)
        """
        if self.prior_mode != 'conditional' or not hasattr(self, 'scaffold_to_prior_net'):
            # 非条件模式，返回固定prior
            return self.atom_type_trans.get_prior_probs(scaffold_context.device).expand(
                batch.size(0), -1
            )

        # 从骨架context预测prior调整
        # scaffold_context: (batch_size, hidden_dim)
        prior_logits = self.scaffold_to_prior_net(scaffold_context)  # (batch_size, num_atom_types)
        log_prior_adjustment = F.log_softmax(prior_logits, dim=-1)

        # 获取基础prior
        base_prior = self.atom_type_trans.get_prior_probs(scaffold_context.device)  # (1, num_classes)

        # 混合：基础prior * 0.7 + 调整prior * 0.3
        log_conditional_prior = torch.log(
            0.7 * base_prior.exp() + 0.3 * log_prior_adjustment.exp()
        )

        # 按batch扩展到每个原子
        log_prior_per_atom = log_conditional_prior[batch]  # (num_atoms, num_atom_types)

        return log_prior_per_atom

    def _sample_position_prior_noise(self, num_atoms: int, device: torch.device,
                                       batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        [新增V3] 从位置Prior采样噪声

        Args:
            num_atoms: 原子数量
            device: 设备
            batch: batch索引 (num_atoms,)

        Returns:
            noise: 位置噪声 (num_atoms, 3)
        """
        # 创建node_mask用于center-of-gravity约束
        node_mask = torch.ones(1, num_atoms, 1, device=device)  # (batch=1, num_atoms, 1)

        # 根据Prior模式采样
        if self.position_prior.mode == 'custom':
            # Custom模式：使用配置的sigma
            sampler = self.position_prior.position_sampler
            noise = sampler.sample((1, num_atoms, 3), device, node_mask).squeeze(0)  # (num_atoms, 3)
        elif self.position_prior.mode == 'learned':
            # Learned模式：使用可学习参数采样
            z_x, z_h = self.position_prior.sample(1, num_atoms, node_mask)
            noise = z_x.squeeze(0)  # (num_atoms, 3)
        elif self.position_prior.mode == 'conditional':
            # Conditional模式：条件注入（这里暂时使用custom作为fallback）
            sampler = self.position_prior.position_sampler
            noise = sampler.sample((1, num_atoms, 3), device, node_mask).squeeze(0)
        else:
            # 默认标准正态分布
            noise = torch.randn(num_atoms, 3, device=device)

        # 确保center-of-gravity为零（E(n)等变约束）
        noise = remove_mean_with_mask(noise.unsqueeze(0), node_mask).squeeze(0)

        return noise

    def _sample_position_prior(self, num_atoms: int, device: torch.device) -> torch.Tensor:
        """
        [新增V3] 从位置Prior采样初始位置（用于sample方法）

        Args:
            num_atoms: 原子数量
            device: 设备

        Returns:
            pos: 初始位置 (num_atoms, 3)
        """
        # 采样噪声作为初始位置
        pos = self._sample_position_prior_noise(num_atoms, device)

        # 归一化（center-of-gravity为零）
        pos = pos - pos.mean(dim=0)

        return pos

    def q_sample_pos(self, pos_start: torch.Tensor, t: torch.Tensor,
                     noise: Optional[torch.Tensor] = None,
                     batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """坐标前向扩散"""
        if noise is None:
            noise = self._sample_position_prior_noise(pos_start.size(0), pos_start.device, batch)

        num_atoms = pos_start.size(0)

        # 使用batch向量来扩展时间步参数到每个原子
        if batch is not None:
            t_per_atom = t[batch]  # (num_atoms,)
            sqrt_alpha = self.sqrt_alphas_cumprod[t_per_atom].unsqueeze(-1)  # (num_atoms, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_per_atom].unsqueeze(-1)
        else:
            # 单样本情况
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 1)

        return sqrt_alpha * pos_start + sqrt_one_minus * noise

    def q_sample_bond(self, bond_types: torch.Tensor, t: torch.Tensor,
                      edge_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        键类型前向扩散（离散扩散）

        Args:
            bond_types: 键类型索引 (num_edges,)
            t: 时间步 (batch_size,) 或 (num_edges,)
            edge_batch: 边所属的batch索引 (num_edges,)

        Returns:
            bond_perturbed_idx: 加噪后的键类型索引
            log_bond_t: 加噪后的log概率分布 (num_edges, num_bond_types)
        """
        # 键类型转为log onehot
        log_bond_0 = index_to_log_onehot(bond_types, CFG.num_bond_types)

        # 使用DiscreteTransition进行离散扩散
        bond_perturbed_idx, log_bond_t = self.bond_type_trans.q_v_sample(
            log_bond_0, t, edge_batch
        )

        return bond_perturbed_idx, log_bond_t

    def forward(self, fragment_data: Data, scaffold_context: torch.Tensor,
                t: torch.Tensor, scaffold_3d_encoding: Optional[torch.Tensor] = None,
                scaffold_data: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 - [P2升级版]

        Args:
            fragment_data: 片段图数据
            scaffold_context: 骨架指纹 (batch_size, 2048)
            t: 时间步 (batch_size,)
            scaffold_3d_encoding: 3D骨架编码 (batch_size, hidden_dim)，可选
            scaffold_data: [P2新增] 骨架图数据（包含节点特征和坐标），用于3D Cross-Attention

        Returns:
            loss字典
        """
        x = fragment_data.x
        pos = fragment_data.pos
        edge_index = fragment_data.edge_index
        atom_types = fragment_data.atom_types
        bond_types = fragment_data.bond_types if hasattr(fragment_data, 'bond_types') else torch.empty((0,), dtype=torch.long, device=x.device)
        batch = fragment_data.batch if hasattr(fragment_data, 'batch') and fragment_data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_atoms = x.size(0)
        batch_size = t.size(0) if t is not None and t.dim() > 0 else 1

        # Clamp atom_types和bond_types到有效范围
        atom_types = atom_types.clamp(0, self.num_atom_types - 1)
        if bond_types.size(0) > 0:
            bond_types = bond_types.clamp(0, CFG.num_bond_types - 1)

        # ========== 原子类型扩散（离散或连续）==========
        # ★ [修复V3-噪声增强] 自重建模式加随机扰动，防止死记硬背
        if self.training and CFG.self_reconstruction_mode:
            # 以20%概率随机替换原子类型（防止模型记住训练集分布）
            noise_mask = torch.rand(atom_types.shape, device=atom_types.device) < 0.2
            random_types = torch.randint(0, self.num_atom_types, atom_types.shape, device=atom_types.device)
            atom_types_noised = torch.where(noise_mask, random_types, atom_types)
        else:
            atom_types_noised = atom_types

        if self.use_discrete:
            # 离散扩散：使用DiscreteTransition
            log_x0 = index_to_log_onehot(atom_types_noised, self.num_atom_types)
            x_perturbed_idx, log_xt = self.atom_type_trans.q_v_sample(log_x0, t, batch)
            x_feat = log_xt.exp()  # 从log空间恢复概率
        else:
            # 连续扩散（向后兼容）
            x_one_hot = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
            x_feat = x_one_hot

        # ========== 键类型扩散（离散）==========
        # 先提取unique边的键类型
        bond_types_unique = None
        edge_batch_unique = None
        log_bond_t = None
        log_bond_0 = None

        if self.use_discrete and bond_types.size(0) > 0 and edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            src = src.clamp(0, num_atoms - 1)
            dst = dst.clamp(0, num_atoms - 1)
            unique_mask = src < dst
            unique_indices = unique_mask.nonzero().squeeze(-1)

            if unique_mask.sum() > 0:
                bond_types_unique = bond_types[unique_indices].clamp(0, CFG.num_bond_types - 1)
                # 边的batch索引（从原子batch继承）
                # src[unique_mask]获取unique边的源原子索引，然后用batch获取对应的batch索引
                src_unique_indices = src[unique_mask]
                # 确保索引在有效范围内
                src_unique_indices = src_unique_indices.clamp(0, batch.size(0) - 1)
                edge_batch_unique = batch[src_unique_indices]

                # 键类型前向扩散
                log_bond_0 = index_to_log_onehot(bond_types_unique, CFG.num_bond_types)
                bond_perturbed_idx, log_bond_t = self.bond_type_trans.q_v_sample(
                    log_bond_0, t, edge_batch_unique
                )

        # ========== 坐标扩散（连续）==========
        # [改进V3] 使用Flexible Prior采样位置噪声
        pos_noise = self._sample_position_prior_noise(pos.size(0), pos.device, batch)
        pos_noisy = self.q_sample_pos(pos, t, pos_noise, batch)

        # ========== Cross-Attention条件注入==========
        # [修复] CFG训练：以15%概率将条件置零，迫使模型学会无条件生成
        cfg_dropout_prob = 0.15
        if self.training and torch.rand(1).item() < cfg_dropout_prob:
            # 条件置零
            scaffold_context = torch.zeros_like(scaffold_context)
            if scaffold_3d_encoding is not None:
                scaffold_3d_encoding = torch.zeros_like(scaffold_3d_encoding)
            scaffold_data_cfg = None  # 3D条件也置零
        else:
            scaffold_data_cfg = scaffold_data

        # 投影骨架条件
        c_emb = self.context_proj(scaffold_context)  # (batch_size, hidden_dim)

        # 如果有3D骨架编码，融合进来
        if scaffold_3d_encoding is not None:
            c_3d = self.scaffold_3d_proj(scaffold_3d_encoding)
            c_emb = c_emb + c_3d

        # Cross-Attention注入
        h_init = self.x_feat_proj(x_feat)  # (num_atoms, hidden_dim)

        # [P2新增] 3D Cross-Attention（如果提供了骨架节点数据）
        if scaffold_data_cfg is not None and hasattr(scaffold_data_cfg, 'x') and hasattr(scaffold_data_cfg, 'pos'):
            # 骨架节点特征
            h_scaffold = self.scaffold_node_proj(scaffold_data_cfg.x)
            pos_scaffold = scaffold_data_cfg.pos

            # [P2] 提取骨架的药效团特征
            if hasattr(scaffold_data_cfg, 'mol') and scaffold_data_cfg.mol is not None:
                # 检查mol是否是单个Mol对象（batch情况下mol可能是list，暂时跳过）
                if isinstance(scaffold_data_cfg.mol, Chem.Mol):
                    pharm_types, pharm_pos = extract_pharmacophore_features(scaffold_data_cfg.mol)
                    pharm_types = pharm_types.to(x.device)
                    pharm_pos = pharm_pos.to(x.device)
                else:
                    # batch情况：跳过药效团提取（后续可扩展支持）
                    pharm_types = None
                    pharm_pos = None
            else:
                pharm_types = None
                pharm_pos = None

            # [P2] 3D Cross-Attention注入
            h_cond_3d = self.cross_attention_3d(
                h_fragment=h_init,
                pos_fragment=pos_noisy,  # 使用加噪后的片段坐标
                h_scaffold=h_scaffold,
                pos_scaffold=pos_scaffold,
                pharm_types=pharm_types,
                pharm_pos=pharm_pos,
                batch_fragment=batch
            )

            # 仍然使用原有的1D条件作为补充
            h_cond_1d = self.cross_attention(h_init, c_emb, batch)

            # [修复] 降低3D条件权重，防止过度依赖骨架特征
            h_cond = h_cond_3d * 0.5 + h_cond_1d * 0.5  # 3D和1D条件平衡
        else:
            # 没有骨架节点数据时，使用原有的1D条件
            h_cond = self.cross_attention(h_init, c_emb, batch)

        # ========== 网络预测==========
        # EGNN使用原始特征(x)和条件特征(h_cond)的融合
        # 注意: EGNN的node_proj期望6维输入，但我们传递融合后的特征
        # 需要先投影h_cond到EGNN期望的输入维度
        pos_pred, atom_logits, bond_logits = self.egnn(
            x,  # 使用原始原子特征(6维)
            pos_noisy, edge_index, t, scaffold_context, batch
        )

        # 使用条件特征增强原子类型预测
        if self.use_discrete:
            # 结合EGNN输出和条件特征
            combined_feat = atom_logits + self.atom_type_head(h_cond)
            log_x_recon = F.log_softmax(combined_feat, dim=-1)

            # [修复V2] KL损失计算改进
            # 原方法：计算后验概率的KL，当预测接近真实时KL≈0，导致loss塌缩
            # 新方法：直接计算预测分布与真实分布的KL，始终有惩罚

            # 方法1：计算预测分布与真实one-hot分布的KL（解码器损失）
            # KL(预测 || 真实) = -log p(真实类别) + 常数
            # 这等同于交叉熵损失，但我们使用KL形式
            log_x0_onehot = index_to_log_onehot(atom_types, self.atom_type_trans.num_classes)

            # KL(预测 || 真实_onehot)
            kl_pred_true = categorical_kl(log_x_recon, log_x0_onehot)

            # 方法2：时间步相关的混合损失
            # t=0时：使用纯解码器损失（KL预测||真实）
            # t>0时：使用后验KL + 解码器损失的混合
            mask_t0 = (t == 0).float()[batch]

            # 后验概率KL（用于t>0时的扩散过程一致性）
            log_x_model_prob = self.atom_type_trans.q_v_posterior(log_x_recon, log_xt, t, batch)
            log_x_true_prob = self.atom_type_trans.q_v_posterior(log_x0, log_xt, t, batch)
            kl_posterior = categorical_kl(log_x_model_prob, log_x_true_prob)

            # [修复V2] 使用类别权重
            atom_weights = ATOM_TYPE_WEIGHTS.to(x.device)

            # 解码器损失（预测vs真实）
            decoder_loss = kl_pred_true * atom_weights[atom_types]

            # 后验KL损失（也加权）
            kl_posterior_weighted = kl_posterior * atom_weights[atom_types]

            # 混合损失：t=0时纯解码器，t>0时混合
            # [关键] t>0时也保留解码器损失成分，防止塌缩
            loss_atom = (mask_t0 * decoder_loss + (1. - mask_t0) * (kl_posterior_weighted + 0.5 * decoder_loss)).mean()
        else:
            # [修复] 使用类别权重
            atom_weights = ATOM_TYPE_WEIGHTS.to(x.device)
            loss_atom = F.cross_entropy(atom_logits, atom_types, weight=atom_weights)

        # ========== 损失计算==========
        # 位置损失：预测噪声
        loss_pos = F.mse_loss(pos_pred, pos_noise)

        # ========== [新增V4] x0预测损失（片段重构损失）==========
        # 核心思想：强制模型在每一步t预测出x0，直接与真实片段比较
        # 公式：x0_pred = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        loss_x0_pos = torch.tensor(0.0, device=x.device)
        loss_x0_atom = torch.tensor(0.0, device=x.device)
        if CFG.use_x0_prediction_loss:
            # 计算x0预测（从噪声预测反推）
            # pos_noisy = sqrt(alpha_bar) * pos + sqrt(1-alpha_bar) * pos_noise
            # pos_pred是预测的noise，所以：
            # x0_pred = (pos_noisy - sqrt(1-alpha_bar) * pos_pred) / sqrt(alpha_bar)
            # 第3230-3231行，将两行替换为：
            sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][batch].unsqueeze(-1)  # [num_atoms, 1]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][batch].unsqueeze(-1)  # [num_atoms, 1]

            x0_pred_pos = (pos_noisy - sqrt_one_minus_alpha_bar * pos_pred) / (sqrt_alpha_bar + 1e-8)
            x0_pred_pos = x0_pred_pos.clamp(-10, 10)  # 防止极端值

            # 位置x0损失：预测的原始坐标与真实坐标的差异
            loss_x0_pos = F.mse_loss(x0_pred_pos, pos)

            # 原子类型x0损失：预测的原子类型与真实类型的差异
            if self.use_discrete:
                # 从atom_logits计算预测的原子类型
                atom_pred_types = atom_logits.argmax(dim=-1)
                # ★ [修复V3] 使用加权CrossEntropy，避免过拟合到常见原子类型
                loss_x0_atom = F.cross_entropy(
                    atom_logits, atom_types,
                    weight=ATOM_TYPE_WEIGHTS.to(x.device)
                )

        # 总位置损失 = 噪声预测损失 + x0预测损失
        loss_pos_total = loss_pos + CFG.x0_loss_weight * loss_x0_pos

        # ========== 键类型损失（离散扩散KL损失）==========
        loss_bond = torch.tensor(0.0, device=x.device)
        if self.use_discrete and bond_logits is not None and log_bond_t is not None and log_bond_0 is not None:
            # 键类型重建的log概率
            log_bond_recon = F.log_softmax(bond_logits, dim=-1)

            # 后验概率
            log_bond_model_prob = self.bond_type_trans.q_v_posterior(
                log_bond_recon, log_bond_t, t, edge_batch_unique
            )
            log_bond_true_prob = self.bond_type_trans.q_v_posterior(
                log_bond_0, log_bond_t, t, edge_batch_unique
            )

            # KL损失
            kl_bond = categorical_kl(log_bond_model_prob, log_bond_true_prob)

            # t=0时使用解码器损失，否则使用KL
            if edge_batch_unique is not None:
                mask_t0_edge = (t == 0).float()[edge_batch_unique]

                # [修复] 使用类别权重的解码器损失，防止双键泛滥
                bond_weights = BOND_TYPE_WEIGHTS.to(x.device)
                decoder_nll_bond_weighted = F.cross_entropy(
                    log_bond_recon.exp(),
                    bond_types_unique,
                    weight=bond_weights,
                    reduction='none'
                )

                # [修复] 对KL损失按键类型加权
                kl_bond_weighted = kl_bond * bond_weights[bond_types_unique]

                loss_bond = (mask_t0_edge * decoder_nll_bond_weighted + (1. - mask_t0_edge) * kl_bond_weighted).mean()
            else:
                loss_bond = kl_bond.mean()

        elif bond_logits is not None and bond_types.size(0) > 0 and edge_index.size(1) > 0:
            # 非离散扩散时使用交叉熵（向后兼容）
            src, dst = edge_index
            src = src.clamp(0, num_atoms - 1)
            dst = dst.clamp(0, num_atoms - 1)
            unique_mask = src < dst

            if unique_mask.sum() > 0 and bond_logits.size(0) == unique_mask.sum():
                unique_indices = unique_mask.nonzero().squeeze(-1)
                bond_types_unique_fallback = bond_types[unique_indices].clamp(0, CFG.num_bond_types - 1)
                # [修复] 使用类别权重
                bond_weights = BOND_TYPE_WEIGHTS.to(x.device)
                loss_bond = F.cross_entropy(bond_logits, bond_types_unique_fallback, weight=bond_weights)

        # 价键约束损失（片段内部）- 传入键类型预测和时间步
        # [修复] 动态调整价键约束权重：早期(t大)放松，晚期(t小)严格
        # 计算时间比例 (t/max_t)，用于动态调整权重
        if batch is not None and t is not None:
            t_per_atom = t[batch]
            t_ratio = t_per_atom.float() / self.num_timesteps  # 0~1，越大噪声越多
            avg_t_ratio = t_ratio.mean()  # 平均时间比例
        else:
            avg_t_ratio = 0.5  # 默认中间值

        # [修复V3] 基于SOTA分析调整：大幅降低价键权重防止atom loss爆炸
        # TargetDiff: loss_v_weight=100（原子类型主导），无价键损失
        # DiffSBDD: 无显式价键约束，依赖后处理reconstruct
        # 新范围：0.1 (t=100,高噪声) -> 0.5 (t=0,低噪声)
        dynamic_valence_weight = 0.1 + (1.0 - avg_t_ratio) * 0.4

        loss_valence = self.compute_valence_loss(atom_logits, edge_index, bond_logits, avg_t_ratio)

        # ========== [新增V3] 键长-键类型联合约束 ==========
        loss_bond_distance = torch.tensor(0.0, device=x.device)
        if self.use_discrete and bond_types.size(0) > 0:
            # [改进] 传入原子类型以使用atom-pair specific键长表
            loss_bond_distance = self.compute_bond_distance_loss(
                pos_noisy, edge_index, bond_types, atom_types, bond_logits
            )

        # ========== [新增V3] 连通性先验 ==========
        loss_connectivity = torch.tensor(0.0, device=x.device)
        if edge_index.size(1) > 0 and num_atoms > 2:
            # 使用键存在概率（softmax最大值）
            if bond_logits is not None:
                bond_probs = F.softmax(bond_logits, dim=-1).max(dim=-1).values
            else:
                bond_probs = None
            loss_connectivity = self.compute_connectivity_loss(
                edge_index, bond_probs, num_atoms
            )

        # ========== [新增V2] 多样性约束（熵惩罚）==========
        # [修复] 始终惩罚低熵，防止预测分布过于集中
        loss_diversity = torch.tensor(0.0, device=x.device)
        if self.use_discrete and atom_logits is not None:
            # 计算原子类型预测分布的熵
            atom_probs = F.softmax(atom_logits, dim=-1)
            # 熵 = -sum(p * log(p))
            entropy_per_atom = -(atom_probs * torch.log(atom_probs + 1e-8)).sum(dim=-1)
            # 平均熵
            avg_entropy = entropy_per_atom.mean()
            # 理论最大熵（均匀分布）：log(num_classes)
            max_entropy = math.log(self.atom_type_trans.num_classes)
            # 熵比：实际熵 / 最大熵
            entropy_ratio = avg_entropy / max_entropy

            # [修复] 始终惩罚熵低于理想值的情况
            # 理想熵比：0.8（80%的最大熵，接近均匀分布但有合理差异）
            # 使用平滑惩罚：熵越低，惩罚越大
            ideal_entropy_ratio = 0.8
            # 惩罚公式：(ideal - actual)^2，平方惩罚使低熵惩罚更重
            entropy_gap = ideal_entropy_ratio - entropy_ratio
            if entropy_gap > 0:
                # 惩罚系数随gap增大而增大（非线性）
                loss_diversity = entropy_gap ** 2 * 10.0
            else:
                # 熵过高时轻微惩罚（避免过度均匀）
                loss_diversity = (entropy_ratio - ideal_entropy_ratio) ** 2 * 1.0

        # [修复V2] 使用动态权重 + 多样性约束 + [新增V3] 键长和连通性 + [新增V4] x0预测损失
        total_loss = loss_pos_total + loss_atom + CFG.x0_loss_weight * loss_x0_atom \
                     + 0.5 * loss_bond \
                     + dynamic_valence_weight * loss_valence \
                     + CFG.bond_distance_weight * loss_bond_distance \
                     + CFG.connectivity_weight * loss_connectivity \
                     + loss_diversity

        return {
            'total': total_loss,
            'pos': loss_pos.item(),
            'x0_pos': loss_x0_pos.item(),  # [新增V4]
            'x0_atom': loss_x0_atom.item(),  # [新增V4]
            'atom': loss_atom.item(),
            'bond': loss_bond.item(),
            'valence': loss_valence.item(),
            'bond_distance': loss_bond_distance.item(),
            'connectivity': loss_connectivity.item(),
            'diversity': loss_diversity.item(),
        }

    def compute_valence_loss(self, atom_logits: torch.Tensor,
                              edge_index: torch.Tensor,
                              bond_logits: Optional[torch.Tensor] = None,
                              t_ratio: float = 0.5) -> torch.Tensor:
        """
        价键约束损失 - 升级版（考虑键类型权重和动态时间调整）

        改进：
        1. 键类型权重：SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1
        2. 原子类型-键类型联合约束：N/O不应承受DOUBLE键
        3. 基于预测键概率计算期望价键负载
        4. [修复] 动态权重：t_ratio大（噪声多）时放松，t_ratio小（噪声少）时严格

        Args:
            t_ratio: 时间比例 (t/max_t)，0表示无噪声，1表示最大噪声
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=atom_logits.device)

        num_atoms = atom_logits.size(0)
        edge_index = edge_index.clamp(0, num_atoms - 1)

        # ========== 键类型权重（任务A）==========
        # 键类型对价键负载的贡献：SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1
        bond_valence_weights = torch.tensor([1.0, 2.0, 3.0, 1.0], device=atom_logits.device)

        src, dst = edge_index
        src = src.clamp(0, num_atoms - 1)
        dst = dst.clamp(0, num_atoms - 1)

        # 只考虑unique边（src < dst）
        unique_mask = src < dst
        src_unique = src[unique_mask]
        dst_unique = dst[unique_mask]
        num_edges = src_unique.size(0)

        if num_edges == 0:
            return torch.tensor(0.0, device=atom_logits.device)

        # ========== 计算期望价键负载（考虑键类型）==========
        # 如果有键类型预测，使用预测概率计算期望负载
        if bond_logits is not None and bond_logits.size(0) == num_edges:
            # 键类型概率
            bond_probs = F.softmax(bond_logits, dim=-1)  # (num_edges, 4)

            # 期望键类型权重
            expected_bond_weight = (bond_probs * bond_valence_weights.unsqueeze(0)).sum(dim=-1)  # (num_edges,)
        else:
            # 没有键类型预测，默认单键权重=1
            expected_bond_weight = torch.ones(num_edges, device=atom_logits.device)

        # 计算每个原子的价键负载
        valence_load = torch.zeros(num_atoms, device=atom_logits.device)
        valence_load.scatter_add_(0, src_unique.long(), expected_bond_weight)
        valence_load.scatter_add_(0, dst_unique.long(), expected_bond_weight)

        # ========== 原子类型价键限制（任务B）==========
        # 使用softmax概率计算期望价键限制
        atom_probs = F.softmax(atom_logits, dim=-1)  # (num_atoms, num_atom_types)

        # 价键限制表（与ATOM_TYPES对应）
        # ATOM_TYPES = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        # 索引:       [0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11]
        # 价键:       [1, 3, 4, 5, 2, 1, 5,  6,  1,  1,  1,  4]
        max_valence = torch.tensor([1.0, 3.0, 4.0, 5.0, 2.0, 1.0, 5.0, 6.0, 1.0, 1.0, 1.0, 4.0],
                                    device=atom_logits.device)

        # 期望最大价键
        expected_max_valence = (atom_probs * max_valence.unsqueeze(0)).sum(dim=-1)  # (num_atoms,)

        # ========== 原子类型-键类型联合约束（任务B升级）==========
        # 某些原子类型不应承受DOUBLE/TRIPLE键
        # O(索引4): 最大键数2，不应有DOUBLE键（除非是羰基等特殊情况）
        # N(索引3): 可以有DOUBLE，但要小心
        # 这里用软约束：对不合理的组合增加惩罚权重

        atom_type_restricted_for_double = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                                                         device=atom_logits.device)  # O/F/Cl/Br/I禁止DOUBLE

        if bond_logits is not None and bond_logits.size(0) == num_edges:
            bond_double_probs = bond_probs[:, 1]  # DOUBLE键概率

            # 对每条边，如果任一端是"禁止DOUBLE"的原子类型，惩罚DOUBLE键预测
            atom_double_penalty = atom_probs * atom_type_restricted_for_double.unsqueeze(0)
            src_double_penalty = atom_double_penalty[src_unique].sum(dim=-1)  # src端的惩罚权重
            dst_double_penalty = atom_double_penalty[dst_unique].sum(dim=-1)  # dst端的惩罚权重

            # 边的总惩罚 = 两端惩罚之和 * DOUBLE键概率
            edge_type_penalty = (src_double_penalty + dst_double_penalty) * bond_double_probs
            loss_type_constraint = edge_type_penalty.mean()
        else:
            loss_type_constraint = torch.tensor(0.0, device=atom_logits.device)

        # ========== 总价键损失 ==========
        # 1. 超价惩罚（价键负载超过最大限制） - [P1增强]权重提高到3.0
        excess = valence_load - expected_max_valence
        loss_over_valence = F.relu(excess).mean()

        # 2. 原子类型-键类型不合理组合惩罚
        # 3. 权重分配
        loss_type_constraint_final = loss_type_constraint * 1.0

        # ========== [P1新增] TRIPLE键严格禁止 ==========
        # 片段中TRIPLE键极少出现，对大多数原子类型应该禁止
        # TRIPLE键（索引2）只允许在特定情况下出现（如炔烃C≡C）
        loss_triple_penalty = torch.tensor(0.0, device=atom_logits.device)
        if bond_logits is not None and bond_logits.size(0) == num_edges:
            bond_triple_probs = bond_probs[:, 2]  # TRIPLE键概率

            # 检查每条边两端是否都是碳（只有碳碳炔键允许）
            atom_is_carbon = atom_probs[:, 2]  # 碳的概率（索引2是C）
            src_carbon_prob = atom_is_carbon[src_unique]
            dst_carbon_prob = atom_is_carbon[dst_unique]

            # 如果任一端不是碳，惩罚TRIPLE键
            # 允许条件：两端都是碳（概率都接近1）
            non_carbon_penalty = (1.0 - src_carbon_prob) + (1.0 - dst_carbon_prob)
            triple_penalty_per_edge = non_carbon_penalty * bond_triple_probs * 5.0  # 高惩罚
            loss_triple_penalty = triple_penalty_per_edge.mean()

        # ========== [P1新增] 欠价惩罚（可选）==========
        # 某些原子需要足够的键才能稳定
        # 例如：碳原子如果只有0-1条键，可能是不合理的
        # 但片段边界原子可能确实只有1条键，所以这里用较弱的惩罚
        loss_under_valence = torch.tensor(0.0, device=atom_logits.device)

        # 碳原子至少需要2条键才能稳定（片段边界可以是1条）
        # 这里只检查明显的欠价情况（碳原子键数<1）
        carbon_min_bonds = 1.0  # 最少1条键
        atom_expected_min_bonds = torch.zeros(num_atoms, device=atom_logits.device)
        atom_expected_min_bonds += atom_probs[:, 2] * carbon_min_bonds  # 碳原子需要至少1键

        under_valence = atom_expected_min_bonds - valence_load
        loss_under_valence = F.relu(under_valence).mean() * 0.5  # 较弱惩罚

        # ========== [P1增强] 总价键损失 ==========
        # [修复V3] 基于SOTA分析：大幅降低超价惩罚权重
        # 原权重10.0导致atom loss爆炸，SOTA方法无显式价键约束
        # 新策略：温和约束，不干扰原子类型学习
        relax_factor = 0.5 + (1.0 - t_ratio) * 0.5  # 0.5~1.0，温和的动态调整
        total_loss = loss_over_valence * 2.0 * relax_factor + \
                     loss_type_constraint_final * 1.0 * relax_factor + \
                     loss_triple_penalty * 2.0 * relax_factor + \
                     loss_under_valence * 0.2 * relax_factor

        return total_loss

    def compute_bond_distance_loss(self, pos: torch.Tensor,
                                     edge_index: torch.Tensor,
                                     bond_types: torch.Tensor,
                                     atom_types: Optional[torch.Tensor] = None,
                                     bond_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        [新增V3] 键长-键类型联合约束损失

        [改进] 使用atom-pair specific键长表（借鉴DiffSBDD）
        - C-C单键: 1.54Å, C-C双键: 1.34Å, C-N单键: 1.47Å 等
        - 若未找到atom-pair组合，回退到基础BOND_LENGTHS表

        Args:
            pos: 原子坐标 (num_atoms, 3)
            edge_index: 边索引 (2, num_edges)
            bond_types: 真实键类型索引 (num_edges,)
            atom_types: 原子类型索引 (num_atoms,)，用于atom-pair键长查询
            bond_logits: 预测键类型logits (num_edges, num_bond_types)

        Returns:
            loss: 键长一致性损失
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device)

        num_atoms = pos.size(0)
        edge_index = edge_index.clamp(0, num_atoms - 1)

        # 提取unique边（src < dst）
        src, dst = edge_index
        unique_mask = src < dst
        src_u = src[unique_mask].clamp(0, num_atoms - 1)
        dst_u = dst[unique_mask].clamp(0, num_atoms - 1)
        num_edges_u = src_u.size(0)

        if num_edges_u == 0:
            return torch.tensor(0.0, device=pos.device)

        # 计算预测的原子间距离
        pred_dist = (pos[src_u] - pos[dst_u]).norm(dim=-1)  # (num_edges_u,)

        # 获取理想键长
        bond_types_u = bond_types[unique_mask].clamp(0, len(BOND_LENGTHS) - 1)

        # [改进] 使用atom-pair specific键长表（如果配置启用且有原子类型）
        if CFG.use_atom_pair_bond_lengths and atom_types is not None and atom_types.size(0) >= num_atoms:
            # 获取边的原子类型
            atom_types_src = atom_types[src_u].clamp(0, len(ATOM_SYMBOL_MAP) - 1)
            atom_types_dst = atom_types[dst_u].clamp(0, len(ATOM_SYMBOL_MAP) - 1)

            # 转换为符号并查询键长表
            bond_type_keys = ['single', 'double', 'triple', 'aromatic']
            ideal_lengths = []
            for i in range(num_edges_u):
                bt_idx = bond_types_u[i].item()
                src_symbol = ATOM_SYMBOL_MAP.get(atom_types_src[i].item(), 'C')
                dst_symbol = ATOM_SYMBOL_MAP.get(atom_types_dst[i].item(), 'C')

                # 排序atom pair以便查询（表中的key按字母序）
                pair = tuple(sorted([src_symbol, dst_symbol]))
                bond_key = bond_type_keys[bt_idx]

                # 查询atom-pair键长表
                pair_lengths = BOND_LENGTHS_PAIR.get(bond_key, {})
                ideal_len = pair_lengths.get(pair, None)

                # 若未找到，尝试反向pair
                if ideal_len is None:
                    pair_rev = (pair[1], pair[0])
                    ideal_len = pair_lengths.get(pair_rev, None)

                # 回退到基础BOND_LENGTHS
                if ideal_len is None:
                    ideal_len = BOND_LENGTHS[bt_idx]

                ideal_lengths.append(ideal_len)

            ideal_lengths_tensor = torch.tensor(
                ideal_lengths, device=pos.device, dtype=pred_dist.dtype
            )
        else:
            # 回退：使用基础键长表（仅按键类型）
            ideal_lengths_tensor = torch.tensor(
                [BOND_LENGTHS[bt.item()] for bt in bond_types_u],
                device=pos.device, dtype=pred_dist.dtype
            )

        # 计算距离损失：(pred - ideal)^2
        distance_loss = F.mse_loss(pred_dist, ideal_lengths_tensor)

        # [可选] 如果有键类型预测，可以计算预测键类型对应的理想距离
        if bond_logits is not None and bond_logits.size(0) >= num_edges_u:
            # 使用预测的键类型概率计算期望距离
            bond_probs = F.softmax(bond_logits[:num_edges_u], dim=-1)  # (num_edges_u, 4)
            ideal_length_tensor = torch.tensor(
                list(BOND_LENGTHS.values()), device=pos.device, dtype=bond_probs.dtype
            )
            expected_dist = (bond_probs * ideal_length_tensor.unsqueeze(0)).sum(dim=-1)
            # 惩罚预测距离与期望距离的差异
            expected_dist_loss = F.mse_loss(pred_dist, expected_dist)
            distance_loss = distance_loss + 0.5 * expected_dist_loss

        return distance_loss

    def compute_connectivity_loss(self, edge_index: torch.Tensor,
                                   bond_probs: Optional[torch.Tensor] = None,
                                   num_atoms: int = 0) -> torch.Tensor:
        """
        [新增V3] 连通性先验损失

        核心思想：惩罚断开的图（分子碎裂）
        使用拉普拉斯矩阵的第二小特征值（代数连通度）

        Args:
            edge_index: 边索引 (2, num_edges)
            bond_probs: 键存在概率 (num_edges,)，用于软连通性
            num_atoms: 原子数量

        Returns:
            loss: 连通性损失
        """
        if edge_index.size(1) == 0 or num_atoms < 2:
            return torch.tensor(0.0, device=edge_index.device)

        # 构建邻接矩阵（软邻接，使用键概率）
        device = edge_index.device
        adj = torch.zeros(num_atoms, num_atoms, device=device)

        src, dst = edge_index.clamp(0, num_atoms - 1)

        # 确保unique边
        unique_mask = src < dst
        src_u, dst_u = src[unique_mask], dst[unique_mask]

        if src_u.size(0) == 0:
            return torch.tensor(0.0, device=device)

        # 使用键概率作为边权重（如果提供）
        if bond_probs is not None and bond_probs.size(0) >= src_u.size(0):
            edge_weights = bond_probs[:src_u.size(0)]
        else:
            edge_weights = torch.ones(src_u.size(0), device=device)

        # 填充邻接矩阵（双向）
        adj[src_u, dst_u] = edge_weights
        adj[dst_u, src_u] = edge_weights

        # 计算拉普拉斯矩阵 L = D - A
        degree = adj.sum(dim=1)
        # 处理孤立节点（degree=0）
        degree = degree.clamp(min=1e-6)
        L = torch.diag(degree) - adj

        # 计算特征值（只计算最小的几个）
        try:
            eigenvalues = torch.linalg.eigvalsh(L)
            # 第二小特征值 = 代数连通度
            algebraic_connectivity = eigenvalues[1]

            # 惩罚连通度接近0（图断开）
            # [调整] 使用Config中的阈值，默认0.1（更适合小片段）
            connectivity_threshold = CFG.connectivity_threshold  # 0.1
            if algebraic_connectivity < connectivity_threshold:
                loss = (connectivity_threshold - algebraic_connectivity) ** 2 * 10.0
            else:
                loss = torch.tensor(0.0, device=device)
        except Exception:
            # 特征值计算失败时返回0
            loss = torch.tensor(0.0, device=device)

        return loss

    @torch.no_grad()
    def sample(self, num_atoms: int, scaffold_context: torch.Tensor,
               edge_index_template: Optional[torch.Tensor] = None,
               fix_noise: bool = False,
               fragment_reference: Optional[Dict] = None,
               guidance_scale: float = 1.0,
               scaffold_data: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """
        采样生成片段 - [P2升级版] 使用离散扩散采样

        [改进] 支持原片段信息约束，提高生成相似度
        [P2新增] 支持骨架节点3D特征和药效团特征注入

        Args:
            num_atoms: 片段原子数
            scaffold_context: 骨架指纹 (1, 2048)
            edge_index_template: 片段内部边模板
            fix_noise: 是否固定噪声
            fragment_reference: 原片段参考信息（用于约束生成）
                - 'atom_types': 原片段原子类型索引 (num_atoms,)
                - 'pos': 原片段坐标 (num_atoms, 3)
            guidance_scale: 条件引导强度（越大越忠实于原片段）
            scaffold_data: [P2新增] 骨架图数据（用于3D Cross-Attention）

        Returns:
            生成的片段数据
        """
        if fix_noise:
            torch.manual_seed(42)

        # [改进] 使用原片段信息初始化
        ref_atom_types = None
        ref_pos = None
        ref_bond_types = None  # ★ [修复V3-新增] 键类型引导
        if fragment_reference is not None:
            if 'atom_types' in fragment_reference:
                ref_atom_types = fragment_reference['atom_types'].to(device)
            if 'pos' in fragment_reference:
                ref_pos = fragment_reference['pos'].to(device)
                ref_pos = ref_pos - ref_pos.mean(dim=0)  # 中心化
            if 'bond_types' in fragment_reference:  # ★ 新增
                ref_bond_types = fragment_reference['bond_types'].to(device)

        # 初始化坐标
        if ref_pos is not None:
            # [改进] 从原片段坐标附近开始，而不是完全随机
            pos = ref_pos + torch.randn(num_atoms, 3, device=device) * 0.5
        else:
            # [改进V3] 使用Flexible Prior采样初始位置
            pos = self._sample_position_prior(num_atoms, device)
        pos = pos - pos.mean(dim=0)

        # 原子类型初始化
        if ref_atom_types is not None:
            # [改进] 用原片段原子类型作为soft prior
            # 添加少量噪声，让模型有一定探索空间
            log_atom_types = index_to_log_onehot(ref_atom_types, self.num_atom_types)
            # 加一点噪声防止完全固定
            noise_scale = 0.1
            uniform_log = torch.log(torch.ones(num_atoms, self.num_atom_types, device=device) / self.num_atom_types)
            log_atom_types = log_atom_types + noise_scale * uniform_log
            log_atom_types = F.log_softmax(log_atom_types.exp(), dim=-1)
            atom_types = ref_atom_types.clone()
        else:
            # 原子类型：从均匀分布开始（离散扩散）
            atom_types = torch.randint(0, self.num_atom_types, (num_atoms,), device=device)
            log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

        # ========== 键类型初始化（离散扩散）==========
        # 提取unique边用于键类型扩散
        src_edges, dst_edges = edge_index_template[0], edge_index_template[1]
        unique_edge_mask = src_edges < dst_edges
        edge_index_unique = edge_index_template[:, unique_edge_mask]
        num_edges_unique = edge_index_unique.size(1)

        # 键类型初始化
        if num_edges_unique > 0:
            # 从均匀分布开始（离散扩散）
            bond_types_unique = torch.randint(0, CFG.num_bond_types, (num_edges_unique,), device=device)
            log_bond_types = index_to_log_onehot(bond_types_unique, CFG.num_bond_types)
            edge_batch = torch.zeros(num_edges_unique, dtype=torch.long, device=device)
        else:
            bond_types_unique = torch.empty((0,), dtype=torch.long, device=device)
            log_bond_types = None
            edge_batch = None

        # 原子特征（初始化）
        x = torch.zeros(num_atoms, 6, device=device)
        if ref_atom_types is not None:
            # 使用原片段的原子类型
            for i, at in enumerate(ref_atom_types):
                if at < len(ATOM_TYPES):
                    x[i, 0] = ATOM_TYPES[at]  # 原子序数
        else:
            x[:, 0] = 6  # 默认碳

        # 默认边（如果没有模板）
        if edge_index_template is None or edge_index_template.size(1) == 0:
            # 简单连接图
            if num_atoms > 1:
                src = torch.arange(num_atoms - 1, device=device)
                dst = torch.arange(1, num_atoms, device=device)
                edge_index_template = torch.stack([src, dst], dim=0)
            else:
                edge_index_template = torch.empty((2, 0), dtype=torch.long, device=device)

        batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # [P2新增] 骨架节点特征和药效团特征处理
        h_scaffold = None
        pos_scaffold = None
        pharm_types = None
        pharm_pos = None

        if scaffold_data is not None and hasattr(scaffold_data, 'x') and hasattr(scaffold_data, 'pos'):
            # 骨架节点特征投影
            h_scaffold = self.scaffold_node_proj(scaffold_data.x.to(device))
            pos_scaffold = scaffold_data.pos.to(device)

            # [P2] 提取骨架的药效团特征
            if hasattr(scaffold_data, 'mol') and scaffold_data.mol is not None:
                pharm_types, pharm_pos = extract_pharmacophore_features(scaffold_data.mol)
                pharm_types = pharm_types.to(device)
                pharm_pos = pharm_pos.to(device)

        # 逆向扩散
        for t_int in tqdm(reversed(range(self.num_timesteps)),
                          desc="Sampling", total=self.num_timesteps):
            t = torch.full((1,), t_int, dtype=torch.long, device=device)

            # ========== [新增V3] Classifier-Free Guidance ==========
            if CFG.use_cfg_inference and CFG.cfg_scale > 1.0:
                # 条件预测
                pos_pred_cond, atom_logits_cond, bond_logits_cond = self.egnn(
                    x, pos, edge_index_template, t, scaffold_context, batch
                )

                # 无条件预测（使用zero context）
                zero_context = torch.zeros_like(scaffold_context)
                pos_pred_uncond, atom_logits_uncond, bond_logits_uncond = self.egnn(
                    x, pos, edge_index_template, t, zero_context, batch
                )

                # CFG组合: Pred = Pred_uncond + w * (Pred_cond - Pred_uncond)
                cfg_scale = CFG.cfg_scale
                pos_pred = pos_pred_uncond + cfg_scale * (pos_pred_cond - pos_pred_uncond)
                atom_logits = atom_logits_uncond + cfg_scale * (atom_logits_cond - atom_logits_uncond)
                bond_logits = bond_logits_uncond + cfg_scale * (bond_logits_cond - bond_logits_uncond)
            else:
                # 不使用CFG，直接预测
                pos_pred, atom_logits, bond_logits = self.egnn(
                    x, pos, edge_index_template, t, scaffold_context, batch
                )

            # [P2新增] 3D Cross-Attention增强原子类型预测
            if h_scaffold is not None:
                # 片段原子特征投影
                x_feat = F.one_hot(atom_logits.argmax(dim=-1), num_classes=self.num_atom_types).float()
                h_fragment = self.x_feat_proj(x_feat)

                # 3D Cross-Attention
                atom_logits_3d = self.cross_attention_3d(
                    h_fragment=h_fragment,
                    pos_fragment=pos,
                    h_scaffold=h_scaffold,
                    pos_scaffold=pos_scaffold,
                    pharm_types=pharm_types,
                    pharm_pos=pharm_pos,
                    batch_fragment=batch
                )

                # 使用3D增强的特征增强原子类型预测
                atom_logits_3d_proj = self.atom_type_head(atom_logits_3d)
                atom_logits = atom_logits + atom_logits_3d_proj * 0.3  # 轻度增强

            # ========== 原子位置采样（连续扩散） ==========
            x0_pred = (pos - self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 3) * pos_pred) / \
                      (self.sqrt_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 3) + 1e-8)
            x0_pred = x0_pred.clamp(-10, 10)

            # 后验均值
            alpha_t = self.alphas[t]
            alpha_cumprod_prev = torch.cat([torch.ones(1, device=device),
                                            self.alphas_cumprod[:-1]])[t]
            beta_t = self.betas[t]

            posterior_mean = (alpha_cumprod_prev.sqrt() * beta_t * x0_pred +
                             alpha_t.sqrt() * (1 - alpha_cumprod_prev) * pos) / \
                            (1 - self.alphas_cumprod[t] + 1e-8)

            # 添加噪声（t>0时）
            if t_int > 0:
                noise = torch.randn_like(pos)
                pos = posterior_mean + self.posterior_variance[t].sqrt() * noise
            else:
                pos = posterior_mean

            pos = pos - pos.mean(dim=0)  # 中心化

            # ========== 原子类型采样（离散扩散） ==========
            if self.use_discrete:
                # 使用离散扩散的后验采样
                log_x_recon = F.log_softmax(atom_logits, dim=-1)

                # [改进] 条件引导：如果有参考原子类型，增强对其的倾向
                if ref_atom_types is not None and guidance_scale > 1.0:
                    # 创建参考原子类型的log onehot
                    log_ref = index_to_log_onehot(ref_atom_types, self.num_atom_types)
                    # ★ [修复V4-根因修复] t=0时先应用引导权重，再用argmax
                    # 根因：之前t=0时argmax用在未引导的log_x_recon上，导致模型预测O就输出O
                    # 解决：t=0时先混合模型预测和参考分布，再取argmax
                    # t=99（纯噪声）→ ref_weight≈0.1（让模型自由探索）
                    # t=0（最终步）→ ref_weight≈0.9（强约束到参考）
                    t_ratio = t_int / (self.num_timesteps - 1)  # 1.0 → 0.0
                    ref_weight = (1.0 - t_ratio) * 0.8 + 0.1  # 动态权重：0.1 → 0.9
                    recon_weight = 1.0 - ref_weight

                    # ★★★ 关键修复：无论t=0还是t>0，都先应用引导权重 ★★★
                    guided_probs = (log_x_recon.exp() * recon_weight + log_ref.exp() * ref_weight)
                    guided_probs = guided_probs / (guided_probs.sum(dim=-1, keepdim=True) + 1e-10)

                    if t_int == 0:
                        # t=0: 用argmax（确定性输出，避免后验塌缩）
                        atom_types = guided_probs.argmax(dim=-1)
                        log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)
                    else:
                        # t>0: 用采样（允许探索）
                        log_guided = torch.log(guided_probs + 1e-10)
                        log_model_prob = self.atom_type_trans.q_v_posterior(
                            log_guided, log_atom_types, t, batch)
                        atom_types = log_sample_categorical(log_model_prob)
                        log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)
                else:
                    # 无引导时的正常采样
                    log_model_prob = self.atom_type_trans.q_v_posterior(
                        log_x_recon, log_atom_types, t, batch)
                    atom_types = log_sample_categorical(log_model_prob)
                    log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)
            else:
                # 简单softmax采样
                atom_probs = F.softmax(atom_logits, dim=-1)

                # [改进] 条件引导
                if ref_atom_types is not None and guidance_scale > 1.0:
                    ref_probs = F.one_hot(ref_atom_types, self.num_atom_types).float()
                    atom_probs = atom_probs * (1 - 0.3) + ref_probs * 0.3 * (guidance_scale - 1.0)
                    atom_probs = atom_probs / atom_probs.sum(dim=-1, keepdim=True)

                atom_types = torch.multinomial(atom_probs, 1).squeeze(-1)
            # [修复] Clamp到有效范围
            atom_types = atom_types.clamp(0, self.num_atom_types - 1)

            # ========== 键类型采样（离散扩散）==========
            if self.use_discrete and bond_logits is not None and log_bond_types is not None:
                # 键类型重建的log概率
                log_bond_recon = F.log_softmax(bond_logits, dim=-1)

                # ★ [修复V3-新增] 键类型引导：如果有参考键类型，增强对其的倾向
                if ref_bond_types is not None and guidance_scale > 1.0 and ref_bond_types.size(0) >= bond_logits.size(0):
                    # 只引导存在的边（ref_bond_types可能比当前边少）
                    num_edges_to_guide = min(bond_logits.size(0), ref_bond_types.size(0))
                    # ★ [关键修复] 检测芳香键并强化引导
                    t_ratio = t_int / (self.num_timesteps - 1)
                    has_aromatic = (ref_bond_types == 3).any().item()  # 3=AROMATIC
                    if has_aromatic:
                        # 含芳香键的片段：大幅强化引导权重，防止芳香键被单键替代
                        # 先验分布单键占65%，需要更强的引导来对抗
                        ref_weight = (1.0 - t_ratio) * 0.65 + 0.3  # 范围 0.3 → 0.95
                        if t_int <= 10:  # 临近结束时日志
                            logger.info(f"    [Bond guidance] Aromatic fragment detected, ref_weight={ref_weight:.2f}")
                    else:
                        ref_weight = (1.0 - t_ratio) * 0.7 + 0.1  # 原来的范围 0.1 → 0.8
                    recon_weight = 1.0 - ref_weight

                    # 创建参考键类型的log onehot
                    log_ref_bond = index_to_log_onehot(ref_bond_types[:num_edges_to_guide], CFG.num_bond_types)
                    guided_bond_probs = (log_bond_recon[:num_edges_to_guide].exp() * recon_weight +
                                         log_ref_bond.exp() * ref_weight)
                    guided_bond_probs = guided_bond_probs / (guided_bond_probs.sum(dim=-1, keepdim=True) + 1e-10)
                    log_bond_recon[:num_edges_to_guide] = torch.log(guided_bond_probs + 1e-10)

                # [P1新增] 价键约束：在采样前调整键类型概率分布
                bond_probs_for_constraint = log_bond_recon.exp()

                # [P1] TRIPLE键严格禁止（除非两端都是碳）
                # 检查原子类型概率
                atom_probs_for_constraint = F.softmax(atom_logits, dim=-1)
                atom_is_carbon = atom_probs_for_constraint[:, 2]  # 碳概率

                for edge_idx in range(bond_probs_for_constraint.size(0)):
                    if edge_idx < edge_index_unique.size(1):
                        s = edge_index_unique[0, edge_idx].item()
                        d = edge_index_unique[1, edge_idx].item()
                        s = min(s, num_atoms - 1)
                        d = min(d, num_atoms - 1)

                        src_carbon = atom_is_carbon[s].item()
                        dst_carbon = atom_is_carbon[d].item()

                        # 如果任一端不是碳（概率<0.8），大幅降低TRIPLE键概率
                        if src_carbon < 0.8 or dst_carbon < 0.8:
                            bond_probs_for_constraint[edge_idx, 2] *= 0.01  # 几乎禁止TRIPLE

                        # [P1] 检查氧原子：氧（索引4）不应该有DOUBLE键（除非羰基）
                        src_oxygen = atom_probs_for_constraint[s, 4].item()  # 氧概率
                        dst_oxygen = atom_probs_for_constraint[d, 4].item()
                        if src_oxygen > 0.5 or dst_oxygen > 0.5:
                            bond_probs_for_constraint[edge_idx, 1] *= 0.3  # 降低DOUBLE键概率

                # 重新归一化并转为log
                bond_probs_for_constraint = bond_probs_for_constraint / bond_probs_for_constraint.sum(dim=-1, keepdim=True)
                log_bond_recon = torch.log(bond_probs_for_constraint + 1e-10)

                # 后验概率采样
                log_bond_model_prob = self.bond_type_trans.q_v_posterior(
                    log_bond_recon, log_bond_types, t, edge_batch
                )
                bond_types_unique = log_sample_categorical(log_bond_model_prob)
                log_bond_types = index_to_log_onehot(bond_types_unique, CFG.num_bond_types)
                # Clamp到有效范围
                bond_types_unique = bond_types_unique.clamp(0, CFG.num_bond_types - 1)
            elif bond_logits is not None and num_edges_unique > 0:
                # 非离散扩散时使用softmax采样（向后兼容）
                bond_probs = F.softmax(bond_logits, dim=-1)

                # [P1增强] 价键约束调整
                # TRIPLE键几乎禁止
                bond_probs[:, 2] *= 0.01

                # 增加SINGLE键概率（片段内部最常见）
                bond_probs[:, 0] += 1.0

                bond_probs = bond_probs / bond_probs.sum(dim=-1, keepdim=True)
                bond_types_unique = torch.multinomial(bond_probs, 1).squeeze(-1)
                bond_types_unique = bond_types_unique.clamp(0, CFG.num_bond_types - 1)

            # 价键约束（强制）- 使用当前扩散采样的键类型
            # 原子类型索引: ATOM_TYPES = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            #               H, B, C, N, O, F, P,  S,  Cl, Br, I,  其他
            # 索引:         [0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11]
            # 价键:         [1, 3, 4, 5, 2, 1, 5,  6,  1,  1,  1,  4]
            # 键类型: SINGLE=0(1键位), DOUBLE=1(2键位), TRIPLE=2(3键位), AROMATIC=3(1.5键位)
            if edge_index_unique.size(1) > 0 and bond_types_unique.size(0) > 0:
                # 价键负载：SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1
                bond_valence_weights = torch.tensor([1, 2, 3, 1], device=device)
                src_unique = edge_index_unique[0].clamp(0, num_atoms - 1)
                dst_unique = edge_index_unique[1].clamp(0, num_atoms - 1)

                # 计算每个原子的价键负载（使用扩散采样的键类型）
                valence_load = torch.zeros(num_atoms, device=device)
                for i, (s, d) in enumerate(zip(src_unique.tolist(), dst_unique.tolist())):
                    if i < bond_types_unique.size(0):
                        bond_type = bond_types_unique[i].item()
                        bond_weight = bond_valence_weights[bond_type].item()
                        valence_load[s] += bond_weight
                        valence_load[d] += bond_weight
                    else:
                        # 超出范围，默认单键
                        valence_load[s] += 1
                        valence_load[d] += 1

                max_valence = torch.tensor([1, 3, 4, 5, 2, 1, 5, 6, 1, 1, 1, 4], device=device)

                for atom_idx in range(num_atoms):
                    num_bonds = int(valence_load[atom_idx].item())
                    # 连接点原子（默认第一个）需要额外考虑骨架连接（单键=1）
                    if atom_idx == 0:
                        num_bonds += 1  # 预留骨架连接键

                    current_type = int(atom_types[atom_idx].item())
                    current_valence = max_valence[current_type].item()

                    if num_bonds > current_valence:
                        # 根据需要的键数选择合适的原子类型
                        if num_bonds >= 6:
                            atom_types[atom_idx] = 7  # S (索引7，价6)
                        elif num_bonds == 5:
                            atom_types[atom_idx] = 3  # N (索引3，价5)
                        elif num_bonds == 4:
                            atom_types[atom_idx] = 2  # C (索引2，价4)
                        elif num_bonds == 3:
                            atom_types[atom_idx] = 2  # C (索引2，价4)
                        elif num_bonds == 2:
                            atom_types[atom_idx] = 2  # C (索引2，价4)
                        # num_bonds == 1 的情况不需要修改

                # 更新log_atom_types
                if self.use_discrete:
                    log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

                # ★ [修复V3-根因1] 关键修复：每步同步更新x的原子序数特征
                # 原因：atom_types每步都在变化，但EGNN收到的x一直是初始特征
                # 导致100步下来atom_types和x完全脱节，EGNN预测严重偏离
                for i, at in enumerate(atom_types):
                    if at < len(ATOM_TYPES):
                        x[i, 0] = float(ATOM_TYPES[at])
                # 处理未知类型（索引11或越界）→ 默认碳
                x[:, 0] = torch.where(x[:, 0] == 0, torch.tensor(6.0, device=device), x[:, 0])

        # 返回扩散采样的结果
        # 注意：键类型已经在逆向扩散循环中采样完成
        # 确保edge_index_unique与bond_types_unique一致
        src, dst = edge_index_template
        src = src.clamp(0, num_atoms - 1)
        dst = dst.clamp(0, num_atoms - 1)
        unique_mask = src < dst
        edge_index_unique = edge_index_template[:, unique_mask]

        # 如果扩散采样没有产生键类型，使用默认值
        if bond_types_unique.size(0) == 0 and edge_index_unique.size(1) > 0:
            bond_types_unique = torch.zeros(edge_index_unique.size(1), dtype=torch.long, device=device)

        return {
            'pos': pos,
            'atom_types': atom_types,
            'bond_types': bond_types_unique,
            'edge_index': edge_index_unique,
        }

    @torch.no_grad()
    def inpaint_sample(self, num_atoms: int, scaffold_context: torch.Tensor,
                       fixed_atom_mask: torch.Tensor,
                       fixed_atom_types: Optional[torch.Tensor] = None,
                       fixed_pos: Optional[torch.Tensor] = None,
                       edge_index_template: Optional[torch.Tensor] = None,
                       resamplings: int = 1, jump_length: int = 1,
                       scaffold_data: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """
        Inpaint采样：固定部分原子，只生成其余部分（借鉴DiffSBDD inpaint方法第676-836行）

        应用场景：
        - 等排体替换：保持连接点原子类型和位置，生成新的等排体片段
        - 局部修改：固定部分片段结构，只修改需要变化的部分

        算法：RePaint（Lugmayr et al., 2022）
        - 在每个时间步，固定部分从原始数据采样噪声，可变部分从模型采样
        - 通过resampling和jump back增加一致性

        Args:
            num_atoms: 片段总原子数
            scaffold_context: 骨架指纹 (1, 2048)
            fixed_atom_mask: 固定原子掩码 (num_atoms,)，1表示固定，0表示可变
            fixed_atom_types: 固定原子的类型 (num_fixed,)，仅fixed位置有效
            fixed_pos: 固定原子的坐标 (num_fixed, 3)，仅fixed位置有效
            edge_index_template: 边模板
            resamplings: 重采样次数（越大越一致）
            jump_length: 回跳步数（越大越全局一致）
            scaffold_data: 骨架图数据

        Returns:
            生成的片段数据
        """
        # 初始化可变部分
        num_fixed = fixed_atom_mask.sum().item()
        num_variable = num_atoms - num_fixed

        # 创建完整的原子类型和坐标数组
        atom_types = torch.randint(0, self.num_atom_types, (num_atoms,), device=device)
        log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

        # [改进V3] 使用Flexible Prior采样初始位置
        pos = self._sample_position_prior(num_atoms, device)
        pos = pos - pos.mean(dim=0)

        # 设置固定部分
        if fixed_atom_types is not None and num_fixed > 0:
            fixed_indices = torch.where(fixed_atom_mask == 1)[0]
            atom_types[fixed_indices] = fixed_atom_types[:num_fixed]
            log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

        if fixed_pos is not None and num_fixed > 0:
            fixed_indices = torch.where(fixed_atom_mask == 1)[0]
            pos[fixed_indices] = fixed_pos[:num_fixed]

        # 固定部分的原始状态（用于RePaint）
        x0_fixed = pos[fixed_atom_mask == 1].clone() if fixed_pos is not None else None
        atom_types0_fixed = atom_types[fixed_atom_mask == 1].clone() if fixed_atom_types is not None else None

        # 键类型初始化
        if edge_index_template is not None and edge_index_template.size(1) > 0:
            src_edges, dst_edges = edge_index_template[0], edge_index_template[1]
            unique_edge_mask = src_edges < dst_edges
            edge_index_unique = edge_index_template[:, unique_edge_mask]
            num_edges_unique = edge_index_unique.size(1)
            bond_types_unique = torch.randint(0, CFG.num_bond_types, (num_edges_unique,), device=device)
            log_bond_types = index_to_log_onehot(bond_types_unique, CFG.num_bond_types)
            edge_batch = torch.zeros(num_edges_unique, dtype=torch.long, device=device)
        else:
            edge_index_unique = torch.empty((2, 0), dtype=torch.long, device=device)
            bond_types_unique = torch.empty((0,), dtype=torch.long, device=device)
            log_bond_types = None
            edge_batch = None

        # 原子特征初始化
        x = torch.zeros(num_atoms, 6, device=device)
        for i, at in enumerate(atom_types):
            if at < len(ATOM_TYPES):
                x[i, 0] = ATOM_TYPES[at]
        x[:, 0] = torch.where(x[:, 0] == 0, torch.tensor(6.0, device=device), x[:, 0])

        batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # 骨架节点特征处理
        h_scaffold = None
        pos_scaffold = None
        if scaffold_data is not None and hasattr(scaffold_data, 'x'):
            h_scaffold = self.scaffold_node_proj(scaffold_data.x.to(device))
            pos_scaffold = scaffold_data.pos.to(device)

        # RePaint采样调度
        schedule = self._get_repaint_schedule(resamplings, jump_length)

        # 从T开始逆向采样
        s = self.num_timesteps - 1
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                # 时间步
                t = torch.full((1,), s, dtype=torch.long, device=device)

                # ===== 固定部分：从原始数据加噪声 =====
                if x0_fixed is not None and num_fixed > 0:
                    # 对固定部分加噪到当前时间步
                    alpha_t = self.alphas_cumprod[t]
                    noise_fixed = torch.randn_like(x0_fixed)
                    pos_fixed_noised = alpha_t.sqrt() * x0_fixed + (1 - alpha_t).sqrt() * noise_fixed
                    fixed_indices = torch.where(fixed_atom_mask == 1)[0]
                    pos[fixed_indices] = pos_fixed_noised

                # ===== 可变部分：从模型预测 =====
                pos_pred, atom_logits, bond_logits = self.egnn(
                    x, pos, edge_index_template, t, scaffold_context, batch
                )

                # 位置采样
                x0_pred = (pos - self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 3) * pos_pred) / \
                          (self.sqrt_alphas_cumprod[t].unsqueeze(-1).expand(num_atoms, 3) + 1e-8)
                x0_pred = x0_pred.clamp(-10, 10)

                alpha_t = self.alphas[t]
                alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])[t]
                beta_t = self.betas[t]

                posterior_mean = (alpha_cumprod_prev.sqrt() * beta_t * x0_pred +
                                  alpha_t.sqrt() * (1 - alpha_cumprod_prev) * pos) / \
                                 (1 - self.alphas_cumprod[t] + 1e-8)

                if s > 0:
                    noise = torch.randn_like(pos)
                    pos_new = posterior_mean + self.posterior_variance[t].sqrt() * noise
                else:
                    pos_new = posterior_mean

                # 只更新可变部分的位置
                variable_mask = (fixed_atom_mask == 0)
                pos[variable_mask] = pos_new[variable_mask]
                pos = pos - pos.mean(dim=0)

                # 原子类型采样（仅可变部分）
                if self.use_discrete:
                    log_x_recon = F.log_softmax(atom_logits, dim=-1)
                    log_model_prob = self.atom_type_trans.q_v_posterior(
                        log_x_recon, log_atom_types, t, batch)
                    atom_types_new = log_sample_categorical(log_model_prob)

                    # 只更新可变部分
                    atom_types[variable_mask] = atom_types_new[variable_mask]
                    log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

                # 键类型采样
                if self.use_discrete and bond_logits is not None and log_bond_types is not None:
                    log_bond_recon = F.log_softmax(bond_logits, dim=-1)
                    log_bond_model_prob = self.bond_type_trans.q_v_posterior(
                        log_bond_recon, log_bond_types, t, edge_batch)
                    bond_types_unique = log_sample_categorical(log_bond_model_prob)
                    log_bond_types = index_to_log_onehot(bond_types_unique, CFG.num_bond_types)

                # 更新原子特征
                for i, at in enumerate(atom_types):
                    if at < len(ATOM_TYPES):
                        x[i, 0] = ATOM_TYPES[at]

                # ===== RePaint回跳 =====
                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    # 回跳jump_length步
                    t_jump = s + jump_length
                    t_jump = min(t_jump, self.num_timesteps - 1)

                    # 对可变部分加噪到回跳时间步
                    alpha_jump = self.alphas_cumprod[t_jump]
                    alpha_s = self.alphas_cumprod[s]

                    # 逆向加噪：从s到t_jump
                    pos_variable = pos[variable_mask]
                    noise_jump = torch.randn_like(pos_variable)
                    # z_t = alpha_t/z_s * z_s + sigma_t/z_s * noise
                    sigma_ratio = ((1 - alpha_jump) / (1 - alpha_s)).sqrt()
                    alpha_ratio = (alpha_jump / alpha_s).sqrt()
                    pos[variable_mask] = alpha_ratio * pos_variable + sigma_ratio * noise_jump

                    s = t_jump

                s -= 1

        # 最终处理
        if edge_index_template is not None:
            src, dst = edge_index_template
            unique_mask = src < dst
            edge_index_unique = edge_index_template[:, unique_mask]
            if bond_types_unique.size(0) == 0 and edge_index_unique.size(1) > 0:
                bond_types_unique = torch.zeros(edge_index_unique.size(1), dtype=torch.long, device=device)
        else:
            edge_index_unique = torch.empty((2, 0), dtype=torch.long, device=device)

        return {
            'pos': pos,
            'atom_types': atom_types,
            'bond_types': bond_types_unique,
            'edge_index': edge_index_unique,
        }

    def _get_repaint_schedule(self, resamplings: int, jump_length: int) -> List[int]:
        """
        生成RePaint采样调度（借鉴DiffSBDD第653-674行）

        返回一个整数列表，每个整数表示需要多少次去噪步骤后才回跳

        Args:
            resamplings: 每个时间步的重采样次数
            jump_length: 回跳的步数

        Returns:
            调度列表
        """
        schedule = []
        curr_t = 0
        while curr_t < self.num_timesteps:
            if curr_t + jump_length < self.num_timesteps:
                if len(schedule) > 0:
                    schedule[-1] += jump_length
                    schedule.extend([jump_length] * (resamplings - 1))
                else:
                    schedule.extend([jump_length] * resamplings)
                curr_t += jump_length
            else:
                residual = self.num_timesteps - curr_t
                if len(schedule) > 0:
                    schedule[-1] += residual
                else:
                    schedule.append(residual)
                curr_t += residual

        return list(reversed(schedule))


# =============================================================================
# 7. 主模型
# =============================================================================
class BioIsostericFragmentModel(nn.Module):
    """
    片段替换模型 - 升级版

    Stage 1: 3D骨架编码器预训练（EGNN编码骨架结构）
    Stage 2: 对比学习训练（学习骨架相似性）
    Stage 3: 片段扩散训练（batch训练）
    """

    def __init__(self, hidden_dim: int = 128, lr: float = 1e-4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        # 3D骨架编码器（EGNN）
        self.scaffold_3d_encoder = ScaffoldEncoder(
            node_dim=6,
            hidden_dim=hidden_dim,
            num_layers=3,
            edge_dim=20,
        )

        # 骨架指纹编码器（备用）
        self.scaffold_fp_encoder = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 片段扩散模型
        self.fragment_diffusion = FragmentDiffusion(
            hidden_dim=hidden_dim,
            num_timesteps=CFG.stage3_timesteps,
            use_discrete=CFG.use_discrete_diffusion,
        )

        # 对比学习头（Stage 2）
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 对比学习损失函数
        from torch.nn import TripletMarginLoss
        self.contrastive_criterion = TripletMarginLoss(margin=0.5)

        # 损失历史
        self.stage1_loss_history = []
        self.stage2_loss_history = []
        self.stage3_loss_history = []

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def train_stage1(self, dataset: FragmentDataset, epochs: int, batch_size: int):
        """
        Stage 1: 预训练3D骨架编码器

        目标：学习将骨架3D结构映射到有意义的嵌入空间
        方法：重建骨架节点特征（类似v2_3的context_encoder）
        """
        logger.info(f"Stage 1: 3D Scaffold encoder pretraining, epochs={epochs}")

        # 创建重建头
        recon_head = nn.Linear(self.hidden_dim, 6).to(self.device)  # 重建原子特征

        # 参数列表
        all_params = list(self.scaffold_3d_encoder.parameters()) + list(recon_head.parameters())
        optimizer = optim.Adam(all_params, lr=1e-3)

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            # 随机采样
            indices = np.random.permutation(len(dataset))

            pbar = tqdm(range(0, len(indices), batch_size),
                       desc=f"Stage 1 - Epoch {epoch+1}/{epochs}")

            for start in pbar:
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]

                # 收集骨架数据
                scaffold_data_list = []
                for idx in batch_indices:
                    decomp = dataset[idx]
                    scaffold_data_list.append(decomp['scaffold'])

                if len(scaffold_data_list) == 0:
                    continue

                # Batch化
                scaffold_batch = Batch.from_data_list(scaffold_data_list).to(self.device)

                optimizer.zero_grad()

                # 3D编码
                scaffold_encoding = self.scaffold_3d_encoder(
                    scaffold_batch.x,
                    scaffold_batch.pos,
                    scaffold_batch.edge_index,
                    scaffold_batch.batch
                )

                # 重建任务：预测节点特征
                # 扩展编码到每个节点
                scaffold_encoding_per_node = scaffold_encoding[scaffold_batch.batch]
                recon_feat = recon_head(scaffold_encoding_per_node)

                loss = F.mse_loss(recon_feat, scaffold_batch.x)

                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(count, 1)
            self.stage1_loss_history.append(avg_loss)
            logger.info(f"Stage 1 Epoch {epoch+1} avg_loss={avg_loss:.6f}")

    def train_stage2(self, dataset: FragmentDataset, epochs: int, batch_size: int) -> List[Tuple[int, int]]:
        """
        Stage 2: 对比学习训练 + 片段配对挖掘

        目标：学习骨架相似性，挖掘可互换片段配对
        """
        logger.info(f"Stage 2: Contrastive learning + pair mining, epochs={epochs}")

        # 挖掘配对
        pairs = mine_fragment_pairs(
            dataset,
            max_pairs=CFG.stage2_max_pairs,
            fp_threshold=CFG.stage2_fp_threshold
        )

        if len(pairs) == 0:
            logger.warning("No valid pairs found, using random pairs")
            n = len(dataset)
            pairs = [(i, (i + 1) % n) for i in range(min(n, CFG.stage2_max_pairs))]

        logger.info(f"Generated {len(pairs)} fragment pairs")

        # 对比学习训练（使用batch）
        # 创建配对数据集
        class ContrastivePairDataset(Dataset):
            def __init__(self, dataset, pairs):
                self.dataset = dataset
                self.pairs = pairs

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                i, j = self.pairs[idx]
                return self.dataset[i], self.dataset[j]

        pair_dataset = ContrastivePairDataset(dataset, pairs[:CFG.stage2_max_pairs])

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            # 随机采样batch - 限制采样数量不超过配对总数
            actual_batch_size = min(batch_size, len(pair_dataset))
            batch_indices = np.random.choice(len(pair_dataset), actual_batch_size, replace=False)

            pbar = tqdm(batch_indices, desc=f"Stage 2 - Epoch {epoch+1}/{epochs}")

            for idx in pbar:
                decomp_i, decomp_j = pair_dataset[idx]

                # 编码两个骨架
                scaffold_i = decomp_i['scaffold'].to(self.device)
                scaffold_j = decomp_j['scaffold'].to(self.device)

                # 添加batch向量
                scaffold_i.batch = torch.zeros(scaffold_i.x.size(0), dtype=torch.long, device=self.device)
                scaffold_j.batch = torch.zeros(scaffold_j.x.size(0), dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()

                # 3D编码
                enc_i = self.scaffold_3d_encoder(
                    scaffold_i.x, scaffold_i.pos,
                    scaffold_i.edge_index, scaffold_i.batch
                )
                enc_j = self.scaffold_3d_encoder(
                    scaffold_j.x, scaffold_j.pos,
                    scaffold_j.edge_index, scaffold_j.batch
                )

                # 对比学习头
                proj_i = self.contrast_head(enc_i)
                proj_j = self.contrast_head(enc_j)

                # 正样本对：相似骨架应该接近
                # 简化的对比损失：鼓励相似骨架编码接近
                loss = F.mse_loss(proj_i, proj_j)  # 正样本对应该接近

                # 添加负样本（随机不同的骨架）
                neg_idx = np.random.randint(0, len(dataset))
                while neg_idx == idx or neg_idx == (idx + 1) % len(dataset):
                    neg_idx = np.random.randint(0, len(dataset))

                decomp_neg = dataset[neg_idx]
                scaffold_neg = decomp_neg['scaffold'].to(self.device)
                scaffold_neg.batch = torch.zeros(scaffold_neg.x.size(0), dtype=torch.long, device=self.device)

                enc_neg = self.scaffold_3d_encoder(
                    scaffold_neg.x, scaffold_neg.pos,
                    scaffold_neg.edge_index, scaffold_neg.batch
                )
                proj_neg = self.contrast_head(enc_neg)

                # 负样本损失：不同骨架应该远离
                loss_neg = -F.logsigmoid(-F.cosine_similarity(proj_i, proj_neg)).mean()

                total_contrast_loss = loss + loss_neg

                if torch.isfinite(total_contrast_loss):
                    total_contrast_loss.backward()
                    self.optimizer.step()
                    total_loss += total_contrast_loss.item()
                    count += 1
                    pbar.set_postfix(loss=f"{total_contrast_loss.item():.4f}")

            avg_loss = total_loss / max(count, 1)
            self.stage2_loss_history.append(avg_loss)
            logger.info(f"Stage 2 Epoch {epoch+1} avg_loss={avg_loss:.6f}")

        return pairs[:CFG.stage3_max_pairs]

    def train_stage3(self, dataset: FragmentDataset, epochs: int,
                     batch_size: int, pairs: List[Tuple[int, int]], phase: int = 1):
        """
        Stage 3: 片段扩散训练（batch训练）

        目标：给定骨架编码，学习生成合适的片段
        [新增V5] 支持两阶段课程学习：
        - Phase 1: 自重建模式，学习完美重建原片段
        - Phase 2: 等排体生成模式，学习生成不同的替换片段

        Args:
            dataset: 片段数据集
            epochs: 训练轮数
            batch_size: batch大小
            pairs: 训练配对（Phase 1 为空列表，Phase 2 为等排体配对）
            phase: 训练阶段（1=自重建，2=等排体）
        """
        logger.info(f"Stage 3: Fragment diffusion training (batch), epochs={epochs}, phase={phase}")

        # ========== [V5改进] 显式phase参数处理 ==========
        if phase == 1:
            # Phase 1: 自重建模式 - 学习完美重建原片段
            logger.info("[V5] Phase 1: Self-reconstruction mode")
            pairs = [(i, i) for i in range(min(len(dataset), CFG.stage3_max_pairs))]
            logger.info(f"  Using {len(pairs)} self-reconstruction pairs (scaffold_i -> fragment_i)")
            actual_epochs = min(epochs, CFG.self_reconstruction_epochs)
            logger.info(f"  Training for {actual_epochs} epochs")

            # Phase 1 特殊设置
            use_noise_augmentation = True  # 20%随机扰动
            noise_augmentation_ratio = 0.2

        elif phase == 2:
            # Phase 2: 等排体生成模式 - 使用挖掘的等排体配对
            logger.info("[V5] Phase 2: Isosteric generation mode")
            if len(pairs) == 0:
                # 如果没有传入配对，重新挖掘
                logger.warning("No isostere pairs provided, mining fresh pairs...")
                pairs = mine_isostere_pairs(dataset, max_pairs=CFG.stage3_max_pairs)
            if len(pairs) < 100:
                logger.warning(f"Only {len(pairs)} isostere pairs, this may limit training effectiveness")
            logger.info(f"  Using {len(pairs)} isostere pairs (scaffold_i -> fragment_j)")
            actual_epochs = min(epochs, CFG.isostere_epochs)
            logger.info(f"  Training for {actual_epochs} epochs")

            # Phase 2 特殊设置：增强探索能力
            use_noise_augmentation = False  # 不使用噪声增强，让模型学习真实的等排体转换
            noise_augmentation_ratio = 0.0

        else:
            # 默认使用传入的配对
            if len(pairs) == 0:
                logger.warning("No pairs for Stage 3, using self-reconstruction as fallback")
                pairs = [(i, i) for i in range(min(len(dataset), CFG.stage3_max_pairs))]
            pairs = pairs[:CFG.stage3_max_pairs]
            actual_epochs = epochs
            use_noise_augmentation = False
            noise_augmentation_ratio = 0.0

        # 调整学习率
        original_lr = self.optimizer.param_groups[0]['lr']
        for pg in self.optimizer.param_groups:
            pg['lr'] = CFG.stage3_lr
        logger.info(f"Stage 3 LR adjusted to: {CFG.stage3_lr}")

        # [新增] 计算配对权重：基于片段稀有原子比例
        pair_weights = []
        for i, j in pairs:
            # 使用j索引的片段权重（因为训练时用j的片段）
            weight = dataset.get_sample_weight(j)
            pair_weights.append(weight)

        # [新增] 创建加权采样器
        # [优化] 每个epoch采样len(pairs)个样本，避免num_samples太大导致训练过慢
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=pair_weights,
            num_samples=len(pairs),  # 每个epoch采样pairs数个样本
            replacement=True  # 允许重复采样，这样稀有片段会被多次采样
        )

        # 创建扩散训练数据集
        class DiffusionPairDataset(Dataset):
            def __init__(self, dataset, pairs):
                self.dataset = dataset
                self.pairs = pairs

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                i, j = self.pairs[idx]
                return self.dataset[i], self.dataset[j]

        pair_dataset = DiffusionPairDataset(dataset, pairs)

        def diffusion_collate_fn(batch):
            """将配对batch化"""
            fragments = []
            scaffold_fps = []
            scaffold_data_list = []

            for decomp_i, decomp_j in batch:
                frag = decomp_j['fragment']
                # 确保所有数据在CPU上（统一处理）
                frag = frag.to('cpu')
                # 确保bond_types属性存在
                if not hasattr(frag, 'bond_types'):
                    frag.bond_types = torch.zeros(frag.edge_index.size(1), dtype=torch.long)
                fragments.append(frag)

                scaffold_fps.append(decomp_j['scaffold_fp'].to('cpu'))
                scaffold_data_list.append(decomp_j['scaffold'].to('cpu'))

            if len(fragments) == 0:
                return None

            # Batch.from_data_list在CPU上执行
            fragment_batch = Batch.from_data_list(fragments)
            scaffold_fp_batch = torch.stack(scaffold_fps)

            return fragment_batch, scaffold_fp_batch, scaffold_data_list

        # [改进] 使用WeightedRandomSampler过采样稀有原子片段
        # 注意：使用sampler时必须设置shuffle=False
        loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=False,
                          sampler=sampler, collate_fn=diffusion_collate_fn, num_workers=0)
        logger.info(f"Using WeightedRandomSampler for rare atom fragment oversampling")

        for epoch in range(actual_epochs):  # [V4] 使用actual_epochs（自重建或等排体模式）
            total_loss = 0
            # [修复V3] 添加新的loss类型 + [V4] x0损失
            loss_details = {'pos': 0, 'x0_pos': 0, 'x0_atom': 0, 'atom': 0, 'bond': 0, 'valence': 0,
                           'bond_distance': 0, 'connectivity': 0, 'diversity': 0}
            count = 0

            pbar = tqdm(loader, desc=f"Stage 3 - Epoch {epoch+1}/{actual_epochs}")

            for batch_data in pbar:
                if batch_data is None:
                    continue

                fragment_batch, scaffold_fp_batch, scaffold_data_list = batch_data
                fragment_batch = fragment_batch.to(self.device)
                scaffold_fp_batch = scaffold_fp_batch.to(self.device)

                batch_size_actual = scaffold_fp_batch.size(0)

                # 计算3D骨架编码（先移动到device再batch）
                scaffold_batch_list = []
                for scaffold in scaffold_data_list:
                    scaffold = scaffold.to(self.device)
                    scaffold.batch = torch.zeros(scaffold.x.size(0), dtype=torch.long, device=self.device)
                    scaffold_batch_list.append(scaffold)
                scaffold_batch = Batch.from_data_list(scaffold_batch_list)

                # 随机时间步
                t = torch.randint(0, CFG.stage3_timesteps, (batch_size_actual,), device=self.device)

                # 计算3D骨架编码
                with torch.no_grad():
                    scaffold_3d_encoding = self.scaffold_3d_encoder(
                        scaffold_batch.x, scaffold_batch.pos,
                        scaffold_batch.edge_index, scaffold_batch.batch
                    )

                self.optimizer.zero_grad()

                # [P2新增] 扩散训练（传入骨架节点数据用于3D Cross-Attention）
                losses = self.fragment_diffusion(
                    fragment_batch, scaffold_fp_batch, t,
                    scaffold_3d_encoding=scaffold_3d_encoding,
                    scaffold_data=scaffold_batch  # [P2] 传入骨架节点数据
                )

                if torch.isfinite(losses['total']):
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    total_loss += losses['total'].item()
                    for k in loss_details:
                        loss_details[k] += losses[k]
                    count += 1

                    pbar.set_postfix(
                        pos=f"{losses['pos']:.3f}",
                        atom=f"{losses['atom']:.3f}",
                        bond=f"{losses['bond']:.3f}",
                        bd=f"{losses['bond_distance']:.3f}",
                        conn=f"{losses['connectivity']:.3f}",
                        loss=f"{losses['total'].item():.4f}"
                    )

            avg_loss = total_loss / max(count, 1)
            avg_details = {k: v / max(count, 1) for k, v in loss_details.items()}

            self.stage3_loss_history.append(avg_loss)

            # [新增] Early Stopping机制 - 防止塌缩
            # 监控atom_loss：如果atom_loss过低（接近0）表示预测过于完美，可能塌缩
            # 或者如果atom_loss连续上升表示模型收敛到稳定模式
            current_atom_loss = avg_details['atom']

            # 记录atom_loss历史
            if not hasattr(self, 'atom_loss_history'):
                self.atom_loss_history = []
            self.atom_loss_history.append(current_atom_loss)

            # Early stopping条件：
            # [调整] 降低早停触发阈值，避免过早停止训练
            # 1. atom_loss < 0.005 连续10个epoch（预测接近完美，可能收敛）
            # 2. atom_loss连续10个epoch上升（收敛到稳定模式）
            # 3. atom_loss连续5个epoch为0（完全塌缩）
            early_stop = False
            early_stop_reason = ""

            if len(self.atom_loss_history) >= 10:
                # 检查条件1：atom_loss过低（训练收敛）
                if all(l < 0.005 for l in self.atom_loss_history[-10:]):
                    early_stop = True
                    early_stop_reason = f"atom_loss收敛(最近10个epoch: {self.atom_loss_history[-10:]})"

            if len(self.atom_loss_history) >= 10:
                # 检查条件2：atom_loss连续上升
                recent = self.atom_loss_history[-10:]
                if all(recent[i] >= recent[i-1] for i in range(1, 10)):
                    early_stop = True
                    early_stop_reason = f"atom_loss连续上升(最近10个epoch: {recent})"

                # 检查条件3：atom_loss连续为0（完全塌缩）
                if all(l == 0 for l in self.atom_loss_history[-5:]):
                    early_stop = True
                    early_stop_reason = "atom_loss连续5个epoch为0(完全塌缩)"

            logger.info(f"Stage 3 Epoch {epoch+1} avg_loss={avg_loss:.4f} "
                       f"(pos={avg_details['pos']:.3f}, x0_pos={avg_details['x0_pos']:.3f}, "
                       f"x0_atom={avg_details['x0_atom']:.3f}, atom={avg_details['atom']:.3f}, "
                       f"bond={avg_details['bond']:.3f}, bd={avg_details['bond_distance']:.3f}, "
                       f"conn={avg_details['connectivity']:.3f}, val={avg_details['valence']:.3f})")

            if early_stop:
                logger.warning(f"Early stopping triggered at epoch {epoch+1}: {early_stop_reason}")
                break

        # 恢复学习率
        for pg in self.optimizer.param_groups:
            pg['lr'] = original_lr

    def fit(self, dataset: FragmentDataset):
        """
        ★ [V5改进] 完整的两阶段课程学习训练流程

        支持灵活的测试配置：
        - test_stage_only: 只运行特定阶段
        - skip_stage1/skip_stage2/skip_phase1: 跳过特定阶段
        - load_phase1_weights: 加载预训练权重

        Stage 3 Phase 1: 自重建（学习基本生成能力）
        Stage 3 Phase 2: 等排体生成（学习真正的替换）
        """
        logger.info("=" * 60)
        logger.info("[V5] Training Configuration:")
        logger.info(f"  test_stage_only: {CFG.test_stage_only}")
        logger.info(f"  skip_stage1: {CFG.skip_stage1}")
        logger.info(f"  skip_stage2: {CFG.skip_stage2}")
        logger.info(f"  skip_phase1: {CFG.skip_phase1}")
        logger.info("=" * 60)

        pairs_stage2 = []

        # ========== Stage 1: 骨架编码器预训练 ==========
        if CFG.test_stage_only == 1:
            logger.info("[TEST MODE] Only running Stage 1")
            self.train_stage1(dataset, CFG.stage1_epochs, CFG.stage1_batch_size)
            logger.info("Stage 1 complete. Exiting.")
            return

        if not CFG.skip_stage1:
            self.train_stage1(dataset, CFG.stage1_epochs, CFG.stage1_batch_size)
        else:
            logger.info("[SKIP] Stage 1 skipped")

        # ========== Stage 2: 对比学习 + 配对挖掘 ==========
        if CFG.test_stage_only == 2:
            logger.info("[TEST MODE] Only running Stage 2")
            pairs_stage2 = self.train_stage2(dataset, CFG.stage2_epochs, CFG.stage2_batch_size)
            logger.info("Stage 2 complete. Exiting.")
            return

        if not CFG.skip_stage2:
            pairs_stage2 = self.train_stage2(dataset, CFG.stage2_epochs, CFG.stage2_batch_size)
        else:
            logger.info("[SKIP] Stage 2 skipped")

        # ========== Stage 3 Phase 1: 自重建 ==========
        if CFG.test_stage_only == 3:
            logger.info("[TEST MODE] Only running Stage 3 Phase 1")
            CFG.curriculum_phase = 1
            CFG.x0_loss_weight = CFG.x0_loss_weight_phase1
            self.train_stage3(dataset, CFG.self_reconstruction_epochs, CFG.stage3_batch_size,
                              pairs=[], phase=1)
            logger.info("Stage 3 Phase 1 complete. Exiting.")
            return

        if not CFG.skip_phase1:
            logger.info("=" * 60)
            logger.info("Stage 3 Phase 1: Self-reconstruction")
            logger.info(f"Goal: Learn to reconstruct fragments ({CFG.self_reconstruction_epochs} epochs)")
            logger.info("=" * 60)

            CFG.curriculum_phase = 1
            CFG.x0_loss_weight = CFG.x0_loss_weight_phase1

            self.train_stage3(dataset, CFG.self_reconstruction_epochs, CFG.stage3_batch_size,
                              pairs=[], phase=1)

            # 保存Phase 1权重
            phase1_weights_path = "bioisosteric_model_v2_5_phase1.pth"
            torch.save(self.state_dict(), phase1_weights_path)
            logger.info(f"Phase 1 weights saved: {phase1_weights_path}")
        else:
            logger.info("[SKIP] Phase 1 skipped")
            # 加载预训练权重
            if CFG.load_phase1_weights:
                logger.info(f"Loading Phase 1 weights from: {CFG.load_phase1_weights}")
                self.load_state_dict(torch.load(CFG.load_phase1_weights))
            else:
                logger.warning("No Phase 1 weights provided! Model may not be properly initialized.")

        # ========== Stage 3 Phase 2: 等排体生成 ==========
        if CFG.test_stage_only == 4:
            logger.info("[TEST MODE] Only running Stage 3 Phase 2")
            CFG.curriculum_phase = 2
            CFG.x0_loss_weight = CFG.x0_loss_weight_phase2
            isostere_pairs = mine_isostere_pairs(dataset, max_pairs=CFG.stage3_max_pairs)
            self.train_stage3(dataset, CFG.isostere_epochs, CFG.stage3_batch_size,
                              pairs=isostere_pairs, phase=2)
            logger.info("Stage 3 Phase 2 complete. Exiting.")
            return

        logger.info("=" * 60)
        logger.info("Stage 3 Phase 2: Isostere Generation")
        logger.info(f"Goal: Learn isostere replacements ({CFG.isostere_epochs} epochs)")
        logger.info("=" * 60)

        CFG.curriculum_phase = 2
        CFG.x0_loss_weight = CFG.x0_loss_weight_phase2

        # 挖掘等排体配对
        isostere_pairs = mine_isostere_pairs(dataset, max_pairs=CFG.stage3_max_pairs)

        if len(isostere_pairs) < 100:
            logger.warning(f"Only {len(isostere_pairs)} isostere pairs found!")
            logger.warning("This may be due to insufficient training data.")
            logger.warning("Consider using more data or relaxing thresholds.")
            # 如果等排体配对太少，用Stage 2的配对作为fallback
            if len(pairs_stage2) > 0:
                logger.info("Falling back to Stage 2 pairs for Phase 2 training")
                isostere_pairs = pairs_stage2[:CFG.stage3_max_pairs]

        self.train_stage3(dataset, CFG.isostere_epochs, CFG.stage3_batch_size,
                          pairs=isostere_pairs, phase=2)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Phase 1: Self-reconstruction ({CFG.self_reconstruction_epochs} epochs)")
        logger.info(f"Phase 2: Isostere generation ({CFG.isostere_epochs} epochs)")
        logger.info(f"Total Stage 3 epochs: {CFG.self_reconstruction_epochs + CFG.isostere_epochs}")
        logger.info("=" * 60)
        logger.info("=" * 60)

    @torch.no_grad()
    def generate_fragment(self, scaffold_fp: torch.Tensor,
                          fragment_template: Optional[Data] = None,
                          num_atoms: Optional[int] = None,
                          use_reference_guidance: bool = True,
                          guidance_scale: float = 2.0,
                          scaffold_data: Optional[Data] = None,
                          generation_mode: str = 'isostere') -> Dict:  # ★ [V5新增] generation_mode参数
        """
        给定骨架指纹，生成片段

        [改进] 支持原片段信息引导，提高生成相似度
        [修复V3] 新增scaffold_data参数，启用3D Cross-Attention
        [V5新增] generation_mode参数支持三种生成模式

        Args:
            scaffold_fp: 骨架指纹 (1, 2048)
            fragment_template: 参考片段（提供原子数和边结构）
            num_atoms: 如果没有模板，指定原子数
            use_reference_guidance: 是否使用原片段信息约束生成
            guidance_scale: 条件引导强度（越大越忠实于原片段）
            scaffold_data: [新增] 骨架图数据，用于3D Cross-Attention条件
            generation_mode: [V5新增] 生成模式
                - 'self_reconstruct': 自重建模式（用原片段引导，生成与原片段相似）
                - 'isostere': 等排体模式（不用原片段原子类型引导，让模型自由生成不同片段）
                - 'guided_isostere': 引导等排体（用原片段的位置和边结构，但不约束原子类型）

        Returns:
            生成的片段数据（包含original_indices如果模板有）
        """
        scaffold_fp = scaffold_fp.to(self.device)

        # 使用骨架指纹编码器处理
        scaffold_context = self.scaffold_fp_encoder(scaffold_fp)

        # ★★★ [V5新增] 根据generation_mode调整引导策略 ★★★
        if generation_mode == 'self_reconstruct':
            # 自重建模式：完全使用原片段引导，生成与原片段尽可能相似
            guidance_scale = 3.0  # 更强的引导
            use_reference_guidance = True
            logger.info("[V5] Generation mode: self_reconstruct (strong reference guidance)")

        elif generation_mode == 'isostere':
            # 等排体模式：不用原片段原子类型引导，让模型自由探索
            # 只用骨架指纹作为条件，允许模型生成不同的片段
            guidance_scale = 1.0  # 较弱的引导，允许更多探索
            use_reference_guidance = False  # 关闭参考引导
            logger.info("[V5] Generation mode: isostere (minimal guidance, free exploration)")

        elif generation_mode == 'guided_isostere':
            # 引导等排体：保留位置和边结构引导，但不约束原子类型
            # 这允许生成结构相似但原子组成不同的片段
            guidance_scale = 1.5  # 中等引导
            use_reference_guidance = True  # 开启参考引导，但只用于位置
            logger.info("[V5] Generation mode: guided_isostere (position guidance only)")

        # 片段扩散采样
        original_indices = None
        fragment_reference = None

        if fragment_template is not None:
            num_atoms = fragment_template.x.size(0)
            edge_index_template = fragment_template.edge_index.to(self.device)
            # 保存原始索引映射
            if hasattr(fragment_template, 'original_indices'):
                original_indices = fragment_template.original_indices.cpu().numpy()

            # [改进] 提取原片段信息用于引导
            if use_reference_guidance:
                # ★★★ [V5新增] guided_isostere模式：只保留位置和键信息，不保留原子类型 ★★★
                if generation_mode == 'guided_isostere':
                    # 只提取位置和键类型，不提取原子类型
                    logger.info("  [V5] guided_isostere: Using position/bond guidance only, NO atom type guidance")

                    # 提取坐标（如果有）
                    ref_pos = None
                    if hasattr(fragment_template, 'pos') and fragment_template.pos is not None:
                        ref_pos = fragment_template.pos.clone()

                    # 提取键类型用于引导
                    ref_bond_types = None
                    if hasattr(fragment_template, 'bond_types') and fragment_template.bond_types is not None:
                        ref_bond_types = fragment_template.bond_types.clone()

                    # ★ 关键：不包含atom_types，让模型自由决定原子类型
                    fragment_reference = {
                        'pos': ref_pos,
                        'bond_types': ref_bond_types,
                        # atom_types不包含！
                    }

                else:
                    # self_reconstruct模式：完整使用所有参考信息
                    # ★ [修复V4] 直接使用 atom_types 字段（已经是类型索引）
                    if hasattr(fragment_template, 'atom_types') and fragment_template.atom_types is not None:
                        atom_type_indices = fragment_template.atom_types.clone()
                        atom_types_raw = fragment_template.x[:, 0].long()  # 原子序数（仅用于显示）
                    else:
                        # 向后兼容：如果没有atom_types字段，从x[:,0]转换
                        atom_types_raw = fragment_template.x[:, 0].long()  # 原子序数
                        atom_type_indices = torch.zeros(num_atoms, dtype=torch.long)
                        for i, z in enumerate(atom_types_raw):
                            z_int = int(z.item())
                            if z_int in atom_type_map:
                                atom_type_indices[i] = atom_type_map[z_int]
                            else:
                                atom_type_indices[i] = len(ATOM_TYPES) - 1  # 未知类型

                    # ★ [诊断] 打印参考原子类型（修复：indices和atomic numbers对应正确）
                    ref_atom_indices = atom_type_indices.tolist()
                    ref_atom_z = [ATOM_TYPES[i] if i < len(ATOM_TYPES) else 0 for i in ref_atom_indices]
                    logger.info(f"  Reference atom types (indices): {ref_atom_indices}")
                    logger.info(f"  Reference atom types (atomic numbers): {ref_atom_z}")

                    # 提取坐标（如果有）
                    ref_pos = None
                    if hasattr(fragment_template, 'pos') and fragment_template.pos is not None:
                        ref_pos = fragment_template.pos.clone()

                    # ★ [修复V3-新增] 提取键类型用于引导
                    ref_bond_types = None
                    if hasattr(fragment_template, 'bond_types') and fragment_template.bond_types is not None:
                        ref_bond_types = fragment_template.bond_types.clone()

                    fragment_reference = {
                        'atom_types': atom_type_indices,
                        'pos': ref_pos,
                        'bond_types': ref_bond_types,  # ★ 新增：键类型引导
                    }
        else:
            if num_atoms is None:
                num_atoms = 5  # 默认
            edge_index_template = None

        result = self.fragment_diffusion.sample(
            num_atoms=num_atoms,
            scaffold_context=scaffold_fp,  # 直接用指纹
            edge_index_template=edge_index_template,
            fragment_reference=fragment_reference,
            guidance_scale=guidance_scale,
            scaffold_data=scaffold_data,  # ★ [修复V3] 传入骨架数据，启用3D Cross-Attention
        )

        # 添加原始索引信息
        if original_indices is not None:
            result['original_indices'] = original_indices

        # ★ [P0修复+边对齐+坐标修复] 添加原始键类型、边索引和坐标用于大片段重建
        if fragment_template is not None:
            # 原始键类型
            if hasattr(fragment_template, 'bond_types') and fragment_template.bond_types is not None:
                result['original_bond_types'] = fragment_template.bond_types.clone().cpu()
            # ★ [关键修复] 原始边索引 - 直接使用，不要再次过滤unique_mask
            # 因为 extract_subgraph 已经是单向边存储，每条键只存储一次
            if hasattr(fragment_template, 'edge_index') and fragment_template.edge_index is not None:
                result['original_edge_index'] = fragment_template.edge_index.clone().cpu()
            # ★★★ [关键修复] 原始坐标（确保环几何正确）
            if hasattr(fragment_template, 'pos') and fragment_template.pos is not None:
                result['original_pos'] = fragment_template.pos.clone().cpu()

        return result

    def reconstruct_molecule(self, scaffold_data: Data, fragment_result: Dict,
                             attachment_info: Tuple[int, int]) -> Chem.Mol:
        """
        将生成的片段拼接回骨架，重建完整分子

        Args:
            scaffold_data: 骨架数据
            fragment_result: 生成的片段结果
            attachment_info: (scaffold_attachment_idx, fragment_attachment_idx)

        Returns:
            重建的分子
        """
        # 创建骨架分子
        scaffold_mol = Chem.RWMol()
        scaffold_conf = Chem.Conformer()

        scaffold_atom_nums = [atom_type_map_rev.get(int(idx.item()), 6) or 6
                             for idx in scaffold_data.atom_types]
        scaffold_pos = scaffold_data.pos.cpu().numpy()

        for i, atom_num in enumerate(scaffold_atom_nums):
            atom = Chem.Atom(atom_num)
            scaffold_mol.AddAtom(atom)
            scaffold_conf.SetAtomPosition(i, tuple(scaffold_pos[i].tolist()))

        scaffold_mol.AddConformer(scaffold_conf)

        # 添加骨架内部键 - 边现在是单向存储，直接遍历
        src, dst = scaffold_data.edge_index.cpu()
        bond_types = scaffold_data.bond_types.cpu()

        # 诊断日志
        logger.info(f"    Scaffold edges: {src.size(0)}")
        logger.info(f"    Scaffold bond_types: {bond_types.tolist()[:10]}")

        scaffold_added_bonds = 0
        for i in range(src.size(0)):
            s, d = int(src[i].item()), int(dst[i].item())
            bond_idx = int(bond_types[i].item()) if i < bond_types.size(0) else 0
            # [修复] 保留AROMATIC键，让SanitizeMol自动处理芳香性，而不是降级为SINGLE
            bond_type = BOND_TYPE_MAP_REV.get(bond_idx, Chem.rdchem.BondType.SINGLE)
            scaffold_mol.AddBond(s, d, bond_type)
            scaffold_added_bonds += 1
        logger.info(f"    Added scaffold bonds: {scaffold_added_bonds}")

        # 诊断：检查骨架原子键数分布
        num_scaffold_atoms_local = scaffold_mol.GetNumAtoms()
        scaffold_bond_counts = []
        max_valence_map = {1: 1, 5: 3, 6: 4, 7: 5, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1, 0: 4}
        for i in range(num_scaffold_atoms_local):
            atom = scaffold_mol.GetAtomWithIdx(i)
            scaffold_bond_counts.append(atom.GetDegree())
        logger.info(f"    Scaffold bond counts (top 15): {scaffold_bond_counts[:15]}")
        # 检查是否有超过最大价键的原子
        for i, count in enumerate(scaffold_bond_counts):
            atom_num = scaffold_atom_nums[i]
            max_val = max_valence_map.get(atom_num, 4)
            if count > max_val:
                logger.warning(f"    Scaffold atom {i} ({atom_num}) has {count} bonds > max {max_val} BEFORE adding fragment!")
                # 这个骨架本身就有问题，跳过重建
                return None

        # 添加片段原子
        fragment_atom_nums = [atom_type_map_rev.get(int(idx.item()), 6) or 6   # 0（通配符）→ 6（碳）
                             for idx in fragment_result['atom_types']]

        # ★★★ [关键修复] 大片段使用原始坐标（确保环几何正确）
        num_frag_atoms = len(fragment_atom_nums)
        has_original_pos = (
            'original_pos' in fragment_result and
            fragment_result['original_pos'] is not None and
            fragment_result['original_pos'].size(0) >= num_frag_atoms
        )

        if num_frag_atoms >= 5 and has_original_pos:
            # 大片段：使用原始坐标（保留环几何）
            fragment_pos = fragment_result['original_pos'].cpu().numpy()
            logger.info(f"    Large fragment ({num_frag_atoms} atoms): using original coordinates to preserve ring geometry")
        else:
            # 小片段：使用扩散生成坐标
            fragment_pos = fragment_result['pos'].cpu().numpy()

        num_scaffold_atoms = scaffold_mol.GetNumAtoms()

        for i, atom_num in enumerate(fragment_atom_nums):
            atom = Chem.Atom(atom_num)
            scaffold_mol.AddAtom(atom)
            new_idx = num_scaffold_atoms + i
            scaffold_conf.SetAtomPosition(new_idx, tuple(fragment_pos[i].tolist()))

        # 添加片段内部键 - [P0修复+边对齐] 大片段使用原始边索引和键类型
        frag_src_gen, frag_dst_gen = fragment_result['edge_index'].cpu()
        frag_bond_types_gen = fragment_result['bond_types'].cpu()

        # ★ [关键修复] 大片段（≥5原子）优先使用原始边索引+键类型（完全对齐）
        num_frag_atoms = len(fragment_atom_nums)
        has_original_edge = (
            'original_edge_index' in fragment_result and
            fragment_result['original_edge_index'] is not None
        )
        has_original_bond = (
            'original_bond_types' in fragment_result and
            fragment_result['original_bond_types'] is not None
        )

        use_original_bonds = False
        if num_frag_atoms >= 5 and has_original_edge and has_original_bond:
            # 使用原始边索引和键类型（完全对齐，保留芳香性）
            original_ei = fragment_result['original_edge_index']
            frag_src = original_ei[0]
            frag_dst = original_ei[1]
            frag_bond_types = fragment_result['original_bond_types']
            use_original_bonds = True
            logger.info(f"    Large fragment ({num_frag_atoms} atoms): using original edge_index + bond_types to preserve aromaticity")
        else:
            frag_src = frag_src_gen
            frag_dst = frag_dst_gen
            frag_bond_types = frag_bond_types_gen

        # 添加诊断日志
        logger.info(f"    Fragment edges: {frag_src.size(0)}")
        logger.info(f"    Fragment bond_types: {frag_bond_types.tolist()[:10]}")
        if use_original_bonds:
            logger.info(f"    Original bond_types (used): {frag_bond_types.tolist()[:10]}")
            # 检查是否包含芳香键
            aromatic_count = (frag_bond_types == 3).sum().item()
            logger.info(f"    Aromatic bonds in original: {aromatic_count}")

        # 保存键类型信息用于后续fallback
        frag_bond_types_backup = frag_bond_types.clone()
        frag_added_bond_info = []  # 记录添加的键信息用于fallback

        added_bonds = 0
        for i in range(frag_src.size(0)):
            s = int(frag_src[i].item()) + num_scaffold_atoms
            d = int(frag_dst[i].item()) + num_scaffold_atoms
            bond_idx = int(frag_bond_types[i].item()) if i < frag_bond_types.size(0) else 0
            # [修复] 保留AROMATIC键，让SanitizeMol自动处理芳香性
            bond_type = BOND_TYPE_MAP_REV.get(bond_idx, Chem.rdchem.BondType.SINGLE)
            frag_added_bond_info.append((s, d, bond_idx))  # 记录用于fallback
            try:
                scaffold_mol.AddBond(s, d, bond_type)
                added_bonds += 1
            except:
                pass
        logger.info(f"    Added fragment bonds: {added_bonds}")

        # 连接骨架和片段（在attachment点）
        scaffold_attach_idx_original = attachment_info[0]  # 原始分子中的骨架连接点索引
        fragment_attach_idx_original = attachment_info[1]  # 原始分子中的片段连接点索引

        # 需要映射到新的骨架/片段内部索引
        scaffold_original_indices = scaffold_data.original_indices.cpu().numpy()

        # 诊断：打印attachment点信息
        logger.info(f"    Original attachment: scaffold={scaffold_attach_idx_original}, fragment={fragment_attach_idx_original}")
        logger.info(f"    Scaffold original indices: {scaffold_original_indices[:10]}...")

        # 找到原始索引在新骨架中的位置
        new_scaffold_attach = np.where(scaffold_original_indices == scaffold_attach_idx_original)[0]
        if len(new_scaffold_attach) == 0:
            logger.warning(f"Scaffold attachment point {scaffold_attach_idx_original} not found in scaffold")
            return None
        new_scaffold_attach = int(new_scaffold_attach[0])

        # 诊断：检查该原子在骨架中的当前状态
        scaffold_atom_num = scaffold_atom_nums[new_scaffold_attach]
        scaffold_atom = scaffold_mol.GetAtomWithIdx(new_scaffold_attach)
        current_bonds = scaffold_atom.GetDegree()
        max_valence_map = {1: 1, 5: 3, 6: 4, 7: 5, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1, 0: 4}
        max_valence = max_valence_map.get(scaffold_atom_num, 4)

        logger.info(f"    New scaffold attach idx: {new_scaffold_attach}, atom type: {scaffold_atom_num}")
        logger.info(f"    Current bonds: {current_bonds}, max valence: {max_valence}")

        if current_bonds >= max_valence:
            logger.warning(f"Attachment atom already has max bonds ({current_bonds}/{max_valence})")
            # 尝试找一个有空余键位的骨架原子
            found_alternative = False
            for alt_idx in range(num_scaffold_atoms):
                alt_atom_num = scaffold_atom_nums[alt_idx]
                alt_max = max_valence_map.get(alt_atom_num, 4)
                alt_current = scaffold_mol.GetAtomWithIdx(alt_idx).GetDegree()
                if alt_current < alt_max:
                    new_scaffold_attach = alt_idx
                    found_alternative = True
                    logger.info(f"    Alternative found: atom {alt_idx} ({alt_atom_num}) has {alt_current}/{alt_max} bonds")
                    break
            if not found_alternative:
                logger.warning(f"No alternative attachment point found, skipping...")
                return None

        # 片段连接点映射
        fragment_original_indices = fragment_result.get('original_indices')
        if fragment_original_indices is not None:
            new_fragment_attach_local = np.where(fragment_original_indices == fragment_attach_idx_original)[0]
            if len(new_fragment_attach_local) == 0:
                new_fragment_attach_local = 0
            else:
                new_fragment_attach_local = int(new_fragment_attach_local[0])
        else:
            new_fragment_attach_local = 0

        new_fragment_attach = num_scaffold_atoms + new_fragment_attach_local

        logger.info(f"    Fragment attach: local={new_fragment_attach_local}, global={new_fragment_attach}")

        # 添加连接键（默认单键）
        try:
            scaffold_mol.AddBond(new_scaffold_attach, new_fragment_attach,
                                 Chem.rdchem.BondType.SINGLE)
            logger.info(f"    Added attachment bond: {new_scaffold_attach} -> {new_fragment_attach}")
        except Exception as e:
            logger.warning(f"Failed to add attachment bond: {e}")
            return None

        # 尝试Sanitize（带价键修复）
        try:
            scaffold_mol = scaffold_mol.GetMol()
            Chem.SanitizeMol(scaffold_mol)

            # ★ [修复] 新增：3D几何优化，解决应变能nan问题
            # diffusion生成的坐标通常不物理合理，需要用ETKDGv3+MMFF优化
            mol_h = Chem.AddHs(scaffold_mol)
            embed_result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if embed_result == 0:  # 嵌入成功
                mmff_result = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)  # ★ 减少迭代次数避免卡住
                scaffold_mol = Chem.RemoveHs(mol_h)
                logger.info(f"    3D geometry optimized with ETKDGv3+MMFF")
            # embed失败时仍返回原mol（只是没有优化构象，应变能可能仍nan）

            logger.info(f"    Molecule reconstruction successful")
            return scaffold_mol
        except Exception as e:
            logger.warning(f"Initial sanitize failed: {e}, attempting valence fix...")

            # 应用价键修复策略（借鉴DecompDiff）
            scaffold_mol_fixed, fixed = fix_molecule_valence(scaffold_mol)

            if fixed:
                # ★ [修复] 对价键修复后的分子也进行几何优化
                mol_h = Chem.AddHs(scaffold_mol_fixed)
                embed_result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                if embed_result == 0:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                    scaffold_mol_fixed = Chem.RemoveHs(mol_h)
                    logger.info(f"    3D geometry optimized after valence fix")

                logger.info(f"    Molecule reconstruction successful after valence fix")
                return scaffold_mol_fixed
            else:
                # ★ [P0修复] 策略3：对大片段将所有内部键降级为单键，让RDKit推断芳香性
                if num_frag_atoms >= 5 and 'frag_added_bond_info' in dir():
                    logger.info(f"    Attempting fallback strategy: downgrade fragment bonds to SINGLE for large fragment")
                    try:
                        # 重建骨架部分（不包括片段键）
                        scaffold_mol_fallback = Chem.RWMol()
                        scaffold_conf_fallback = Chem.Conformer()

                        # 添加骨架原子
                        for i, atom_num in enumerate(scaffold_atom_nums):
                            atom = Chem.Atom(atom_num)
                            scaffold_mol_fallback.AddAtom(atom)
                            scaffold_conf_fallback.SetAtomPosition(i, tuple(scaffold_pos[i].tolist()))

                        # 添加骨架内部键
                        for i in range(src.size(0)):
                            s, d = int(src[i].item()), int(dst[i].item())
                            scaffold_mol_fallback.AddBond(s, d, Chem.rdchem.BondType.SINGLE)

                        # 添加片段原子
                        for i, atom_num in enumerate(fragment_atom_nums):
                            atom = Chem.Atom(atom_num)
                            scaffold_mol_fallback.AddAtom(atom)
                            new_idx = num_scaffold_atoms + i
                            scaffold_conf_fallback.SetAtomPosition(new_idx, tuple(fragment_pos[i].tolist()))

                        scaffold_mol_fallback.AddConformer(scaffold_conf_fallback)

                        # ★ 添加片段内部键 - 全部使用单键，让RDKit推断芳香性
                        for i in range(frag_src.size(0)):
                            s = int(frag_src[i].item()) + num_scaffold_atoms
                            d = int(frag_dst[i].item()) + num_scaffold_atoms
                            scaffold_mol_fallback.AddBond(s, d, Chem.rdchem.BondType.SINGLE)

                        # 添加连接键
                        scaffold_mol_fallback.AddBond(new_scaffold_attach, new_fragment_attach,
                                                       Chem.rdchem.BondType.SINGLE)

                        # 尝试Sanitize - RDKit会自动推断芳香性
                        scaffold_mol_fallback = scaffold_mol_fallback.GetMol()
                        Chem.SanitizeMol(scaffold_mol_fallback)

                        # 几何优化
                        mol_h = Chem.AddHs(scaffold_mol_fallback)
                        embed_result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                        if embed_result == 0:
                            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                            scaffold_mol_fallback = Chem.RemoveHs(mol_h)
                            logger.info(f"    3D geometry optimized after bond downgrade fallback")

                        logger.info(f"    Molecule reconstruction successful with bond downgrade fallback")
                        return scaffold_mol_fallback
                    except Exception as e2:
                        logger.warning(f"    Bond downgrade fallback also failed: {e2}")

                logger.warning(f"Molecule reconstruction failed even after all fallback strategies")
                return None

    def get_neural_3d_fingerprint(self, mol_or_data, use_pos: bool = True) -> np.ndarray:
        """
        提取分子的神经3D指纹（Neural 3D Fingerprint）

        [核心功能] 使用EGNN编码器提取语义化的3D分子表示
        - SE(3)不变：旋转/平移不影响指纹
        - 语义化：通过对比学习编码化学意义

        Args:
            mol_or_data: RDKit分子对象或PyG Data对象
            use_pos: 是否使用3D坐标（True时为纯3D指纹，False时退化为拓扑指纹）

        Returns:
            128维神经指纹向量 (numpy array)
        """
        self.eval()

        with torch.no_grad():
            # 处理输入
            if isinstance(mol_or_data, Chem.Mol):
                # 从RDKit分子转换为PyG Data
                data = self._mol_to_data(mol_or_data, use_pos)
            else:
                data = mol_or_data

            if data is None:
                return np.zeros(self.hidden_dim)

            data = data.to(self.device)

            # 使用EGNN编码器提取特征
            h = self.scaffold_3d_encoder.node_proj(data.x)

            if use_pos and hasattr(data, 'pos') and data.pos is not None:
                # 3D编码：使用EGNN层处理空间信息
                edge_index = data.edge_index if hasattr(data, 'edge_index') else self._get_fully_connected_edges(data.x.size(0))
                num_atoms = data.x.size(0)

                for egnn_layer in self.scaffold_3d_encoder.egnn_layers:
                    # EGNNLayer内部自己计算距离编码，只需要传入num_atoms
                    h, pos_update = egnn_layer(h, data.pos, edge_index, num_atoms)

            # [改进] 使用Attention Pooling聚合
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            attn_logits = self.scaffold_3d_encoder.attention_pool(h)  # (num_atoms, 1)
            attn_weights = self.scaffold_3d_encoder._batch_softmax(attn_logits, batch)
            weighted_h = h * attn_weights
            fp = global_mean_pool(weighted_h, batch)

            # 输出投影
            fp = self.scaffold_3d_encoder.output_proj(fp)

            # 不使用L2归一化，保持欧氏距离的区分性
            # 归一化会压缩相似分子的差异

            return fp.cpu().numpy().flatten()

    def _mol_to_data(self, mol: Chem.Mol, use_pos: bool = True) -> Data:
        """
        将RDKit分子转换为PyG Data对象

        Args:
            mol: RDKit分子
            use_pos: 是否生成3D坐标

        Returns:
            PyG Data对象
        """
        try:
            if mol is None:
                return None

            # 获取原子特征
            atoms = list(mol.GetAtoms())
            num_atoms = len(atoms)

            if num_atoms == 0:
                return None

            # 原子特征: [原子序数, 形式电荷, 杂化状态, 是否芳香, 是否在环中, 自由电子数]
            x = torch.zeros(num_atoms, 6)
            for i, atom in enumerate(atoms):
                x[i, 0] = atom.GetAtomicNum()
                x[i, 1] = atom.GetFormalCharge()
                x[i, 2] = int(atom.GetHybridization())
                x[i, 3] = float(atom.GetIsAromatic())
                x[i, 4] = float(atom.IsInRing())
                x[i, 5] = atom.GetNumRadicalElectrons()

            # 获取边索引
            bonds = list(mol.GetBonds())
            if len(bonds) > 0:
                edge_index = torch.zeros(2, len(bonds) * 2, dtype=torch.long)
                for i, bond in enumerate(bonds):
                    edge_index[0, i] = bond.GetBeginAtomIdx()
                    edge_index[1, i] = bond.GetEndAtomIdx()
                    edge_index[0, i + len(bonds)] = bond.GetEndAtomIdx()
                    edge_index[1, i + len(bonds)] = bond.GetBeginAtomIdx()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 3D坐标
            pos = None
            if use_pos:
                # 尝试获取现有构象或生成新构象
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    pos = torch.zeros(num_atoms, 3)
                    for i in range(num_atoms):
                        pos[i] = torch.tensor([conf.GetAtomPosition(i).x,
                                              conf.GetAtomPosition(i).y,
                                              conf.GetAtomPosition(i).z])
                else:
                    # 生成3D构象
                    mol_copy = Chem.RWMol(mol)
                    AllChem.EmbedMolecule(mol_copy, AllChem.ETKDGv3())
                    if mol_copy.GetNumConformers() > 0:
                        conf = mol_copy.GetConformer()
                        pos = torch.zeros(num_atoms, 3)
                        for i in range(num_atoms):
                            pos[i] = torch.tensor([conf.GetAtomPosition(i).x,
                                                  conf.GetAtomPosition(i).y,
                                                  conf.GetAtomPosition(i).z])

            return Data(x=x, edge_index=edge_index, pos=pos)

        except Exception as e:
            logger.warning(f"Failed to convert mol to data: {e}")
            return None

    def _get_fully_connected_edges(self, num_atoms: int) -> torch.Tensor:
        """生成完全连接图的边索引"""
        src = torch.arange(num_atoms, device=self.device).repeat_interleave(num_atoms)
        dst = torch.arange(num_atoms, device=self.device).repeat(num_atoms)
        mask = src != dst
        return torch.stack([src[mask], dst[mask]], dim=0)

    def compute_neural_similarity(self, mol1, mol2) -> float:
        """
        计算两个分子的神经3D指纹相似度

        Args:
            mol1, mol2: RDKit分子对象

        Returns:
            相似度分数 (0-1)
        """
        fp1 = self.get_neural_3d_fingerprint(mol1)
        fp2 = self.get_neural_3d_fingerprint(mol2)

        # Cosine相似度
        similarity = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2) + 1e-10)

        # 转换到0-1范围
        similarity = (similarity + 1) / 2

        return float(similarity)

    def train_conformation_consistency(self, dataset, num_epochs: int = 10, batch_size: int = 32,
                                        num_conformers: int = 10, temperature: float = 0.1,
                                        variance_weight: float = 0.5):
        """
        [改进版] 构象一致性对比学习

        训练目标：
        - 正样本：同分子不同构象 → 指纹相似
        - 负样本：不同分子构象 → 指纹不同（困难负样本挖掘）

        改进点：
        1. 多构象生成：每个分子生成num_conformers个构象，选取RMSD最大的两个
        2. MMFF优化：构象经过力场优化，更接近真实能量极小点
        3. 困难负样本：按2D指纹相似度挖掘相似分子作为负样本
        4. 温度参数：τ=0.07，更适合InfoNCE

        Args:
            dataset: FragmentDataset
            num_epochs: 训练轮数（默认10）
            batch_size: 批大小
            num_conformers: 每分子生成的构象数（默认10）
            temperature: InfoNCE温度参数（默认0.07）
        """
        logger.info("=" * 60)
        logger.info("Stage 2.5: Conformation Consistency Training (Improved)")
        logger.info("目标：神经3D指纹学会抓取核心拓扑特征，忽略构象噪音")
        logger.info("=" * 60)

        # 只微调EGNN编码器，冻结其他部分
        self.scaffold_3d_encoder.train()
        for name, param in self.named_parameters():
            if 'scaffold_3d_encoder' not in name:
                param.requires_grad = False

        optimizer = optim.Adam(self.scaffold_3d_encoder.parameters(), lr=1e-3)
        logger.info(f"Training {sum(p.numel() for p in self.scaffold_3d_encoder.parameters())} encoder params")

        # 预计算分子2D指纹用于困难负样本挖掘
        logger.info("Precomputing 2D fingerprints for hard negative mining...")
        mol_2d_fps = []
        mol_list = []
        for idx in range(len(dataset)):
            decomp = dataset[idx]
            mol = decomp['original_mol']
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                mol_2d_fps.append(fp)
                mol_list.append(mol)
            except:
                mol_2d_fps.append(None)
                mol_list.append(None)

        logger.info(f"Precomputed {len([f for f in mol_2d_fps if f is not None])} 2D fingerprints")

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            total_loss = 0
            total_pos_sim = 0  # 正样本相似度（监控）
            total_neg_sim = 0  # 负样本相似度（监控）
            total_attn_entropy = 0  # attention熵监控（新增）
            num_batches = 0

            # 随机采样
            valid_indices = [i for i in range(len(dataset)) if mol_2d_fps[i] is not None]
            indices = np.random.permutation(valid_indices)[:batch_size * 20]

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]

                # 直接在训练模式下提取指纹（保留梯度）
                batch_data_anchor = []
                batch_data_positive = []

                for idx in batch_indices:
                    mol = mol_list[idx]
                    if mol is None:
                        continue

                    try:
                        # 生成多个构象并选取差异最大的两个
                        conf_pair = self._generate_diverse_conformers(mol, num_conformers)
                        if conf_pair is None:
                            continue

                        mol1, mol2 = conf_pair

                        # 转换为PyG Data（保留梯度）
                        data1 = self._mol_to_data(mol1, use_pos=True)
                        data2 = self._mol_to_data(mol2, use_pos=True)

                        if data1 is not None and data2 is not None and data1.pos is not None:
                            batch_data_anchor.append(data1)
                            batch_data_positive.append(data2)

                    except Exception as e:
                        continue

                if len(batch_data_anchor) < 2:
                    continue

                # 批量处理：使用编码器直接提取指纹（保留梯度）
                batch_anchor = Batch.from_data_list(batch_data_anchor).to(self.device)
                batch_positive = Batch.from_data_list(batch_data_positive).to(self.device)

                # 提取指纹（不使用no_grad）- 同时获取attention weights用于多样性约束
                anchor_fp, _, anchor_attn = self._extract_fingerprint_batch(batch_anchor, return_intermediate=True)
                positive_fp, _, positive_attn = self._extract_fingerprint_batch(batch_positive, return_intermediate=True)

                batch_size_cur = anchor_fp.size(0)

                # [改进] 使用欧氏距离损失 + 方差正则化，防止collapse
                # 正样本：同分子不同构象，距离应该小
                pos_dist = torch.norm(anchor_fp - positive_fp, dim=-1)  # (batch_size,)

                # 负样本：不同分子，距离应该大
                # 使用L2归一化后的cosine来计算等价的欧氏距离
                anchor_norm = F.normalize(anchor_fp, p=2, dim=-1)
                neg_dist_matrix = 2 - 2 * torch.mm(anchor_norm, anchor_norm.t())  # 等价欧氏距离
                mask = ~torch.eye(batch_size_cur, dtype=torch.bool, device=self.device)
                neg_dist = neg_dist_matrix[mask].view(batch_size_cur, batch_size_cur - 1)

                # Triplet-like损失：pos_dist应小，neg_dist应大
                # margin确保正负样本有足够间隔
                margin = 1.0
                triplet_loss = F.relu(pos_dist.unsqueeze(1) - neg_dist + margin).mean()

                # [关键] 方差正则化：确保指纹维度有足够方差，防止collapse
                # 每个维度的方差应该大于阈值
                fp_variance = torch.var(anchor_fp, dim=0)  # (hidden_dim,)
                variance_loss = F.relu(0.1 - fp_variance).mean()  # 方差小于0.1时惩罚

                # [新增] Attention权重多样性约束：防止所有原子权重相同
                # 计算每个分子内部attention weights的熵，熵越高说明权重越分散
                # 理想状态：不同原子有不同权重，熵应该接近最大值
                def compute_attention_entropy(attn_weights, batch):
                    """计算每个分子的attention熵"""
                    entropies = []
                    for mol_idx in range(batch.max().item() + 1):
                        mol_mask = (batch == mol_idx)
                        mol_attn = attn_weights[mol_mask]
                        # 熵计算: -sum(p * log(p))
                        # 归一化到概率分布
                        mol_attn_prob = mol_attn / (mol_attn.sum() + 1e-8)
                        entropy = -(mol_attn_prob * torch.log(mol_attn_prob + 1e-8)).sum()
                        max_entropy = torch.log(torch.tensor(mol_attn.shape[0], dtype=torch.float, device=self.device))
                        # 归一化熵：0-1范围，1表示完全均匀，0表示单一原子主导
                        normalized_entropy = entropy / (max_entropy + 1e-8)
                        entropies.append(normalized_entropy)
                    return torch.stack(entropies)

                # Attention熵应该高（权重分散），否则惩罚
                anchor_entropy = compute_attention_entropy(anchor_attn, batch_anchor.batch)
                positive_entropy = compute_attention_entropy(positive_attn, batch_positive.batch)
                # 目标：熵应该大于0.5（至少有一定分散度）
                attention_entropy_loss = F.relu(0.5 - anchor_entropy).mean() + F.relu(0.5 - positive_entropy).mean()

                # 总损失
                # 总损失（添加attention多样性约束）
                loss = triplet_loss + variance_weight * variance_loss + 0.5 * attention_entropy_loss

                # 监控指标
                total_pos_sim += (1 - pos_dist.mean() / 10).item()  # 转换为相似度格式
                total_neg_sim += (neg_dist.mean() / 10).item()
                total_attn_entropy += anchor_entropy.mean().item()  # 新增监控

                total_loss += loss.item()
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / max(num_batches, 1)
            avg_pos_sim = total_pos_sim / max(num_batches, 1)
            avg_neg_sim = total_neg_sim / max(num_batches, 1)
            avg_attn_entropy = total_attn_entropy / max(num_batches, 1)

            # 计算当前指纹方差
            fp_var = torch.var(anchor_fp, dim=0).mean().item()

            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} | "
                        f"Pos Dist: {avg_pos_sim:.3f} | Neg Dist: {avg_neg_sim:.3f} | "
                        f"FP Var: {fp_var:.4f} | Attn Entropy: {avg_attn_entropy:.3f}")

            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复梯度状态
        for param in self.parameters():
            param.requires_grad = True

        logger.info("Conformation consistency training complete!")

    def _extract_fingerprint_batch(self, batch_data: Batch, return_intermediate: bool = False) -> torch.Tensor:
        """
        从批量Data中提取神经指纹（保留梯度）

        [改进] 添加方差正则化信号，防止collapse

        Args:
            batch_data: PyG Batch对象
            return_intermediate: 是否返回中间特征（用于方差监控）

        Returns:
            指纹tensor [batch_size, hidden_dim]
        """
        # 使用编码器提取特征（不使用no_grad）
        h = self.scaffold_3d_encoder.node_proj(batch_data.x)

        if batch_data.pos is not None:
            edge_index = batch_data.edge_index if hasattr(batch_data, 'edge_index') else \
                         self._get_fully_connected_edges(batch_data.x.size(0))
            num_atoms = batch_data.x.size(0)

            for egnn_layer in self.scaffold_3d_encoder.egnn_layers:
                h, pos_update = egnn_layer(h, batch_data.pos, edge_index, num_atoms)

        # [改进] 使用Attention Pooling聚合
        attn_logits = self.scaffold_3d_encoder.attention_pool(h)  # (num_atoms, 1)
        attn_weights = self.scaffold_3d_encoder._batch_softmax(attn_logits, batch_data.batch)
        weighted_h = h * attn_weights
        fp = global_mean_pool(weighted_h, batch_data.batch)

        # 输出投影
        fp = self.scaffold_3d_encoder.output_proj(fp)

        # 添加方差正则化信号（防止collapse）
        # 这会在训练损失中添加方差惩罚
        if return_intermediate:
            return fp, h, attn_weights

        return fp

    def _generate_diverse_conformers(self, mol: Chem.Mol, num_conformers: int = 10) -> Tuple[Chem.Mol, Chem.Mol]:
        """
        生成多个构象并选取差异最大的两个

        Args:
            mol: RDKit分子
            num_conformers: 生成的构象数量

        Returns:
            (mol1, mol2): 两个差异最大的构象（作为不同分子对象）
        """
        try:
            mol_copy = Chem.RWMol(mol)

            # 生成多个构象
            conf_ids = AllChem.EmbedMultipleConfs(mol_copy, num_conformers, AllChem.ETKDGv3())

            if len(conf_ids) < 2:
                return None

            # MMFF力场优化（可选）
            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol_copy, numThreads=1)
            except:
                pass  # MMFF可能失败，跳过

            # 计算所有构象对的RMSD，选取最大的
            best_pair = None
            best_rmsd = 0

            for i in range(len(conf_ids)):
                for j in range(i+1, len(conf_ids)):
                    rmsd = AllChem.GetBestRMS(mol_copy, mol_copy, conf_ids[i], conf_ids[j])
                    if rmsd > best_rmsd:
                        best_rmsd = rmsd
                        best_pair = (conf_ids[i], conf_ids[j])

            if best_pair is None:
                return None

            # 提取两个构象作为独立分子
            conf1_id, conf2_id = best_pair

            mol1 = Chem.RWMol(mol)
            mol2 = Chem.RWMol(mol)

            # 复制坐标
            conf1 = mol_copy.GetConformer(conf1_id)
            conf2 = mol_copy.GetConformer(conf2_id)

            AllChem.EmbedMolecule(mol1, AllChem.ETKDGv3())
            AllChem.EmbedMolecule(mol2, AllChem.ETKDGv3())

            if mol1.GetNumConformers() > 0 and mol2.GetNumConformers() > 0:
                # 替换坐标
                c1 = mol1.GetConformer()
                c2 = mol2.GetConformer()
                for atom_idx in range(mol.GetNumAtoms()):
                    pos1 = conf1.GetAtomPosition(atom_idx)
                    pos2 = conf2.GetAtomPosition(atom_idx)
                    c1.SetAtomPosition(atom_idx, pos1)
                    c2.SetAtomPosition(atom_idx, pos2)

                logger.debug(f"Selected conformer pair with RMSD={best_rmsd:.2f}")
                return mol1, mol2

            return None

        except Exception as e:
            logger.warning(f"Conformer generation failed: {e}")
            return None

    def _find_hard_negative(self, anchor_idx: int, mol_2d_fps: list, mol_list: list,
                            valid_indices: list, similarity_threshold: float = 0.3) -> Optional[int]:
        """
        困难负样本挖掘：找相似但不同的分子

        Args:
            anchor_idx:锚点分子索引
            mol_2d_fps: 预计算的2D指纹列表
            mol_list: 分子列表
            valid_indices: 有效索引列表
            similarity_threshold: 相似度阈值（找相似但不完全相同的）

        Returns:
            困难负样本的索引，或None
        """
        anchor_fp = mol_2d_fps[anchor_idx]
        if anchor_fp is None:
            return None

        # 随机采样候选
        candidate_indices = np.random.choice(valid_indices, size=min(50, len(valid_indices)), replace=False)
        candidate_indices = [i for i in candidate_indices if i != anchor_idx]

        if len(candidate_indices) == 0:
            return None

        best_idx = None
        best_sim = 0

        for cand_idx in candidate_indices:
            cand_fp = mol_2d_fps[cand_idx]
            if cand_fp is None:
                continue

            sim = DataStructs.TanimotoSimilarity(anchor_fp, cand_fp)

            # 困难负样本：相似度在阈值附近（相似但不完全相同）
            if sim > best_sim and sim < similarity_threshold + 0.2:
                best_sim = sim
                best_idx = cand_idx

        return best_idx


# =============================================================================
# 8. 主函数
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("BioIsosteric V2.5 Fragment Diffusion Model")
    logger.info("[V5改进] 两阶段课程学习：Phase 1 自重建 + Phase 2 等排体")
    logger.info("方案A：只扩散片段，骨架作为条件")
    logger.info("=" * 60)
    logger.info(f"Config: {CFG}")

    # 加载数据
    logger.info("Loading and decomposing molecules...")
    dataset = FragmentDataset(
        data_dir=CFG.data_dir,
        max_mols=CFG.test_mode_limit if CFG.test_mode else None
    )

    if len(dataset) == 0:
        logger.error("No valid decompositions found!")
        return

    logger.info(f"Dataset size: {len(dataset)} decomposed molecules")

    # 创建模型
    model = BioIsostericFragmentModel(
        hidden_dim=128,
        lr=1e-4,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 训练（包含两阶段课程学习）
    model.fit(dataset)

    # 保存
    torch.save(model.state_dict(), "bioisosteric_fragment_model_v2_5.pth")
    logger.info("Model saved: bioisosteric_fragment_model_v2_5.pth")

    # 绘制损失曲线
    plot_loss_curves(model)

    # ★★★ [V5改进] Demo推理：使用等排体模式 ★★★
    logger.info("\n[V5] Running demo inference in isostere mode...")
    demo_inference(model, dataset, num_samples=15, generation_mode='isostere')


def plot_loss_curves(model):
    """绘制损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if model.stage1_loss_history:
        axes[0].plot(model.stage1_loss_history)
        axes[0].set_title('Stage 1: Scaffold Encoder')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

    if model.stage2_loss_history:
        axes[1].plot(model.stage2_loss_history)
        axes[1].set_title('Stage 2: Contrastive Learning')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')

    if model.stage3_loss_history:
        axes[2].plot(model.stage3_loss_history)
        axes[2].set_title('Stage 3: Fragment Diffusion')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_loss_curves_v2_5.png')
    logger.info("Loss curves saved: training_loss_curves_v2_5.png")


# =============================================================================
# [V5新增] 等排体质量评估函数
# =============================================================================
def evaluate_isostere_quality(generated_mol, original_mol, scaffold_mol=None):
    """
    ★ [V5新增] 等排体质量评估

    等排体的目标不是高相似度，而是功能等价。因此需要新的评估指标：

    指标：
    1. 化学有效性（能否通过sanitize）
    2. 分子量相近（±50 Da）
    3. logP相近（±1.5）
    4. QED（药物相似性）相近
    5. 整体分子相似度（骨架+新片段 vs 原分子）
    6. 片段多样性（与原片段的差异程度）
    7. 综合等排体评分

    Args:
        generated_mol: 生成的分子（骨架+新片段）
        original_mol: 原始分子
        scaffold_mol: 骨架分子（可选，用于进一步分析）

    Returns:
        dict: 包含各项评估指标的字典
    """
    from rdkit.Chem import Descriptors, QED, AllChem
    from rdkit import DataStructs

    results = {}

    # 1. 化学有效性检查
    try:
        Chem.SanitizeMol(generated_mol)
        results['valid'] = True
        results['validity_error'] = None
    except Exception as e:
        results['valid'] = False
        results['validity_error'] = str(e)
        # 如果无效，后续指标无法计算，返回基础结果
        results['isostere_score'] = 0.0
        return results

    # 2. 分子量相近性（等排体应该有相近的分子量）
    mw_orig = Descriptors.MolWt(original_mol)
    mw_gen = Descriptors.MolWt(generated_mol)
    results['mw_orig'] = mw_orig
    results['mw_gen'] = mw_gen
    results['mw_diff'] = abs(mw_orig - mw_gen)
    results['mw_similar'] = results['mw_diff'] < 50  # 50 Da阈值
    results['mw_ratio'] = mw_gen / mw_orig if mw_orig > 0 else 0

    # 3. logP相近性（等排体应该保持相近的脂溶性）
    logp_orig = Descriptors.MolLogP(original_mol)
    logp_gen = Descriptors.MolLogP(generated_mol)
    results['logp_orig'] = logp_orig
    results['logp_gen'] = logp_gen
    results['logp_diff'] = abs(logp_orig - logp_gen)
    results['logp_similar'] = results['logp_diff'] < 1.5  # 1.5阈值
    results['logp_ratio'] = logp_gen / logp_orig if abs(logp_orig) > 0.01 else 1.0

    # 4. QED药物相似性（等排体替换应该保持或改善药物相似性）
    qed_orig = QED.qed(original_mol)
    qed_gen = QED.qed(generated_mol)
    results['qed_orig'] = qed_orig
    results['qed_gen'] = qed_gen
    results['qed_diff'] = abs(qed_orig - qed_gen)
    results['qed_improved'] = qed_gen >= qed_orig * 0.9  # 允许10%下降

    # 5. 整体分子相似度（骨架+新片段 vs 原分子）
    # 使用Morgan指纹计算Tanimoto相似度
    fp_gen = AllChem.GetMorganGenerator(radius=2).GetFingerprint(generated_mol)
    fp_orig = AllChem.GetMorganGenerator(radius=2).GetFingerprint(original_mol)
    whole_mol_sim = DataStructs.TanimotoSimilarity(fp_orig, fp_gen)
    results['whole_mol_sim'] = whole_mol_sim

    # 6. 片段多样性（等排体的核心指标：片段应该不同）
    # 注意：对于等排体，片段相似度低是好的（说明发生了真正的替换）
    results['fragment_diversity'] = 1.0 - whole_mol_sim
    results['isostere_replacement'] = whole_mol_sim < 0.4  # 相似度<0.4表示发生了替换

    # 7. TPSA（拓扑极性表面积）- 影响药物渗透性
    tpsa_orig = Descriptors.TPSA(original_mol)
    tpsa_gen = Descriptors.TPSA(generated_mol)
    results['tpsa_orig'] = tpsa_orig
    results['tpsa_gen'] = tpsa_gen
    results['tpsa_diff'] = abs(tpsa_orig - tpsa_gen)

    # 8. 氢键 donors/acceptors（影响药物-靶点相互作用）
    hbd_orig = Descriptors.NumHDonors(original_mol)
    hbd_gen = Descriptors.NumHDonors(generated_mol)
    hba_orig = Descriptors.NumHAcceptors(original_mol)
    hba_gen = Descriptors.NumHAcceptors(generated_mol)
    results['hbd_diff'] = abs(hbd_orig - hbd_gen)
    results['hba_diff'] = abs(hba_orig - hba_gen)
    results['hbond_preserved'] = results['hbd_diff'] <= 1 and results['hba_diff'] <= 1

    # 9. 环系统（保留或改变环结构）
    rings_orig = original_mol.GetRingInfo().NumRings()
    rings_gen = generated_mol.GetRingInfo().NumRings()
    results['rings_orig'] = rings_orig
    results['rings_gen'] = rings_gen
    results['rings_diff'] = abs(rings_orig - rings_gen)

    # 10. 综合等排体评分
    # 等排体评分 = 有效性 × 分子量相近 × logP相近 × 片段有差异 × QED保持
    # 注意：片段多样性是加分项（与原片段不同才是等排体）
    score_components = [
        float(results['valid']),  # 必须有效
        float(results['mw_similar']),  # 分子量相近
        float(results['logp_similar']),  # logP相近
        min(results['fragment_diversity'] * 2, 1.0),  # 片段差异（越大越好，上限1）
        float(results['qed_improved']),  # QED保持或改善
    ]
    results['isostere_score'] = sum(score_components) / len(score_components)

    # 11. 等排体分类（基于评分）
    if results['isostere_score'] >= 0.8:
        results['isostere_class'] = 'Excellent'
    elif results['isostere_score'] >= 0.6:
        results['isostere_class'] = 'Good'
    elif results['isostere_score'] >= 0.4:
        results['isostere_class'] = 'Moderate'
    else:
        results['isostere_class'] = 'Poor'

    return results


def demo_inference(model, dataset, num_samples=15, generation_mode='isostere'):
    """Demo推理测试 - 显示ACCFG官能团信息，并保存结果到results文件夹"""
    from datetime import datetime

    # 生成带时间戳的结果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"v2_5_results_{timestamp}.txt"  # ★ [V5] 更新版本号
    result_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", result_filename)

    # 确保results目录存在
    results_dir = os.path.dirname(result_filepath)
    os.makedirs(results_dir, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("Demo Inference (ACCFG-based decomposition)")
    logger.info(f"Results will be saved to: {result_filepath}")
    logger.info("=" * 60)

    success_count = 0
    total_count = 0
    results_data = []  # 存储所有结果用于汇总

    # 打开结果文件
    with open(result_filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Bioisosteric Fragment Diffusion Results - v2_5\n")  # ★ [V5] 更新版本号
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generation mode: {generation_mode}\n")  # ★ [V5] 记录生成模式
        f.write("=" * 60 + "\n\n")

        # 选多个样本测试
        for idx in range(min(num_samples, len(dataset))):
            decomp = dataset[idx]

            logger.info(f"\nSample {idx}:")
            f.write(f"\n{'='*40}\n")
            f.write(f"Sample {idx}:\n")
            f.write(f"{'='*40}\n")

            logger.info(f"  Scaffold atoms: {decomp['scaffold'].x.size(0)}")
            logger.info(f"  Fragment atoms: {decomp['fragment'].x.size(0)}")
            f.write(f"Scaffold atoms: {decomp['scaffold'].x.size(0)}\n")
            f.write(f"Fragment atoms: {decomp['fragment'].x.size(0)}\n")

            # 显示官能团信息
            if 'fg_info' in decomp:
                fg_name, fg_indices = decomp['fg_info']
                logger.info(f"  Functional group: {fg_name} (atoms: {fg_indices})")
                f.write(f"Functional group: {fg_name} (atoms: {fg_indices})\n")

            # 参考原子类型 - 使用 atom_types 字段（已经是类型索引）
            ref_atom_indices = decomp['fragment'].atom_types.tolist()
            # 转换为原子序数用于显示
            ref_atom_z = [ATOM_TYPES[i] if i < len(ATOM_TYPES) else 0 for i in ref_atom_indices]
            logger.info(f"  Reference atom types (indices): {ref_atom_indices}")
            logger.info(f"  Reference atom types (atomic numbers): {ref_atom_z}")
            f.write(f"Reference atom types (indices): {ref_atom_indices}\n")
            f.write(f"Reference atom types (atomic numbers): {ref_atom_z}\n")

            # ★★★ [V5改进] 使用generation_mode参数 ★★★
            # 生成片段 - [修复] 传入scaffold_data启用3D Cross-Attention
            result = model.generate_fragment(
                scaffold_fp=decomp['scaffold_fp'].unsqueeze(0),
                fragment_template=decomp['fragment'],
                scaffold_data=decomp['scaffold'],  # ★ 新增：传入骨架数据，启用3D条件
                generation_mode=generation_mode,  # ★ [V5新增]：指定生成模式
            )

            logger.info(f"  Generated fragment atoms: {result['atom_types'].size(0)}")
            # ★ [诊断] 打印生成的原子类型具体值
            generated_atom_indices = result['atom_types'].tolist()
            generated_atom_z = [ATOM_TYPES[i] if i < len(ATOM_TYPES) else 0 for i in generated_atom_indices]
            logger.info(f"  Generated atom types (indices): {generated_atom_indices}")
            logger.info(f"  Generated atom types (atomic numbers): {generated_atom_z}")
            f.write(f"Generated atom types (indices): {generated_atom_indices}\n")
            f.write(f"Generated atom types (atomic numbers): {generated_atom_z}\n")

            # 记录原子类型匹配情况
            atom_match = generated_atom_indices == ref_atom_indices
            f.write(f"Atom types match: {atom_match}\n")

            # ★★★ [修复] 使用try-except包裹重建过程，防止某个样本卡住导致后续样本无法处理 ★★★
            sample_result = {
                'idx': idx,
                'fg_name': fg_name if 'fg_info' in decomp else 'unknown',
                'ref_atoms': ref_atom_z,
                'gen_atoms': generated_atom_z,
                'atom_match': atom_match,
                'success': False,
                'smiles_gen': None,
                'smiles_orig': None,
                'strain_energy': None,
                'fp_similarity': None
            }

            try:
                # 重建分子（可能耗时较长）
                reconstructed = model.reconstruct_molecule(
                    decomp['scaffold'],
                    result,
                    decomp['attachment'],
                )

                if reconstructed is not None:
                    smiles = Chem.MolToSmiles(reconstructed)
                    logger.info(f"  Reconstructed SMILES: {smiles}")
                    f.write(f"Reconstructed SMILES: {smiles}\n")

                    # 原始分子对比
                    original_smiles = Chem.MolToSmiles(decomp['original_mol'])
                    logger.info(f"  Original SMILES: {original_smiles}")
                    f.write(f"Original SMILES: {original_smiles}\n")

                    sample_result['success'] = True
                    sample_result['smiles_gen'] = smiles
                    sample_result['smiles_orig'] = original_smiles

                    # [新增V4] 物理合理性评估
                    try:
                        from evaluate_3d import compute_strain_energy, compute_bond_length_distribution

                        # 计算构象应变能
                        strain_result = compute_strain_energy(reconstructed)
                        strain_energy = strain_result.get('strain_mmff', float('inf'))
                        is_reasonable = strain_result.get('is_reasonable', False)
                        logger.info(f"  Strain energy: {strain_energy:.2f} kcal/mol (reasonable={is_reasonable})")
                        f.write(f"Strain energy: {strain_energy:.2f} kcal/mol (reasonable={is_reasonable})\n")

                        sample_result['strain_energy'] = strain_energy

                        # 计算键长分布
                        bond_stats = compute_bond_length_distribution(reconstructed)
                        mean_lengths = bond_stats.get('mean_lengths', {})
                        if mean_lengths:
                            # 显示主要键类型的键长
                            for pair, mean_len in list(mean_lengths.items())[:3]:
                                std_len = bond_stats.get('std_lengths', {}).get(pair, 0)
                                logger.info(f"  Bond {pair}: {mean_len:.3f} ± {std_len:.3f} Å")
                                f.write(f"Bond {pair}: {mean_len:.3f} ± {std_len:.3f} Å\n")

                        # 计算指纹相似度
                        from rdkit import DataStructs
                        from rdkit.Chem import AllChem
                        fp_gen = AllChem.GetMorganGenerator(radius=2)
                        fp_orig = fp_gen.GetFingerprint(decomp['original_mol'])
                        fp_new = fp_gen.GetFingerprint(reconstructed)
                        fp_sim = DataStructs.TanimotoSimilarity(fp_orig, fp_new)
                        logger.info(f"  Fingerprint similarity: {fp_sim:.3f}")
                        f.write(f"Fingerprint similarity: {fp_sim:.3f}\n")

                        sample_result['fp_similarity'] = fp_sim

                        # ★★★ [V5新增] 等排体质量评估 ★★★
                        if generation_mode in ['isostere', 'guided_isostere']:
                            isostere_quality = evaluate_isostere_quality(reconstructed, decomp['original_mol'])

                            logger.info(f"  [V5] Isostere Quality:")
                            f.write(f"\n[V5] Isostere Quality Evaluation:\n")

                            # 输出关键等排体指标
                            key_metrics = ['valid', 'mw_diff', 'mw_similar', 'logp_diff', 'logp_similar',
                                           'qed_diff', 'qed_improved', 'fragment_diversity',
                                           'isostere_score', 'isostere_class']
                            for metric in key_metrics:
                                if metric in isostere_quality:
                                    value = isostere_quality[metric]
                                    logger.info(f"    {metric}: {value}")
                                    f.write(f"  {metric}: {value}\n")

                            sample_result['isostere_quality'] = isostere_quality
                            sample_result['isostere_score'] = isostere_quality.get('isostere_score', 0.0)
                            sample_result['isostere_class'] = isostere_quality.get('isostere_class', 'Unknown')

                    except ImportError:
                        logger.info("  [Note] evaluate_3d module not available for physical plausibility check")
                        f.write("[Note] evaluate_3d module not available\n")
                    except Exception as e:
                        logger.warning(f"  Physical plausibility check failed: {e}")
                        f.write(f"Physical plausibility check failed: {e}\n")

                    success_count += 1
                else:
                    logger.warning("  Reconstruction failed")
                    f.write("Reconstruction failed\n")

            except Exception as e:
                # ★★★ [关键修复] 捕获重建过程中的任何错误，确保循环继续 ★★★
                logger.warning(f"  Reconstruction error: {e}")
                f.write(f"Reconstruction error: {e}\n")
            total_count += 1

            results_data.append(sample_result)

        # 统计成功率
        logger.info("\n" + "=" * 60)
        logger.info(f"Summary: {success_count}/{total_count} molecules reconstructed successfully")
        logger.info(f"Success rate: {success_count/total_count*100:.1f}%")
        logger.info("=" * 60)

        # 写入汇总
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Reconstruction success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)\n")

        # 原子类型匹配统计
        atom_match_count = sum(1 for r in results_data if r['atom_match'])
        f.write(f"Atom types exact match: {atom_match_count}/{total_count} ({atom_match_count/total_count*100:.1f}%)\n")

        # 指纹相似度统计
        valid_fp = [r['fp_similarity'] for r in results_data if r['fp_similarity'] is not None]
        if valid_fp:
            avg_fp = sum(valid_fp) / len(valid_fp)
            max_fp = max(valid_fp)
            min_fp = min(valid_fp)
            f.write(f"Fingerprint similarity: avg={avg_fp:.3f}, max={max_fp:.3f}, min={min_fp:.3f}\n")

        # 应变能统计
        valid_strain = [r['strain_energy'] for r in results_data if r['strain_energy'] is not None]
        if valid_strain:
            avg_strain = sum(valid_strain) / len(valid_strain)
            f.write(f"Strain energy: avg={avg_strain:.2f} kcal/mol\n")

        # ★★★ [V5新增] 等排体质量统计 ★★★
        if generation_mode in ['isostere', 'guided_isostere']:
            valid_isostere_scores = [r.get('isostere_score', 0) for r in results_data if r.get('isostere_score') is not None]
            if valid_isostere_scores:
                avg_isostere = sum(valid_isostere_scores) / len(valid_isostere_scores)
                max_isostere = max(valid_isostere_scores)
                min_isostere = min(valid_isostere_scores)
                f.write(f"\n[V5] Isostere Quality:\n")
                f.write(f"  Isostere score: avg={avg_isostere:.3f}, max={max_isostere:.3f}, min={min_isostere:.3f}\n")

                # 统计各级别数量
                class_counts = {}
                for r in results_data:
                    cls = r.get('isostere_class', 'Unknown')
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                f.write(f"  Isostere classes: {class_counts}\n")

                # 统计分子量保持率
                mw_similar_count = sum(1 for r in results_data
                                        if r.get('isostere_quality') and r['isostere_quality'].get('mw_similar'))
                f.write(f"  MW preserved (±50 Da): {mw_similar_count}/{len(results_data)}\n")

                # 统计logP保持率
                logp_similar_count = sum(1 for r in results_data
                                          if r.get('isostere_quality') and r['isostere_quality'].get('logp_similar'))
                f.write(f"  LogP preserved (±1.5): {logp_similar_count}/{len(results_data)}\n")

                # 统计片段多样性（真正发生了替换）
                diverse_count = sum(1 for r in results_data
                                    if r.get('isostere_quality') and r['isostere_quality'].get('fragment_diversity', 0) > 0.3)
                f.write(f"  Fragment diversity (>0.3): {diverse_count}/{len(results_data)}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("Detailed Results Table:\n")
        f.write("-" * 60 + "\n")

        # ★ [V5改进] 根据生成模式调整表格格式
        if generation_mode in ['isostere', 'guided_isostere']:
            f.write(f"{'Idx':<5} {'FG':<15} {'Atoms Match':<12} {'Success':<8} {'FP Sim':<8} {'Strain':<8} {'IsoScore':<8} {'IsoClass':<10}\n")
            for r in results_data:
                fp_str = f"{r['fp_similarity']:.3f}" if r['fp_similarity'] else "N/A"
                strain_str = f"{r['strain_energy']:.2f}" if r['strain_energy'] else "N/A"
                iso_score_str = f"{r.get('isostere_score', 0):.2f}" if r.get('isostere_score') else "N/A"
                iso_class_str = r.get('isostere_class', 'N/A') if r.get('isostere_class') else "N/A"
                f.write(f"{r['idx']:<5} {r['fg_name']:<15} {str(r['atom_match']):<12} {str(r['success']):<8} {fp_str:<8} {strain_str:<8} {iso_score_str:<8} {iso_class_str:<10}\n")
        else:
            f.write(f"{'Idx':<5} {'FG':<15} {'Atoms Match':<12} {'Success':<8} {'FP Sim':<8} {'Strain':<8}\n")
            for r in results_data:
                fp_str = f"{r['fp_similarity']:.3f}" if r['fp_similarity'] else "N/A"
                strain_str = f"{r['strain_energy']:.2f}" if r['strain_energy'] else "N/A"
                f.write(f"{r['idx']:<5} {r['fg_name']:<15} {str(r['atom_match']):<12} {str(r['success']):<8} {fp_str:<8} {strain_str:<8}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Results\n")
        f.write("=" * 60 + "\n")

    

if __name__ == "__main__":
    main()