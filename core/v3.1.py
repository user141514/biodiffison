import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import to_undirected, to_dense_adj, to_dense_batch
from rdkit import Chem, DataStructs, RDLogger, rdBase, RDConfig
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm
import logging
import pickle
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple, List, Dict
import warnings

# [新增] 从 v2_3 导入通用组件
from v2_3 import (
    GaussianSmearing,
    FallbackEGNNLayer,
    AttentionX2HLayer,
    AttentionH2XLayer,
    AttentionEGNNLayer,
    EGNNWrapper,
    SinusoidalPositionalEmbedding,
    # [新增] 离散扩散组件
    DiscreteTransition,
    index_to_log_onehot,
    log_onehot_to_index,
    log_sample_categorical,
    log_add_exp,
    categorical_kl,
)

# 设置 Matplotlib 后端
matplotlib.use('Agg')

# 屏蔽警告
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# 0. 全局配置与依赖检查
# =============================================================================

try:
    from egnn_pytorch import EGNN

    HAS_EGNN = True
except ImportError:
    EGNN = None
    HAS_EGNN = False
    print("WARNING: egnn_pytorch not found. Using fallback MLP.")

# --- 全局初始化药效团工厂 ---
fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
PHARM_FACTORY = AllChem.BuildFeatureFactory(fdef_path)

# 动态获取当前环境的真实维度（因为我们强制重算，所以这里动态获取是最安全的，它会和 v2.py 保持绝对一致！）
PHARM_DIM = len(PHARM_FACTORY.GetFeatureFamilies())
# [新增] 硬编码检查，确保与 v2_3.py 的 PHARM_DIM=8 一致
if PHARM_DIM != 8:
    print(f"⚠️ WARNING: PHARM_DIM={PHARM_DIM}，但 v2_3.py 使用 PHARM_DIM=8，可能导致维度不匹配！")
    print("建议检查 RDKit 版本或 BaseFeatures.fdef 文件。")
print(f"INFO: Global Pharmacophore Dimension = {PHARM_DIM} (Will force recalculation to match this)")

# =============================================================================
# 1. 基础设置
# =============================================================================
logger = logging.getLogger("BioIsostericDiffusion")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")


# =============================================================================
# 2. 数据集定义
# =============================================================================

class CrossDockedDataset(Dataset):
    def __init__(self, packed_file_path, mode='train'):
        self.mode = mode
        self.data_list = []

        if not os.path.exists(packed_file_path):
            logger.error(f"[{mode}] 找不到文件: {packed_file_path}")
            return

        logger.info(f"[{mode}] 加载数据: {packed_file_path} ...")
        try:
            self.data_list = torch.load(packed_file_path, weights_only=False)
            logger.info(f"[{mode}] 加载完成，共 {len(self.data_list)} 条")
        except Exception as e:
            logger.error(f"[{mode}] 加载失败: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# =============================================================================
# 3. 关键函数：特征提取与 Collate
# =============================================================================

def extract_pharmacophore_features(mol):
    try:
        feats = PHARM_FACTORY.GetFeaturesForMol(mol)
        if not feats: return None, None

        fam_list = list(PHARM_FACTORY.GetFeatureFamilies())
        fam_map = {fam: i for i, fam in enumerate(fam_list)}

        f_list, p_list = [], []
        for f in feats:
            fam = f.GetFamily()
            if fam in fam_map:
                vec = np.zeros(PHARM_DIM)
                vec[fam_map[fam]] = 1
                f_list.append(vec)
                p_list.append(list(f.GetPos()))

        if not f_list: return None, None
        return torch.tensor(np.array(f_list), dtype=torch.float), torch.tensor(np.array(p_list), dtype=torch.float)
    except:
        return None, None


def collate_dual(batch_list):
    valid = [b for b in batch_list if b is not None]
    if not valid: return None

    ligands, proteins = [], []
    pharm_feats_all, pharm_pos_all, pharm_batch_all = [], [], []
    ed_vec_all = []

    # [新增] 导入 ed_context 计算函数
    try:
        from ed_context import build_ed_context_vector
        HAS_ED_CONTEXT = True
    except ImportError:
        HAS_ED_CONTEXT = False
        print("WARNING: ed_context.py not found, ed_vec will be zeros")

    for i, b in enumerate(valid):
        if isinstance(b, dict):
            lig = b['ligand']
            prot = b['protein']
        else:
            continue

        # [新增] 从 mol 提取键类型（如果有 mol）
        if hasattr(lig, 'mol') and lig.mol is not None and not hasattr(lig, 'bond_types'):
            try:
                mol = lig.mol
                if mol.GetNumBonds() > 0:
                    bond_type_map = {
                        Chem.rdchem.BondType.SINGLE: 0,
                        Chem.rdchem.BondType.DOUBLE: 1,
                        Chem.rdchem.BondType.TRIPLE: 2,
                        Chem.rdchem.BondType.AROMATIC: 3,
                    }
                    # 获取原始键（单向）
                    bonds = list(mol.GetBonds())
                    bond_types_raw = torch.tensor(
                        [bond_type_map.get(b.GetBondType(), 0) for b in bonds],
                        dtype=torch.long
                    )
                    # 转为无向图（双向）
                    lig.bond_types = torch.cat([bond_types_raw, bond_types_raw], dim=0)
                else:
                    lig.bond_types = torch.empty((0,), dtype=torch.long)
            except Exception:
                lig.bond_types = torch.empty((0,), dtype=torch.long)

        # [新增] 确保 bond_types 存在（即使为空）
        if not hasattr(lig, 'bond_types'):
            lig.bond_types = torch.empty((0,), dtype=torch.long)

        ligands.append(lig)
        proteins.append(prot)

        p_feats, p_pos = None, None

        # =============================================================================
        # 🚨 [核心修复]：强制重算！无视旧数据里的残缺特征，保证物理意义 100% 对齐
        # =============================================================================
        if hasattr(lig, 'mol') and lig.mol is not None:
            p_feats, p_pos = extract_pharmacophore_features(lig.mol)

        if p_feats is not None and p_pos is not None and p_feats.size(0) > 0:
            p_feats = p_feats.cpu()
            p_pos = p_pos.cpu()
            n = p_feats.size(0)
            pharm_feats_all.append(p_feats)
            pharm_pos_all.append(p_pos)
            pharm_batch_all.append(torch.full((n,), i, dtype=torch.long))

        # [新增] 计算 ED 向量
        ed_vec = None
        if hasattr(lig, 'mol') and lig.mol is not None and HAS_ED_CONTEXT:
            try:
                ed_vec_np = build_ed_context_vector(lig.mol)
                ed_vec = torch.from_numpy(ed_vec_np.astype(np.float32))
            except Exception as e:
                pass
        if ed_vec is None:
            ed_vec = torch.zeros(15, dtype=torch.float32)  # ED_DIM = 15
        ed_vec_all.append(ed_vec)

    if not ligands: return None

    ligand_batch = Batch.from_data_list(ligands)
    protein_batch = Batch.from_data_list(proteins)

    if pharm_feats_all:
        ligand_batch.pharm_feats = torch.cat(pharm_feats_all, dim=0)
        ligand_batch.pharm_pos = torch.cat(pharm_pos_all, dim=0)
        ligand_batch.pharm_batch = torch.cat(pharm_batch_all, dim=0)
    else:
        ligand_batch.pharm_feats = torch.zeros(1, PHARM_DIM)
        ligand_batch.pharm_pos = torch.zeros(1, 3)
        ligand_batch.pharm_batch = torch.zeros(1, dtype=torch.long)

    # [新增] 添加 ED 向量到 batch
    if ed_vec_all:
        ligand_batch.ed_vec = torch.stack(ed_vec_all, dim=0)

    # [新增] bond_types 会由 Batch.from_data_list 自动处理
    # 如果所有 ligands 都有 bond_types，ligand_batch.bond_types 会自动存在

    return ligand_batch, protein_batch


# =============================================================================
# 4. 模型组件（从 v2_3 导入）
# =============================================================================
# 注意：SinusoidalPositionalEmbedding, FallbackEGNNLayer, AttentionX2HLayer,
# AttentionH2XLayer, AttentionEGNNLayer, EGNNWrapper 已从 v2_3 导入


class ProteinEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3, use_attention=False, n_heads=4):
        super().__init__()
        # [更新] 使用 v2_3 的 EGNNWrapper，启用 layer_norm
        self.egnn = EGNNWrapper(in_dim, hidden_dim, out_dim, n_layers,
                                use_attention=use_attention, n_heads=n_heads,
                                use_layer_norm=True)

    def forward(self, x, pos, edge_index, batch_vec):
        h = self.egnn(x, pos, edge_index, batch_vec)
        return global_mean_pool(h, batch_vec)


class LigandEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3, use_attention=False, n_heads=4):
        super().__init__()
        # [更新] 使用 v2_3 的 EGNNWrapper，启用 layer_norm
        self.egnn = EGNNWrapper(in_dim, hidden_dim, out_dim, n_layers,
                                use_attention=use_attention, n_heads=n_heads,
                                use_layer_norm=True)

    def forward(self, x, pos, edge_index, batch_vec):
        h = self.egnn(x, pos, edge_index, batch_vec)
        return global_mean_pool(h, batch_vec)


class PharmacophoreEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pharm_feats, pharm_pos, pharm_batch=None, batch_size=None):
        device = pharm_feats.device
        if pharm_feats.size(0) == 0:
            B = batch_size if batch_size is not None else (1 if pharm_batch is None else (pharm_batch.max().item() + 1))
            return torch.zeros(B, self.out_dim, device=device)

        combined = torch.cat([pharm_feats, pharm_pos], dim=-1)
        h = self.encoder(combined)
        if pharm_batch is None:
            return h.mean(dim=0, keepdim=True)
        return global_mean_pool(h, pharm_batch, size=batch_size)


class EDContextEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, v): return self.net(v) if v is not None else None


class JointDiffusionGenerator(nn.Module):
    """
    联合扩散生成器（v3.1版本 - 双塔架构）
    改进版本：
    - 原子类型使用离散扩散
    - 新增键类型生成
    - 实现完整采样过程

    借鉴: DecompDiff, EDM
    """
    def __init__(self, hidden_dim, context_dim, num_atom_types, atom_type_map, num_timesteps=60, time_dim=64,
                 use_attention=False, n_heads=4, r_feat_dim=20,
                 use_discrete_diffusion=True, bond_loss_weight=0.3, num_bond_types=4):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.num_timesteps = num_timesteps
        self.use_discrete_diffusion = use_discrete_diffusion
        self.bond_loss_weight = bond_loss_weight
        self.hidden_dim = hidden_dim

        # 原子类型查找表
        self.register_buffer('atom_type_lookup', torch.full((150,), self.num_atom_types - 1, dtype=torch.long))
        for z, idx in atom_type_map.items():
            self.atom_type_lookup[z] = idx

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.context_embed = nn.Linear(context_dim, hidden_dim)

        # 去噪网络
        self.denoising_net = EGNNWrapper(
            num_atom_types + hidden_dim, hidden_dim, hidden_dim, n_layers=4,
            use_attention=use_attention, n_heads=n_heads, r_feat_dim=r_feat_dim
        )

        # 原子位置预测
        self.pos_noise_predictor = nn.Linear(hidden_dim, 3)

        # 原子类型预测
        if use_discrete_diffusion:
            self.x_logits_predictor = nn.Linear(hidden_dim, num_atom_types)
            self.atom_type_trans = DiscreteTransition(
                num_timesteps=num_timesteps, num_classes=num_atom_types, s=0.008
            )
        else:
            self.x_noise_predictor = nn.Linear(hidden_dim, num_atom_types)

        # [新增] 键类型预测
        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + r_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_bond_types)
        )
        self.bond_type_trans = DiscreteTransition(
            num_timesteps=num_timesteps, num_classes=num_bond_types, s=0.008
        )
        self.distance_expansion = GaussianSmearing(0.0, 5.0, num_gaussians=r_feat_dim)

        # 噪声调度
        self._init_schedule(num_timesteps)

    def _init_schedule(self, T, s=0.008):
        steps = T + 1
        t = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 1e-4, 0.999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())

        # 后验方差
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_logvar', torch.log(posterior_variance.clamp(min=1e-20)))

    def forward(self, batch, context):
        """
        训练时的前向传播
        """
        x_raw, pos = batch.x, batch.pos
        edge_index, bvec = batch.edge_index, batch.batch
        device = pos.device

        # 获取原子类型
        atom_idx = self.atom_type_lookup[x_raw[:, 0].long()]

        B = bvec.max().item() + 1
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        t_node = t[bvec]

        # ========== 原子类型扩散 ==========
        if self.use_discrete_diffusion:
            log_x0 = index_to_log_onehot(atom_idx, self.num_atom_types)
            x_perturbed_idx, log_xt = self.atom_type_trans.q_v_sample(log_x0, t, bvec)
            x_feat = log_xt.exp()  # 从log空间恢复概率
        else:
            x_onehot = F.one_hot(atom_idx, self.num_atom_types).float().detach()
            noise_x = torch.randn_like(x_onehot)
            sqrt_a = self.sqrt_alphas_cumprod[t_node].unsqueeze(-1)
            sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t_node].unsqueeze(-1)
            x_feat = sqrt_a * x_onehot + sqrt_1ma * noise_x

        # ========== 原子位置扩散 ==========
        noise_pos = torch.randn_like(pos)
        sqrt_a = self.sqrt_alphas_cumprod[t_node].unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t_node].unsqueeze(-1)
        noisy_pos = sqrt_a * pos + sqrt_1ma * noise_pos
        noisy_pos = noisy_pos.clamp(-50.0, 50.0)

        # ========== 网络预测 ==========
        t_emb = self.time_embed(t)
        c_emb = self.context_embed(context)
        cond = (t_emb + c_emb)[bvec]

        h_in = torch.cat([x_feat, cond], dim=-1)
        h_out = self.denoising_net(h_in, noisy_pos, edge_index, bvec)

        # 预测位置噪声
        pred_noise_pos = self.pos_noise_predictor(h_out)
        loss_pos = F.mse_loss(pred_noise_pos, noise_pos)

        # ========== 原子类型损失 ==========
        if self.use_discrete_diffusion:
            pred_x_logits = self.x_logits_predictor(h_out)
            log_x_recon = F.log_softmax(pred_x_logits, dim=-1)

            log_x_model_prob = self.atom_type_trans.q_v_posterior(log_x_recon, log_xt, t, bvec)
            log_x_true_prob = self.atom_type_trans.q_v_posterior(log_x0, log_xt, t, bvec)

            kl_x = categorical_kl(log_x_model_prob, log_x_true_prob)
            mask_t0 = (t == 0).float()[bvec]
            decoder_nll_x = -log_x_recon[atom_idx]
            loss_x = (mask_t0 * decoder_nll_x + (1. - mask_t0) * kl_x).mean()
        else:
            pred_noise_x = self.x_noise_predictor(h_out)
            loss_x = F.mse_loss(pred_noise_x, noise_x)

        # ========== 键类型损失 ==========
        # 注意：检查 bond_types（键类型）或 edge_labels（兼容旧数据）
        loss_bond = torch.tensor(0.0, device=device)
        bond_types_data = None
        if hasattr(batch, 'bond_types') and batch.bond_types is not None:
            bond_types_data = batch.bond_types.long()
        elif hasattr(batch, 'edge_labels') and batch.edge_labels is not None:
            # 兼容旧数据格式（edge_labels 可能存储键类型）
            bond_types_data = batch.edge_labels.long()

        if bond_types_data is not None and edge_index.size(1) > 0:
            log_b0 = index_to_log_onehot(bond_types_data, self.num_bond_types)

            src, dst = edge_index
            dist = torch.norm(noisy_pos[dst] - noisy_pos[src], p=2, dim=-1)
            r_feat = self.distance_expansion(dist)
            hi, hj = h_out[src], h_out[dst]
            bond_input = torch.cat([r_feat, hi, hj], dim=-1)

            pred_bond_logits = self.bond_predictor(bond_input)
            log_b_recon = F.log_softmax(pred_bond_logits, dim=-1)

            # 简化的键损失（CE）
            loss_bond = F.cross_entropy(pred_bond_logits, bond_types_data)

        # 总损失
        loss = loss_pos + 0.5 * loss_x + self.bond_loss_weight * loss_bond
        return loss

    @torch.no_grad()
    def sample(self, num_atoms, context, atom_type_map_rev, edge_index_template=None, fix_noise=False):
        """
        采样生成分子
        """
        self.eval()
        device = self.device

        # 初始化
        if fix_noise:
            torch.manual_seed(42)

        pos = torch.randn(num_atoms, 3, device=device)
        pos = pos - pos.mean(dim=0, keepdim=True)

        atom_types = torch.randint(0, self.num_atom_types, (num_atoms,), device=device)
        log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

        bond_types = None
        if edge_index_template is not None and edge_index_template.size(1) > 0:
            bond_types = torch.randint(0, self.num_bond_types, (edge_index_template.size(1),), device=device)

        # 扩展context
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # 逆向采样
        for t_int in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            t = torch.full((1,), t_int, dtype=torch.long, device=device)
            bvec = torch.zeros(num_atoms, dtype=torch.long, device=device)

            t_emb = self.time_embed(t)
            c_emb = self.context_embed(context)
            cond = (t_emb + c_emb).expand(num_atoms, -1)

            if self.use_discrete_diffusion:
                x_feat = log_atom_types.exp()
            else:
                x_feat = F.one_hot(atom_types, self.num_atom_types).float()

            # 边索引
            if edge_index_template is not None:
                edge_index = edge_index_template
            else:
                src = torch.arange(num_atoms, device=device).repeat_interleave(num_atoms)
                dst = torch.arange(num_atoms, device=device).repeat(num_atoms)
                mask = src != dst
                edge_index = torch.stack([src[mask], dst[mask]], dim=0)

            h_out = self.denoising_net(torch.cat([x_feat, cond], dim=-1), pos, edge_index, bvec)

            # 位置采样
            pred_noise_pos = self.pos_noise_predictor(h_out)
            x0_pred = (pos - self.sqrt_one_minus_alphas_cumprod[t].expand(num_atoms, -1) * pred_noise_pos) / \
                      self.sqrt_alphas_cumprod[t].expand(num_atoms, -1)
            x0_pred = x0_pred.clamp(-50., 50.)

            # 简化的后验均值
            alpha_cumprod_prev = torch.cat([torch.ones(1, device=device),
                                           (self.sqrt_alphas_cumprod ** 2)[:-1]])[t]
            posterior_mean = (alpha_cumprod_prev * pos + (1. - alpha_cumprod_prev) * x0_pred) / \
                            (1. - (self.sqrt_alphas_cumprod[t] ** 2) + 1e-8)

            pos_var = self.posterior_variance[t]
            noise = torch.randn_like(pos) if t_int > 0 else torch.zeros_like(pos)
            pos = posterior_mean + pos_var.sqrt() * noise
            pos = pos - pos.mean(dim=0, keepdim=True)

            # 原子类型采样
            if self.use_discrete_diffusion:
                pred_x_logits = self.x_logits_predictor(h_out)
                log_x_recon = F.log_softmax(pred_x_logits, dim=-1)
                log_model_prob = self.atom_type_trans.q_v_posterior(log_x_recon, log_atom_types, t, bvec)
                atom_types = log_sample_categorical(log_model_prob)
                log_atom_types = index_to_log_onehot(atom_types, self.num_atom_types)

            # 键类型采样
            if edge_index_template is not None and bond_types is not None:
                src, dst = edge_index_template
                dist = torch.norm(pos[dst] - pos[src], p=2, dim=-1)
                r_feat = self.distance_expansion(dist)
                hi, hj = h_out[src], h_out[dst]
                bond_input = torch.cat([r_feat, hi, hj], dim=-1)
                pred_bond_logits = self.bond_predictor(bond_input)
                bond_types = pred_bond_logits.argmax(dim=-1)

        return {'pos': pos, 'atom_types': atom_types, 'bond_types': bond_types}


# =============================================================================
# 5. 主模型：BioIsostericDiffusion
# =============================================================================

class BioIsostericDiffusion(nn.Module):
    def __init__(self, atom_dim, hidden_dim, lr, atom_types, pharm_factory, device, fragment_fp_dim=2048,
                 ed_vec_dim=15, use_attention=False, n_heads=4, r_feat_dim=20,
                 use_discrete_diffusion=True, bond_loss_weight=0.3, num_bond_types=4):
        """
        Args:
            atom_dim: 原子特征维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            atom_types: 原子类型列表
            pharm_factory: 药效团工厂
            device: 设备
            fragment_fp_dim: 片段指纹维度
            ed_vec_dim: ED向量维度
            use_attention: 是否启用注意力机制
            n_heads: 注意力头数
            r_feat_dim: 高斯距离编码维度
            use_discrete_diffusion: 是否使用离散扩散处理原子类型
            bond_loss_weight: 键类型损失权重
            num_bond_types: 键类型数量
        """
        super().__init__()
        self.device = device
        self.atom_types = atom_types
        self.num_atom_types = len(atom_types)
        self.atom_type_map = {z: i for i, z in enumerate(atom_types)}
        self.pharm_factory = pharm_factory
        self.hidden_dim = hidden_dim
        self.pharm_dim = PHARM_DIM
        self.ed_vec_dim = ed_vec_dim
        self.use_attention = use_attention
        self.use_discrete_diffusion = use_discrete_diffusion

        if use_attention:
            logger.info(f"v3.1 注意力机制已启用: n_heads={n_heads}, r_feat_dim={r_feat_dim}")

        if use_discrete_diffusion:
            logger.info("v3.1 离散扩散已启用：原子类型将使用D3PM风格的离散扩散")

        # [关键修复] 命名统一为 v2_3.py 的风格，确保权重正确加载
        self.context_encoder = EGNNWrapper(
            atom_dim, hidden_dim, hidden_dim, n_layers=3,
            use_attention=use_attention, n_heads=n_heads, r_feat_dim=r_feat_dim
        )
        self.pharmacophore_encoder = PharmacophoreEncoder(self.pharm_dim, hidden_dim, hidden_dim)  # 原 pharm_encoder
        self.ed_context_encoder = EDContextEncoder(ed_vec_dim, hidden_dim, hidden_dim)  # 原 ed_encoder

        # v3.1 特有：蛋白质编码器
        self.protein_encoder = ProteinEncoder(
            in_dim=25, hidden_dim=hidden_dim, out_dim=hidden_dim,
            use_attention=use_attention, n_heads=n_heads
        )

        # [改进] 生成模型：支持离散扩散和键类型生成
        self.generative_model = JointDiffusionGenerator(
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            num_atom_types=self.num_atom_types,
            atom_type_map=self.atom_type_map,
            num_timesteps=60,  # 与 v2_3.py 的 Config.stage3_timesteps 一致
            time_dim=64,       # 与 v2_3.py 一致
            use_attention=use_attention,
            n_heads=n_heads,
            r_feat_dim=r_feat_dim,
            use_discrete_diffusion=use_discrete_diffusion,
            bond_loss_weight=bond_loss_weight,
            num_bond_types=num_bond_types,
        )

        # v3.1 特有：融合层（四路 context 合一）
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.ln_geom = nn.LayerNorm(hidden_dim)
        self.ln_pharm = nn.LayerNorm(hidden_dim)
        self.ln_ed = nn.LayerNorm(hidden_dim)
        self.ln_prot = nn.LayerNorm(hidden_dim)  # v3.1 特有

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = None

    def setup_scheduler(self, total_epochs, warmup_epochs=5):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_context(self, ligand_batch, protein_batch):
        device = ligand_batch.x.device
        B = ligand_batch.batch.max().item() + 1

        # [修复] 使用统一的命名
        geom_ctx = self.context_encoder(ligand_batch.x, ligand_batch.pos, ligand_batch.edge_index, ligand_batch.batch)
        # [修复] 添加 global_mean_pool，与 v2_3.py 一致
        geom_ctx = global_mean_pool(geom_ctx, ligand_batch.batch)  # (B, hidden_dim)

        if hasattr(ligand_batch, 'pharm_feats') and ligand_batch.pharm_feats is not None:
            pharm_batch = getattr(ligand_batch, 'pharm_batch', None)
            pharm_ctx = self.pharmacophore_encoder(
                ligand_batch.pharm_feats.to(device),
                ligand_batch.pharm_pos.to(device),
                pharm_batch.to(device) if pharm_batch is not None else None,
                batch_size=B
            )
        else:
            pharm_ctx = torch.zeros(B, self.hidden_dim, device=device)

        if hasattr(ligand_batch, 'ed_vec') and ligand_batch.ed_vec is not None:
            ed_ctx = self.ed_context_encoder(ligand_batch.ed_vec.to(device))
        else:
            ed_ctx = torch.zeros(B, self.hidden_dim, device=device)
        if ed_ctx.size(0) != B: ed_ctx = torch.zeros(B, self.hidden_dim, device=device)

        prot_ctx = self.protein_encoder(protein_batch.x, protein_batch.pos, protein_batch.edge_index,
                                        protein_batch.batch)

        combined = torch.cat(
            [self.ln_geom(geom_ctx), self.ln_pharm(pharm_ctx), self.ln_ed(ed_ctx), self.ln_prot(prot_ctx)], dim=-1)
        return self.fusion_layer(combined)

    def train_step(self, ligand_batch, protein_batch, accum_steps, step_idx):
        # 纯 FP32 训练，防坐标爆炸
        context = self.compute_context(ligand_batch, protein_batch)
        loss = self.generative_model(ligand_batch, context)  # [修复] 使用统一命名

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Loss is NaN/Inf, skipping batch")
            return None

        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        if (step_idx + 1) % accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning("Gradient Explosion, skipping update")
                self.optimizer.zero_grad()
                return None

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def fit(self, dataloader, epochs=50, save_interval=10, accum_steps=4):
        self.train()
        self.setup_scheduler(epochs, 5)

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            self.optimizer.zero_grad()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch_idx, batch_data in enumerate(pbar):
                if batch_data is None: continue

                try:
                    ligand_batch, protein_batch = batch_data
                    ligand_batch = ligand_batch.to(self.device)
                    protein_batch = protein_batch.to(self.device)

                    loss_val = self.train_step(ligand_batch, protein_batch, accum_steps, batch_idx)

                    if loss_val is not None:
                        total_loss += loss_val
                        n_batches += 1
                        pbar.set_postfix({'loss': f'{loss_val:.4f}'})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM, skipping batch")
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        logger.error(f"Batch {batch_idx} Runtime Error: {e}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} Error: {e}")

            avg_loss = total_loss / max(n_batches, 1)
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.2e}")

            if self.scheduler: self.scheduler.step()

            if (epoch + 1) % save_interval == 0:
                torch.save(self.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")


# =============================================================================
# 7. 主函数
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("BioIsosteric Diffusion - Dual Tower Training (FP32 Version)")
    logger.info("=" * 60)

    # [新增] 数据路径检查
    packed_path = "packed_data/train_packed.pt"
    if not os.path.exists(packed_path):
        logger.error(f"❌ 数据文件不存在: {packed_path}")
        logger.error("v3.1 需要 CrossDocked 配体-蛋白质配对数据，格式为:")
        logger.error("  每条数据: {'ligand': Data(mol, x, pos, edge_index, ed_vec), 'protein': Data(x, pos, edge_index)}")
        logger.error("请准备数据后重新运行，或使用以下脚本生成:")
        logger.error("  1. 从 CrossDocked 数据集提取 pocket-ligand 对")
        logger.error("  2. 使用 torch.save(data_list, packed_path) 保存")
        return

    dataset = CrossDockedDataset(
        packed_file_path=packed_path,
        mode='train'
    )

    if len(dataset) == 0:
        logger.error("Dataset is empty! Check the data path.")
        return

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_dual,
        num_workers=0,
        pin_memory=True
    )

    # 原子类型列表必须与 v2.py 完全一致！
    atom_types = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

    model = BioIsostericDiffusion(
        atom_dim=6,
        hidden_dim=128,
        lr=1e-4,
        atom_types=atom_types,
        pharm_factory=PHARM_FACTORY,
        device=device
    ).to(device)

    # [改进] 支持多种预训练权重路径，按兼容性优先级尝试
    # v3.1从v2_3演化，结构更相似，所以v2_3权重优先
    # v2_4是片段替换模型，结构差异大，兼容率低但可尝试
    pretrained_paths = [
        "bioisosteric_model_final1.pth",         # v2_3完整模型（优先，结构最相似）
        "bioisosteric_model_final.pth",          # v2_3早期版本
        "bioisosteric_fragment_model_v2_4.pth",  # v2_4片段替换模型（兼容率低）
    ]

    pretrained_path = None
    loaded_from = None
    for p in pretrained_paths:
        if os.path.exists(p):
            pretrained_path = p
            loaded_from = p.split('_')[-1].replace('.pth', '')  # 提取版本号
            break

    if pretrained_path:
        logger.info(f"发现预训练权重！来源: {pretrained_path}")
            try:
            # [改进] 详细报告加载情况
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(pretrained_path, map_location=device, weights_only=False),
                strict=False
            )

            # 报告加载结果
            loaded_keys = [k for k in model.state_dict().keys() if k not in missing_keys]
            logger.info(f"✅ 成功加载 {len(loaded_keys)} 个参数:")

            # 关键参数检查
            critical_keys = ['generative_model', 'context_encoder', 'pharmacophore_encoder', 'ed_context_encoder']
            for ck in critical_keys:
                matched = [k for k in loaded_keys if k.startswith(ck)]
                if matched:
                    logger.info(f"   ✓ {ck}: {len(matched)} 个参数已加载")
                else:
                    logger.warning(f"   ✗ {ck}: 未加载（将从随机初始化开始）")

            # v3.1 特有模块报告
            v31_specific = ['protein_encoder', 'fusion_layer', 'ln_prot']
            logger.info("v3.1 特有模块（无法从预训练权重加载）:")
            for sk in v31_specific:
                logger.info(f"   • {sk}: 随机初始化")

            if missing_keys:
                logger.warning(f"缺失 {len(missing_keys)} 个参数（将使用默认值）")
            if unexpected_keys:
                logger.warning(f"忽略 {len(unexpected_keys)} 个额外参数（预训练模型特有）")

        except Exception as e:
            logger.error(f"❌ 预训练权重加载失败: {e}")
    else:
        logger.warning(f"⚠️ 未找到预训练权重 {pretrained_path}，模型将从零开始盲学（极不推荐）！")

    logger.info("Starting training...")
    model.fit(
        dataloader=loader,
        epochs=50,
        save_interval=10,
        accum_steps=4
    )

    torch.save(model.state_dict(), "bioisosteric_dual_tower_final.pth")
    logger.info("Training complete. Model saved.")


if __name__ == "__main__":
    main()