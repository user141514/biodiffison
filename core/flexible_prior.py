"""
Flexible Prior Manager for BioIsosteric Diffusion Models

支持三种模式:
- 'custom': 手动设定prior参数
- 'conditional': 条件注入动态prior
- 'learned': 端到端学习prior参数

支持四种prior类型:
- 'isotropic_gaussian': 各向同性高斯 (cov = scalar * I)
- 'anisotropic_gaussian': 各向异性高斯 (cov = full matrix)
- 'categorical_uniform': 均匀离散分布 (用于原子类型)
- 'golden_prior': 数据驱动prior (从数据计算)

四种条件类型:
- 'topology': 拓扑和化学语义条件
- 'geometry': 3D几何和空间构象条件
- 'pharmacophore': 药效团功能条件
- 'electron_density': 电子密度条件

参考: DecompDiff/utils/prior.py, DiffSBDD, targetdiff

移植日期: 2026-04-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# 基础工具函数
# =============================================================================

def sum_except_batch(x):
    """Sum all dimensions except batch dimension."""
    return x.view(x.size(0), -1).sum(-1)


def remove_mean_with_mask(x, node_mask):
    """Remove mean from positions while respecting node mask."""
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_correctly_masked(variable, node_mask):
    """Assert that masked values are zero."""
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


# =============================================================================
# Prior Sampler Classes
# =============================================================================

class IsotropicGaussianSampler(nn.Module):
    """各向同性高斯prior采样器 (cov = sigma * I)"""

    def __init__(self, n_dim: int, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        self.n_dim = n_dim
        self.mu = mu
        self.sigma = sigma

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor) -> torch.Tensor:
        z = torch.randn(size, device=device) * self.sigma + self.mu
        z_masked = z * node_mask
        z_projected = remove_mean_with_mask(z_masked, node_mask)
        return z_projected

    def log_likelihood(self, z: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        assert_correctly_masked(z, node_mask)
        r2 = sum_except_batch(z.pow(2))
        N = node_mask.squeeze(2).sum(1)
        degrees_of_freedom = (N - 1) * self.n_dim
        log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2 * np.pi * self.sigma**2)
        log_pz = -0.5 * r2 / (self.sigma**2) + log_normalizing_constant
        return log_pz


class AnisotropicGaussianSampler(nn.Module):
    """各向异性高斯prior采样器 (cov = full matrix) - 参考DecompDiff"""

    def __init__(self, n_dim: int, mu: np.ndarray = None, cov: np.ndarray = None):
        super().__init__()
        self.n_dim = n_dim
        if mu is None:
            mu = np.zeros(n_dim)
        if cov is None:
            cov = np.eye(n_dim)
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('cov', torch.tensor(cov, dtype=torch.float32))
        self._compute_cov_inverse()

    def _compute_cov_inverse(self):
        cov_np = self.cov.numpy()
        try:
            cov_inv = np.linalg.inv(cov_np)
            cov_logdet = np.linalg.slogdet(cov_np)[1]
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_np)
            cov_logdet = np.sum(np.log(np.abs(np.linalg.eigvals(cov_np))))
        self.register_buffer('cov_inv', torch.tensor(cov_inv, dtype=torch.float32))
        self.cov_logdet = cov_logdet

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor) -> torch.Tensor:
        n_samples, n_nodes, n_dim = size
        try:
            L = torch.linalg.cholesky(self.cov.to(device))
        except RuntimeError:
            eigenvalues, eigenvectors = torch.linalg.eigh(self.cov.to(device))
            L = eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=1e-6)))
        eps = torch.randn(n_samples, n_nodes, n_dim, device=device)
        z = self.mu.to(device).unsqueeze(0).unsqueeze(0) + torch.matmul(eps, L.T)
        z_masked = z * node_mask
        z_projected = remove_mean_with_mask(z_masked, node_mask)
        return z_projected

    def log_likelihood(self, z: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        assert_correctly_masked(z, node_mask)
        z_centered = z - self.mu.to(z.device).unsqueeze(0).unsqueeze(0)
        mahalanobis = torch.matmul(z_centered, self.cov_inv.to(z.device))
        mahalanobis = sum_except_batch(torch.matmul(mahalanobis, z_centered.transpose(-1, -2)))
        N = node_mask.squeeze(2).sum(1)
        degrees_of_freedom = (N - 1) * self.n_dim
        log_normalizing_constant = -0.5 * degrees_of_freedom * (np.log(2 * np.pi) + self.cov_logdet)
        log_pz = -0.5 * mahalanobis + log_normalizing_constant
        return log_pz


class CategoricalUniformSampler(nn.Module):
    """均匀离散分布采样器 (用于原子类型等离散特征)"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor) -> torch.Tensor:
        n_samples, n_nodes, _ = size
        indices = torch.randint(0, self.num_classes, (n_samples, n_nodes), device=device)
        samples = F.one_hot(indices, self.num_classes).float()
        samples = samples * node_mask
        return samples

    def log_likelihood(self, z: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        log_prob_per_class = -np.log(self.num_classes)
        N = node_mask.squeeze(2).sum(1)
        log_pz = N * log_prob_per_class
        return log_pz


class GoldenPriorSampler(nn.Module):
    """数据驱动prior采样器 - 参考DecompDiff的compute_golden_prior_from_data"""

    def __init__(self, n_dim: int, golden_prior_dict: Dict = None):
        super().__init__()
        self.n_dim = n_dim
        self.golden_prior_dict = golden_prior_dict or {}
        self._register_prior_params()

    def _register_prior_params(self):
        if 'arms_prior' in self.golden_prior_dict:
            arms = []
            for arm_prior in self.golden_prior_dict['arms_prior']:
                num_atoms, mu_iso, cov_iso, mu_aniso, cov_aniso = arm_prior
                arms.append({
                    'num_atoms': num_atoms,
                    'mu_iso': torch.tensor(mu_iso).float() if mu_iso is not None else None,
                    'cov_iso': torch.tensor(cov_iso).float() if cov_iso is not None else None,
                })
            self.arms_prior = arms
        if 'scaffold_prior' in self.golden_prior_dict:
            scaffold = []
            for sca_prior in self.golden_prior_dict['scaffold_prior']:
                num_atoms, mu_iso, cov_iso, mu_aniso, cov_aniso = sca_prior
                scaffold.append({
                    'num_atoms': num_atoms,
                    'mu_iso': torch.tensor(mu_iso).float() if mu_iso is not None else None,
                    'cov_iso': torch.tensor(cov_iso).float() if cov_iso is not None else None,
                })
            self.scaffold_prior = scaffold

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor,
               component_idx: int = 0) -> torch.Tensor:
        if component_idx < len(self.arms_prior):
            prior_params = self.arms_prior[component_idx]
        else:
            scaffold_idx = component_idx - len(self.arms_prior)
            if scaffold_idx < len(self.scaffold_prior):
                prior_params = self.scaffold_prior[scaffold_idx]
            else:
                prior_params = None

        if prior_params is None or prior_params['mu_iso'] is None:
            return torch.randn(size, device=device) * node_mask

        mu = prior_params['mu_iso'].to(device)
        cov = prior_params['cov_iso'].to(device)
        n_samples, n_nodes, n_dim = size

        if cov.dim() == 0:
            z = torch.randn(size, device=device) * cov.sqrt() + mu.unsqueeze(0).unsqueeze(0)
        else:
            try:
                L = torch.linalg.cholesky(cov)
            except RuntimeError:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                L = eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=1e-6)))
            eps = torch.randn(n_samples, n_nodes, n_dim, device=device)
            z = mu.unsqueeze(0).unsqueeze(0) + torch.matmul(eps, L.T)

        z_masked = z * node_mask
        z_projected = remove_mean_with_mask(z_masked, node_mask)
        return z_projected

    def log_likelihood(self, z: torch.Tensor, node_mask: torch.Tensor,
                       component_idx: int = 0) -> torch.Tensor:
        if component_idx < len(self.arms_prior):
            prior_params = self.arms_prior[component_idx]
        else:
            scaffold_idx = component_idx - len(self.arms_prior)
            if scaffold_idx < len(self.scaffold_prior):
                prior_params = self.scaffold_prior[scaffold_idx]
            else:
                prior_params = None

        if prior_params is None or prior_params['cov_iso'] is None:
            N = node_mask.squeeze(2).sum(1)
            degrees_of_freedom = (N - 1) * self.n_dim
            r2 = sum_except_batch(z.pow(2))
            log_pz = -0.5 * r2 - 0.5 * degrees_of_freedom * np.log(2 * np.pi)
            return log_pz

        cov = prior_params['cov_iso'].to(z.device)
        if cov.dim() == 0:
            sigma2 = cov.item()
            r2 = sum_except_batch(z.pow(2))
            N = node_mask.squeeze(2).sum(1)
            degrees_of_freedom = (N - 1) * self.n_dim
            log_pz = -0.5 * r2 / sigma2 - 0.5 * degrees_of_freedom * np.log(2 * np.pi * sigma2)
        else:
            cov_inv = torch.linalg.inv(cov)
            cov_logdet = torch.linalg.slogdet(cov)[1].item()
            mahalanobis = torch.matmul(z, cov_inv)
            mahalanobis = sum_except_batch(torch.matmul(mahalanobis, z.transpose(-1, -2)))
            N = node_mask.squeeze(2).sum(1)
            degrees_of_freedom = (N - 1) * self.n_dim
            log_pz = -0.5 * mahalanobis - 0.5 * degrees_of_freedom * (np.log(2 * np.pi) + cov_logdet)
        return log_pz


# =============================================================================
# Learned Prior Classes
# =============================================================================

class LearnableGaussianPrior(nn.Module):
    """可学习的高斯prior"""

    def __init__(self, n_dim: int, in_node_nf: int,
                 learn_mu: bool = True, learn_sigma: bool = True):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf

        if learn_mu:
            self.mu_x = nn.Parameter(torch.zeros(n_dim))
            self.mu_h = nn.Parameter(torch.zeros(in_node_nf))
        else:
            self.register_buffer('mu_x', torch.zeros(n_dim))
            self.register_buffer('mu_h', torch.zeros(in_node_nf))

        if learn_sigma:
            self.log_sigma_x = nn.Parameter(torch.zeros(n_dim))
            self.log_sigma_h = nn.Parameter(torch.zeros(in_node_nf))
        else:
            self.register_buffer('log_sigma_x', torch.zeros(n_dim))
            self.register_buffer('log_sigma_h', torch.zeros(in_node_nf))

    def get_sigma_x(self):
        return torch.exp(self.log_sigma_x)

    def get_sigma_h(self):
        return torch.exp(self.log_sigma_h)

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples, n_nodes, n_dim = size
        sigma_x = self.get_sigma_x().to(device)
        z_x = torch.randn(n_samples, n_nodes, n_dim, device=device) * sigma_x + self.mu_x.to(device)
        z_x = z_x * node_mask
        z_x = remove_mean_with_mask(z_x, node_mask)
        sigma_h = self.get_sigma_h().to(device)
        z_h = torch.randn(n_samples, n_nodes, self.in_node_nf, device=device) * sigma_h + self.mu_h.to(device)
        z_h = z_h * node_mask
        return z_x, z_h

    def log_likelihood(self, z_x: torch.Tensor, z_h: torch.Tensor,
                       node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_x = self.get_sigma_x().to(z_x.device)
        sigma_h = self.get_sigma_h().to(z_h.device)
        N = node_mask.squeeze(2).sum(1)
        degrees_of_freedom_x = (N - 1) * self.n_dim
        r2_x = sum_except_batch(z_x.pow(2))
        log_pz_x = -0.5 * r2_x / (sigma_x.pow(2).sum()) - 0.5 * degrees_of_freedom_x * np.log(2 * np.pi)
        log_pz_h_elementwise = -0.5 * z_h.pow(2) / sigma_h.pow(2).unsqueeze(0).unsqueeze(0) \
                               - 0.5 * np.log(2 * np.pi) - self.log_sigma_h.unsqueeze(0).unsqueeze(0)
        log_pz_h = sum_except_batch(log_pz_h_elementwise * node_mask)
        return log_pz_x, log_pz_h


class MixtureGaussianPrior(nn.Module):
    """混合高斯prior"""

    def __init__(self, n_dim: int, in_node_nf: int, num_components: int = 5):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf
        self.num_components = num_components
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))
        self.mu_x_components = nn.Parameter(torch.zeros(num_components, n_dim))
        self.log_sigma_x_components = nn.Parameter(torch.zeros(num_components, n_dim))
        self.mu_h_components = nn.Parameter(torch.zeros(num_components, in_node_nf))
        self.log_sigma_h_components = nn.Parameter(torch.zeros(num_components, in_node_nf))

    def get_mixture_weights(self):
        return F.softmax(self.mixture_logits, dim=0)

    def get_sigma_x_components(self):
        return torch.exp(self.log_sigma_x_components)

    def get_sigma_h_components(self):
        return torch.exp(self.log_sigma_h_components)

    def sample(self, size: tuple, device: torch.device, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples, n_nodes, n_dim = size
        weights = self.get_mixture_weights().to(device)
        component_indices = torch.multinomial(weights, n_samples, replacement=True)
        z_x_list = []
        z_h_list = []
        for i in range(n_samples):
            k = component_indices[i]
            sigma_x = torch.exp(self.log_sigma_x_components[k]).to(device)
            mu_x = self.mu_x_components[k].to(device)
            sigma_h = torch.exp(self.log_sigma_h_components[k]).to(device)
            mu_h = self.mu_h_components[k].to(device)
            x_i = torch.randn(1, n_nodes, n_dim, device=device) * sigma_x + mu_x
            h_i = torch.randn(1, n_nodes, self.in_node_nf, device=device) * sigma_h + mu_h
            z_x_list.append(x_i)
            z_h_list.append(h_i)
        z_x = torch.cat(z_x_list, dim=0)
        z_h = torch.cat(z_h_list, dim=0)
        z_x = z_x * node_mask
        z_x = remove_mean_with_mask(z_x, node_mask)
        z_h = z_h * node_mask
        return z_x, z_h

    def log_likelihood(self, z_x: torch.Tensor, z_h: torch.Tensor,
                       node_mask: torch.Tensor) -> torch.Tensor:
        weights = self.get_mixture_weights().to(z_x.device)
        log_pz_components = []
        for k in range(self.num_components):
            sigma_x = torch.exp(self.log_sigma_x_components[k]).to(z_x.device)
            sigma_h = torch.exp(self.log_sigma_h_components[k]).to(z_h.device)
            mu_x = self.mu_x_components[k].to(z_x.device)
            mu_h = self.mu_h_components[k].to(z_h.device)
            N = node_mask.squeeze(2).sum(1)
            degrees_of_freedom_x = (N - 1) * self.n_dim
            z_x_centered = z_x - mu_x.unsqueeze(0).unsqueeze(0)
            r2_x = sum_except_batch(z_x_centered.pow(2))
            log_pz_x_k = -0.5 * r2_x / sigma_x.pow(2).sum() - 0.5 * degrees_of_freedom_x * np.log(2 * np.pi)
            z_h_centered = z_h - mu_h.unsqueeze(0).unsqueeze(0)
            log_pz_h_k = sum_except_batch(
                -0.5 * z_h_centered.pow(2) / sigma_h.pow(2).unsqueeze(0).unsqueeze(0) * node_mask
            )
            log_pz_components.append(log_pz_x_k + log_pz_h_k + torch.log(weights[k]))
        log_pz_stack = torch.stack(log_pz_components, dim=1)
        log_pz = torch.logsumexp(log_pz_stack, dim=1)
        return log_pz


# =============================================================================
# Condition Encoder Classes
# =============================================================================

class TopologyEncoder(nn.Module):
    """拓扑和化学语义编码器"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, topology_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(topology_features)


class GeometryEncoder(nn.Module):
    """3D几何和空间构象编码器"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, geometry_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(geometry_features)


class PharmacophoreEncoder(nn.Module):
    """药效团功能编码器"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pharmacophore_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(pharmacophore_features)


class ElectronDensityEncoder(nn.Module):
    """电子密度编码器"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, electron_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(electron_features)


class CrossAttentionFusion(nn.Module):
    """条件融合网络"""

    def __init__(self, num_conditions: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoded_conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        condition_list = list(encoded_conditions.values())
        condition_stack = torch.stack(condition_list, dim=1)
        fused, _ = self.cross_attention(condition_stack, condition_stack, condition_stack)
        fused = fused.mean(dim=1)
        fused = self.output_proj(fused)
        return fused


class ConditionEncoder(nn.Module):
    """四种条件类型的统一编码器"""

    def __init__(self, condition_types: List[str], condition_dims: Dict[str, int],
                 output_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.condition_types = condition_types
        self.output_dim = output_dim
        self.encoders = nn.ModuleDict()

        if 'topology' in condition_types:
            self.encoders['topology'] = TopologyEncoder(
                input_dim=condition_dims.get('topology', 64),
                output_dim=output_dim, hidden_dim=hidden_dim
            )
        if 'geometry' in condition_types:
            self.encoders['geometry'] = GeometryEncoder(
                input_dim=condition_dims.get('geometry', 128),
                output_dim=output_dim, hidden_dim=hidden_dim
            )
        if 'pharmacophore' in condition_types:
            self.encoders['pharmacophore'] = PharmacophoreEncoder(
                input_dim=condition_dims.get('pharmacophore', 64),
                output_dim=output_dim, hidden_dim=hidden_dim
            )
        if 'electron_density' in condition_types:
            self.encoders['electron_density'] = ElectronDensityEncoder(
                input_dim=condition_dims.get('electron_density', 32),
                output_dim=output_dim, hidden_dim=hidden_dim
            )

        self.fusion_net = CrossAttentionFusion(
            num_conditions=len(condition_types), hidden_dim=output_dim
        )

    def forward(self, conditions_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {}
        for key, encoder in self.encoders.items():
            if key in conditions_dict:
                encoded[key] = encoder(conditions_dict[key])
        if len(encoded) == 0:
            batch_size = conditions_dict.get('batch_size', 1)
            return torch.zeros(batch_size, self.output_dim,
                              device=next(self.parameters()).device if len(self.encoders) > 0 else 'cpu')
        fused_condition = self.fusion_net(encoded)
        return fused_condition


class PriorParameterPredictor(nn.Module):
    """从条件向量预测prior参数"""

    def __init__(self, condition_dim: int, n_dim: int, in_node_nf: int, hidden_dim: int = 256):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf
        self.mu_x_predictor = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_dim),
        )
        self.log_sigma_x_predictor = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_dim),
        )
        self.mu_h_predictor = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_node_nf),
        )
        self.log_sigma_h_predictor = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_node_nf),
        )

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_x = self.mu_x_predictor(condition)
        sigma_x = torch.exp(self.log_sigma_x_predictor(condition))
        mu_h = self.mu_h_predictor(condition)
        sigma_h = torch.exp(self.log_sigma_h_predictor(condition))
        return mu_x, sigma_x, mu_h, sigma_h


# =============================================================================
# Main Class: FlexiblePriorManager
# =============================================================================

class FlexiblePriorManager(nn.Module):
    """
    统一的三模式Prior管理器

    支持模式:
    - 'custom': 手动设定prior参数
    - 'conditional': 条件注入动态prior
    - 'learned': 端到端学习prior参数

    支持prior类型 (custom模式):
    - 'isotropic_gaussian': 各向同性高斯
    - 'anisotropic_gaussian': 各向异性高斯
    - 'categorical_uniform': 均匀离散分布
    - 'golden_prior': 数据驱动prior
    """

    VALID_MODES = ['custom', 'conditional', 'learned']
    VALID_PRIOR_TYPES = ['isotropic_gaussian', 'anisotropic_gaussian',
                          'categorical_uniform', 'golden_prior']

    def __init__(
        self,
        n_dim: int,
        in_node_nf: int,
        mode: str = 'custom',
        prior_type: str = 'isotropic_gaussian',
        # Custom模式参数
        custom_mu_x: float = 0.0,
        custom_sigma_x: float = 1.0,
        custom_cov_x: np.ndarray = None,
        custom_mu_h: float = 0.0,
        custom_sigma_h: float = 1.0,
        golden_prior_dict: Dict = None,
        # Conditional模式参数
        condition_types: List[str] = None,
        condition_dims: Dict[str, int] = None,
        conditioning_net_hidden: int = 256,
        # Learned模式参数
        learnable_components: int = 5,
        learn_mu: bool = True,
        learn_sigma: bool = True,
        use_mixture: bool = False,
    ):
        super().__init__()

        assert mode in self.VALID_MODES, f"Invalid mode: {mode}. Valid modes: {self.VALID_MODES}"
        if mode == 'custom':
            assert prior_type in self.VALID_PRIOR_TYPES, \
                f"Invalid prior_type: {prior_type}. Valid types: {self.VALID_PRIOR_TYPES}"

        self.n_dim = n_dim
        self.in_node_nf = in_node_nf
        self.mode = mode
        self.prior_type = prior_type

        # ==================== Custom模式 ====================
        if mode == 'custom':
            if prior_type == 'isotropic_gaussian':
                self.position_sampler = IsotropicGaussianSampler(n_dim=n_dim, mu=custom_mu_x, sigma=custom_sigma_x)
                self.feature_sampler = IsotropicGaussianSampler(n_dim=in_node_nf, mu=custom_mu_h, sigma=custom_sigma_h)
            elif prior_type == 'anisotropic_gaussian':
                self.position_sampler = AnisotropicGaussianSampler(
                    n_dim=n_dim, mu=np.full(n_dim, custom_mu_x) if isinstance(custom_mu_x, float) else custom_mu_x,
                    cov=custom_cov_x
                )
                self.feature_sampler = IsotropicGaussianSampler(n_dim=in_node_nf, mu=custom_mu_h, sigma=custom_sigma_h)
            elif prior_type == 'categorical_uniform':
                self.position_sampler = IsotropicGaussianSampler(n_dim=n_dim, mu=custom_mu_x, sigma=custom_sigma_x)
                self.feature_sampler = CategoricalUniformSampler(num_classes=in_node_nf)
            elif prior_type == 'golden_prior':
                self.position_sampler = GoldenPriorSampler(n_dim=n_dim, golden_prior_dict=golden_prior_dict)
                self.feature_sampler = IsotropicGaussianSampler(n_dim=in_node_nf, mu=custom_mu_h, sigma=custom_sigma_h)

        # ==================== Conditional模式 ====================
        elif mode == 'conditional':
            default_condition_types = ['topology', 'geometry', 'pharmacophore', 'electron_density']
            default_condition_dims = {'topology': 64, 'geometry': 128, 'pharmacophore': 64, 'electron_density': 32}
            self.condition_encoder = ConditionEncoder(
                condition_types=condition_types or default_condition_types,
                condition_dims=condition_dims or default_condition_dims,
                output_dim=conditioning_net_hidden, hidden_dim=conditioning_net_hidden
            )
            self.prior_predictor = PriorParameterPredictor(
                condition_dim=conditioning_net_hidden, n_dim=n_dim, in_node_nf=in_node_nf, hidden_dim=conditioning_net_hidden
            )

        # ==================== Learned模式 ====================
        elif mode == 'learned':
            if use_mixture:
                self.learned_prior = MixtureGaussianPrior(n_dim=n_dim, in_node_nf=in_node_nf, num_components=learnable_components)
            else:
                self.learned_prior = LearnableGaussianPrior(n_dim=n_dim, in_node_nf=in_node_nf, learn_mu=learn_mu, learn_sigma=learn_sigma)

    def forward(self, z_x: torch.Tensor, z_h: torch.Tensor,
                node_mask: torch.Tensor, conditions_dict: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        if self.mode == 'custom':
            log_pz_x = self.position_sampler.log_likelihood(z_x, node_mask)
            log_pz_h = self.feature_sampler.log_likelihood(z_h, node_mask)
            return log_pz_x + log_pz_h
        elif self.mode == 'conditional':
            fused_condition = self.condition_encoder(conditions_dict or {})
            mu_x, sigma_x, mu_h, sigma_h = self.prior_predictor(fused_condition)
            N = node_mask.squeeze(2).sum(1)
            degrees_of_freedom_x = (N - 1) * self.n_dim
            r2_x = sum_except_batch(z_x.pow(2))
            sigma2_x = sigma_x.pow(2).mean()
            log_pz_x = -0.5 * r2_x / sigma2_x - 0.5 * degrees_of_freedom_x * np.log(2 * np.pi)
            z_h_centered = z_h - mu_h.unsqueeze(1)
            sigma2_h = sigma_h.pow(2).unsqueeze(1)
            log_pz_h_elementwise = -0.5 * z_h_centered.pow(2) / sigma2_h - 0.5 * np.log(2 * np.pi) - torch.log(sigma_h).unsqueeze(1)
            log_pz_h = sum_except_batch(log_pz_h_elementwise * node_mask)
            return log_pz_x + log_pz_h
        elif self.mode == 'learned':
            if hasattr(self.learned_prior, 'log_likelihood'):
                if isinstance(self.learned_prior, MixtureGaussianPrior):
                    return self.learned_prior.log_likelihood(z_x, z_h, node_mask)
                else:
                    log_pz_x, log_pz_h = self.learned_prior.log_likelihood(z_x, z_h, node_mask)
                    return log_pz_x + log_pz_h
            else:
                N = node_mask.squeeze(2).sum(1)
                degrees_of_freedom_x = (N - 1) * self.n_dim
                r2_x = sum_except_batch(z_x.pow(2))
                log_pz_x = -0.5 * r2_x - 0.5 * degrees_of_freedom_x * np.log(2 * np.pi)
                log_pz_h = sum_except_batch(-0.5 * z_h.pow(2) - 0.5 * np.log(2 * np.pi) * node_mask)
                return log_pz_x + log_pz_h

    def sample(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor,
               conditions_dict: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = node_mask.device
        size_x = (n_samples, n_nodes, self.n_dim)
        size_h = (n_samples, n_nodes, self.in_node_nf)

        if self.mode == 'custom':
            z_x = self.position_sampler.sample(size_x, device, node_mask)
            z_h = self.feature_sampler.sample(size_h, device, node_mask)
            return z_x, z_h
        elif self.mode == 'conditional':
            fused_condition = self.condition_encoder(conditions_dict or {})
            mu_x, sigma_x, mu_h, sigma_h = self.prior_predictor(fused_condition)
            eps_x = torch.randn(size_x, device=device)
            z_x = eps_x * sigma_x.unsqueeze(1) + mu_x.unsqueeze(1)
            z_x = z_x * node_mask
            z_x = remove_mean_with_mask(z_x, node_mask)
            eps_h = torch.randn(size_h, device=device)
            z_h = eps_h * sigma_h.unsqueeze(1) + mu_h.unsqueeze(1)
            z_h = z_h * node_mask
            return z_x, z_h
        elif self.mode == 'learned':
            return self.learned_prior.sample(size_x, device, node_mask)

    def compute_golden_prior(self, ligand_pos: torch.Tensor,
                             ligand_atom_mask: torch.Tensor = None) -> Dict:
        """从数据计算golden prior"""
        from sklearn.metrics.pairwise import pairwise_distances

        def get_iso_aniso_mu_cov(pos):
            if pos.shape[0] == 0:
                return np.zeros(3), np.eye(3) * 0, np.zeros(3), np.eye(3) * 0
            pos_np = pos.cpu().numpy() if pos.is_cuda else pos.numpy()
            mu = pos_np.mean(axis=0)
            pos_centered = pos_np - mu
            N = pos_np.shape[0]
            covariance_iso = (pos_centered ** 2).sum() / N * np.eye(3)
            covariance_aniso = np.matmul(pos_centered.T, pos_centered) / N
            return mu, covariance_iso, mu, covariance_aniso

        if ligand_atom_mask is None:
            mu_iso, cov_iso, mu_aniso, cov_aniso = get_iso_aniso_mu_cov(ligand_pos)
            return {'arms_prior': [(ligand_pos.shape[0], mu_iso, cov_iso, mu_aniso, cov_aniso)], 'scaffold_prior': []}

        unique_masks = torch.unique(ligand_atom_mask)
        arms_prior = []
        scaffold_prior = []
        for mask_val in unique_masks:
            atom_pos = ligand_pos[ligand_atom_mask == mask_val]
            mu_iso, cov_iso, mu_aniso, cov_aniso = get_iso_aniso_mu_cov(atom_pos)
            if mask_val == -1:
                scaffold_prior.append((atom_pos.shape[0], mu_iso, cov_iso, mu_aniso, cov_aniso))
            else:
                arms_prior.append((atom_pos.shape[0], mu_iso, cov_iso, mu_aniso, cov_aniso))

        return {'arms_prior': arms_prior, 'scaffold_prior': scaffold_prior,
                'num_arms': len(arms_prior), 'num_scaffold': len(scaffold_prior)}

    def kl_divergence(self, q_mu_x: torch.Tensor, q_sigma_x: torch.Tensor,
                      q_mu_h: torch.Tensor, q_sigma_h: torch.Tensor,
                      node_mask: torch.Tensor,
                      conditions_dict: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """计算KL(q || p)"""
        if self.mode == 'custom':
            if isinstance(self.position_sampler, IsotropicGaussianSampler):
                p_mu_x = torch.zeros_like(q_mu_x) + self.position_sampler.mu
                p_sigma_x = torch.ones_like(q_sigma_x) * self.position_sampler.sigma
            elif isinstance(self.position_sampler, AnisotropicGaussianSampler):
                p_mu_x = self.position_sampler.mu.to(q_mu_x.device).unsqueeze(0).unsqueeze(0)
                p_sigma_x = torch.ones_like(q_sigma_x)
            if isinstance(self.feature_sampler, IsotropicGaussianSampler):
                p_mu_h = torch.zeros_like(q_mu_h) + self.feature_sampler.mu
                p_sigma_h = torch.ones_like(q_sigma_h) * self.feature_sampler.sigma
        elif self.mode == 'conditional':
            fused_condition = self.condition_encoder(conditions_dict or {})
            p_mu_x, p_sigma_x, p_mu_h, p_sigma_h = self.prior_predictor(fused_condition)
            p_mu_x = p_mu_x.unsqueeze(1).expand_as(q_mu_x)
            p_sigma_x = p_sigma_x.unsqueeze(1).expand_as(q_sigma_x)
            p_mu_h = p_mu_h.unsqueeze(1).expand_as(q_mu_h)
            p_sigma_h = p_sigma_h.unsqueeze(1).expand_as(q_sigma_h)
        elif self.mode == 'learned':
            if isinstance(self.learned_prior, LearnableGaussianPrior):
                p_mu_x = self.learned_prior.mu_x.to(q_mu_x.device).unsqueeze(0).unsqueeze(0)
                p_sigma_x = self.learned_prior.get_sigma_x().to(q_sigma_x.device).unsqueeze(0).unsqueeze(0)
                p_mu_h = self.learned_prior.mu_h.to(q_mu_h.device).unsqueeze(0).unsqueeze(0)
                p_sigma_h = self.learned_prior.get_sigma_h().to(q_sigma_h.device).unsqueeze(0).unsqueeze(0)
            else:
                weights = self.learned_prior.get_mixture_weights()
                p_mu_x = self.learned_prior.mu_x_components[0].to(q_mu_x.device).unsqueeze(0).unsqueeze(0)
                p_sigma_x = self.learned_prior.get_sigma_x_components()[0].to(q_sigma_x.device).unsqueeze(0).unsqueeze(0)
                p_mu_h = self.learned_prior.mu_h_components[0].to(q_mu_h.device).unsqueeze(0).unsqueeze(0)
                p_sigma_h = self.learned_prior.get_sigma_h_components()[0].to(q_sigma_h.device).unsqueeze(0).unsqueeze(0)

        N = node_mask.squeeze(2).sum(1)
        d_x = (N - 1) * self.n_dim
        kl_x = d_x * torch.log(p_sigma_x.mean() / q_sigma_x.mean()) \
               + 0.5 * (d_x * q_sigma_x.pow(2).mean() + sum_except_batch((q_mu_x - p_mu_x).pow(2))) / p_sigma_x.pow(2).mean() \
               - 0.5 * d_x
        kl_h_elementwise = (
            torch.log(p_sigma_h / q_sigma_h)
            + 0.5 * (q_sigma_h.pow(2) + (q_mu_h - p_mu_h).pow(2)) / p_sigma_h.pow(2)
            - 0.5
        )
        kl_h = sum_except_batch(kl_h_elementwise * node_mask)
        return kl_x + kl_h


# =============================================================================
# Factory Function
# =============================================================================

def create_prior(
    n_dim: int = 3,
    in_node_nf: int = 10,
    mode: str = 'custom',
    prior_type: str = 'isotropic_gaussian',
    **kwargs
) -> FlexiblePriorManager:
    """创建prior的工厂函数"""
    return FlexiblePriorManager(
        n_dim=n_dim, in_node_nf=in_node_nf, mode=mode, prior_type=prior_type, **kwargs
    )