"""
neural_3d_fingerprint.py - 神经3D指纹系统

核心功能：
1. 使用EGNN提取神经3D指纹
2. 构象一致性对比学习
3. FAISS向量检索（替代传统Tanimoto）

作者：BioIsosteric项目
"""
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import logging

from rdkit import Chem
from rdkit.Chem import AllChem

# 尝试导入FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: faiss not installed. Vector search will use numpy fallback.")
    print("Install with: pip install faiss-cpu or faiss-gpu")

sys.path.insert(0, '.')
from v2_4 import (
    BioIsostericFragmentModel,
    FragmentDataset,
    device,
)

logger = logging.getLogger("Neural3DFingerprint")


class NeuralFingerprintIndex:
    """
    神经3D指纹索引系统

    功能：
    - 预计算知识库中所有分子的神经指纹
    - 使用FAISS建立高效索引
    - 快速向量检索替代传统Tanimoto
    """

    def __init__(self, model: BioIsostericFragmentModel, hidden_dim: int = 128):
        """
        Args:
            model: 已训练的BioIsostericFragmentModel
            hidden_dim: 神经指纹维度
        """
        self.model = model
        self.hidden_dim = hidden_dim
        self.index = None
        self.mol_list: List[Chem.Mol] = []
        self.smiles_list: List[str] = []
        self.vectors: Optional[np.ndarray] = None

    def build_index(self, mol_list: List[Chem.Mol], smiles_list: Optional[List[str]] = None):
        """
        构建FAISS索引

        Args:
            mol_list: 分子列表
            smiles_list: SMILES列表（可选）
        """
        logger.info(f"Building neural fingerprint index for {len(mol_list)} molecules...")

        self.mol_list = mol_list
        self.smiles_list = smiles_list or [Chem.MolToSmiles(m) for m in mol_list]

        # 预计算所有神经指纹
        vectors = []
        for mol in mol_list:
            try:
                fp = self.model.get_neural_3d_fingerprint(mol, use_pos=True)
                vectors.append(fp)
            except Exception as e:
                logger.warning(f"Failed to extract fingerprint: {e}")
                vectors.append(np.zeros(self.hidden_dim))

        self.vectors = np.array(vectors).astype('float32')

        # 建立FAISS索引
        if HAS_FAISS:
            # 使用L2距离索引（也可以用余弦相似度索引）
            self.index = faiss.IndexFlatL2(self.hidden_dim)
            self.index.add(self.vectors)
            logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        else:
            # 使用numpy作为后备
            self.index = None
            logger.info(f"Using numpy fallback for {len(self.vectors)} vectors")

    def search(self, query_mol: Chem.Mol, top_k: int = 10) -> List[Tuple[int, float, Chem.Mol]]:
        """
        快速检索最相似的分子

        Args:
            query_mol: 查询分子
            top_k: 返回前K个最相似分子

        Returns:
            [(index, distance, mol), ...] 列表
        """
        # 提取查询指纹
        query_fp = self.model.get_neural_3d_fingerprint(query_mol, use_pos=True)
        query_vec = query_fp.astype('float32').reshape(1, -1)

        if HAS_FAISS and self.index is not None:
            # FAISS检索 - O(log N)
            distances, indices = self.index.search(query_vec, top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.mol_list):
                    results.append((int(idx), float(distances[0][i]), self.mol_list[idx]))
            return results
        else:
            # NumPy后备 - O(N)
            distances = np.linalg.norm(self.vectors - query_vec, axis=1)
            sorted_indices = np.argsort(distances)[:top_k]
            results = []
            for idx in sorted_indices:
                results.append((int(idx), float(distances[idx]), self.mol_list[idx]))
            return results

    def search_batch(self, query_mols: List[Chem.Mol], top_k: int = 10) -> List[List[Tuple[int, float, Chem.Mol]]]:
        """
        批量检索

        Args:
            query_mols: 查询分子列表
            top_k: 每个查询返回的前K个

        Returns:
            每个查询的检索结果列表
        """
        # 批量提取指纹
        query_vecs = []
        for mol in query_mols:
            try:
                fp = self.model.get_neural_3d_fingerprint(mol, use_pos=True)
                query_vecs.append(fp)
            except:
                query_vecs.append(np.zeros(self.hidden_dim))

        query_vecs = np.array(query_vecs).astype('float32')

        if HAS_FAISS and self.index is not None:
            distances, indices = self.index.search(query_vecs, top_k)
            results = []
            for q_idx in range(len(query_mols)):
                q_results = []
                for i, idx in enumerate(indices[q_idx]):
                    if idx < len(self.mol_list):
                        q_results.append((int(idx), float(distances[q_idx][i]), self.mol_list[idx]))
                results.append(q_results)
            return results
        else:
            # NumPy后备
            results = []
            for query_vec in query_vecs:
                distances = np.linalg.norm(self.vectors - query_vec.reshape(1, -1), axis=1)
                sorted_indices = np.argsort(distances)[:top_k]
                q_results = [(int(idx), float(distances[idx]), self.mol_list[idx]) for idx in sorted_indices]
                results.append(q_results)
            return results

    def save_index(self, path: str):
        """保存索引到文件"""
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, path)
            logger.info(f"FAISS index saved to {path}")
        # 同时保存分子列表和向量
        np.save(path + '.vectors', self.vectors)
        np.save(path + '.smiles', self.smiles_list)

    def load_index(self, path: str):
        """从文件加载索引"""
        if HAS_FAISS and os.path.exists(path):
            self.index = faiss.read_index(path)
            logger.info(f"FAISS index loaded from {path}")

        # 加载向量
        vectors_path = path + '.vectors.npy'
        if os.path.exists(vectors_path):
            self.vectors = np.load(vectors_path)


def compute_neural_similarity(model: BioIsostericFragmentModel,
                               mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    计算两个分子的神经3D指纹相似度

    Args:
        model: BioIsostericFragmentModel
        mol1, mol2: RDKit分子

    Returns:
        相似度分数 (0-1)
    """
    fp1 = model.get_neural_3d_fingerprint(mol1, use_pos=True)
    fp2 = model.get_neural_3d_fingerprint(mol2, use_pos=True)

    # Cosine相似度
    dot_product = np.dot(fp1, fp2)
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cosine_sim = dot_product / (norm1 * norm2)

    # 转换到0-1范围
    similarity = (cosine_sim + 1) / 2

    return float(similarity)


def build_kb_from_dataset(model: BioIsostericFragmentModel,
                          dataset: FragmentDataset) -> NeuralFingerprintIndex:
    """
    从FragmentDataset构建知识库索引

    Args:
        model: 已训练模型
        dataset: FragmentDataset

    Returns:
        NeuralFingerprintIndex
    """
    mol_list = []
    smiles_list = []

    for idx in range(len(dataset)):
        decomp = dataset[idx]
        mol = decomp['original_mol']
        smiles = Chem.MolToSmiles(mol)

        mol_list.append(mol)
        smiles_list.append(smiles)

    index = NeuralFingerprintIndex(model)
    index.build_index(mol_list, smiles_list)

    return index


# 测试代码
if __name__ == "__main__":
    print("Loading model...")
    model = BioIsostericFragmentModel(hidden_dim=128).to(device)
    model.load_state_dict(torch.load('bioisosteric_fragment_model_v2_4.pth', map_location=device))
    model.eval()

    print("Loading dataset...")
    dataset = FragmentDataset('chembl_data_sample', max_mols=100)

    print("Building KB index...")
    kb_index = build_kb_from_dataset(model, dataset)

    # 测试检索
    print("\nTesting search...")
    test_mol = dataset[0]['original_mol']
    results = kb_index.search(test_mol, top_k=5)

    print(f"Query: {Chem.MolToSmiles(test_mol)}")
    print(f"Top 5 similar molecules:")
    for idx, dist, mol in results:
        print(f"  [{idx}] distance={dist:.3f} SMILES={Chem.MolToSmiles(mol)[:40]}")

    print("\nNeural 3D Fingerprint system ready!")