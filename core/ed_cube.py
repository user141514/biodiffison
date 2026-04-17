# ed_cube.py
# 作用：把真实 ED/ESP 网格（.cube 或 .npz）转成 ed_vec（球壳采样统计向量）
# 接口保持与 ed_context 兼容：主程序无需改训练/推理流程，只改“预计算 ed_vec 的数据源”。

import os
import numpy as np
from typing import Optional, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from ed_context import fibonacci_sphere  # 复用你的球面采样

BOHR_TO_ANG = 0.529177210903  # Bohr→Å

def read_cube(path: str) -> Dict[str, np.ndarray]:
    """读取 .cube，返回 grid、origin（Å）、steps（Å/格）"""
    with open(path, "r") as f:
        _ = f.readline(); _ = f.readline()  # title/comment
        h3 = f.readline().split()
        natoms = int(h3[0]); origin = np.array(list(map(float, h3[1:4])), dtype=np.float64)

        nx_line = f.readline().split(); nx = int(nx_line[0]); ax = np.array(list(map(float, nx_line[1:4])))
        ny_line = f.readline().split(); ny = int(ny_line[0]); ay = np.array(list(map(float, ny_line[1:4])))
        nz_line = f.readline().split(); nz = int(nz_line[0]); az = np.array(list(map(float, nz_line[1:4])))

        for _ in range(abs(natoms)): f.readline()
        vals = []
        n_total = nx * ny * nz
        while len(vals) < n_total:
            line = f.readline()
            if not line: break
            vals.extend(map(float, line.split()))
        arr = np.array(vals, dtype=np.float32)
        if arr.size != n_total:
            raise ValueError(f"Cube size mismatch: expect {n_total}, got {arr.size}")
        # CUBE 通常为 z 外层、x 内层，这里 reshape 为 [nz,ny,nx] 再转置到 [nx,ny,nz]
        grid = arr.reshape(nz, ny, nx).transpose(2, 1, 0).copy()  # [nx,ny,nz]

    origin_A = origin * BOHR_TO_ANG
    steps_A  = np.stack([ax, ay, az], axis=0) * BOHR_TO_ANG  # [3,3] 每行一个轴的步进向量
    return dict(grid=grid.astype(np.float32), origin=origin_A.astype(np.float32), steps=steps_A.astype(np.float32))

def load_npz(path: str) -> Dict[str, np.ndarray]:
    """加载你离线转存的 .npz，字段为 grid、origin、steps"""
    z = np.load(path)
    return dict(grid=z["grid"].astype(np.float32), origin=z["origin"].astype(np.float32), steps=z["steps"].astype(np.float32))

def world_to_grid_frac(points_A: np.ndarray, origin_A: np.ndarray, steps_A: np.ndarray) -> np.ndarray:
    """世界坐标（Å）→分数坐标 u（非整数下标），r = origin + ux*ax + uy*ay + uz*az"""
    A_T = steps_A.T                       # [3,3]
    diff = (points_A - origin_A[None,:]).T  # [3,M]
    u = np.linalg.solve(A_T, diff).T      # [M,3]
    return u

def trilinear_sample(points_A: np.ndarray, cube: Dict[str, np.ndarray]) -> np.ndarray:
    """三线性插值：越界返回0更稳妥"""
    grid = cube["grid"]; nx, ny, nz = grid.shape
    u = world_to_grid_frac(points_A, cube["origin"], cube["steps"])
    ux, uy, uz = u[:,0], u[:,1], u[:,2]
    x0 = np.floor(ux).astype(np.int64); x1 = x0 + 1; fx = ux - x0
    y0 = np.floor(uy).astype(np.int64); y1 = y0 + 1; fy = uy - y0
    z0 = np.floor(uz).astype(np.int64); z1 = z0 + 1; fz = uz - z0
    mask = (x0>=0)&(x1<nx)&(y0>=0)&(y1<ny)&(z0>=0)&(z1<nz)
    out = np.zeros(points_A.shape[0], dtype=np.float32)
    if not mask.any(): return out
    x0m,x1m,y0m,y1m,z0m,z1m = x0[mask],x1[mask],y0[mask],y1[mask],z0[mask],z1[mask]
    fxm,fym,fzm = fx[mask],fy[mask],fz[mask]
    c000=grid[x0m,y0m,z0m]; c100=grid[x1m,y0m,z0m]; c010=grid[x0m,y1m,z0m]; c110=grid[x1m,y1m,z0m]
    c001=grid[x0m,y0m,z1m]; c101=grid[x1m,y0m,z1m]; c011=grid[x0m,y1m,z1m]; c111=grid[x1m,y1m,z1m]
    c00 = c000*(1-fxm)+c100*fxm; c01 = c001*(1-fxm)+c101*fxm
    c10 = c010*(1-fxm)+c110*fxm; c11 = c011*(1-fxm)+c111*fxm
    c0  = c00*(1-fym)+c10*fym;   c1  = c01*(1-fym)+c11*fym
    v   = c0*(1-fzm)+c1*fzm
    out[mask] = v.astype(np.float32)
    return out

def build_ed_context_vector_from_cube(
    mol: Chem.Mol,
    cube: Dict[str, np.ndarray],
    radii=(1.5,2.5,3.5),
    n_dir=64,
    stats=("mean","std","posfrac","p10","p90"),
) -> np.ndarray:
    """用真实网格做球壳采样统计，输出与 ed_context 相同维度的向量"""
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())

    # 采样中心：优先 dummy 原子，没则质心
    def _centers_auto(mm):
        conf = mm.GetConformer()
        d = [a.GetIdx() for a in mm.GetAtoms() if a.GetAtomicNum()==0]
        if d:
            return [np.array(list(conf.GetAtomPosition(i)), dtype=np.float64) for i in d]
        pos = np.array([conf.GetAtomPosition(i) for i in range(mm.GetNumAtoms())], dtype=np.float64)
        return [pos.mean(axis=0)]

    centers = _centers_auto(m)
    dirs    = fibonacci_sphere(n_dir)
    per_c   = []
    for c in centers:
        feat_r = []
        for r in radii:
            pts  = c[None,:] + r*dirs
            vals = trilinear_sample(pts, cube)                 # 真实网格取值
            vn   = vals / (np.linalg.norm(vals)+1e-8)          # 层内范数归一，突出“模式”
            fs   = []
            for s in stats:
                if s=="mean": fs.append(float(vn.mean()))
                elif s=="std": fs.append(float(vn.std()))
                elif s=="posfrac": fs.append(float((vn>0).mean()))
                elif s=="p10": fs.append(float(np.percentile(vn,10)))
                elif s=="p90": fs.append(float(np.percentile(vn,90)))
            feat_r.extend(fs)
        per_c.append(np.array(feat_r, dtype=np.float32))
    if not per_c:
        return np.zeros((len(radii)*len(stats),), dtype=np.float32)
    return np.mean(np.stack(per_c, axis=0), axis=0)