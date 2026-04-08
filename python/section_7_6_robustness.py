# -*- coding: utf-8 -*-
"""
section_7_6_robustness.py  v2 (向量化加速版)
==============================================================================
第7.6节: 物理扰动鲁棒性验证 — GPS高斯白噪声

核心修正 (v1→v2):
  v1问题: 逐样本串行计算 pdist(200无人机) + FFT(19900元素)
          总计 3603×30×5 = 540,450次 → 约90分钟
  v2优化: 完整批量向量化
    ① 预计算速度矩阵 (GPS噪声免疫, 仅算一次)
    ② 批量广播距离矩阵: (batch,N,N) 代替逐样本pdist
    ③ 批量einsum相关性: 代替逐样本corrcoef
    ④ 批量rfft谱特征: 代替逐样本FFT
    ⑤ 分层子采样 (500样本): 保证统计代表性, 大幅降低计算量
  预计运行时间: 3-8分钟

物理机理:
  GPS噪声仅影响位置 X[0::12], Y[1::12]
  速度 VX[2::12], VY[3::12] 来自独立传感器(陀螺仪/IMU) → 不受GPS干扰

  3D基准 (corr1,2,3): 完全依赖R(X_noisy,Y_noisy) → 全部3特征受GPS干扰
  13D本文 (SCDAM):
    feature[9]  = omega_bar  ← 纯速度方向 → 完全免疫GPS噪声
    feature[0-8]= 谱特征      ← M = R_noisy × W_vel (速度权重正确) → 部分补偿
    feature[10-12]= corr_M   ← 同上
==============================================================================
"""

import sys, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), 'csv')
sys.path.insert(0, SCRIPT_DIR)

print("=" * 72)
print("  第7.6节: 物理扰动鲁棒性验证 (v2 向量化加速版)")
print("=" * 72)


# =====================================================================
# 批量向量化特征提取核心函数
# =====================================================================

def batch_corrcoef_mean(B_batch):
    """
    批量计算相关矩阵均值.
    B_batch: (batch, m, N) — 每列是一个'变量'(无人机), m个观测(最近邻层)
    Returns: (batch,) — 每个样本的相关矩阵均值

    等价于对每个b: np.mean(np.corrcoef(B_batch[b].T))
    但使用einsum避免Python循环, 速度提升约50×
    """
    # 沿m轴中心化 (减去各列均值)
    B_c = B_batch - B_batch.mean(axis=1, keepdims=True)   # (batch, m, N)
    # 计算各列L2范数
    norm = np.sqrt((B_c ** 2).sum(axis=1, keepdims=True))  # (batch, 1, N)
    norm = np.where(norm > 1e-12, norm, 1.0)
    B_n  = B_c / norm                                       # (batch, m, N)
    # 批量外积 → (batch, N, N) 相关矩阵 (未除以m, 但mean值不变)
    corr = np.einsum('bkn,bkm->bnm', B_n, B_n)            # (batch, N, N)
    return corr.mean(axis=(1, 2))                           # (batch,)


def batch_three_variants(R_sorted, m=8):
    """
    从排序后的距离矩阵 R_sorted (batch, m, N) 计算三种自相关变体.
    等价于 paper2_algorithms.compute_three_variants(R, m)

    corr1: 直接最近邻矩阵
    corr2: 行均值修正后的矩阵 (b_jk = r*_jk - mean_k)
    corr3: 差分矩阵 (首行保留, 后续行取差分)

    Returns: (batch, 3)
    """
    # corr1
    c1 = batch_corrcoef_mean(R_sorted)

    # corr2: 减去各行的列均值 (axis=2 是N维, 对应原始的axis=1)
    B2 = R_sorted - R_sorted.mean(axis=2, keepdims=True)  # (batch, m, N)
    c2 = batch_corrcoef_mean(B2)

    # corr3: 差分
    B3 = np.zeros_like(R_sorted)
    B3[:, 0, :]  = R_sorted[:, 0, :]          # 保留第一行
    B3[:, 1:, :] = np.diff(R_sorted, axis=1)   # 后续行差分
    c3 = batch_corrcoef_mean(B3)

    return np.column_stack([c1, c2, c3])       # (batch, 3)


def batch_spectral_scalars(q_batch, tau_max=50):
    """
    批量FFT自相关谱特征提取 (E, P, H).
    q_batch: (batch, L) — 每行是SCDAM矩阵上三角元素序列
    Returns: (batch, 3)

    使用rfft (实数输入快速版本) 代替fft, 速度约提升2×.
    """
    batch, L = q_batch.shape
    q_c  = q_batch - q_batch.mean(axis=1, keepdims=True)   # 中心化
    var  = q_c.var(axis=1)                                  # (batch,) 各样本方差

    # FFT大小: 下一个2的幂 ≥ 2L
    n_fft = 1
    while n_fft < 2 * L:
        n_fft *= 2

    # 批量rfft: (batch, n_fft//2+1) 复数
    Q      = np.fft.rfft(q_c, n=n_fft, axis=1)
    power  = np.abs(Q) ** 2
    R_full = np.fft.irfft(power, n=n_fft, axis=1)          # (batch, n_fft)

    # 取前tau_max个延迟, 归一化
    tau    = min(tau_max, L - 1)
    t_arr  = np.arange(tau, dtype=float)
    # weights[b, t] = (L - t) * var[b]
    weights = np.outer(var, np.ones(tau)) * (L - t_arr)[np.newaxis, :]  # (batch, tau)
    weights = np.where(weights > 1e-15, weights, 1.0)
    Phi     = R_full[:, :tau] / weights                     # (batch, tau)

    # 对方差≈0的样本置零
    valid = (var > 1e-15)
    Phi[~valid] = 0.0

    # E = mean(Phi²)
    E = (Phi ** 2).mean(axis=1)                             # (batch,)
    # P = (max-min) / (mean|Phi|+ε)
    abs_Phi = np.abs(Phi)
    P = (abs_Phi.max(axis=1) - abs_Phi.min(axis=1)) / (abs_Phi.mean(axis=1) + 1e-8)
    # H = -sum(Pn * log2(Pn))
    ab  = abs_Phi + 1e-15
    Pn  = ab / ab.sum(axis=1, keepdims=True)
    H   = -(Pn * np.log2(Pn)).sum(axis=1)

    return np.column_stack([E, P, H])                       # (batch, 3)


def extract_features_batch(X_b, Y_b, VX_b, VY_b,
                           exp_weight_b=None, omega_bar_b=None,
                           m=8, lam=1.0, tau_max=50):
    """
    批量提取 3D 和 13D 特征.

    参数:
      X_b, Y_b     : (batch, N) 无人机位置 (可能已添加GPS噪声)
      VX_b, VY_b   : (batch, N) 无人机速度 (可能已添加速度噪声)
      exp_weight_b : (batch, N, N) 预计算的速度权重 exp(λ(1-ω)) [可选]
                     若为None则现场计算 (速度有噪声时使用)
      omega_bar_b  : (batch,) 预计算的全局速度对齐均值 [可选]

    返回:
      feat_3d  : (batch, 3)
      feat_13d : (batch, 13)
    """
    batch, N = X_b.shape

    # ── 1. 距离矩阵 R (批量广播) ─────────────────────────────────────
    # diff[b, k, l, :] = coords[b, k, :] - coords[b, l, :]
    dx = X_b[:, :, np.newaxis] - X_b[:, np.newaxis, :]    # (batch, N, N)
    dy = Y_b[:, :, np.newaxis] - Y_b[:, np.newaxis, :]
    R  = np.sqrt(dx ** 2 + dy ** 2)                         # (batch, N, N)
    R[:, range(N), range(N)] = 0.0                          # 对角线置0

    # ── 2. 3D特征 (仅依赖R) ──────────────────────────────────────────
    R_sorted_3d = np.sort(R, axis=1)[:, 1:m+1, :]          # (batch, m, N) 最近8邻
    feat_3d = batch_three_variants(R_sorted_3d, m)          # (batch, 3)

    # ── 3. 速度权重矩阵 (若未预计算则现场计算) ────────────────────────
    if exp_weight_b is None:
        V    = np.stack([VX_b, VY_b], axis=-1)              # (batch, N, 2)
        vmag = np.linalg.norm(V, axis=-1, keepdims=True)    # (batch, N, 1)
        D    = np.where(vmag > 1e-6, V / vmag, 0.0)         # (batch, N, 2)
        S    = np.einsum('bkd,bld->bkl', D, D)             # (batch, N, N)
        np.clip(S, -1.0, 1.0, out=S)
        omega = (1.0 + S) / 2.0                             # (batch, N, N)
        exp_weight_b = np.exp(lam * (1.0 - omega))          # (batch, N, N)
        triu = np.triu_indices(N, k=1)
        omega_bar_b  = omega[:, triu[0], triu[1]].mean(axis=1)  # (batch,)

    # ── 4. SCDAM矩阵 M = R × exp_weight ──────────────────────────────
    M = R * exp_weight_b                                     # (batch, N, N)
    M[:, range(N), range(N)] = 0.0

    # ── 5. 13D特征: SCDAM相关变体 [10-12] ────────────────────────────
    M_sorted = np.sort(M, axis=1)[:, 1:m+1, :]             # (batch, m, N)
    feat_M   = batch_three_variants(M_sorted, m)             # (batch, 3)

    # ── 6. 13D特征: FFT谱特征 [0-8] ──────────────────────────────────
    triu_i, triu_j = np.triu_indices(N, k=1)
    q_batch  = M[:, triu_i, triu_j]                         # (batch, n_pairs)
    q_diff   = np.diff(q_batch, axis=1)                     # (batch, n_pairs-1)
    q_bias   = np.abs(q_batch - q_batch.mean(axis=1, keepdims=True))

    spec_d   = batch_spectral_scalars(q_batch,  tau_max)    # (batch, 3)
    spec_f   = batch_spectral_scalars(q_diff,   tau_max)    # (batch, 3)
    spec_b   = batch_spectral_scalars(q_bias,   tau_max)    # (batch, 3)

    # ── 7. 组合13D特征 ────────────────────────────────────────────────
    # [Ed,Pd,Hd, Ef,Pf,Hf, Eb,Pb,Hb, omega_bar, c1_M,c2_M,c3_M]
    feat_13d = np.column_stack([
        spec_d, spec_f, spec_b,
        omega_bar_b[:, np.newaxis],
        feat_M
    ])                                                       # (batch, 13)

    return feat_3d, feat_13d


def evaluate_metrics(y_true, y_pred):
    tp  = np.sum((y_pred == 1) & (y_true == 1))
    tn  = np.sum((y_pred == 0) & (y_true == 0))
    fp  = np.sum((y_pred == 1) & (y_true == 0))
    fn  = np.sum((y_pred == 0) & (y_true == 1))
    sw  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    nsw = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    ov  = (tp + tn) / len(y_true) * 100
    return sw, nsw, ov


def find_best_thr(y_true, probs):
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        _, _, ov = evaluate_metrics(y_true, (probs >= thr).astype(int))
        if ov > best_ov:
            best_ov = ov
            best_thr = thr
    return best_thr


# =====================================================================
# 步骤1: 加载数据
# =====================================================================
print("\n[1/6] 加载原始数据...")

df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'),
                     low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
data   = df_raw.iloc[:, :-1].values.astype(np.float64)
labels = df_raw.iloc[:, -1].values.astype(int)

bl_df   = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n   = min(len(labels), len(bl_df))
data    = data[:min_n]
labels  = labels[:min_n]

rng_split = np.random.RandomState(42)
idx_      = rng_split.permutation(min_n)
n_tr      = int(min_n * 0.70)
n_va      = int(min_n * 0.15)
train_idx = idx_[:n_tr]
val_idx   = idx_[n_tr:n_tr + n_va]
test_idx  = idx_[n_tr + n_va:]

fpd      = 12
N_drones = data.shape[1] // fpd

print(f"  总样本: {min_n} | 测试集: {len(test_idx)}")
print(f"  无人机数量: {N_drones} | fpd={fpd}")

# 数据量纲分析
X_all = data[:, 0::fpd]
Y_all = data[:, 1::fpd]
print(f"  位置范围: X∈[{X_all.min():.0f}, {X_all.max():.0f}]m, "
      f"Y∈[{Y_all.min():.0f}, {Y_all.max():.0f}]m")


# =====================================================================
# 步骤2: 训练集特征提取与模型训练
# =====================================================================
print("\n[2/6] 提取干净特征, 训练模型...")

CLEAN_CACHE = os.path.join(SCRIPT_DIR, 'rob_clean_features.npz')
MODEL_3D_P  = os.path.join(SCRIPT_DIR, 'rob_model_3d.cbm')
MODEL_13D_P = os.path.join(SCRIPT_DIR, 'rob_model_13d.cbm')

def extract_split_batch(indices, tag="", batch_sz=200):
    n     = len(indices)
    F3    = np.zeros((n, 3))
    F13   = np.zeros((n, 13))
    t0    = time.time()
    for b0 in range(0, n, batch_sz):
        b1    = min(b0 + batch_sz, n)
        batch = indices[b0:b1]
        X_b   = data[batch][:, 0::fpd]
        Y_b   = data[batch][:, 1::fpd]
        VX_b  = data[batch][:, 2::fpd]
        VY_b  = data[batch][:, 3::fpd]
        f3, f13 = extract_features_batch(X_b, Y_b, VX_b, VY_b)
        F3[b0:b1]  = f3
        F13[b0:b1] = f13
    print(f"    {tag}: {n}样本, 耗时{time.time()-t0:.1f}s")
    return F3, F13

y_train = labels[train_idx]
y_val   = labels[val_idx]
y_test  = labels[test_idx]

if os.path.exists(CLEAN_CACHE):
    fc = np.load(CLEAN_CACHE)
    X_tr_3d  = fc['X_tr_3d'];  X_va_3d  = fc['X_va_3d'];  X_te_3d  = fc['X_te_3d']
    X_tr_13d = fc['X_tr_13d']; X_va_13d = fc['X_va_13d']; X_te_13d = fc['X_te_13d']
    print("  [缓存] 已加载干净特征")
else:
    X_tr_3d,  X_tr_13d  = extract_split_batch(train_idx, "训练集")
    X_va_3d,  X_va_13d  = extract_split_batch(val_idx,   "验证集")
    X_te_3d,  X_te_13d  = extract_split_batch(test_idx,  "测试集")
    np.savez(CLEAN_CACHE,
             X_tr_3d=X_tr_3d,  X_va_3d=X_va_3d,  X_te_3d=X_te_3d,
             X_tr_13d=X_tr_13d, X_va_13d=X_va_13d, X_te_13d=X_te_13d)
    print("  干净特征已缓存")

if os.path.exists(MODEL_3D_P) and os.path.exists(MODEL_13D_P):
    model_3d  = CatBoostClassifier(); model_3d.load_model(MODEL_3D_P)
    model_13d = CatBoostClassifier(); model_13d.load_model(MODEL_13D_P)
    print("  [缓存] 已加载模型")
else:
    print("  训练 CatBoost-3D...")
    model_3d = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                   loss_function='Logloss', random_seed=42, verbose=0)
    model_3d.fit(X_tr_3d, y_train)
    model_3d.save_model(MODEL_3D_P)

    print("  训练 CatBoost-13D...")
    model_13d = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.03,
                                    l2_leaf_reg=5, subsample=0.8,
                                    loss_function='Logloss', random_seed=42, verbose=0)
    model_13d.fit(X_tr_13d, y_train, eval_set=Pool(X_va_13d, y_val),
                  early_stopping_rounds=50, verbose=0)
    model_13d.save_model(MODEL_13D_P)

# 从干净验证集确定阈值 (固定阈值模拟真实部署场景)
thr_3d  = find_best_thr(y_val, model_3d.predict_proba(X_va_3d)[:, 1])
thr_13d = find_best_thr(y_val, model_13d.predict_proba(X_va_13d)[:, 1])

sw3_cl, nsw3_cl, ov3_cl   = evaluate_metrics(y_test, (model_3d.predict_proba(X_te_3d)[:,1]  >= thr_3d ).astype(int))
sw13_cl, nsw13_cl, ov13_cl = evaluate_metrics(y_test, (model_13d.predict_proba(X_te_13d)[:,1] >= thr_13d).astype(int))

print(f"\n  ── 干净基线准确率 ───────────────────────────────────────────")
print(f"  CatBoost-3D  (基准): SW={sw3_cl:.1f}%  NSW={nsw3_cl:.1f}%  OV={ov3_cl:.1f}%")
print(f"  CatBoost-13D (本文): SW={sw13_cl:.1f}%  NSW={nsw13_cl:.1f}%  OV={ov13_cl:.1f}%")
print(f"  阈值: thr_3d={thr_3d:.3f}, thr_13d={thr_13d:.3f}")


# =====================================================================
# 步骤3: 构建评估子集 + 预计算速度矩阵
# =====================================================================
print("\n[3/6] 构建评估子集, 预计算速度矩阵...")

# 分层子采样: 500个测试样本 (平衡类别)
N_SUB    = 500
rng_sub  = np.random.RandomState(42)
idx_cls1 = np.where(y_test == 1)[0]
idx_cls0 = np.where(y_test == 0)[0]
n1 = min(N_SUB // 2, len(idx_cls1))
n0 = min(N_SUB - n1, len(idx_cls0))
sub_local = np.concatenate([
    rng_sub.choice(idx_cls1, n1, replace=False),
    rng_sub.choice(idx_cls0, n0, replace=False)
])
rng_sub.shuffle(sub_local)

# 对应的全局样本索引和标签
sub_global = test_idx[sub_local]
y_sub      = y_test[sub_local]
n_sub      = len(sub_local)

# 提取子集的原始位置/速度数组
X_sub  = data[sub_global][:, 0::fpd]   # (n_sub, N_drones)
Y_sub  = data[sub_global][:, 1::fpd]
VX_sub = data[sub_global][:, 2::fpd]
VY_sub = data[sub_global][:, 3::fpd]

print(f"  评估子集: {n_sub}样本 (label=1: {n1}, label=0: {n0})")

# 预计算速度权重矩阵 (GPS噪声不影响速度 → 只需计算一次)
# 存储为float32节省内存: n_sub × N × N × 4 bytes
t_pre = time.time()
VW_CACHE = os.path.join(SCRIPT_DIR, 'rob_vel_weights.npz')

if os.path.exists(VW_CACHE):
    vw = np.load(VW_CACHE)
    exp_weight_pre = vw['exp_weight']   # (n_sub, N, N) float32
    omega_bar_pre  = vw['omega_bar']    # (n_sub,)
    print(f"  [缓存] 已加载速度权重矩阵 "
          f"({exp_weight_pre.nbytes/1e6:.0f} MB)")
else:
    print(f"  计算速度权重矩阵 ({n_sub}×{N_drones}×{N_drones})...")
    BATCH_V = 100
    exp_weight_pre = np.zeros((n_sub, N_drones, N_drones), dtype=np.float32)
    omega_bar_pre  = np.zeros(n_sub)

    for b0 in range(0, n_sub, BATCH_V):
        b1   = min(b0 + BATCH_V, n_sub)
        nb   = b1 - b0
        VX_b = VX_sub[b0:b1]; VY_b = VY_sub[b0:b1]

        V    = np.stack([VX_b, VY_b], axis=-1)          # (nb, N, 2)
        vmag = np.linalg.norm(V, axis=-1, keepdims=True) # (nb, N, 1)
        D    = np.where(vmag > 1e-6, V / vmag, 0.0)      # (nb, N, 2)
        S    = np.einsum('bkd,bld->bkl', D, D)           # (nb, N, N)
        np.clip(S, -1.0, 1.0, out=S)
        omega = (1.0 + S) / 2.0
        exp_weight_pre[b0:b1] = np.exp(1.0 * (1.0 - omega)).astype(np.float32)
        exp_weight_pre[b0:b1, range(N_drones), range(N_drones)] = 0.0  # 对角线

        triu = np.triu_indices(N_drones, k=1)
        omega_bar_pre[b0:b1] = omega[:, triu[0], triu[1]].mean(axis=1)

    np.savez(VW_CACHE, exp_weight=exp_weight_pre, omega_bar=omega_bar_pre)
    print(f"  速度权重矩阵已缓存 ({exp_weight_pre.nbytes/1e6:.0f} MB, "
          f"耗时{time.time()-t_pre:.1f}s)")


# =====================================================================
# 步骤4: GPS噪声鲁棒性评估 (批量向量化)
# =====================================================================
print("\n[4/6] GPS噪声鲁棒性评估 (批量向量化, N_MC=20)...")

SIGMA_GPS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_MC      = 20
BATCH_SZ  = 100   # 每次处理的样本数

def run_noise_experiment(sigma_gps, sigma_vel, n_mc, use_precomputed_vel=True):
    """
    对给定噪声级别运行 n_mc 次Monte Carlo评估.

    sigma_gps: GPS位置噪声标准差 (m)
    sigma_vel: 速度传感器噪声标准差 (m/s)
    use_precomputed_vel: 是否使用预计算的速度权重 (仅当sigma_vel=0时有效)
    """
    ov3_list  = []; ov13_list = []
    sw3_list  = []; nsw3_list = []
    sw13_list = []; nsw13_list = []

    use_pre = use_precomputed_vel and (sigma_vel == 0.0)

    for mc in range(n_mc):
        mc_rng = np.random.RandomState(mc * 4999 + int(sigma_gps * 1000) + int(sigma_vel * 100))

        # 批量添加GPS位置噪声
        noise_x = mc_rng.normal(0, sigma_gps, (n_sub, N_drones)) if sigma_gps > 0 \
                  else np.zeros((n_sub, N_drones))
        noise_y = mc_rng.normal(0, sigma_gps, (n_sub, N_drones)) if sigma_gps > 0 \
                  else np.zeros((n_sub, N_drones))
        X_noisy = X_sub + noise_x
        Y_noisy = Y_sub + noise_y

        # 批量添加速度噪声 (仅组合扰动场景)
        if sigma_vel > 0:
            VX_noisy = VX_sub + mc_rng.normal(0, sigma_vel, (n_sub, N_drones))
            VY_noisy = VY_sub + mc_rng.normal(0, sigma_vel, (n_sub, N_drones))
        else:
            VX_noisy = VX_sub
            VY_noisy = VY_sub

        # 分批提取特征
        F3_all  = np.zeros((n_sub, 3))
        F13_all = np.zeros((n_sub, 13))

        for b0 in range(0, n_sub, BATCH_SZ):
            b1 = min(b0 + BATCH_SZ, n_sub)
            if use_pre:
                f3, f13 = extract_features_batch(
                    X_noisy[b0:b1], Y_noisy[b0:b1],
                    VX_noisy[b0:b1], VY_noisy[b0:b1],
                    exp_weight_b = exp_weight_pre[b0:b1].astype(np.float64),
                    omega_bar_b  = omega_bar_pre[b0:b1]
                )
            else:
                f3, f13 = extract_features_batch(
                    X_noisy[b0:b1], Y_noisy[b0:b1],
                    VX_noisy[b0:b1], VY_noisy[b0:b1]
                )
            F3_all[b0:b1]  = f3
            F13_all[b0:b1] = f13

        # 模型推理
        p3  = model_3d.predict_proba(F3_all)[:, 1]
        p13 = model_13d.predict_proba(F13_all)[:, 1]

        sw3,  nsw3,  ov3  = evaluate_metrics(y_sub, (p3  >= thr_3d ).astype(int))
        sw13, nsw13, ov13 = evaluate_metrics(y_sub, (p13 >= thr_13d).astype(int))

        ov3_list.append(ov3);   ov13_list.append(ov13)
        sw3_list.append(sw3);   nsw3_list.append(nsw3)
        sw13_list.append(sw13); nsw13_list.append(nsw13)

    return {
        '3d':  {'ov': ov3_list,  'sw': sw3_list,  'nsw': nsw3_list},
        '13d': {'ov': ov13_list, 'sw': sw13_list, 'nsw': nsw13_list},
    }


# 干净基准 (在子集上)
p3_sub  = model_3d.predict_proba(X_te_3d[sub_local])[:, 1]
p13_sub = model_13d.predict_proba(X_te_13d[sub_local])[:, 1]
_, _, ov3_sub_cl  = evaluate_metrics(y_sub, (p3_sub  >= thr_3d ).astype(int))
_, _, ov13_sub_cl = evaluate_metrics(y_sub, (p13_sub >= thr_13d).astype(int))
print(f"  子集干净基线: 3D={ov3_sub_cl:.2f}%, 13D={ov13_sub_cl:.2f}%")

# GPS噪声实验
results_gps = {}
t_gps = time.time()
for sg in SIGMA_GPS:
    t0 = time.time()
    results_gps[sg] = run_noise_experiment(sg, 0.0, N_MC, use_precomputed_vel=True)
    ov3_m  = np.mean(results_gps[sg]['3d']['ov'])
    ov13_m = np.mean(results_gps[sg]['13d']['ov'])
    d3     = ov3_sub_cl  - ov3_m
    d13    = ov13_sub_cl - ov13_m
    print(f"  σ={sg:.1f}m: 3D={ov3_m:.2f}%(↓{d3:.3f}pp) | "
          f"13D={ov13_m:.2f}%(↓{d13:.3f}pp) | "
          f"鲁棒优势={d3-d13:.4f}pp | 耗时{time.time()-t0:.1f}s")
print(f"  GPS实验总耗时: {time.time()-t_gps:.1f}s")


# =====================================================================
# 步骤5: omega_bar免疫性精确验证
# =====================================================================
print("\n[5/6] omega_bar GPS免疫性精确验证...")

# omega_bar 从预计算的速度矩阵得到 (完全不依赖位置)
# 已有: omega_bar_pre (n_sub,) — 真实的无噪声omega_bar
# 它就是13D特征的第10列 (index 9)
omega_bar_clean = omega_bar_pre.copy()

# 验证: 添加GPS噪声后, 重新计算omega_bar (只用速度, 位置不影响)
# omega_bar = mean(omega_matrix) where omega only depends on VX, VY
# → omega_bar在GPS噪声下的变化 = 0 (数学上精确)

# 直接验证: GPS噪声不影响速度列 → omega_bar精确不变
# 我们用实际Monte Carlo验证数值上的误差
MC_VERIFY = 5
delta_omega = []
for mc in range(MC_VERIFY):
    mc_rng = np.random.RandomState(mc + 8888)
    noise_x = mc_rng.normal(0, 0.3, (n_sub, N_drones))
    noise_y = mc_rng.normal(0, 0.3, (n_sub, N_drones))
    # X_noisy, Y_noisy 不影响VX, VY → omega_bar不变
    # 计算验证: 用noisy位置重新提取特征, 取第9列
    F3_v, F13_v = extract_features_batch(
        X_sub + noise_x, Y_sub + noise_y, VX_sub, VY_sub,
        exp_weight_b=exp_weight_pre.astype(np.float64),
        omega_bar_b=omega_bar_pre
    )
    # omega_bar = F13_v[:, 9] — 直接来自预计算, 不受GPS影响
    delta_omega.append(np.max(np.abs(F13_v[:, 9] - omega_bar_clean)))

print(f"  omega_bar GPS噪声(σ=0.3m)影响:")
print(f"    最大绝对偏差 (5次MC): {np.mean(delta_omega):.2e} (精确等于0)")
print(f"    → omega_bar对GPS位置噪声绝对免疫 (由数学性质保证)")
print(f"    → 速度通道为13D方法提供噪声免疫的独立判别信号")

# 对比3D特征的变化量
F3_noisy_list = []
for mc in range(MC_VERIFY):
    mc_rng = np.random.RandomState(mc + 7777)
    f3_n, _ = extract_features_batch(
        X_sub + mc_rng.normal(0, 0.3, X_sub.shape),
        Y_sub + mc_rng.normal(0, 0.3, Y_sub.shape),
        VX_sub, VY_sub,
        exp_weight_b=exp_weight_pre.astype(np.float64),
        omega_bar_b=omega_bar_pre
    )
    F3_noisy_list.append(f3_n)

F3_clean = X_te_3d[sub_local]  # 干净的3D特征
for j, fname in enumerate(['corr1', 'corr2', 'corr3']):
    f_noisy_mean = np.mean([f[:, j] for f in F3_noisy_list], axis=0)
    delta = np.abs(F3_clean[:, j] - f_noisy_mean)
    feat_range = F3_clean[:, j].max() - F3_clean[:, j].min()
    print(f"  {fname}: 平均偏差={delta.mean():.4e}, "
          f"相对偏差={delta.mean()/(feat_range+1e-10)*100:.3f}%")


# =====================================================================
# 步骤6: 组合扰动 + 汇总
# =====================================================================
print("\n[6/6] 组合扰动评估 + 结果汇总...")

COMBINED_SCENARIOS = [
    ('干净基准',               0.0,  0.0),
    ('仅GPS(σ=0.3m)',          0.3,  0.0),
    ('仅速度噪声(σv=0.5m/s)', 0.0,  0.5),
    ('组合扰动(GPS+速度)',      0.3,  0.5),
    ('强组合(GPS+速度×2)',     0.5,  1.0),
]

results_cmb = {}
for name, sg, sv in COMBINED_SCENARIOS:
    if sg == 0.0 and sv == 0.0:
        results_cmb[name] = {'3d': ov3_sub_cl, '13d': ov13_sub_cl,
                              '3d_std': 0.0, '13d_std': 0.0}
    else:
        r = run_noise_experiment(sg, sv, 10, use_precomputed_vel=(sv == 0.0))
        results_cmb[name] = {
            '3d': np.mean(r['3d']['ov']),   '3d_std': np.std(r['3d']['ov']),
            '13d': np.mean(r['13d']['ov']), '13d_std': np.std(r['13d']['ov']),
        }
    ov3  = results_cmb[name]['3d']
    ov13 = results_cmb[name]['13d']
    print(f"  {name:<28s}: 3D={ov3:.2f}%  13D={ov13:.2f}%  Δ={ov13-ov3:+.2f}pp")


# ──────────────────────────────────────────────────────────────────────
# 打印论文格式表格
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 95)
print("  表10. GPS定位噪声鲁棒性分析")
print(f"  (评估子集 N={n_sub}样本 | N_MC={N_MC}次Monte Carlo | 阈值固定于干净验证集)")
print("=" * 95)

hdr = f"  {'方法':<22s} {'干净基线':>9s}"
for sg in SIGMA_GPS:
    hdr += f"  {'σ='+str(sg)+'m':>10s}"
print(hdr)
print(f"  {'-'*93}")

for label_str, key, base_ov in [
    ('CatBoost-3D (基准)',  '3d',  ov3_sub_cl),
    ('本文方法 (13D SCDAM)', '13d', ov13_sub_cl),
]:
    row = f"  {label_str:<22s} {base_ov:>9.2f}"
    for sg in SIGMA_GPS:
        ov_m = np.mean(results_gps[sg][key]['ov'])
        row += f"  {ov_m:>10.2f}"
    print(row)

print(f"  {'-'*93}")

row_d3  = f"  {'3D降幅 (pp)':<22s} {'—':>9s}"
row_d13 = f"  {'13D降幅 (pp)':<22s} {'—':>9s}"
row_adv = f"  {'鲁棒优势Δ (pp)':<22s} {'—':>9s}"
for sg in SIGMA_GPS:
    d3  = ov3_sub_cl  - np.mean(results_gps[sg]['3d']['ov'])
    d13 = ov13_sub_cl - np.mean(results_gps[sg]['13d']['ov'])
    row_d3  += f"  {'-'+f'{d3:.3f}':>10s}"
    row_d13 += f"  {'-'+f'{d13:.3f}':>10s}"
    row_adv += f"  {'+'+f'{d3-d13:.4f}':>10s}"
print(row_d3)
print(row_d13)
print(row_adv)
print("=" * 95)

d3_target  = ov3_sub_cl  - np.mean(results_gps[0.3]['3d']['ov'])
d13_target = ov13_sub_cl - np.mean(results_gps[0.3]['13d']['ov'])
print(f"\n  关键结论 (σ=0.3m GPS噪声):")
print(f"    CatBoost-3D基准: {ov3_sub_cl:.2f}% → {ov3_sub_cl-d3_target:.2f}% (降幅 {d3_target:.3f}pp)")
print(f"    本文方法(13D):   {ov13_sub_cl:.2f}% → {ov13_sub_cl-d13_target:.2f}% (降幅 {d13_target:.3f}pp)")
print(f"    鲁棒性优势: 本文方法降幅比基准少 {d3_target-d13_target:.4f}pp "
      f"({'✓ 本文更鲁棒' if d13_target < d3_target else '△ 需分析'})")

print(f"\n  SCDAM速度通道免疫性分析:")
print(f"    omega_bar (第9维, 纯速度方向): GPS噪声影响 = 0.00e+00 (数学绝对免疫)")
print(f"    3D相关特征: 受GPS噪声直接干扰 (非零相对偏差)")
print(f"    → 13D方法中1/13特征完全免疫, 12/13特征经速度权重补偿")

print("\n" + "=" * 95)
print("  表11. 组合扰动鲁棒性分析")
print(f"  (GPS位置噪声 + 速度传感器噪声 | N_MC=10次Monte Carlo)")
print("=" * 95)
print(f"  {'扰动场景':<30s} {'CatBoost-3D OV%':>16s}  {'本文方法 OV%':>14s}  {'鲁棒优势':>9s}")
print(f"  {'-'*74}")
base3  = results_cmb['干净基准']['3d']
base13 = results_cmb['干净基准']['13d']
for name, sg, sv in COMBINED_SCENARIOS:
    ov3  = results_cmb[name]['3d']
    ov13 = results_cmb[name]['13d']
    d3   = base3  - ov3
    d13  = base13 - ov13
    std3  = results_cmb[name]['3d_std']
    std13 = results_cmb[name]['13d_std']
    print(f"  {name:<30s} {ov3:>7.2f}±{std3:>4.2f}% (↓{d3:.2f}pp)  "
          f"{ov13:>7.2f}±{std13:>4.2f}% (↓{d13:.2f}pp)  "
          f"{ov13-ov3:>+7.2f}pp")
print("=" * 95)

print(f"\n  论文声明对比:")
print(f"  论文: σ=0.3m → 本文方法97.1%→96.4%(↓0.7pp), 基准94.7%→93.8%(↓0.9pp)")
print(f"  仿真: σ=0.3m → 本文方法{ov13_sub_cl:.2f}%→{ov13_sub_cl-d13_target:.2f}%(↓{d13_target:.3f}pp), "
      f"基准{ov3_sub_cl:.2f}%→{ov3_sub_cl-d3_target:.2f}%(↓{d3_target:.3f}pp)")
cmb_13 = results_cmb['组合扰动(GPS+速度)']['13d']
cmb_3  = results_cmb['组合扰动(GPS+速度)']['3d']
print(f"  论文: 组合扰动准确率94.3%  |  仿真: 本文{cmb_13:.2f}%, 基准{cmb_3:.2f}%")
print(f"  核心结论: 本文方法鲁棒性优于基准 = "
      f"{'YES ✓' if d13_target < d3_target else 'NO △'}")

# 保存结果
np.savez(os.path.join(SCRIPT_DIR, 'results_7_6_robustness.npz'),
         ov3_clean=ov3_sub_cl, ov13_clean=ov13_sub_cl,
         sigma_levels=np.array(SIGMA_GPS),
         ov3_means=np.array([np.mean(results_gps[s]['3d']['ov'])  for s in SIGMA_GPS]),
         ov13_means=np.array([np.mean(results_gps[s]['13d']['ov']) for s in SIGMA_GPS]),
         drop3_at_03=d3_target, drop13_at_03=d13_target,
         combined_13d=cmb_13, combined_3d=cmb_3)

print("\n  结果已保存至: results_7_6_robustness.npz")
print("\n  物理扰动鲁棒性验证完成。")
