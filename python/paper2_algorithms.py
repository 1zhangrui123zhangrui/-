"""
paper2_algorithms.py
==============================================================================
第二篇论文核心算法库
==============================================================================
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


# =====================================================================
# 1. 基线方法 (论文1)
# =====================================================================

def compute_distance_matrix(X, Y):
    coords = np.column_stack([X, Y])
    return squareform(pdist(coords, metric='euclidean'))


def extract_sorted_neighbors(dist_matrix, m):
    """按列排序，取m个最近邻（排除自身）"""
    sorted_dist = np.sort(dist_matrix, axis=0)
    return sorted_dist[1:m+1, :]   # m x N


def build_rpm(R_star, method_type):
    """构建相对位置矩阵B"""
    if method_type == 1:
        return R_star.copy()
    elif method_type == 2:
        # 修正距离: b_jk = r*_jk - mean_k(r*_jk)
        return R_star - np.mean(R_star, axis=1, keepdims=True)
    elif method_type == 3:
        # 差分: 首行保留，后续行取相邻差
        B = np.zeros_like(R_star)
        B[0, :] = R_star[0, :]
        B[1:, :] = np.diff(R_star, axis=0)
        return B


def compute_autocorrelation(B):
    """计算Pearson相关矩阵均值"""
    corr_mat = np.corrcoef(B.T)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    return float(np.mean(corr_mat))


def compute_three_variants(R, m):
    """从距离矩阵计算三种变体的自相关值"""
    R_star = extract_sorted_neighbors(R, m)
    return tuple(compute_autocorrelation(build_rpm(R_star, i)) for i in [1, 2, 3])


# =====================================================================
# 2. SCDAM (论文2 Section 4)
# =====================================================================

def compute_velocity_directions(VX, VY):
    """计算单位速度方向向量"""
    V = np.column_stack([VX, VY])
    v_mag = np.linalg.norm(V, axis=1, keepdims=True)
    D = np.zeros_like(V)
    valid = (v_mag.flatten() > 1e-6)
    if np.any(valid):
        D[valid] = V[valid] / v_mag[valid]
    return D


def compute_omega_matrix(D):
    """omega_kl = (1 + d_k*d_l) / 2, 取值[0,1]"""
    S = np.clip(D @ D.T, -1.0, 1.0)
    return (1.0 + S) / 2.0


def compute_global_omega(omega):
    """omega_bar: 全局速度对齐均值（上三角）"""
    N = omega.shape[0]
    return float(np.mean(omega[np.triu_indices(N, k=1)]))


def build_scdam(R, omega, lam=1.0):
    """SCDAM矩阵: M_kl = R_kl * exp(lam * (1 - omega_kl))"""
    M = R * np.exp(lam * (1.0 - omega))
    np.fill_diagonal(M, 0.0)
    return M


# =====================================================================
# 3. FFT谱特征 (论文2 Section 5)
# =====================================================================

def fft_autocorrelation(q, tau_max):
    """用FFT快速计算归一化自相关序列"""
    L = len(q)
    if L < 2 or tau_max < 1:
        return np.zeros(max(tau_max, 1))
    q_c = q - np.mean(q)
    var_q = np.var(q)
    if var_q < 1e-15:
        return np.zeros(tau_max)
    n_fft = 1
    while n_fft < 2 * L:
        n_fft *= 2
    Q = np.fft.fft(q_c, n=n_fft)
    R_full = np.fft.ifft(np.abs(Q) ** 2).real
    actual_tau = min(tau_max, L - 1)
    Phi = np.array([R_full[t] / ((L - t) * var_q + 1e-15) for t in range(actual_tau)])
    return Phi


def extract_spectral_scalars(Phi):
    """从自相关序列提取三维谱标量: 能量E, 峰值度P, 熵H"""
    if len(Phi) == 0:
        return 0.0, 0.0, 0.0
    E = float(np.mean(Phi ** 2))
    P = float((np.max(Phi) - np.min(Phi)) / (np.mean(np.abs(Phi)) + 1e-8))
    abs_P = np.abs(Phi) + 1e-15
    Pn = abs_P / np.sum(abs_P)
    H = float(-np.sum(Pn * np.log2(Pn)))
    return E, P, H


# =====================================================================
# 4. 完整特征提取 (单个样本)
# =====================================================================

def extract_features_single(X, Y, VX, VY, m=8, lam=1.0):
    """
    提取单个样本的完整特征集:

    返回字典包含:
      baseline_3d  : [corr1, corr2, corr3] 原始距离矩阵上的三变体
      omega_bar    : 全局速度对齐均值 (SCDAM核心输出)
      scdam_3d     : [corr1_M, corr2_M, corr3_M] SCDAM矩阵上的三变体
      spectral_9d  : [Ed,Pd,Hd, Ef,Pf,Hf, Eb,Pb,Hb] 9维谱特征
      feat_13d     : 完整13维: spectral_9d + omega_bar + scdam_3d
    """
    # -- 距离矩阵 --
    R = compute_distance_matrix(X, Y)

    # -- 基线三变体 --
    bl_corr = compute_three_variants(R, m)

    # -- 速度信息 --
    D = compute_velocity_directions(VX, VY)
    omega = compute_omega_matrix(D)
    omega_bar = compute_global_omega(omega)

    # -- SCDAM矩阵 --
    M = build_scdam(R, omega, lam)
    sc_corr = compute_three_variants(M, m)

    # -- FFT谱特征 (在M的上三角元素序列上) --
    N = M.shape[0]
    q = M[np.triu_indices(N, k=1)]
    tau_max = 50

    Phi_dist = fft_autocorrelation(q, tau_max)
    Phi_diff = fft_autocorrelation(np.diff(q), tau_max)
    Phi_bias = fft_autocorrelation(np.abs(q - np.mean(q)), tau_max)

    Ed, Pd, Hd = extract_spectral_scalars(Phi_dist)
    Ef, Pf, Hf = extract_spectral_scalars(Phi_diff)
    Eb, Pb, Hb = extract_spectral_scalars(Phi_bias)

    return {
        'baseline_3d': np.array(bl_corr),
        'omega_bar':   omega_bar,
        'scdam_3d':    np.array(sc_corr),
        'spectral_9d': np.array([Ed, Pd, Hd, Ef, Pf, Hf, Eb, Pb, Hb]),
        'feat_13d':    np.array([Ed, Pd, Hd, Ef, Pf, Hf, Eb, Pb, Hb,
                                 omega_bar, sc_corr[0], sc_corr[1], sc_corr[2]]),
    }


# =====================================================================
# 5. 迟滞决策 (论文2 Section 6)
# =====================================================================

def apply_hysteresis_static(probs, T_high=0.65, T_low=0.35, T_emg=0.08):
    """
    静态迟滞决策 (独立样本)

    参数:
      T_emg  : 紧急非蜂群阈值 (极低概率强制判非蜂群)
      T_low  : 低置信边界
      T_high : 高置信边界 (≥T_high强制判蜂群)
      死区 [T_low, T_high] 内用中点判断
    """
    mid = (T_high + T_low) / 2.0
    decisions = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p >= T_high:
            decisions[i] = 1
        elif p < T_emg:
            decisions[i] = 0
        elif p < T_low:
            decisions[i] = 0
        else:
            # 死区: 用中点作最终判断
            decisions[i] = 1 if p >= mid else 0
    return decisions
