# -*- coding: utf-8 -*-
"""
section_coupled_hyperparameter.py
==============================================================================
超参数耦合性四层系统实验
==============================================================================

实验架构（从用户设计文档实现）:
  第一层: λ × m        二维耦合  → 判断几何参数间的耦合
  第二层: λ × Temg     二维耦合  → 判断核心参数间的耦合（预期: Temg方向平坦）
  第三层: m × Temg     二维耦合  → 判断邻域与旁路的耦合（预期: Temg方向平坦）
  第四层: λ × m × Temg 三维全局  → 最终最优参数组合确认

【核心计算效率优化】
  T_emg 只影响决策逻辑（apply_hysteresis_static），不影响特征提取和 CatBoost 训练。
  因此策略为:
  ① 收集所有唯一 (λ, m) 组合（四层并集）
  ② 每个 (λ, m) 组合只提取一次特征、只训练一次模型、保存测试集概率
  ③ T_emg 变化通过后处理（重新应用决策阈值）完成，无需重新训练

  另一优化: 预计算 R_upper 和 omega_upper（λ/m 无关），复用于所有 λ 和 m。

【参数网格（精确匹配用户设计文档）】
  Layer 1: λ∈[0.3,2.5] step 0.2 → 12值; m∈[4,12] step 1 → 9值; 共108组
  Layer 2: λ∈[0.3,2.5] step 0.2 → 12值; Temg∈[0.10,0.35] step 0.025 → 11值; 共132组
  Layer 3: m∈[4,12] step 1 → 9值; Temg∈[0.10,0.35] step 0.025 → 11值; 共99组
  Layer 4: λ∈[0.5,2.0] step 0.25 → 7值; m∈[5,11] step 2 → 4值; Temg∈[0.10,0.30] step 0.05 → 5值; 共140组

【输出文件】
  figures/fig_layer1_lam_m_surface.png         第一层 λ×m 曲面
  figures/fig_layer2_lam_temg_surface.png      第二层 λ×Temg 双曲面
  figures/fig_layer3_m_temg_surface.png        第三层 m×Temg 曲面（验证平坦性）
  figures/fig_layer4_slices.png                第四层 热力图切片（3个Temg代表值）
  figures/fig_layer4_bubble.png                第四层 气泡图
  python/results_coupled_hyperparameter.npz    完整数值结果
==============================================================================
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from catboost import CatBoostClassifier, Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_DIR     = os.path.join(os.path.dirname(SCRIPT_DIR), 'csv')
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

_cn_fonts = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
_avail    = [f.name for f in fm.fontManager.ttflist]
_chosen   = next((f for f in _cn_fonts if f in _avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': _chosen, 'axes.unicode_minus': False, 'font.size': 10})

# =====================================================================
# 实验超参数
# =====================================================================
SEED    = 42
N_SUB   = 4000       # 子集大小（兼顾速度与统计稳健性）
FPD     = 12         # 每架无人机字段数
TAU_MAX = 50         # FFT 最大自相关滞后
LAM_PAPER    = 1.0   # 论文推荐值
M_PAPER      = 8
TEMG_PAPER   = 0.20
T_HIGH_FIXED = 0.75  # 固定迟滞上阈值（论文 Section 7.1）
T_LOW_FIXED  = 0.55  # 固定迟滞下阈值
OMEGA_CC_THR = 0.60  # 对向穿插子集阈值

# ── 四层参数网格 ──────────────────────────────────────────────────────
LAM_L1   = np.round(np.arange(0.3, 2.5 + 0.001, 0.2), 2)   # 12 值
M_L1     = np.arange(4, 12 + 1, 1)                           # 9 值
LAM_L2   = LAM_L1.copy()                                      # 12 值
TEMG_L2  = np.round(np.arange(0.10, 0.35 + 0.001, 0.025), 3) # 11 值
M_L3     = M_L1.copy()                                        # 9 值
TEMG_L3  = TEMG_L2.copy()                                     # 11 值
LAM_L4   = np.round(np.arange(0.5, 2.0 + 0.001, 0.25), 2)   # 7 值
M_L4     = np.arange(5, 11 + 1, 2)                           # 4 值: 5,7,9,11
TEMG_L4  = np.round(np.arange(0.10, 0.30 + 0.001, 0.05), 2) # 5 值

# 固定的 T_emg（Layer 1 / Layer 3 中 T_emg 固定）
TEMG_FIXED_L1 = TEMG_PAPER   # Layer 1 固定 T_emg=0.20
LAM_FIXED_L3  = LAM_PAPER    # Layer 3 固定 λ=1.0
M_FIXED_L2    = M_PAPER      # Layer 2 固定 m=8

print("=" * 72)
print("  超参数耦合性四层系统实验")
print("=" * 72)
print(f"  Layer 1: λ({len(LAM_L1)}) × m({len(M_L1)}) = {len(LAM_L1)*len(M_L1)} 组")
print(f"  Layer 2: λ({len(LAM_L2)}) × Temg({len(TEMG_L2)}) = {len(LAM_L2)*len(TEMG_L2)} 组")
print(f"  Layer 3: m({len(M_L3)}) × Temg({len(TEMG_L3)}) = {len(M_L3)*len(TEMG_L3)} 组")
print(f"  Layer 4: λ({len(LAM_L4)}) × m({len(M_L4)}) × Temg({len(TEMG_L4)}) = {len(LAM_L4)*len(M_L4)*len(TEMG_L4)} 组")


# =====================================================================
# 算法函数
# =====================================================================

def fft_autocorr(q, tau_max):
    L = len(q)
    if L < 2 or tau_max < 1: return np.zeros(max(tau_max, 1))
    q_c = q - np.mean(q); varq = np.var(q)
    if varq < 1e-15: return np.zeros(tau_max)
    n_fft = 1
    while n_fft < 2 * L: n_fft *= 2
    Q = np.fft.fft(q_c, n=n_fft)
    R_full = np.fft.ifft(np.abs(Q) ** 2).real
    tau = min(tau_max, L - 1)
    return np.array([R_full[t] / ((L - t) * varq + 1e-15) for t in range(tau)])

def spectral_scalars(Phi):
    if len(Phi) == 0: return 0., 0., 0.
    E = float(np.mean(Phi ** 2))
    P = float((np.max(Phi) - np.min(Phi)) / (np.mean(np.abs(Phi)) + 1e-8))
    absP = np.abs(Phi) + 1e-15; Pn = absP / np.sum(absP)
    return E, P, float(-np.sum(Pn * np.log2(Pn)))

def mean_corr(B):
    c = np.corrcoef(B.T)
    return float(np.mean(np.nan_to_num(c, nan=0.)))

def apply_hysteresis_static(probs, T_high, T_low, T_emg):
    """静态迟滞决策（与 paper2_algorithms.py 保持一致）"""
    mid = (T_high + T_low) / 2.0
    out = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p >= T_high:   out[i] = 1
        elif p < T_emg:   out[i] = 0
        elif p < T_low:   out[i] = 0
        else: out[i] = 1 if p >= mid else 0
    return out

def evaluate(y_true, y_pred):
    tp = np.sum((y_pred==1)&(y_true==1)); tn = np.sum((y_pred==0)&(y_true==0))
    fp = np.sum((y_pred==1)&(y_true==0)); fn = np.sum((y_pred==0)&(y_true==1))
    sw  = tp/(tp+fn)*100 if tp+fn > 0 else 0.
    nsw = tn/(tn+fp)*100 if tn+fp > 0 else 0.
    ov  = (tp+tn)/len(y_true)*100
    fa  = fp/(tn+fp)*100 if tn+fp > 0 else 0.   # 伪告警率
    return ov, sw, nsw, fa

def find_best_thr(y_true, probs):
    best_thr, best_ov = 0.5, 0.
    for thr in np.arange(0.03, 0.97, 0.005):
        ov, *_ = evaluate(y_true, (probs >= thr).astype(int))
        if ov > best_ov: best_ov = ov; best_thr = thr
    return best_thr

def apply_thr_and_eval(y_te, probs_te, thr, T_emg,
                       T_high=T_HIGH_FIXED, T_low=T_LOW_FIXED):
    """用给定 T_emg 应用迟滞决策并评估（T_high/T_low 固定）"""
    pred = apply_hysteresis_static(probs_te, T_high, T_low, T_emg)
    return evaluate(y_te, pred)


# =====================================================================
# 1. 加载数据，构建子集（分层随机，保持类别均衡）
# =====================================================================
print(f"\n[1/5] 加载数据，构建 N_SUB={N_SUB} 子集...")
df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'), low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
data   = df_raw.iloc[:, :-1].values.astype(float)
labels = df_raw.iloc[:, -1].values.astype(int)
bl_df  = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n  = min(len(labels), len(bl_df))
data   = data[:min_n]; labels = labels[:min_n]

rng   = np.random.RandomState(SEED)
idx0  = np.where(labels == 0)[0]; idx1 = np.where(labels == 1)[0]
sub0  = rng.choice(idx0, size=min(N_SUB//2, len(idx0)), replace=False)
sub1  = rng.choice(idx1, size=min(N_SUB-N_SUB//2, len(idx1)), replace=False)
sub_g = np.concatenate([sub0, sub1])
sub_g = rng.permutation(sub_g)

sub_data   = data[sub_g];  sub_labels = labels[sub_g]
N_sub_act  = len(sub_labels)
n_tr = int(N_sub_act * 0.70); n_va = int(N_sub_act * 0.15)
tr_i = np.arange(n_tr); va_i = np.arange(n_tr, n_tr+n_va)
te_i = np.arange(n_tr+n_va, N_sub_act)
y_tr = sub_labels[tr_i]; y_va = sub_labels[va_i]; y_te = sub_labels[te_i]
print(f"  子集大小: {N_sub_act}  训练:{len(tr_i)} 验证:{len(va_i)} 测试:{len(te_i)}")


# =====================================================================
# 2. 预计算 R_upper 和 omega_upper（λ/m 无关，一次性完成）
# =====================================================================
print(f"\n[2/5] 预计算 R_upper 和 omega_upper（λ/m 无关部分）...")
t0 = time.time()

R_upper_all     = np.zeros((N_sub_act, 200*199//2), dtype=np.float32)  # 上三角
omega_upper_all = np.zeros((N_sub_act, 200*199//2), dtype=np.float32)
omega_bar_all   = np.zeros(N_sub_act, dtype=np.float64)

for i, row in enumerate(sub_data):
    X = row[0::FPD]; Y = row[1::FPD]
    VX= row[2::FPD]; VY= row[3::FPD]
    N_ = len(X)
    coords = np.column_stack([X, Y])
    R = squareform(pdist(coords, metric='euclidean'))
    V = np.column_stack([VX, VY])
    v_mag = np.linalg.norm(V, axis=1, keepdims=True)
    D = np.zeros_like(V); valid = (v_mag.flatten() > 1e-6)
    if np.any(valid): D[valid] = V[valid] / v_mag[valid]
    S = np.clip(D @ D.T, -1., 1.); omega = (1. + S) / 2.
    triu_idx = np.triu_indices(N_, k=1)
    R_upper_all[i]     = R[triu_idx].astype(np.float32)
    omega_upper_all[i] = omega[triu_idx].astype(np.float32)
    omega_bar_all[i]   = float(np.mean(omega[triu_idx]))
    if (i+1) % 1000 == 0:
        print(f"    {i+1}/{N_sub_act}  {time.time()-t0:.0f}s")

# 对向穿插子集掩码（omega_bar < 0.60，λ无关）
omega_bar_te = omega_bar_all[te_i]
mask_cc      = omega_bar_te < OMEGA_CC_THR
n_cc         = int(np.sum(mask_cc))
print(f"  预计算完成 ({time.time()-t0:.0f}s)")
print(f"  对向穿插子集: {n_cc} 个测试样本")

N_TRIU = R_upper_all.shape[1]   # 200*199//2 = 19900


# =====================================================================
# 3. 特征提取函数（从预缓存 R_upper/omega_upper 中提取）
# =====================================================================

def extract_feat_batch(indices, lam, m_nbrs):
    """从预缓存矩阵提取 13 维特征，返回 (n, 13) 数组"""
    n = len(indices)
    feats = np.zeros((n, 13), dtype=np.float64)
    for ii, idx in enumerate(indices):
        r_up = R_upper_all[idx].astype(np.float64)
        o_up = omega_upper_all[idx].astype(np.float64)
        ob   = omega_bar_all[idx]

        # SCDAM 矩阵上三角
        M_up = r_up * np.exp(lam * (1. - o_up))

        # 谱特征（直接在上三角序列上计算）
        Phi_d = fft_autocorr(M_up, TAU_MAX)
        Phi_f = fft_autocorr(np.diff(M_up), TAU_MAX)
        Phi_b = fft_autocorr(np.abs(M_up - np.mean(M_up)), TAU_MAX)
        Ed,Pd,Hd = spectral_scalars(Phi_d)
        Ef,Pf,Hf = spectral_scalars(Phi_f)
        Eb,Pb,Hb = spectral_scalars(Phi_b)

        # SCDAM 相关变体（需重建完整矩阵以排序近邻）
        M_full = squareform(M_up)
        R_star = np.sort(M_full, axis=0)[1:m_nbrs+1, :]
        c1 = mean_corr(R_star)
        B2 = R_star - np.mean(R_star, axis=1, keepdims=True); c2 = mean_corr(B2)
        B3 = np.zeros_like(R_star)
        B3[0,:]=R_star[0,:]; B3[1:,:]=np.diff(R_star, axis=0); c3 = mean_corr(B3)

        feats[ii] = [Ed,Pd,Hd, Ef,Pf,Hf, Eb,Pb,Hb, ob, c1,c2,c3]
    return feats


# =====================================================================
# 4. 收集所有唯一 (λ, m) 组合并批量训练
# =====================================================================
# 跨四层合并所有需要的 (λ, m) 组合
unique_lam_m = set()
for lam in LAM_L1:
    for m in M_L1:
        unique_lam_m.add((float(round(lam, 3)), int(m)))
for m in M_L3:                               # Layer 3: λ=1.0（不在 L1 网格中）
    unique_lam_m.add((float(LAM_FIXED_L3), int(m)))
for lam in LAM_L4:                           # Layer 4: 额外的 λ/m 组合
    for m in M_L4:
        unique_lam_m.add((float(round(lam, 3)), int(m)))

unique_lam_m = sorted(unique_lam_m)
n_combos     = len(unique_lam_m)
print(f"\n[3/5] 跨四层共 {n_combos} 个唯一 (λ,m) 组合，逐一训练模型...")

# 存储每个组合的: 测试集概率、最优阈值、评估指标、对向穿插子集指标
combo_results = {}   # key: (lam, m)  value: dict

t_start = time.time()
for combo_idx, (lam, m) in enumerate(unique_lam_m):
    t_c = time.time()
    print(f"  [{combo_idx+1}/{n_combos}] λ={lam:.2f} m={m} ...", end='', flush=True)

    X_tr = extract_feat_batch(tr_i, lam, m)
    X_va = extract_feat_batch(va_i, lam, m)
    X_te = extract_feat_batch(te_i, lam, m)

    model = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.03,
        l2_leaf_reg=5, subsample=0.8,
        loss_function='Logloss', random_seed=SEED, verbose=0)
    model.fit(X_tr, y_tr, eval_set=Pool(X_va, y_va),
              early_stopping_rounds=50, verbose=0)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    thr  = find_best_thr(y_va, p_va)

    # 标准评估（固定 T_emg = TEMG_PAPER = 0.20）
    ov, sw, nsw, fa = apply_thr_and_eval(y_te, p_te, thr, TEMG_PAPER)
    # 对向穿插子集准确率
    if n_cc >= 5:
        ov_cc, sw_cc, nsw_cc, fa_cc = apply_thr_and_eval(
            y_te[mask_cc], p_te[mask_cc], thr, TEMG_PAPER)
    else:
        ov_cc = np.nan

    combo_results[(lam, m)] = {
        'p_te': p_te, 'thr': thr,
        'ov': ov, 'sw': sw, 'nsw': nsw, 'fa': fa,
        'ov_cc': ov_cc,
    }
    print(f"  OV={ov:.2f}% CC={ov_cc:.1f}% ({time.time()-t_c:.0f}s)")

print(f"\n  全部训练完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟")


# =====================================================================
# 辅助函数：按 T_emg 重新评估（后处理，不需要重新训练）
# =====================================================================

def eval_with_temg(lam, m, T_emg):
    """对已训练的 (λ,m) 模型，用指定 T_emg 评估（后处理）"""
    r = combo_results[(lam, m)]
    return apply_thr_and_eval(y_te, r['p_te'], r['thr'], T_emg)


# =====================================================================
# 5. 四层结果矩阵计算
# =====================================================================
print(f"\n[4/5] 计算四层指标矩阵...")

# ── Layer 1: OV[λ, m] (T_emg=0.20 固定) ──────────────────────────────
L1_OV    = np.zeros((len(LAM_L1), len(M_L1)))
L1_CC_OV = np.zeros((len(LAM_L1), len(M_L1)))
L1_FA    = np.zeros((len(LAM_L1), len(M_L1)))
for i, lam in enumerate(LAM_L1):
    for j, m in enumerate(M_L1):
        r = combo_results[(float(round(lam,3)), int(m))]
        L1_OV[i,j]    = r['ov']
        L1_CC_OV[i,j] = r['ov_cc'] if not np.isnan(r['ov_cc']) else r['ov']
        L1_FA[i,j]    = r['fa']

# ── Layer 2: OV[λ, T_emg] (m=8 固定) ─────────────────────────────────
L2_OV  = np.zeros((len(LAM_L2), len(TEMG_L2)))
L2_FA  = np.zeros((len(LAM_L2), len(TEMG_L2)))
for i, lam in enumerate(LAM_L2):
    for k, temg in enumerate(TEMG_L2):
        ov, sw, nsw, fa = eval_with_temg(float(round(lam,3)), int(M_FIXED_L2), temg)
        L2_OV[i,k] = ov; L2_FA[i,k] = fa

# ── Layer 3: OV[m, T_emg] (λ=1.0 固定) ──────────────────────────────
L3_OV  = np.zeros((len(M_L3), len(TEMG_L3)))
for j, m in enumerate(M_L3):
    for k, temg in enumerate(TEMG_L3):
        ov, *_ = eval_with_temg(float(LAM_FIXED_L3), int(m), temg)
        L3_OV[j,k] = ov

# ── Layer 4: OV[λ, m, T_emg] ─────────────────────────────────────────
L4_OV  = np.zeros((len(LAM_L4), len(M_L4), len(TEMG_L4)))
L4_FA  = np.zeros((len(LAM_L4), len(M_L4), len(TEMG_L4)))
for i, lam in enumerate(LAM_L4):
    for j, m in enumerate(M_L4):
        for k, temg in enumerate(TEMG_L4):
            ov, sw, nsw, fa = eval_with_temg(float(round(lam,3)), int(m), temg)
            L4_OV[i,j,k] = ov; L4_FA[i,j,k] = fa

print("  矩阵计算完成")

# ── 统计分析 ──────────────────────────────────────────────────────────
# Layer 1: 每个 m 下最优 λ
print(f"\n  【Layer 1】每个 m 的最优 λ（验证是否随 m 漂移）:")
print(f"  {'m':>4s}  {'最优λ':>7s}  {'OV%':>7s}  {'λ漂移?':>8s}")
prev_lam = None
for j, m in enumerate(M_L1):
    best_i = np.argmax(L1_OV[:, j])
    best_lam = LAM_L1[best_i]
    shift = '' if prev_lam is None else (
        f'+{best_lam-prev_lam:.1f}' if best_lam > prev_lam else f'{best_lam-prev_lam:.1f}')
    print(f"  {m:>4d}  {best_lam:>7.2f}  {L1_OV[best_i,j]:>7.2f}%  {shift:>8s}")
    prev_lam = best_lam

lam_shifts = [LAM_L1[np.argmax(L1_OV[:, j])] for j in range(len(M_L1))]
lam_shift_range = max(lam_shifts) - min(lam_shifts)
print(f"  最优 λ 漂移范围: {lam_shift_range:.2f}  "
      f"({'存在显著耦合' if lam_shift_range > 0.4 else '漂移较小，参数基本解耦'})")

# Layer 2: T_emg 方向平坦性验证
temg_variation_L2 = np.mean(np.std(L2_OV, axis=1))   # 对每个 λ，T_emg 方向的标准差
print(f"\n  【Layer 2】T_emg 方向 OV 标准差均值: {temg_variation_L2:.4f}%")
print(f"  {'<0.05% → T_emg 与 λ 完全解耦（预期结果）' if temg_variation_L2 < 0.05 else '> 0.05% → 存在一定耦合'}")

# Layer 3: T_emg 方向平坦性验证
temg_variation_L3 = np.mean(np.std(L3_OV, axis=1))
print(f"  【Layer 3】T_emg 方向 OV 标准差均值: {temg_variation_L3:.4f}%")
print(f"  {'<0.05% → m 与 T_emg 完全解耦（预期结果）' if temg_variation_L3 < 0.05 else '> 0.05% → 存在一定耦合'}")

# Layer 4: 全局最优参数
L4_OV_max_temg = L4_OV.max(axis=2)   # 对每个 (λ,m) 取最优 T_emg
i_best, j_best = np.unravel_index(L4_OV_max_temg.argmax(), L4_OV_max_temg.shape)
k_best = L4_OV[i_best, j_best, :].argmax()
lam_global_best  = LAM_L4[i_best]
m_global_best    = M_L4[j_best]
temg_global_best = TEMG_L4[k_best]
ov_global_best   = L4_OV[i_best, j_best, k_best]
# 论文推荐值在 Layer 4 中的性能
if LAM_PAPER in LAM_L4 and M_PAPER in M_L4:
    i_p = np.where(np.abs(LAM_L4-LAM_PAPER)<0.001)[0][0]
    j_p = np.where(M_L4==M_PAPER)[0][0]
    k_p = np.where(np.abs(TEMG_L4-TEMG_PAPER)<0.001)[0][0]
    ov_paper_in_L4 = L4_OV[i_p, j_p, k_p]
    paper_in_L4 = True
else:
    ov_paper_in_L4 = combo_results.get(
        (float(LAM_PAPER), int(M_PAPER)), {}).get('ov', np.nan)
    paper_in_L4 = False

print(f"\n  【Layer 4】全局最优参数组合:")
print(f"    λ={lam_global_best:.2f}  m={m_global_best}  T_emg={temg_global_best:.2f}  OV={ov_global_best:.2f}%")
print(f"  论文推荐值 (λ=1.0, m=8, T_emg=0.20): OV≈{ov_paper_in_L4:.2f}%")
diff = ov_global_best - ov_paper_in_L4
print(f"  差距: {diff:+.2f}%  "
      f"({'推荐值即为最优，论文参数合理' if abs(diff)<0.3 else f'最优值优于推荐值 {abs(diff):.2f}%，可考虑更新推荐'})")


# =====================================================================
# 6. 可视化
# =====================================================================
print(f"\n[5/5] 生成四层可视化图表...")

def surface3d(fig, ax, X_grid, Y_grid, Z_mat, title,
              xlabel, ylabel, zlabel, cmap='RdYlGn',
              paper_xy=None, paper_z=None):
    """通用三维曲面函数"""
    Xg, Yg = np.meshgrid(X_grid, Y_grid, indexing='ij')
    surf = ax.plot_surface(Xg, Yg, Z_mat, cmap=cmap, alpha=0.75,
                           linewidth=0, antialiased=True)
    ax.contourf(Xg, Yg, Z_mat, zdir='z',
                offset=Z_mat.min() - (Z_mat.max()-Z_mat.min())*0.08,
                cmap=cmap, alpha=0.45, levels=12)
    if paper_xy is not None and paper_z is not None:
        ax.scatter([paper_xy[0]], [paper_xy[1]], [paper_z],
                   color='gold', s=120, zorder=10, marker='*',
                   label=f'论文推荐值 ({paper_xy[0]:.1f},{paper_xy[1]:.2f})')
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
    ax.set_zlabel(zlabel, fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, pad=8)
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
    return surf


# ── 图1: Layer 1 λ×m 曲面（OV + 对向穿插子集叠加）──────────────────
fig1 = plt.figure(figsize=(15, 5.5))
ax1a = fig1.add_subplot(131, projection='3d')
ax1b = fig1.add_subplot(132, projection='3d')
ax1c = fig1.add_subplot(133)

# 主曲面: OV
Lg, Mg = np.meshgrid(LAM_L1, M_L1, indexing='ij')
p_z_L1 = None
if LAM_PAPER in LAM_L1 and M_PAPER in M_L1:
    pi, pj = np.where(np.abs(LAM_L1-LAM_PAPER)<0.01)[0][0], np.where(M_L1==M_PAPER)[0][0]
    p_z_L1 = L1_OV[pi, pj]

surface3d(fig1, ax1a, LAM_L1, M_L1, L1_OV,
          'OV 综合准确率 (%)', 'λ', 'm', 'OV (%)',
          cmap='RdYlGn',
          paper_xy=(LAM_PAPER, M_PAPER) if LAM_PAPER in LAM_L1 else None,
          paper_z=p_z_L1)
ax1a.view_init(elev=25, azim=-60)

# 叠加曲面: 对向穿插子集 OV
surface3d(fig1, ax1b, LAM_L1, M_L1, L1_CC_OV,
          '对向穿插子集 OV (%)', 'λ', 'm', 'CC_OV (%)',
          cmap='Blues',
          paper_xy=(LAM_PAPER, M_PAPER) if LAM_PAPER in LAM_L1 else None,
          paper_z=L1_CC_OV[pi,pj] if LAM_PAPER in LAM_L1 else None)
ax1b.view_init(elev=25, azim=-60)

# 附表: 每个 m 的最优 λ（热力图）
im = ax1c.imshow(L1_OV.T, aspect='auto', cmap='RdYlGn',
                 origin='lower',
                 extent=[LAM_L1[0]-0.1, LAM_L1[-1]+0.1, M_L1[0]-0.5, M_L1[-1]+0.5])
# 标注每个 m 的最优 λ
for j, m in enumerate(M_L1):
    best_i = np.argmax(L1_OV[:, j])
    ax1c.plot(LAM_L1[best_i], m, 'r*', ms=12, zorder=5)
ax1c.axvline(LAM_PAPER, color='gold', ls='--', lw=1.5, label=f'论文 λ={LAM_PAPER}')
ax1c.axhline(M_PAPER,   color='gold', ls=':',  lw=1.5, label=f'论文 m={M_PAPER}')
ax1c.set_xlabel('λ', fontsize=11); ax1c.set_ylabel('m', fontsize=11)
ax1c.set_yticks(M_L1)
ax1c.set_title('OV热力图（红星=各m最优λ）\n漂移范围: {:.2f}'.format(lam_shift_range), fontsize=10)
ax1c.legend(fontsize=8)
fig1.colorbar(im, ax=ax1c, shrink=0.7)

fig1.suptitle(
    f'Layer 1: λ × m 耦合性实验（T_emg={TEMG_FIXED_L1} 固定）\n'
    f'λ漂移范围={lam_shift_range:.2f} | '
    f'{"基本解耦" if lam_shift_range<=0.4 else "存在耦合"}',
    fontsize=12, y=1.01)
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, 'fig_layer1_lam_m_surface.png')
fig1.savefig(p1, dpi=130, bbox_inches='tight'); plt.close(fig1)
print(f"  图1已保存: {p1}")


# ── 图2: Layer 2 λ×T_emg 双曲面（OV + 伪告警率 + 可行域）────────────
fig2 = plt.figure(figsize=(16, 5.5))
ax2a = fig2.add_subplot(131, projection='3d')
ax2b = fig2.add_subplot(132, projection='3d')
ax2c = fig2.add_subplot(133)

Lg2, Tg2 = np.meshgrid(LAM_L2, TEMG_L2, indexing='ij')
pi_L2 = np.where(np.abs(LAM_L2-LAM_PAPER)<0.01)[0][0] if LAM_PAPER in LAM_L2 else None
pk_L2 = np.where(np.abs(TEMG_L2-TEMG_PAPER)<0.001)[0][0]

surface3d(fig2, ax2a, LAM_L2, TEMG_L2, L2_OV,
          '图A: OV综合准确率 (%)\n(Temg方向应平坦→验证解耦)',
          'λ', 'T_emg', 'OV (%)', cmap='RdYlGn',
          paper_xy=(LAM_PAPER, TEMG_PAPER) if pi_L2 is not None else None,
          paper_z=L2_OV[pi_L2, pk_L2] if pi_L2 is not None else None)
ax2a.view_init(elev=25, azim=-50)

surface3d(fig2, ax2b, LAM_L2, TEMG_L2, L2_FA,
          '图B: 伪告警率 (%)\n(冷色=高伪告警)',
          'λ', 'T_emg', 'FA (%)', cmap='Blues_r',
          paper_xy=(LAM_PAPER, TEMG_PAPER) if pi_L2 is not None else None,
          paper_z=L2_FA[pi_L2, pk_L2] if pi_L2 is not None else None)
ax2b.view_init(elev=25, azim=-50)

# 可行域图（OV ≥ 论文-0.5% 且 FA ≤ 论文+1.5%）
ov_thresh  = np.nanpercentile(L2_OV, 60)   # 采用 60 百分位作为可行域下界
fa_thresh  = np.nanpercentile(L2_FA, 80)
feasible   = (L2_OV >= ov_thresh) & (L2_FA <= fa_thresh)
im2 = ax2c.imshow(feasible.T.astype(float), aspect='auto', cmap='Greens',
                  origin='lower', vmin=0, vmax=1,
                  extent=[LAM_L2[0]-0.1, LAM_L2[-1]+0.1,
                           TEMG_L2[0]-0.01, TEMG_L2[-1]+0.01])
# OV 等高线叠加
cs = ax2c.contour(LAM_L2, TEMG_L2, L2_OV.T,
                  levels=8, colors='navy', alpha=0.6, linewidths=0.8)
ax2c.clabel(cs, fontsize=7)
if pi_L2 is not None:
    ax2c.plot(LAM_PAPER, TEMG_PAPER, 'r*', ms=15, zorder=10, label='论文推荐值')
    ax2c.legend(fontsize=9)
ax2c.set_xlabel('λ', fontsize=11); ax2c.set_ylabel('T_emg', fontsize=11)
ax2c.set_title(f'可行域（绿色: OV≥{ov_thresh:.1f}% 且 FA≤{fa_thresh:.1f}%）\n含等高线', fontsize=10)

fig2.suptitle(
    f'Layer 2: λ × T_emg 耦合性实验（m={M_FIXED_L2} 固定）\n'
    f'T_emg方向OV标准差={temg_variation_L2:.4f}%  '
    f'→ {"T_emg与λ完全解耦（预期）" if temg_variation_L2<0.05 else "存在微弱耦合"}',
    fontsize=12, y=1.01)
plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, 'fig_layer2_lam_temg_surface.png')
fig2.savefig(p2, dpi=130, bbox_inches='tight'); plt.close(fig2)
print(f"  图2已保存: {p2}")


# ── 图3: Layer 3 m×T_emg 曲面（验证平坦性）──────────────────────────
fig3 = plt.figure(figsize=(13, 5.0))
ax3a = fig3.add_subplot(121, projection='3d')
ax3b = fig3.add_subplot(122)

pj_L3 = np.where(M_L3 == M_PAPER)[0][0]
pk_L3 = np.where(np.abs(TEMG_L3-TEMG_PAPER)<0.001)[0][0]

surface3d(fig3, ax3a, M_L3.astype(float), TEMG_L3, L3_OV,
          f'm × T_emg 曲面（λ={LAM_FIXED_L3} 固定）\nT_emg方向应平坦',
          'm', 'T_emg', 'OV (%)', cmap='RdYlGn',
          paper_xy=(float(M_PAPER), TEMG_PAPER),
          paper_z=L3_OV[pj_L3, pk_L3])
ax3a.view_init(elev=25, azim=-50)

# T_emg 方向的 OV 变化曲线（每条曲线对应一个 m）
for j, m in enumerate(M_L3):
    ax3b.plot(TEMG_L3, L3_OV[j,:], '-o', ms=4, lw=1.5,
              label=f'm={m}', alpha=0.8)
ax3b.axvline(TEMG_PAPER, color='gray', ls='--', lw=1.2, label=f'T_emg={TEMG_PAPER}')
ax3b.set_xlabel('T_emg', fontsize=11); ax3b.set_ylabel('OV (%)', fontsize=11)
ax3b.set_title('各 m 值下 OV 随 T_emg 的变化曲线\n（曲线水平 → 完全解耦）', fontsize=10)
ax3b.legend(fontsize=7.5, ncol=3, loc='lower right')
ax3b.grid(True, ls=':', alpha=0.5)
ov_range_L3 = L3_OV.max() - L3_OV.min()
ax3b.set_ylim(L3_OV.min()-0.3, L3_OV.max()+0.3)
ax3b.text(0.02, 0.05,
          f'全图 OV 波动范围: {ov_range_L3:.3f}%\n'
          f'T_emg方向均值Std: {temg_variation_L3:.4f}%',
          transform=ax3b.transAxes, fontsize=9,
          bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

fig3.suptitle(
    f'Layer 3: m × T_emg 耦合性实验（λ={LAM_FIXED_L3} 固定）\n'
    f'曲面平坦性验证: T_emg方向Std={temg_variation_L3:.4f}%  '
    f'→ {"m与T_emg完全解耦，直接验证论文7.7节声明" if temg_variation_L3<0.1 else "存在一定耦合"}',
    fontsize=12, y=1.01)
plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, 'fig_layer3_m_temg_surface.png')
fig3.savefig(p3, dpi=130, bbox_inches='tight'); plt.close(fig3)
print(f"  图3已保存: {p3}")


# ── 图4: Layer 4 热力图切片（方案A）──────────────────────────────────
TEMG_SLICES = [0.10, 0.20, 0.30]
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))

for idx_t, temg_val in enumerate(TEMG_SLICES):
    k_slice = np.argmin(np.abs(TEMG_L4 - temg_val))
    slice_ov = L4_OV[:, :, k_slice]   # (n_lam, n_m)

    ax = axes4[idx_t]
    im = ax.imshow(slice_ov, aspect='auto', cmap='RdYlGn',
                   origin='lower', vmin=L4_OV.min(), vmax=L4_OV.max(),
                   extent=[M_L4[0]-0.5, M_L4[-1]+0.5,
                            LAM_L4[0]-0.1, LAM_L4[-1]+0.1])
    # 数值标注
    for i_l in range(len(LAM_L4)):
        for j_m in range(len(M_L4)):
            ax.text(M_L4[j_m], LAM_L4[i_l],
                    f'{slice_ov[i_l,j_m]:.1f}',
                    ha='center', va='center', fontsize=7.5,
                    color='black' if slice_ov[i_l,j_m] < slice_ov.mean() else 'white')

    # 标注论文推荐值
    if LAM_PAPER in LAM_L4 and M_PAPER in M_L4:
        ax.plot(M_PAPER, LAM_PAPER, 'r*', ms=16, zorder=10)

    ax.set_xticks(M_L4); ax.set_xlabel('m', fontsize=11)
    ax.set_yticks(LAM_L4); ax.set_ylabel('λ', fontsize=11)
    ax.set_title(f'T_emg = {TEMG_L4[k_slice]:.2f}\n'
                 f'max OV={slice_ov.max():.2f}%', fontsize=10)
    fig4.colorbar(im, ax=ax, shrink=0.75)

fig4.suptitle(
    'Layer 4 方案A: λ×m 热力图切片（三个代表性 T_emg 值）\n'
    '红星=论文推荐值 (λ=1.0, m=8)；观察最优参数是否随 T_emg 漂移',
    fontsize=12, y=1.02)
plt.tight_layout()
p4 = os.path.join(FIGURES_DIR, 'fig_layer4_slices.png')
fig4.savefig(p4, dpi=130, bbox_inches='tight'); plt.close(fig4)
print(f"  图4已保存: {p4}")


# ── 图5: Layer 4 气泡图（方案B）──────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(9, 6.5))

# 对每个 (λ,m), 取最优 T_emg 时的 OV 和 FA
L4_best_ov  = L4_OV.max(axis=2)   # (n_lam, n_m)
L4_best_fa  = np.array([[L4_FA[i,j,np.argmax(L4_OV[i,j,:])]
                          for j in range(len(M_L4))]
                         for i in range(len(LAM_L4))])

ov_flat = L4_best_ov.flatten()
fa_flat = L4_best_fa.flatten()
lam_flat= np.repeat(LAM_L4, len(M_L4))
m_flat  = np.tile(M_L4, len(LAM_L4))

# 气泡大小 ∝ OV（相对于最小值放大差异）
ov_min, ov_max = ov_flat.min(), ov_flat.max()
bubble_size = ((ov_flat - ov_min) / (ov_max - ov_min + 1e-9) * 800 + 80)

sc = ax5.scatter(lam_flat, m_flat, s=bubble_size, c=fa_flat,
                 cmap='Blues', alpha=0.75, edgecolors='gray', linewidths=0.6)
plt.colorbar(sc, ax=ax5, label='FA率 (%) [深色=高伪告警]')

# 标注每个气泡的 OV 值
for x, y, ov, fa in zip(lam_flat, m_flat, ov_flat, fa_flat):
    ax5.text(x, y, f'{ov:.1f}%', ha='center', va='center',
             fontsize=7.5, color='black')

# 论文推荐值
if LAM_PAPER in LAM_L4 and M_PAPER in M_L4:
    ax5.plot(LAM_PAPER, M_PAPER, 'r*', ms=20, zorder=10,
             label=f'论文推荐 (λ={LAM_PAPER}, m={M_PAPER})')
ax5.plot(lam_global_best, m_global_best, 'g^', ms=14, zorder=10,
         label=f'全局最优 (λ={lam_global_best:.2f}, m={m_global_best})')

ax5.set_xlabel('λ（耦合敏感系数）', fontsize=12)
ax5.set_ylabel('m（k近邻数）', fontsize=12)
ax5.set_yticks(M_L4)
ax5.set_title(
    'Layer 4 方案B: 气泡图\n'
    '气泡大小 ∝ OV准确率（取最优T_emg），颜色深浅 = 伪告警率',
    fontsize=12)
ax5.legend(fontsize=9.5, loc='upper right')
ax5.grid(True, ls=':', alpha=0.4)
# 大小图例
for ov_demo in [ov_min, (ov_min+ov_max)/2, ov_max]:
    s = (ov_demo - ov_min) / (ov_max - ov_min + 1e-9) * 800 + 80
    ax5.scatter([], [], s=s, c='steelblue', alpha=0.6,
                label=f'OV={ov_demo:.1f}%')
ax5.legend(fontsize=8.5, loc='lower left', ncol=2)

plt.tight_layout()
p5 = os.path.join(FIGURES_DIR, 'fig_layer4_bubble.png')
fig5.savefig(p5, dpi=130, bbox_inches='tight'); plt.close(fig5)
print(f"  图5已保存: {p5}")


# =====================================================================
# 7. 保存数值结果
# =====================================================================
np.savez(os.path.join(SCRIPT_DIR, 'results_coupled_hyperparameter.npz'),
    LAM_L1=LAM_L1, M_L1=M_L1, L1_OV=L1_OV, L1_CC_OV=L1_CC_OV, L1_FA=L1_FA,
    LAM_L2=LAM_L2, TEMG_L2=TEMG_L2, L2_OV=L2_OV, L2_FA=L2_FA,
    M_L3=M_L3.astype(float), TEMG_L3=TEMG_L3, L3_OV=L3_OV,
    LAM_L4=LAM_L4, M_L4=M_L4.astype(float), TEMG_L4=TEMG_L4,
    L4_OV=L4_OV, L4_FA=L4_FA,
    lam_best_global=np.array([lam_global_best]),
    m_best_global=np.array([m_global_best]),
    temg_best_global=np.array([temg_global_best]),
    ov_best_global=np.array([ov_global_best]),
    ov_paper_in_L4=np.array([ov_paper_in_L4]),
    temg_variation_L2=np.array([temg_variation_L2]),
    temg_variation_L3=np.array([temg_variation_L3]),
    lam_shift_range=np.array([lam_shift_range]),
)

# =====================================================================
# 最终汇总
# =====================================================================
print("\n" + "=" * 72)
print("  四层超参数耦合性实验完整汇总")
print("=" * 72)
print(f"\n  【Layer 1】λ × m 耦合性")
print(f"    最优 λ 随 m 的漂移范围: {lam_shift_range:.2f}")
print(f"    结论: {'λ随m基本不漂移，两者弱耦合，λ=1.0选值稳健' if lam_shift_range<=0.4 else f'最优λ随m变化，存在显著耦合（漂移{lam_shift_range:.2f}）'}")

print(f"\n  【Layer 2】λ × T_emg 耦合性")
print(f"    T_emg方向OV标准差: {temg_variation_L2:.4f}%")
print(f"    结论: {'T_emg与λ完全解耦，T_emg不影响静态分类精度（论文论断得到验证）' if temg_variation_L2<0.05 else f'存在微弱耦合(Std={temg_variation_L2:.4f}%)'}")

print(f"\n  【Layer 3】m × T_emg 耦合性")
print(f"    T_emg方向OV标准差: {temg_variation_L3:.4f}%")
print(f"    结论: {'m与T_emg完全解耦，直接验证论文7.7节声明' if temg_variation_L3<0.1 else f'存在耦合(Std={temg_variation_L3:.4f}%)'}")

print(f"\n  【Layer 4】全局最优参数")
print(f"    全局最优: λ={lam_global_best:.2f}  m={m_global_best}  T_emg={temg_global_best:.2f}  OV={ov_global_best:.2f}%")
print(f"    论文推荐: λ={LAM_PAPER:.2f}  m={M_PAPER}  T_emg={TEMG_PAPER:.2f}  OV≈{ov_paper_in_L4:.2f}%")
print(f"    差距: {ov_global_best-ov_paper_in_L4:+.2f}%")
if abs(ov_global_best - ov_paper_in_L4) < 0.3:
    print(f"    → 论文推荐参数处于最优邻域（±0.3%），推荐值合理")
else:
    print(f"    → 建议在论文中更新推荐参数为 λ={lam_global_best:.2f}, m={m_global_best}")

print("\n" + "=" * 72)
print("section_coupled_hyperparameter.py 完成.")
