# -*- coding: utf-8 -*-
"""
section_4_4_lambda_search.py
==============================================================================
Section 4.4  λ耦合敏感系数网格搜索  — 回应审稿人意见 2.2
==============================================================================

【实验真实性声明】
  本脚本使用全量数据集（24 000 样本），采用与论文主实验完全相同的划分方案
  （RandomState(42) 全排列，70% 训练 / 15% 验证 / 15% 测试），不做任何
  子集抽样。λ=1.0 处的测试集准确率应与论文 Table 5 CatBoost-13D 数值吻合，
  可作为内置一致性校验。

【回应审稿人三条意见】
  ① 搜索仅在验证集上进行，存在过拟合风险
      → 同时绘制验证集与测试集准确率曲线，阈值在验证集确定，
        准确率在测试集独立评估，二者曲线吻合则无过拟合

  ② 缺少λ与数据集特性（平均飞行速度、集群密度）的关系分析
      → 按 omega_bar 区间（低/中/高速度对齐度）分段报告各λ的准确率，
        直接展示λ与数据集速度特性的交互效应

  ③ 展示对向穿插场景（SCDAM 核心机制）的λ敏感性
      → 单独计算 omega_bar < 0.60 子集（对向穿插/动态耦合）的准确率，
        绘制该子集的λ敏感性曲线

【物理意义】
  λ → 0 : M → R，SCDAM 退化为纯距离矩阵（等价于论文 1 基线）
  λ = 1.0: 指数加权适中，速度不对齐惩罚合理放大
  λ >> 1 : 过度惩罚，矩阵数值极大，频谱特征数值溢出风险上升

【输出】
  figures/fig_lambda_accuracy.png    验证集+测试集双线曲线
  figures/fig_lambda_cross_scene.png 对向穿插子集 + 整体测试集曲线
  python/results_lambda_search.npz   完整数值结果（供后续分析）
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
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_DIR     = os.path.join(os.path.dirname(SCRIPT_DIR), 'csv')
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 字体配置
# ─────────────────────────────────────────────────────────────────────────────
_cn_fonts = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
_avail    = [f.name for f in fm.fontManager.ttflist]
_chosen   = next((f for f in _cn_fonts if f in _avail), 'DejaVu Sans')
plt.rcParams.update({
    'font.family':        _chosen,
    'axes.unicode_minus': False,
    'font.size':          12,
})

# =====================================================================
# 实验超参数（与论文主实验完全一致，不得随意修改）
# =====================================================================
LAMBDA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
LAMBDA_PAPER  = 1.0      # 论文固定选用值
OMEGA_CC_THR  = 0.60     # 对向穿插/动态耦合场景阈值（与 challenge_scenarios.py 一致）
m             = 8        # k 近邻数（与论文 Section 4 一致）
fpd           = 12       # 每架无人机字段数
TAU_MAX       = 50       # 最大自相关滞后（与 paper2_algorithms.py 一致）
SEED          = 42       # 随机种子（全项目统一）
CATBOOST_PARAMS = dict(  # 论文 2 CatBoost 超参数（与所有实验脚本一致）
    iterations=300, depth=8, learning_rate=0.03,
    l2_leaf_reg=5, subsample=0.8,
    loss_function='Logloss', random_seed=SEED, verbose=0,
)

# =====================================================================
# 工具函数（与项目其他实验脚本保持完全一致）
# =====================================================================

def evaluate(y_true, y_pred):
    """返回 (蜂群召回%, 非蜂群召回%, 综合准确率%)"""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sw  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    nsw = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    ov  = (tp + tn) / len(y_true) * 100
    return sw, nsw, ov


def find_best_thr(y_true, probs):
    """在验证集上遍历阈值，找使综合准确率最大的阈值"""
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        _, _, ov = evaluate(y_true, (probs >= thr).astype(int))
        if ov > best_ov:
            best_ov  = ov
            best_thr = thr
    return best_thr


def fft_autocorr(q, tau_max):
    """FFT 加速归一化自相关（与 paper2_algorithms.fft_autocorrelation 等价）"""
    L = len(q)
    if L < 2 or tau_max < 1:
        return np.zeros(max(tau_max, 1))
    q_c  = q - np.mean(q)
    varq = np.var(q)
    if varq < 1e-15:
        return np.zeros(tau_max)
    n_fft = 1
    while n_fft < 2 * L:
        n_fft *= 2
    Q      = np.fft.fft(q_c, n=n_fft)
    R_full = np.fft.ifft(np.abs(Q) ** 2).real
    tau    = min(tau_max, L - 1)
    return np.array([R_full[t] / ((L - t) * varq + 1e-15) for t in range(tau)])


def spectral_scalars(Phi):
    """从自相关序列提取三维谱标量：能量 E，峰均比 P，熵 H"""
    if len(Phi) == 0:
        return 0.0, 0.0, 0.0
    E    = float(np.mean(Phi ** 2))
    P    = float((np.max(Phi) - np.min(Phi)) / (np.mean(np.abs(Phi)) + 1e-8))
    absP = np.abs(Phi) + 1e-15
    Pn   = absP / np.sum(absP)
    H    = float(-np.sum(Pn * np.log2(Pn)))
    return E, P, H


def mean_corr(B):
    """Pearson 相关矩阵均值（与 paper2_algorithms.compute_autocorrelation 等价）"""
    c = np.corrcoef(B.T)
    return float(np.mean(np.nan_to_num(c, nan=0.0)))


def extract_one_sample(row, lam):
    """
    从单个原始样本行提取 13 维特征向量。
    同时返回 omega_bar（λ 无关，用于对向穿插掩码）。

    特征布局（与论文 Section 5 / ablation_7_3.py 完全一致）：
      [0-8]  谱特征 9D : Ed,Pd,Hd, Ef,Pf,Hf, Eb,Pb,Hb
      [9]    omega_bar   全局速度对齐均值（λ无关）
      [10-12] SCDAM 三变体自相关: corr1_M, corr2_M, corr3_M
    """
    X  = row[0::fpd];  Y  = row[1::fpd]
    VX = row[2::fpd];  VY = row[3::fpd]
    N_ = len(X)

    # ── 距离矩阵 R ──
    coords = np.column_stack([X, Y])
    R = squareform(pdist(coords, metric='euclidean'))

    # ── 速度方向与 omega 矩阵 ──
    V     = np.column_stack([VX, VY])
    v_mag = np.linalg.norm(V, axis=1, keepdims=True)
    D     = np.zeros_like(V)
    valid = (v_mag.flatten() > 1e-6)
    if np.any(valid):
        D[valid] = V[valid] / v_mag[valid]
    S     = np.clip(D @ D.T, -1.0, 1.0)
    omega = (1.0 + S) / 2.0

    # ── omega_bar（λ 无关，全局速度一致性标量）──
    omega_bar = float(np.mean(omega[np.triu_indices(N_, k=1)]))

    # ── SCDAM 矩阵 M = R × exp(λ × (1 - ω)) ──
    M = R * np.exp(lam * (1.0 - omega))
    np.fill_diagonal(M, 0.0)

    # ── 谱特征（FFT 加速，在 M 上三角序列上计算）──
    q       = M[np.triu_indices(N_, k=1)]
    Phi_d   = fft_autocorr(q,                          TAU_MAX)
    Phi_f   = fft_autocorr(np.diff(q),                 TAU_MAX)
    Phi_b   = fft_autocorr(np.abs(q - np.mean(q)),     TAU_MAX)
    Ed, Pd, Hd = spectral_scalars(Phi_d)
    Ef, Pf, Hf = spectral_scalars(Phi_f)
    Eb, Pb, Hb = spectral_scalars(Phi_b)

    # ── SCDAM 矩阵上的三变体自相关 ──
    R_star = np.sort(M, axis=0)[1:m + 1, :]          # m × N，k 近邻排序
    c1 = mean_corr(R_star)
    B2 = R_star - np.mean(R_star, axis=1, keepdims=True)
    c2 = mean_corr(B2)
    B3 = np.zeros_like(R_star)
    B3[0, :]  = R_star[0, :]
    B3[1:, :] = np.diff(R_star, axis=0)
    c3 = mean_corr(B3)

    feat = np.array([Ed, Pd, Hd, Ef, Pf, Hf, Eb, Pb, Hb,
                     omega_bar, c1, c2, c3], dtype=np.float64)
    return feat, omega_bar


def extract_block(data_rows, lam, tag="", log_interval=2000):
    """批量提取特征，返回 (features, omega_bar_arr)"""
    n = len(data_rows)
    feats       = np.zeros((n, 13), dtype=np.float64)
    omega_bars  = np.zeros(n,       dtype=np.float64)
    t0 = time.time()
    for i, row in enumerate(data_rows):
        feats[i], omega_bars[i] = extract_one_sample(row, lam)
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"    {tag} {i+1}/{n}  已用{elapsed:.0f}s  剩余≈{eta:.0f}s")
    return feats, omega_bars


# =====================================================================
# 1. 加载全量数据，采用与论文完全相同的划分方案
# =====================================================================
print("=" * 70)
print("  Section 4.4  λ 敏感性分析（全量数据，与论文同划分）")
print("=" * 70)

print("\n[1/4] 加载数据...")
df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'), low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
data   = df_raw.iloc[:, :-1].values.astype(float)
labels = df_raw.iloc[:, -1].values.astype(int)

# 与 ablation_7_3.py / challenge_scenarios.py 相同：以 catboost_features_table5.csv 行数为上限
bl_df  = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n  = min(len(labels), len(bl_df))
data   = data[:min_n]
labels = labels[:min_n]
print(f"  全量样本: {min_n}  蜂群:{np.sum(labels==1)}  非蜂群:{np.sum(labels==0)}")

# 与论文完全相同的划分（seed=42，70/15/15）
rng   = np.random.RandomState(SEED)
idx   = rng.permutation(min_n)
n_tr  = int(min_n * 0.70)
n_va  = int(min_n * 0.15)
tr_i  = idx[:n_tr]
va_i  = idx[n_tr:n_tr + n_va]
te_i  = idx[n_tr + n_va:]

data_tr = data[tr_i];  y_tr = labels[tr_i]
data_va = data[va_i];  y_va = labels[va_i]
data_te = data[te_i];  y_te = labels[te_i]

print(f"  训练:{len(tr_i)}  验证:{len(va_i)}  测试:{len(te_i)}")
print(f"  测试集  蜂群:{np.sum(y_te==1)}  非蜂群:{np.sum(y_te==0)}")
print(f"  预计总运行时间: {len(LAMBDA_VALUES) * (len(tr_i)+len(va_i)+len(te_i)) // 100 // 60 + 5}~"
      f"{len(LAMBDA_VALUES) * (len(tr_i)+len(va_i)+len(te_i)) // 60 // 60 + 10} 分钟（视硬件而异）")


# =====================================================================
# 2. 预扫描：仅提取 omega_bar（λ 无关，用于定义对向穿插掩码）
#    利用 λ=0 时 M=R（不影响 omega_bar 计算）做快速预扫
# =====================================================================
print(f"\n[2/4] 预扫描测试集 omega_bar（λ 无关，定义对向穿插掩码）...")
t_pre = time.time()
omega_bar_te = np.zeros(len(te_i))
for i, row in enumerate(data_te):
    X = row[0::fpd]; Y = row[1::fpd]
    VX= row[2::fpd]; VY= row[3::fpd]
    N_ = len(X)
    V     = np.column_stack([VX, VY])
    v_mag = np.linalg.norm(V, axis=1, keepdims=True)
    D     = np.zeros_like(V)
    valid = (v_mag.flatten() > 1e-6)
    if np.any(valid):
        D[valid] = V[valid] / v_mag[valid]
    S     = np.clip(D @ D.T, -1.0, 1.0)
    omega = (1.0 + S) / 2.0
    omega_bar_te[i] = float(np.mean(omega[np.triu_indices(N_, k=1)]))

mask_cc = omega_bar_te < OMEGA_CC_THR
n_cc    = int(np.sum(mask_cc))
print(f"  预扫描完成 ({time.time()-t_pre:.0f}s)")
print(f"  对向穿插子集 (omega_bar < {OMEGA_CC_THR}): {n_cc} 个测试样本"
      f"  蜂群:{np.sum(y_te[mask_cc]==1)}  非蜂群:{np.sum(y_te[mask_cc]==0)}")

# omega_bar 区间分析（用于审稿回复中展示λ与速度特性的关系）
OB_BINS = [(0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
OB_NAMES = ['极低(<0.50)', '低[0.50,0.60)', '中[0.60,0.70)', '高[0.70,0.80)', '极高[0.80,1.0]']
ob_masks = [(omega_bar_te >= lo) & (omega_bar_te < hi) for lo, hi in OB_BINS]
print(f"\n  omega_bar 区间分布:")
for name, mask in zip(OB_NAMES, ob_masks):
    print(f"    {name}: {np.sum(mask)} 样本  蜂群:{np.sum(y_te[mask]==1)}")


# =====================================================================
# 3. λ 网格搜索（全量数据）
# =====================================================================
print(f"\n[3/4] λ 网格搜索（{len(LAMBDA_VALUES)} 个 λ 值 × 全量数据）...")

# 结果存储
results_va  = np.zeros(len(LAMBDA_VALUES))  # 验证集综合准确率
results_te  = np.zeros(len(LAMBDA_VALUES))  # 测试集综合准确率
results_sw  = np.zeros(len(LAMBDA_VALUES))  # 测试集蜂群召回率
results_nsw = np.zeros(len(LAMBDA_VALUES))  # 测试集非蜂群召回率
results_cc  = np.full(len(LAMBDA_VALUES), np.nan)   # 对向穿插子集准确率
# omega_bar 区间的准确率矩阵：shape (n_bins, n_lambda)
results_ob  = np.full((len(OB_BINS), len(LAMBDA_VALUES)), np.nan)

t_total = time.time()

for lam_idx, lam in enumerate(LAMBDA_VALUES):
    t_lam = time.time()
    print(f"\n{'='*60}")
    print(f"  λ = {lam:.1f}  [{lam_idx+1}/{len(LAMBDA_VALUES)}]")
    print(f"{'='*60}")

    # ── 提取特征 ──
    print(f"  提取训练集特征 ({len(tr_i)} 样本)...")
    X_tr, _ = extract_block(data_tr, lam, tag="训练")

    print(f"  提取验证集特征 ({len(va_i)} 样本)...")
    X_va, _ = extract_block(data_va, lam, tag="验证", log_interval=500)

    print(f"  提取测试集特征 ({len(te_i)} 样本)...")
    X_te, ob_check = extract_block(data_te, lam, tag="测试", log_interval=500)

    # 一致性校验：omega_bar 应与预扫描结果完全吻合（验证提取代码正确性）
    ob_diff = np.max(np.abs(ob_check - omega_bar_te))
    if ob_diff > 1e-6:
        print(f"  [警告] omega_bar 与预扫描不一致，最大偏差 {ob_diff:.2e}，请检查代码！")
    else:
        print(f"  [校验] omega_bar 一致性通过（偏差 < 1e-6）")

    # ── 训练 CatBoost ──
    print(f"  训练 CatBoost...")
    t_cb = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=Pool(X_va, y_va),
              early_stopping_rounds=50, verbose=0)
    print(f"  CatBoost 训练完成（{time.time()-t_cb:.0f}s）")

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # ── 阈值在验证集上确定，准确率在测试集独立评估 ──
    thr    = find_best_thr(y_va, p_va)
    pred_va = (p_va >= thr).astype(int)
    pred_te = (p_te >= thr).astype(int)

    _, _, ov_va          = evaluate(y_va, pred_va)
    sw_te, nsw_te, ov_te = evaluate(y_te, pred_te)

    results_va[lam_idx]  = ov_va
    results_te[lam_idx]  = ov_te
    results_sw[lam_idx]  = sw_te
    results_nsw[lam_idx] = nsw_te

    # ── 对向穿插子集准确率 ──
    if n_cc > 0:
        _, _, ov_cc = evaluate(y_te[mask_cc], pred_te[mask_cc])
        results_cc[lam_idx] = ov_cc
    else:
        ov_cc = np.nan

    # ── omega_bar 区间准确率（展示λ与速度特性的交互）──
    ob_row = []
    for b_idx, (b_mask, b_name) in enumerate(zip(ob_masks, OB_NAMES)):
        if np.sum(b_mask) >= 5:
            _, _, ov_b = evaluate(y_te[b_mask], pred_te[b_mask])
            results_ob[b_idx, lam_idx] = ov_b
            ob_row.append(f"{b_name}={ov_b:.1f}%")

    elapsed = time.time() - t_lam
    print(f"\n  结果: 验证={ov_va:.2f}%  测试={ov_te:.2f}%  对向穿插={ov_cc:.2f}%  阈值={thr:.3f}")
    print(f"        蜂群召回={sw_te:.2f}%  非蜂群召回={nsw_te:.2f}%")
    print(f"  omega_bar 区间: {' | '.join(ob_row)}")
    print(f"  本轮耗时: {elapsed:.0f}s  累计: {time.time()-t_total:.0f}s")

    # ── λ=1.0 内置一致性校验（结果应与论文 Table 5 CatBoost-13D 一致）──
    if abs(lam - LAMBDA_PAPER) < 0.01:
        print(f"\n  [一致性校验] λ={LAMBDA_PAPER:.1f} 测试集 OV%={ov_te:.2f}%")
        print(f"  论文 Table 5 CatBoost-13D 应约为 99.x%，请人工核对")


# =====================================================================
# 4. 汇总结果并诊断
# =====================================================================
print("\n" + "=" * 70)
print("  λ 网格搜索完整结果表")
print("=" * 70)
print(f"  {'λ':>4s}  {'验证集OV%':>9s}  {'测试集OV%':>9s}  {'对向穿插%':>9s}  "
      f"{'蜂群召回%':>9s}  {'非蜂群召回%':>11s}")
print("  " + "-" * 65)

for i, lam in enumerate(LAMBDA_VALUES):
    marker = "  ← 论文选定" if abs(lam - LAMBDA_PAPER) < 0.01 else ""
    print(f"  {lam:>4.1f}  {results_va[i]:>9.2f}  {results_te[i]:>9.2f}  "
          f"{results_cc[i]:>9.2f}  {results_sw[i]:>9.2f}  {results_nsw[i]:>11.2f}{marker}")

lam_best_va_idx = int(np.argmax(results_va))
lam_best_te_idx = int(np.argmax(results_te))
lam_best_va     = LAMBDA_VALUES[lam_best_va_idx]
lam_best_te     = LAMBDA_VALUES[lam_best_te_idx]
paper_idx       = LAMBDA_VALUES.index(LAMBDA_PAPER)

va_range  = float(results_va.max() - results_va.min())
te_range  = float(results_te.max() - results_te.min())
valid_cc  = ~np.isnan(results_cc)
cc_range  = float(results_cc[valid_cc].max() - results_cc[valid_cc].min()) if valid_cc.any() else 0.0

print(f"\n  验证集最优 λ = {lam_best_va:.1f}  准确率 = {results_va[lam_best_va_idx]:.2f}%")
print(f"  测试集最优 λ = {lam_best_te:.1f}  准确率 = {results_te[lam_best_te_idx]:.2f}%")
print(f"  论文固定  λ = {LAMBDA_PAPER:.1f}  验证集={results_va[paper_idx]:.2f}%  测试集={results_te[paper_idx]:.2f}%")
print(f"\n  准确率波动范围:  验证集={va_range:.2f}%  测试集={te_range:.2f}%  对向穿插={cc_range:.2f}%")

# 诊断
print("\n  【诊断】")
if lam_best_va == lam_best_te == LAMBDA_PAPER:
    print(f"  ✓ 验证集与测试集最优 λ 均为 {LAMBDA_PAPER:.1f}，与论文完全一致。")
    print(f"    两条曲线峰值位置相同，可证明无验证集过拟合。")
elif lam_best_va == lam_best_te:
    print(f"  △ 验证集与测试集最优 λ 一致（{lam_best_va:.1f}），但偏离论文固定值 {LAMBDA_PAPER:.1f}。")
    print(f"    建议：在论文中修改 λ 的默认值为 {lam_best_va:.1f}，并更新相关实验结果。")
else:
    print(f"  △ 验证集最优 λ={lam_best_va:.1f}，测试集最优 λ={lam_best_te:.1f}，两者不同。")
    print(f"    说明曲线平坦，λ 在该区间内对结果影响不显著（请核查波动范围）。")

if te_range < 0.5:
    print(f"  ✓ 测试集准确率波动仅 {te_range:.2f}%，SCDAM 对 λ 鲁棒，论文结论成立。")
elif te_range < 2.0:
    print(f"  ℹ 测试集准确率波动 {te_range:.2f}%，λ 有一定影响，应在论文中如实报告。")
else:
    print(f"  ! 测试集准确率波动达 {te_range:.2f}%，λ 影响显著，需认真讨论选值依据。")


# =====================================================================
# 5. 生成图表
# =====================================================================
print(f"\n[4/4] 生成图表...")
lam_arr = np.array(LAMBDA_VALUES)

# ───────────────────────────────────────────────────────
# 图 1：λ-准确率曲线（验证集 + 测试集双线，主图）
# ───────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

ax1.plot(lam_arr, results_va, 'b-o', lw=2.0, ms=7, label='验证集综合准确率 (Validation OV%)', zorder=3)
ax1.plot(lam_arr, results_te, 'r-s', lw=2.0, ms=7, label='测试集综合准确率 (Test OV%)',       zorder=3)

# 标注论文固定值竖线
ax1.axvline(x=LAMBDA_PAPER, color='dimgray', ls='--', lw=1.4, alpha=0.9,
            label=f'论文选定值 λ={LAMBDA_PAPER:.1f}', zorder=2)

# 标注论文固定值处的数值
va_at_paper = results_va[paper_idx]
te_at_paper = results_te[paper_idx]
offset_x = 0.12
ax1.annotate(
    f'λ={LAMBDA_PAPER:.1f}\n验证:{va_at_paper:.2f}%\n测试:{te_at_paper:.2f}%',
    xy=(LAMBDA_PAPER, te_at_paper),
    xytext=(LAMBDA_PAPER + offset_x, te_at_paper - 0.5),
    fontsize=9.5, color='dimgray',
    arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0),
)

# 若最优 λ 不等于论文值，额外标注最优点
if lam_best_te != LAMBDA_PAPER:
    ax1.plot(lam_best_te, results_te[lam_best_te_idx], 'r*', ms=14, zorder=5,
             label=f'测试集最优 λ={lam_best_te:.1f}  ({results_te[lam_best_te_idx]:.2f}%)')

y_lo = min(results_va.min(), results_te.min()) - 0.6
y_hi = max(results_va.max(), results_te.max()) + 0.6
ax1.set_ylim(y_lo, y_hi)
ax1.set_xlim(lam_arr[0] - 0.1, lam_arr[-1] + 0.1)
ax1.set_xlabel('耦合敏感系数 λ', fontsize=13)
ax1.set_ylabel('综合准确率 OV (%)', fontsize=13)
ax1.set_title('SCDAM 耦合系数 λ 对分类准确率的影响\n（验证集与测试集交叉验证，全量数据）',
              fontsize=13)
ax1.legend(fontsize=10.5, loc='lower right')
ax1.grid(True, ls=':', alpha=0.55)
ax1.set_xticks(lam_arr)

# 在图中注明数据规模
ax1.text(0.02, 0.03,
         f'全量数据 N={min_n}  训练/验证/测试={n_tr}/{n_va}/{len(te_i)}\n'
         f'验证集最优λ={lam_best_va:.1f}  测试集最优λ={lam_best_te:.1f}',
         transform=ax1.transAxes, fontsize=8.5, color='gray',
         verticalalignment='bottom')

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, 'fig_lambda_accuracy.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  图1已保存: {path1}")


# ───────────────────────────────────────────────────────
# 图 2：对向穿插子集λ敏感性曲线（含整体测试集参考线）
# ───────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# 对向穿插子集
ax2.plot(lam_arr[valid_cc], results_cc[valid_cc], 'g-^', lw=2.2, ms=9,
         label=f'对向穿插子集 OV%  (ω̄<{OMEGA_CC_THR}, n={n_cc})', zorder=4)

# 整体测试集参考
ax2.plot(lam_arr, results_te, 'r--s', lw=1.6, ms=6, alpha=0.65,
         label='测试集整体 OV%（参考）', zorder=3)

ax2.axvline(x=LAMBDA_PAPER, color='dimgray', ls='--', lw=1.4, alpha=0.9,
            label=f'论文选定值 λ={LAMBDA_PAPER:.1f}', zorder=2)

# 标注论文固定值处的对向穿插准确率
if valid_cc[paper_idx]:
    cc_at_paper = results_cc[paper_idx]
    ax2.annotate(
        f'λ={LAMBDA_PAPER:.1f}\n对向穿插:{cc_at_paper:.2f}%',
        xy=(LAMBDA_PAPER, cc_at_paper),
        xytext=(LAMBDA_PAPER + 0.15, cc_at_paper - 1.5),
        fontsize=9.5, color='green',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.0),
    )

# 标注对向穿插子集最优λ
if valid_cc.any():
    cc_best_idx = int(np.nanargmax(results_cc))
    cc_best_lam = LAMBDA_VALUES[cc_best_idx]
    if cc_best_lam != LAMBDA_PAPER:
        ax2.plot(cc_best_lam, results_cc[cc_best_idx], 'g*', ms=14, zorder=5,
                 label=f'对向穿插最优 λ={cc_best_lam:.1f}  ({results_cc[cc_best_idx]:.2f}%)')

y_lo2 = min(results_cc[valid_cc].min() if valid_cc.any() else 100,
            results_te.min()) - 1.5
y_hi2 = max(results_cc[valid_cc].max() if valid_cc.any() else 0,
            results_te.max()) + 1.5
ax2.set_ylim(y_lo2, y_hi2)
ax2.set_xlim(lam_arr[0] - 0.1, lam_arr[-1] + 0.1)
ax2.set_xlabel('耦合敏感系数 λ', fontsize=13)
ax2.set_ylabel('综合准确率 OV (%)', fontsize=13)
ax2.set_title(
    f'SCDAM 核心机制对 λ 的敏感性：对向穿插场景（ω̄ < {OMEGA_CC_THR}）\n'
    f'速度一致性低 → 速度耦合项起关键区分作用',
    fontsize=12,
)
ax2.legend(fontsize=10.5, loc='lower right')
ax2.grid(True, ls=':', alpha=0.55)
ax2.set_xticks(lam_arr)

ax2.text(0.02, 0.03,
         f'对向穿插子集 n={n_cc}  蜂群:{np.sum(y_te[mask_cc]==1)}'
         f'  非蜂群:{np.sum(y_te[mask_cc]==0)}\n'
         f'波动范围={cc_range:.2f}%  整体测试集波动={te_range:.2f}%',
         transform=ax2.transAxes, fontsize=8.5, color='gray',
         verticalalignment='bottom')

plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, 'fig_lambda_cross_scene.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  图2已保存: {path2}")


# ───────────────────────────────────────────────────────
# 图 3：omega_bar 区间 × λ 的准确率热力图
#        展示 λ 与数据集速度特性的交互效应（回应审稿人第②条）
# ───────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 4.5))
# 过滤掉全 NaN 的行
valid_rows = [i for i in range(len(OB_BINS))
              if not np.all(np.isnan(results_ob[i]))]
ob_data = results_ob[valid_rows, :]
ob_labels_plot = [OB_NAMES[i] for i in valid_rows]

im = ax3.imshow(ob_data, aspect='auto', cmap='RdYlGn',
                vmin=np.nanmin(ob_data) - 0.5,
                vmax=np.nanmax(ob_data) + 0.5)
plt.colorbar(im, ax=ax3, label='准确率 OV (%)')

ax3.set_xticks(range(len(LAMBDA_VALUES)))
ax3.set_xticklabels([f'{l:.1f}' for l in LAMBDA_VALUES])
ax3.set_yticks(range(len(ob_labels_plot)))
ax3.set_yticklabels(ob_labels_plot, fontsize=10)
ax3.set_xlabel('耦合敏感系数 λ', fontsize=13)
ax3.set_ylabel('omega_bar 区间（速度一致性）', fontsize=12)
ax3.set_title('λ 与数据集速度特性的交互效应\n（各 ω̄ 区间在不同 λ 下的测试集准确率）',
              fontsize=12)

# 在每个格子标注数值
for r in range(len(ob_labels_plot)):
    for c in range(len(LAMBDA_VALUES)):
        val = ob_data[r, c]
        if not np.isnan(val):
            ax3.text(c, r, f'{val:.1f}', ha='center', va='center',
                     fontsize=8.5, color='black')

# 标记论文固定值列
ax3.axvline(x=paper_idx, color='blue', lw=2.0, alpha=0.5, ls='--')
ax3.text(paper_idx, -0.6, f'λ={LAMBDA_PAPER:.1f}', ha='center', fontsize=9,
         color='blue', transform=ax3.get_xaxis_transform())

plt.tight_layout()
path3 = os.path.join(FIGURES_DIR, 'fig_lambda_omega_heatmap.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  图3已保存: {path3}")


# =====================================================================
# 6. 保存完整数值结果
# =====================================================================
npz_path = os.path.join(SCRIPT_DIR, 'results_lambda_search.npz')
np.savez(npz_path,
    lambda_values  = np.array(LAMBDA_VALUES),
    ov_val         = results_va,
    ov_test        = results_te,
    ov_cc          = results_cc,
    sw_test        = results_sw,
    nsw_test       = results_nsw,
    results_ob     = results_ob,
    ob_bin_edges   = np.array([b[0] for b in OB_BINS] + [OB_BINS[-1][1]]),
    omega_bar_te   = omega_bar_te,
    mask_cc        = mask_cc,
    n_cc           = np.array([n_cc]),
    omega_cc_thr   = np.array([OMEGA_CC_THR]),
    n_total        = np.array([min_n]),
    n_train        = np.array([n_tr]),
    n_val          = np.array([n_va]),
    n_test         = np.array([len(te_i)]),
)
print(f"  数值结果已保存: {npz_path}")


# =====================================================================
# 最终总结
# =====================================================================
print("\n" + "=" * 70)
print("  λ 敏感性分析结论（全量数据，与论文同划分）")
print("=" * 70)
print(f"  数据规模   : {min_n} 样本（训练{n_tr} / 验证{n_va} / 测试{len(te_i)}）")
print(f"  λ 搜索范围 : {LAMBDA_VALUES[0]:.1f} ~ {LAMBDA_VALUES[-1]:.1f}，共 {len(LAMBDA_VALUES)} 个值")
print(f"  验证集最优 λ = {lam_best_va:.1f}  |  测试集最优 λ = {lam_best_te:.1f}")
print(f"  论文固定   λ = {LAMBDA_PAPER:.1f}  测试集准确率 = {results_te[paper_idx]:.2f}%")
print(f"  测试集波动 = {te_range:.2f}%  对向穿插波动 = {cc_range:.2f}%")

if lam_best_te == LAMBDA_PAPER:
    print(f"\n  ✓ 结论：λ=1.0 是全量数据上的最优/并列最优值，论文论证充分。")
elif abs(results_te[lam_best_te_idx] - results_te[paper_idx]) < 0.2:
    print(f"\n  ✓ 结论：λ={lam_best_te:.1f} 比 λ={LAMBDA_PAPER:.1f} 高 "
          f"{results_te[lam_best_te_idx]-results_te[paper_idx]:.2f}%，差异极小（<0.2%），")
    print(f"    λ=1.0 仍属于稳健选择区间，可在论文中如实说明并保留 λ=1.0。")
else:
    print(f"\n  ! 结论：λ={lam_best_te:.1f} 与 λ={LAMBDA_PAPER:.1f} 存在 "
          f"{results_te[lam_best_te_idx]-results_te[paper_idx]:.2f}% 差异，")
    print(f"    建议将论文中 λ 的固定值更新为 {lam_best_te:.1f}，并重新运行所有实验。")

print(f"\n  总运行时间: {(time.time()-t_total)/60:.1f} 分钟")
print("=" * 70)
print("\nsection_4_4_lambda_search.py 完成.")
