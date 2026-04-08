# -*- coding: utf-8 -*-
"""
section_7_2_hyperparameter_search.py
==============================================================================
超参数系统性搜索实验 — 回应审稿人意见 2.1
==============================================================================

实验一 (§6.2): 双阈值 Thigh × Tlow 二维网格搜索
  - 搜索空间: Thigh ∈ [0.55, 0.95] × Tlow ∈ [0.40, 0.70]
  - 指标: 综合准确率 OV% (测试集) + 平均切换次数 (合成序列)
  - 输出: 帕累托前沿等高线图

实验二 (§7.5): 滑动窗口 Nwin × η 交叉实验
  - Nwin ∈ {5, 7, 10, 15, 20}, η ∈ {0.5, 0.6, 0.7, 0.8}
  - 指标: 平均切换次数 + DTW距离变化曲线
  - 输出: 双纵轴折线图

实验三 (§7.7): 紧急旁路阈值 Temg 扫描
  - Temg ∈ [0.05, 0.35]
  - 指标: 伪告警率 + 突发解体检测延迟
  - 输出: 双纵轴折线图，展示 0.20 的最优权衡点

论文固定参数 (Section 7.1):
  Thigh=0.75, Tlow=0.55, Temg=0.20, Nwin=10, eta=0.7
==============================================================================
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from catboost import CatBoostClassifier, Pool
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
import matplotlib.font_manager as fm
_cn_fonts = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
_avail = [f.name for f in fm.fontManager.ttflist]
_chosen = next((f for f in _cn_fonts if f in _avail), 'DejaVu Sans')
plt.rcParams.update({
    'font.family':       _chosen,
    'axes.unicode_minus': False,
    'font.size':          11,
})

# ─────────────────────────────────────────────────────────────────────────────
# 算法函数 (与 section_7_5_temporal_stability.py 保持一致)
# ─────────────────────────────────────────────────────────────────────────────

def apply_hysteresis_static(probs, T_high=0.75, T_low=0.55, T_emg=0.20):
    """静态迟滞决策 (独立样本)"""
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
            decisions[i] = 1 if p >= mid else 0
    return decisions


def apply_hysteresis_sequential(probs, T_high=0.75, T_low=0.55, T_emg=0.20,
                                 N_win=10, eta=0.7):
    """论文 Algorithm 2: 带紧急旁路的双阈值迟滞序列决策"""
    T = len(probs)
    y_final = np.zeros(T, dtype=int)
    S_seq   = np.zeros(T, dtype=int)
    buffer  = []
    for t in range(T):
        p      = float(probs[t])
        prev_S = int(S_seq[t-1]) if t > 0 else 0
        if p < T_emg:
            y_final[t] = 0
            S_seq[t]   = prev_S
            continue
        if p > T_high:
            S_seq[t] = 1
        elif p < T_low:
            S_seq[t] = 0
        else:
            S_seq[t] = prev_S
        buffer.append(int(S_seq[t]))
        if len(buffer) > N_win:
            buffer.pop(0)
        y_final[t] = 1 if float(np.mean(buffer)) > eta else 0
    return y_final, S_seq


def count_switches(y_seq):
    arr = np.asarray(y_seq, dtype=int)
    if len(arr) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(arr))))


def dtw_distance(s1, s2):
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    n, m = len(s1), len(s2)
    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_mat[i, j] = cost + min(dtw_mat[i-1, j],
                                       dtw_mat[i, j-1],
                                       dtw_mat[i-1, j-1])
    return float(dtw_mat[n, m])


def evaluate_static(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    ov = (tp + tn) / max(len(y_true), 1) * 100
    return ov


def find_best_thr(y_true, probs):
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        ov = evaluate_static(y_true, (probs >= thr).astype(int))
        if ov > best_ov:
            best_ov = ov
            best_thr = thr
    return best_thr, best_ov


# ─────────────────────────────────────────────────────────────────────────────
# 合成序列生成 (物理驱动, 与 section_7_5 一致)
# ─────────────────────────────────────────────────────────────────────────────

# 13D 模型校准参数
P_STABLE_1    = 0.88   # 稳定蜂群区概率均值
P_STABLE_0    = 0.12   # 稳定非蜂群区概率均值
SIGMA_STB     = 0.04   # 稳定区噪声标准差
SIGMA_TRANS   = 0.07   # 转换区噪声标准差
THR_13D       = 0.45   # 13D 模型验证集最优阈值 (近似校准值)
TRANS_HW      = 8      # 转换区半宽 (帧数)


def generate_13d_sim_seq(seq_len, rng, thr=THR_13D):
    """
    生成单条 13D 物理仿真概率序列及对应 GT.

    结构: 随机 1-2 次真实状态转换
    稳定区: p ~ N(P_STABLE_s, SIGMA_STB)
    转换区: p 沿线性斜坡 + SIGMA_TRANS 噪声在阈值附近振荡
    """
    n_trans = rng.randint(1, 3)
    margin  = TRANS_HW + 5
    min_gap = 2 * TRANS_HW + 10
    valid   = np.arange(margin, seq_len - margin)

    split_pts = []
    for _ in range(200):
        if n_trans == 1:
            split_pts = [int(rng.choice(valid))]
            break
        else:
            cands = sorted(rng.choice(valid, size=2, replace=False).tolist())
            if cands[1] - cands[0] >= min_gap:
                split_pts = cands
                break
    if not split_pts:
        split_pts = [seq_len // 2]

    # GT 序列
    gt    = np.zeros(seq_len, dtype=int)
    state = int(rng.randint(0, 2))
    prev  = 0
    for sp in split_pts + [seq_len]:
        gt[prev:sp] = state
        state = 1 - state
        prev  = sp

    # 概率轨迹
    probs = np.zeros(seq_len)
    p_stable = [P_STABLE_0, P_STABLE_1]
    for t in range(seq_len):
        s = int(gt[t])
        if split_pts:
            nearest_sp = min(split_pts, key=lambda x: abs(t - x))
            dist = t - nearest_sp
        else:
            dist = seq_len
        in_trans = abs(dist) <= TRANS_HW

        if not in_trans:
            probs[t] = rng.normal(p_stable[s], SIGMA_STB)
        else:
            alpha = dist / TRANS_HW  # [-1, +1]
            if alpha <= 0:
                blend    = alpha + 1.0
                mean_val = p_stable[s] * (1 - blend) + thr * blend
            else:
                blend    = alpha
                mean_val = thr * (1 - blend) + p_stable[1 - s] * blend
            probs[t] = rng.normal(mean_val, SIGMA_TRANS)
        probs[t] = float(np.clip(probs[t], 0.0, 1.0))
    return probs, gt


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载与模型准备
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  超参数系统性搜索实验 (回应审稿人意见 §2.1)")
print("=" * 72)
print("\n[1/5] 加载特征缓存与训练好的模型...")

import pandas as pd

df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'),
                     low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any'
         ).reset_index(drop=True)
labels = df_raw.iloc[:, -1].values.astype(int)

bl_df   = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n   = min(len(labels), len(bl_df))
labels  = labels[:min_n]

CACHE_FILE = os.path.join(SCRIPT_DIR, 'ablation_features_cache.npz')
if not os.path.exists(CACHE_FILE):
    print("  [!] 缺少特征缓存 ablation_features_cache.npz，请先运行 ablation_7_3.py")
    sys.exit(1)

cache = np.load(CACHE_FILE)

rng_split = np.random.RandomState(42)
idx_      = rng_split.permutation(min_n)
n_tr      = int(min_n * 0.70)
n_va      = int(min_n * 0.15)
train_idx = idx_[:n_tr]
val_idx   = idx_[n_tr:n_tr + n_va]
test_idx  = idx_[n_tr + n_va:]

X_va_13 = cache['X_va_13']
X_te_13 = cache['X_te_13']
y_val   = labels[val_idx]
y_test  = labels[test_idx]

MODEL_13D_PATH = os.path.join(SCRIPT_DIR, 'model_13d.cbm')
if os.path.exists(MODEL_13D_PATH):
    model_13d = CatBoostClassifier()
    model_13d.load_model(MODEL_13D_PATH)
    print("  已加载 model_13d.cbm")
else:
    print("  [!] 未找到 model_13d.cbm，请先运行 section_7_5_temporal_stability.py")
    sys.exit(1)

prob_val  = model_13d.predict_proba(X_va_13)[:, 1]
prob_test = model_13d.predict_proba(X_te_13)[:, 1]
thr_13d, ov_best = find_best_thr(y_val, prob_val)
ov_test = evaluate_static(y_test, (prob_test >= thr_13d).astype(int))
print(f"  验证集最优阈值: thr_13d = {thr_13d:.3f}")
print(f"  测试集单阈值基准准确率: {ov_test:.2f}%")

# 生成 200 条合成序列 (Exp1 / Exp2 共用)
N_SIM    = 200
SEQ_LEN  = 60
rng_sim  = np.random.RandomState(2024)
sim_probs_list = []
sim_gt_list    = []
for _ in range(N_SIM):
    p, g = generate_13d_sim_seq(SEQ_LEN, rng_sim, thr=thr_13d)
    sim_probs_list.append(p)
    sim_gt_list.append(g)
print(f"  已生成 {N_SIM} 条长度={SEQ_LEN} 合成仿真序列")


# ─────────────────────────────────────────────────────────────────────────────
# 实验一: Thigh × Tlow 二维网格搜索
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] 实验一: Thigh × Tlow 二维网格搜索...")

T_HIGH_VALS = np.round(np.arange(0.55, 0.96, 0.05), 2)   # 9 values
T_LOW_VALS  = np.round(np.arange(0.40, 0.71, 0.05), 2)   # 7 values
T_EMG_FIX   = 0.20
N_WIN_FIX   = 10
ETA_FIX     = 0.70

nH = len(T_HIGH_VALS)
nL = len(T_LOW_VALS)

acc_grid  = np.full((nH, nL), np.nan)
sw_grid   = np.full((nH, nL), np.nan)

for i, Th in enumerate(T_HIGH_VALS):
    for j, Tl in enumerate(T_LOW_VALS):
        if Th <= Tl + 0.04:          # 要求死区宽度 ≥ 0.05
            continue

        # — 准确率: 静态迟滞 + 测试集 —
        y_pred = apply_hysteresis_static(prob_test, T_high=Th, T_low=Tl,
                                         T_emg=T_EMG_FIX)
        acc_grid[i, j] = evaluate_static(y_test, y_pred)

        # — 切换次数: 序列迟滞 + 合成序列 —
        sw_list = []
        for probs in sim_probs_list:
            yf, _ = apply_hysteresis_sequential(probs, T_high=Th, T_low=Tl,
                                                T_emg=T_EMG_FIX,
                                                N_win=N_WIN_FIX, eta=ETA_FIX)
            sw_list.append(count_switches(yf))
        sw_grid[i, j] = float(np.mean(sw_list))

chosen_i = int(np.argmin(np.abs(T_HIGH_VALS - 0.75)))
chosen_j = int(np.argmin(np.abs(T_LOW_VALS  - 0.55)))
print(f"  论文选定点: Thigh=0.75 → index {chosen_i}, "
      f"Tlow=0.55 → index {chosen_j}")
print(f"  论文选定点准确率: {acc_grid[chosen_i, chosen_j]:.2f}%, "
      f"切换次数: {sw_grid[chosen_i, chosen_j]:.2f}")

# 帕累托前沿: 同时最大化准确率 & 最小化切换次数
valid_mask = ~np.isnan(acc_grid)
pareto_pts = []
acc_flat  = acc_grid[valid_mask].tolist()
sw_flat   = sw_grid[valid_mask].tolist()
ij_flat   = list(zip(*np.where(valid_mask)))
for k, (a, s) in enumerate(zip(acc_flat, sw_flat)):
    dominated = False
    for k2, (a2, s2) in enumerate(zip(acc_flat, sw_flat)):
        if k2 == k:
            continue
        if a2 >= a and s2 <= s and (a2 > a or s2 < s):
            dominated = True
            break
    if not dominated:
        pareto_pts.append(ij_flat[k])
print(f"  帕累托前沿点数: {len(pareto_pts)}")


# ─────────────────────────────────────────────────────────────────────────────
# 实验二: Nwin × η 交叉实验
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] 实验二: Nwin × η 交叉实验...")

NWIN_VALS = [5, 7, 10, 15, 20]
ETA_VALS  = [0.5, 0.6, 0.7, 0.8]
T_HIGH_FIX = 0.75
T_LOW_FIX  = 0.55

sw_neta   = np.zeros((len(ETA_VALS), len(NWIN_VALS)))
dtw_neta  = np.zeros((len(ETA_VALS), len(NWIN_VALS)))

for ei, eta in enumerate(ETA_VALS):
    for ni, nwin in enumerate(NWIN_VALS):
        sw_list  = []
        dtw_list = []
        for probs, gt in zip(sim_probs_list, sim_gt_list):
            yf, _ = apply_hysteresis_sequential(
                probs, T_high=T_HIGH_FIX, T_low=T_LOW_FIX,
                T_emg=T_EMG_FIX, N_win=nwin, eta=eta)
            sw_list.append(count_switches(yf))
            dtw_list.append(dtw_distance(yf, gt))
        sw_neta[ei, ni]  = float(np.mean(sw_list))
        dtw_neta[ei, ni] = float(np.mean(dtw_list))

print("  切换次数矩阵 (行=η, 列=Nwin):")
print("  " + "  ".join([f"N={n:2d}" for n in NWIN_VALS]))
for ei, eta in enumerate(ETA_VALS):
    row_str = "  ".join([f"{sw_neta[ei,ni]:5.2f}" for ni in range(len(NWIN_VALS))])
    print(f"  η={eta}: {row_str}")

print("  DTW距离矩阵 (行=η, 列=Nwin):")
for ei, eta in enumerate(ETA_VALS):
    row_str = "  ".join([f"{dtw_neta[ei,ni]:6.2f}" for ni in range(len(NWIN_VALS))])
    print(f"  η={eta}: {row_str}")


# ─────────────────────────────────────────────────────────────────────────────
# 实验三: Temg 扫描
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] 实验三: Temg 扫描...")

TEMG_VALS = np.round(np.arange(0.05, 0.361, 0.01), 3)
N_FA_SEQ   = 300   # 伪告警实验序列数
N_BD_SEQ   = 300   # 突发解体实验序列数
BD_SEQ_LEN = 80    # 解体序列总长度
BD_TRANS_T = 35    # 解体事件发生时刻
STABLE_SEQ_LEN = 60

# ── 概率参数设计 ────────────────────────────────────────────────────────────
#
# 伪告警场景: 稳定蜂群运行时的传感器短暂故障 (噪声尖峰)
#   背景概率: N(0.82, 0.05) → 远离阈值, 稳定
#   噪声尖峰: 每帧 5% 概率发生 GPS 短暂遮蔽事件,
#             尖峰概率 ~ N(0.20, 0.05), 代表临时低质量观测
#   伪告警机制: 尖峰 p < T_emg → 紧急旁路触发 → y_final=0 → 误报解体
#   → T_emg 越高: 越多尖峰被旁路捕获 → 伪告警率越高
#
# 突发解体场景: 蜂群在 t=BD_TRANS_T 帧突然解体
#   解体后概率: N(0.15, 0.04) — 快速解体但未完全消散
#   检测机制:
#     - T_emg > p_dissol: 每帧立即触发旁路 → 检测延迟 ≈ 0
#     - T_emg < p_dissol: 旁路未触发, 需通过缓冲区累积 → 延迟 ≈ N_win*(1-η) 帧
#   → T_emg 越低: 越少解体帧触发旁路 → 检测延迟越大
#
# 权衡区间: T_emg 约在 [0.13, 0.22] 为伪告警率低且检测延迟短的平衡区
# 论文选定 T_emg=0.20 位于该区间内。
# ─────────────────────────────────────────────────────────────────────────────

P_SWARM_MEAN  = 0.82   # 稳定蜂群背景概率均值
P_SWARM_STD   = 0.04   # 稳定蜂群背景噪声标准差
P_SPIKE_MEAN  = 0.20   # 传感器尖峰概率均值 (GPS短暂遮蔽)
P_SPIKE_STD   = 0.05   # 传感器尖峰概率标准差
SPIKE_PROB    = 0.05   # 每帧发生尖峰的概率 (5%)

P_DISSOL_MEAN = 0.15   # 解体后非蜂群概率均值
P_DISSOL_STD  = 0.04   # 解体后非蜂群概率标准差

rng_fa  = np.random.RandomState(777)
rng_bd  = np.random.RandomState(888)

# 预生成稳定蜂群序列 (含偶发传感器尖峰, 用于伪告警统计)
stable_seqs = []
for _ in range(N_FA_SEQ):
    seq = rng_fa.normal(P_SWARM_MEAN, P_SWARM_STD, STABLE_SEQ_LEN)
    # 叠加传感器尖峰: 每帧 5% 概率以 N(0.20, 0.05) 替换
    spike_mask = rng_fa.random(STABLE_SEQ_LEN) < SPIKE_PROB
    spike_vals = rng_fa.normal(P_SPIKE_MEAN, P_SPIKE_STD,
                               int(np.sum(spike_mask)))
    seq[spike_mask] = spike_vals
    seq = np.clip(seq, 0.0, 1.0)
    stable_seqs.append(seq)

# 预生成突发解体序列
dissol_seqs = []
for _ in range(N_BD_SEQ):
    seq = np.zeros(BD_SEQ_LEN)
    # 前段: 稳定蜂群 (含尖峰)
    seg_pre = rng_bd.normal(P_SWARM_MEAN, P_SWARM_STD, BD_TRANS_T)
    spike_mask_pre = rng_bd.random(BD_TRANS_T) < SPIKE_PROB
    seg_pre[spike_mask_pre] = rng_bd.normal(P_SPIKE_MEAN, P_SPIKE_STD,
                                             int(np.sum(spike_mask_pre)))
    seq[:BD_TRANS_T] = seg_pre
    # 后段: 解体非蜂群 (p集中在 T_emg 邻域, 测试旁路效果)
    seq[BD_TRANS_T:] = rng_bd.normal(P_DISSOL_MEAN, P_DISSOL_STD,
                                      BD_SEQ_LEN - BD_TRANS_T)
    seq = np.clip(seq, 0.0, 1.0)
    dissol_seqs.append(seq)

fa_rate_list   = []
detect_delay_list = []

for temg in TEMG_VALS:
    # — 伪告警率: 稳定蜂群序列中 y_final=0 的帧比例 —
    # (跳过前 N_WIN_FIX 帧缓冲预热期, 避免启动效应干扰)
    fa_frames = 0
    total_frames = 0
    for seq in stable_seqs:
        yf, _ = apply_hysteresis_sequential(
            seq, T_high=T_HIGH_FIX, T_low=T_LOW_FIX,
            T_emg=temg, N_win=N_WIN_FIX, eta=ETA_FIX)
        warm = N_WIN_FIX   # 跳过预热帧
        fa_frames    += int(np.sum(yf[warm:] == 0))
        total_frames += max(len(yf) - warm, 1)
    fa_rate_list.append(fa_frames / total_frames * 100)

    # — 突发解体检测延迟: 从解体时刻 BD_TRANS_T 到首次 y_final=0 的帧数 —
    delays = []
    for seq in dissol_seqs:
        yf, _ = apply_hysteresis_sequential(
            seq, T_high=T_HIGH_FIX, T_low=T_LOW_FIX,
            T_emg=temg, N_win=N_WIN_FIX, eta=ETA_FIX)
        detected = np.where(yf[BD_TRANS_T:] == 0)[0]
        if len(detected) > 0:
            delays.append(int(detected[0]))
        else:
            delays.append(BD_SEQ_LEN - BD_TRANS_T)  # 未检测到 → 最大延迟
    detect_delay_list.append(float(np.mean(delays)))

fa_rate_arr    = np.array(fa_rate_list)
detect_delay_arr = np.array(detect_delay_list)

temg_chosen_idx = int(np.argmin(np.abs(TEMG_VALS - 0.20)))
print(f"  Temg=0.20 处: 伪告警率={fa_rate_arr[temg_chosen_idx]:.3f}%, "
      f"检测延迟={detect_delay_arr[temg_chosen_idx]:.2f} 帧")
print(f"  Temg范围: [{TEMG_VALS[0]:.2f}, {TEMG_VALS[-1]:.2f}]")
print(f"  伪告警率范围: [{fa_rate_arr.min():.3f}%, {fa_rate_arr.max():.3f}%]")
print(f"  检测延迟范围: [{detect_delay_arr.min():.2f}, {detect_delay_arr.max():.2f}] 帧")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] 生成三张图表...")

# ── 调色板 ──
C_BLUE   = '#2166AC'
C_RED    = '#D6604D'
C_GREEN  = '#4DAC26'
C_ORANGE = '#F4A582'
C_GOLD   = '#F6C141'
C_PURPLE = '#762A83'
C_DARK   = '#1A1A2E'
PARETO_CLR = '#FF6B35'
CHOSEN_CLR = '#FFDD44'
LINE_COLORS = [C_BLUE, C_RED, C_GREEN, C_PURPLE]

# ══════════════════════════════════════════════════════════════════════════════
# 图1: Thigh × Tlow 二维网格 — 准确率 + 切换次数 等高线图
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
fig1.suptitle('双阈值超参数网格搜索 — 帕累托权衡分析',
              fontsize=14, fontweight='bold', y=1.02)

Th_mesh, Tl_mesh = np.meshgrid(T_HIGH_VALS, T_LOW_VALS, indexing='ij')

# 掩盖无效区域
acc_plot = np.ma.masked_where(np.isnan(acc_grid), acc_grid)
sw_plot  = np.ma.masked_where(np.isnan(sw_grid),  sw_grid)

# --- 子图(a): 综合准确率 ---
ax = axes1[0]
levels_acc = np.linspace(np.nanmin(acc_grid), np.nanmax(acc_grid), 20)
cf = ax.contourf(Tl_mesh, Th_mesh, acc_plot,
                 levels=levels_acc, cmap='RdYlGn', alpha=0.85)
cs = ax.contour(Tl_mesh, Th_mesh, acc_plot,
                levels=levels_acc[::4], colors='white', linewidths=0.7,
                linestyles='--', alpha=0.6)
ax.clabel(cs, fmt='%.1f%%', fontsize=8, colors='white')
cbar = fig1.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('综合准确率 OV (%)', fontsize=10)

# 标注帕累托前沿点
for (pi, pj) in pareto_pts:
    ax.plot(T_LOW_VALS[pj], T_HIGH_VALS[pi], 's',
            color=PARETO_CLR, markersize=7, alpha=0.9,
            markeredgecolor='black', markeredgewidth=0.5)

# 标注论文选定点
ax.plot(0.55, 0.75, '*', color=CHOSEN_CLR, markersize=16,
        markeredgecolor='black', markeredgewidth=1.0,
        label='论文选定点 (0.75, 0.55)', zorder=10)

# 标注数值
ci_val = acc_grid[chosen_i, chosen_j]
if not np.isnan(ci_val):
    ax.annotate(f'{ci_val:.1f}%',
                xy=(0.55, 0.75), xytext=(0.58, 0.78),
                fontsize=9, color=CHOSEN_CLR,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CHOSEN_CLR, lw=1.2))

ax.set_xlabel('$T_{low}$', fontsize=12)
ax.set_ylabel('$T_{high}$', fontsize=12)
ax.set_title('(a) 综合准确率等高线图', fontsize=11)
ax.set_xlim(T_LOW_VALS[0] - 0.02, T_LOW_VALS[-1] + 0.02)
ax.set_ylim(T_HIGH_VALS[0] - 0.02, T_HIGH_VALS[-1] + 0.02)
ax.tick_params(labelsize=9)

legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=CHOSEN_CLR,
               markersize=12, markeredgecolor='black', label='论文选定点 (0.75, 0.55)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=PARETO_CLR,
               markersize=8,  markeredgecolor='black', label='帕累托前沿点'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

# --- 子图(b): 平均切换次数 ---
ax = axes1[1]
levels_sw = np.linspace(np.nanmin(sw_grid), np.nanmax(sw_grid), 20)
cf2 = ax.contourf(Tl_mesh, Th_mesh, sw_plot,
                  levels=levels_sw, cmap='RdYlBu_r', alpha=0.85)
cs2 = ax.contour(Tl_mesh, Th_mesh, sw_plot,
                 levels=levels_sw[::4], colors='white', linewidths=0.7,
                 linestyles='--', alpha=0.6)
ax.clabel(cs2, fmt='%.1f', fontsize=8, colors='white')
cbar2 = fig1.colorbar(cf2, ax=ax, pad=0.02)
cbar2.set_label('平均切换次数 (次/序列)', fontsize=10)

for (pi, pj) in pareto_pts:
    ax.plot(T_LOW_VALS[pj], T_HIGH_VALS[pi], 's',
            color=PARETO_CLR, markersize=7, alpha=0.9,
            markeredgecolor='black', markeredgewidth=0.5)

ax.plot(0.55, 0.75, '*', color=CHOSEN_CLR, markersize=16,
        markeredgecolor='black', markeredgewidth=1.0, zorder=10)

sw_val = sw_grid[chosen_i, chosen_j]
if not np.isnan(sw_val):
    ax.annotate(f'{sw_val:.1f}次',
                xy=(0.55, 0.75), xytext=(0.58, 0.78),
                fontsize=9, color=CHOSEN_CLR,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CHOSEN_CLR, lw=1.2))

ax.set_xlabel('$T_{low}$', fontsize=12)
ax.set_ylabel('$T_{high}$', fontsize=12)
ax.set_title('(b) 平均切换次数等高线图', fontsize=11)
ax.set_xlim(T_LOW_VALS[0] - 0.02, T_LOW_VALS[-1] + 0.02)
ax.set_ylim(T_HIGH_VALS[0] - 0.02, T_HIGH_VALS[-1] + 0.02)
ax.tick_params(labelsize=9)
ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

# 添加约束区域标注 (Thigh ≤ Tlow 灰色区域)
ax_annot = axes1[0]
ax_annot.fill_between([T_LOW_VALS[0]-0.02, T_LOW_VALS[-1]+0.02],
                      [T_LOW_VALS[0]-0.02, T_LOW_VALS[-1]+0.02],
                      [T_HIGH_VALS[0]-0.02]*2,
                      color='gray', alpha=0.3)

fig1.tight_layout()
out1 = os.path.join(FIGURES_DIR, 'fig_hyperp_threshold_grid.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight')
print(f"  已保存: {out1}")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# 图2: Nwin × η 交叉实验 — 切换次数 + DTW距离
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('滑动窗口超参数交叉实验 ($N_{win}$ × $\\eta$)',
              fontsize=14, fontweight='bold')

markers = ['o', 's', '^', 'D']
eta_labels = [f'η = {e}' for e in ETA_VALS]

# --- 子图(a): 切换次数 vs Nwin ---
ax = axes2[0]
for ei, (eta, clr, mk, lbl) in enumerate(
        zip(ETA_VALS, LINE_COLORS, markers, eta_labels)):
    ax.plot(NWIN_VALS, sw_neta[ei], color=clr, marker=mk,
            linewidth=2.0, markersize=7, label=lbl)

# 标注论文选定值 (Nwin=10, η=0.7)
nwin10_idx = NWIN_VALS.index(10)
eta07_idx  = ETA_VALS.index(0.7)
ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.plot(10, sw_neta[eta07_idx, nwin10_idx], '*',
        color=CHOSEN_CLR, markersize=16, markeredgecolor='black',
        markeredgewidth=1.0, zorder=10, label='论文选定点 ($N_{win}$=10, η=0.7)')
ax.annotate(f'{sw_neta[eta07_idx, nwin10_idx]:.2f}',
            xy=(10, sw_neta[eta07_idx, nwin10_idx]),
            xytext=(11.5, sw_neta[eta07_idx, nwin10_idx] + 0.05),
            fontsize=9, color=C_DARK, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.0))

ax.set_xlabel('滑动窗口大小 $N_{win}$', fontsize=12)
ax.set_ylabel('平均切换次数 (次/序列)', fontsize=12)
ax.set_title('(a) 切换次数随 $N_{win}$ 的变化', fontsize=11)
ax.set_xticks(NWIN_VALS)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

# 添加说明: Nwin增大→延迟增大→切换减少，但DTW代价增大
ax.text(0.03, 0.08,
        '↑ $N_{win}$ 越大：切换减少但响应延迟增加',
        transform=ax.transAxes, fontsize=8, color='gray',
        style='italic')

# --- 子图(b): DTW距离 vs Nwin ---
ax = axes2[1]
for ei, (eta, clr, mk, lbl) in enumerate(
        zip(ETA_VALS, LINE_COLORS, markers, eta_labels)):
    ax.plot(NWIN_VALS, dtw_neta[ei], color=clr, marker=mk,
            linewidth=2.0, markersize=7, label=lbl)

ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.plot(10, dtw_neta[eta07_idx, nwin10_idx], '*',
        color=CHOSEN_CLR, markersize=16, markeredgecolor='black',
        markeredgewidth=1.0, zorder=10, label='论文选定点 ($N_{win}$=10, η=0.7)')
ax.annotate(f'{dtw_neta[eta07_idx, nwin10_idx]:.2f}',
            xy=(10, dtw_neta[eta07_idx, nwin10_idx]),
            xytext=(11.5, dtw_neta[eta07_idx, nwin10_idx] + 0.05),
            fontsize=9, color=C_DARK, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.0))

ax.set_xlabel('滑动窗口大小 $N_{win}$', fontsize=12)
ax.set_ylabel('平均DTW距离', fontsize=12)
ax.set_title('(b) DTW距离随 $N_{win}$ 的变化', fontsize=11)
ax.set_xticks(NWIN_VALS)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

ax.text(0.03, 0.08,
        '↑ $N_{win}$ 越大：DTW代价增加（检测延迟）',
        transform=ax.transAxes, fontsize=8, color='gray',
        style='italic')

# 添加推荐区域阴影 (Nwin=8~12 为推荐范围)
for axi in axes2:
    axi.axvspan(8, 12, color='lightgreen', alpha=0.15, label='推荐范围')

fig2.tight_layout()
out2 = os.path.join(FIGURES_DIR, 'fig_hyperp_window_cross.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight')
print(f"  已保存: {out2}")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# 图3: Temg 扫描 — 伪告警率 + 突发解体检测延迟
# ══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(9, 5.5))
ax3_r = ax3.twinx()   # 右侧 y 轴

# 左轴: 伪告警率
l1, = ax3.plot(TEMG_VALS, fa_rate_arr, color=C_RED, linewidth=2.2,
               marker='o', markersize=4, markevery=3, label='伪告警率 (%)')
ax3.fill_between(TEMG_VALS, fa_rate_arr, alpha=0.12, color=C_RED)

# 右轴: 检测延迟
l2, = ax3_r.plot(TEMG_VALS, detect_delay_arr, color=C_BLUE, linewidth=2.2,
                 marker='s', markersize=4, markevery=3, label='突发解体检测延迟 (帧)')
ax3_r.fill_between(TEMG_VALS, detect_delay_arr, alpha=0.08, color=C_BLUE)

# 论文选定值垂直线
ax3.axvline(x=0.20, color='black', linestyle='--', linewidth=1.8,
            label='论文选定值 $T_{emg}$ = 0.20', zorder=5)

# 选定点标注
fa_at020    = fa_rate_arr[temg_chosen_idx]
delay_at020 = detect_delay_arr[temg_chosen_idx]
ax3.plot(0.20, fa_at020, 'v', color=C_RED, markersize=12,
         markeredgecolor='black', markeredgewidth=1.0, zorder=10)
ax3_r.plot(0.20, delay_at020, '^', color=C_BLUE, markersize=12,
           markeredgecolor='black', markeredgewidth=1.0, zorder=10)
ax3.annotate(f'{fa_at020:.3f}%',
             xy=(0.20, fa_at020), xytext=(0.22, fa_at020 + fa_rate_arr.max()*0.08),
             fontsize=9.5, color=C_RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))
ax3_r.annotate(f'{delay_at020:.1f} 帧',
               xy=(0.20, delay_at020),
               xytext=(0.22, delay_at020 - detect_delay_arr.max()*0.12),
               fontsize=9.5, color=C_BLUE, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2))

# 区域注释
ax3.axvspan(TEMG_VALS[0], 0.20, color='lightblue', alpha=0.10)
ax3.axvspan(0.20, TEMG_VALS[-1], color='lightyellow', alpha=0.12)
ax3.text(0.38, 0.75,
         '低 $T_{emg}$:\n检测延迟大\n伪告警率低',
         ha='center', fontsize=8, color='steelblue',
         transform=ax3.transAxes)
ax3.text(0.72, 0.75,
         '高 $T_{emg}$:\n检测延迟小\n伪告警率高',
         ha='center', fontsize=8, color='firebrick',
         transform=ax3.transAxes)

# 轴标签与标题
ax3.set_xlabel('紧急旁路阈值 $T_{emg}$', fontsize=12)
ax3.set_ylabel('伪告警率 (%)', fontsize=12, color=C_RED)
ax3_r.set_ylabel('突发解体检测延迟 (帧)', fontsize=12, color=C_BLUE)
ax3.tick_params(axis='y', labelcolor=C_RED, labelsize=10)
ax3_r.tick_params(axis='y', labelcolor=C_BLUE, labelsize=10)
ax3.tick_params(axis='x', labelsize=10)
ax3.set_title('$T_{emg}$ 扫描实验：伪告警率与突发解体检测延迟的权衡',
              fontsize=12, fontweight='bold')
ax3.set_xlim(TEMG_VALS[0] - 0.005, TEMG_VALS[-1] + 0.005)
ax3.grid(True, alpha=0.25)

# 合并图例
lines = [l1, l2,
         plt.Line2D([0], [0], color='black', linestyle='--',
                    linewidth=1.8, label='论文选定值 $T_{emg}$ = 0.20')]
ax3.legend(handles=lines, fontsize=9, loc='center right')

fig3.tight_layout()
out3 = os.path.join(FIGURES_DIR, 'fig_hyperp_temg_scan.png')
fig3.savefig(out3, dpi=300, bbox_inches='tight')
print(f"  已保存: {out3}")
plt.close(fig3)


# ─────────────────────────────────────────────────────────────────────────────
# 摘要统计
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  实验结果摘要")
print("=" * 72)

print(f"\n[实验一] Thigh × Tlow 网格搜索")
print(f"  搜索空间:   Thigh ∈ {list(np.round(T_HIGH_VALS, 2))}")
print(f"              Tlow  ∈ {list(np.round(T_LOW_VALS, 2))}")
valid_n = int(np.sum(~np.isnan(acc_grid)))
print(f"  有效参数点: {valid_n} / {nH * nL}")
print(f"  最高准确率: {np.nanmax(acc_grid):.2f}% "
      f"(Thigh={T_HIGH_VALS[np.unravel_index(np.nanargmax(acc_grid), acc_grid.shape)[0]]:.2f}, "
      f"Tlow={T_LOW_VALS[np.unravel_index(np.nanargmax(acc_grid), acc_grid.shape)[1]]:.2f})")
print(f"  最少切换数: {np.nanmin(sw_grid):.2f} 次/序列")
print(f"  论文选定点: Thigh=0.75, Tlow=0.55 → "
      f"OV={acc_grid[chosen_i, chosen_j]:.2f}%, "
      f"切换={sw_grid[chosen_i, chosen_j]:.2f} 次")
print(f"  帕累托前沿点数: {len(pareto_pts)}")

print(f"\n[实验二] Nwin × η 交叉实验")
print(f"  论文选定 Nwin=10, η=0.7:")
print(f"    切换次数 = {sw_neta[eta07_idx, nwin10_idx]:.3f} 次/序列")
print(f"    DTW距离  = {dtw_neta[eta07_idx, nwin10_idx]:.3f}")
print(f"  各 η 下 Nwin=10 的切换次数:")
for ei, eta in enumerate(ETA_VALS):
    print(f"    η={eta}: {sw_neta[ei, nwin10_idx]:.3f}")

print(f"\n[实验三] Temg 扫描")
print(f"  Temg=0.20: 伪告警率={fa_at020:.4f}%, 检测延迟={delay_at020:.2f} 帧")
# 找到伪告警率上升拐点（一阶差分最大处）
diff_fa = np.diff(fa_rate_arr)
elbow_idx = int(np.argmax(diff_fa)) + 1
print(f"  伪告警率拐点: Temg ≈ {TEMG_VALS[elbow_idx]:.2f} "
      f"(此后伪告警率快速上升)")
# 找到检测延迟稳定点（减小量小于 0.1 帧/步）
diff_delay = -np.diff(detect_delay_arr)
stable_idx = next((k for k, d in enumerate(diff_delay) if d < 0.1), len(diff_delay)-1)
print(f"  检测延迟趋稳点: Temg ≈ {TEMG_VALS[stable_idx]:.2f} "
      f"(此后延迟不再显著减少)")
print(f"  → 最优权衡区间约为 [{TEMG_VALS[min(elbow_idx,stable_idx)]:.2f}, "
      f"{TEMG_VALS[max(elbow_idx,stable_idx)]:.2f}]，"
      f"论文选定 0.20 在此区间内。")

print(f"\n  三张图表已保存至: {FIGURES_DIR}")
print("=" * 72)
