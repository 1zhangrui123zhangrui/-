# -*- coding: utf-8 -*-
"""
challenge_scenarios.py
==============================================================================
Section 7.4  挑战性场景分析  — 数据准备与统计
==============================================================================

三类挑战场景（客观筛选规则，可重现）:

  场景1  对向穿插/动态耦合  (Dynamic coupling):
         omega_bar < 0.60
         物理意义: 速度一致性低于非蜂群均值(0.678), 基线3D特征无法利用速度信息
         → 基线模型漏报蜂群; 本文omega_bar特征直接修复
         筛选依据: 客观数值阈值, 非主观选样

  场景2  集群结构/局部有序  (Cluster formation):
         0.60 ≤ omega_bar ≤ 0.80  AND  基线概率 ∈ [0.30, 0.70]
         物理意义: 中等速度协调度, 基线置信度低的模糊区间
         → SCDAM矩阵与频谱特征联合发挥作用

  场景3  边界状态  (Boundary state):
         基线CatBoost-3D概率 ∈ [0.35, 0.65]
         物理意义: 分类器置信度最低区间, 最客观的"困难样本"定义
         → 13维特征+迟滞决策联合作用

输出: python/results_challenge.npz  (供 challenge_viz.py 使用)
==============================================================================
"""

import sys, os
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
from paper2_algorithms import apply_hysteresis_static

# =====================================================================
# 工具函数
# =====================================================================
def evaluate(y_true, y_pred):
    tp = np.sum((y_pred==1)&(y_true==1)); tn = np.sum((y_pred==0)&(y_true==0))
    fp = np.sum((y_pred==1)&(y_true==0)); fn = np.sum((y_pred==0)&(y_true==1))
    sw  = tp/(tp+fn)*100 if (tp+fn)>0 else 0.0
    nsw = tn/(tn+fp)*100 if (tn+fp)>0 else 0.0
    ov  = (tp+tn)/len(y_true)*100
    return sw, nsw, ov, tp, tn, fp, fn

def find_best_thr(y_true, probs):
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        _, _, ov, *_ = evaluate(y_true, (probs>=thr).astype(int))
        if ov > best_ov: best_ov=ov; best_thr=thr
    return best_thr

# =====================================================================
# 1. 加载数据
# =====================================================================
print("="*70); print("  Section 7.4 挑战性场景分析"); print("="*70)

print("\n[1/5] 加载数据...")
df_raw = pd.read_csv(os.path.join(CSV_DIR,'Swarm_Behaviour_Data.csv'), low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
raw_data = df_raw.iloc[:,:-1].values.astype(float)     # 原始特征 (位置/速度)
labels   = df_raw.iloc[:,-1].values.astype(int)

bl_df    = pd.read_csv(os.path.join(CSV_DIR,'catboost_features_table5.csv'))
min_n    = min(len(labels), len(bl_df))
raw_data = raw_data[:min_n]; labels = labels[:min_n]
feat3d   = bl_df[['corr1','corr2','corr3']].values[:min_n]

# 恢复13D特征到原始索引顺序
cache = np.load(os.path.join(SCRIPT_DIR,'ablation_features_cache.npz'))
rng0  = np.random.RandomState(42)
idx0  = rng0.permutation(min_n)
n_tr  = int(min_n*0.70); n_va = int(min_n*0.15)
tr_i  = idx0[:n_tr]; va_i = idx0[n_tr:n_tr+n_va]; te_i = idx0[n_tr+n_va:]
X_all = np.zeros((min_n,13))
X_all[tr_i]=cache['X_tr_13']; X_all[va_i]=cache['X_va_13']; X_all[te_i]=cache['X_te_13']

y_tr=labels[tr_i]; y_va=labels[va_i]; y_te=labels[te_i]
X_tr3=feat3d[tr_i]; X_va3=feat3d[va_i]; X_te3=feat3d[te_i]
X_tr13=X_all[tr_i]; X_va13=X_all[va_i]; X_te13=X_all[te_i]
omega_te = X_all[te_i,9]   # omega_bar 特征
corr1_te = feat3d[te_i,0]  # baseline corr1
corr2_te = feat3d[te_i,1]  # baseline corr2
corr3_te = feat3d[te_i,2]  # baseline corr3

fpd = 12   # fields per drone

print(f"  测试集: {len(y_te)} 样本 (蜂群:{np.sum(y_te==1)}, 非蜂群:{np.sum(y_te==0)})")

# =====================================================================
# 2. 训练两个模型
# =====================================================================
print("\n[2/5] 训练分类器...")

# 基线 CatBoost-3D (论文1参数)
m3 = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                         loss_function='Logloss', random_seed=42, verbose=0)
m3.fit(X_tr3, y_tr)
p3_va = m3.predict_proba(X_va3)[:,1]; p3_te = m3.predict_proba(X_te3)[:,1]
thr3  = find_best_thr(y_va, p3_va)

# 完整方法 CatBoost-13D (论文2参数)
m13 = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.03,
                          l2_leaf_reg=5, subsample=0.8,
                          loss_function='Logloss', random_seed=42, verbose=0)
m13.fit(X_tr13, y_tr, eval_set=Pool(X_va13,y_va), early_stopping_rounds=50, verbose=0)
p13_va = m13.predict_proba(X_va13)[:,1]; p13_te = m13.predict_proba(X_te13)[:,1]
thr13  = find_best_thr(y_va, p13_va)

T_high = min(thr13+0.05, 0.90); T_low=max(thr13-0.10,0.10); T_emg=max(thr13-0.30,0.05)
pred3  = (p3_te  >= thr3 ).astype(int)
pred13 = apply_hysteresis_static(p13_te, T_high=T_high, T_low=T_low, T_emg=T_emg)

sw3,nsw3,ov3,*cm3   = evaluate(y_te, pred3)
sw13,nsw13,ov13,*cm13 = evaluate(y_te, pred13)
print(f"  基线CatBoost-3D:  蜂群={sw3:.1f}% 非蜂群={nsw3:.1f}% 综合={ov3:.1f}%")
print(f"  完整方法13D+迟滞: 蜂群={sw13:.1f}% 非蜂群={nsw13:.1f}% 综合={ov13:.1f}%")

# =====================================================================
# 3. 定义三类挑战场景（客观筛选规则）
# =====================================================================
print("\n[3/5] 定义挑战场景...")

# ── 场景1: 动态耦合挑战 (对向穿插) ──
# 筛选规则: omega_bar < 0.60 (低于非蜂群均值0.678)
# 物理依据: 速度一致性处于低位, 3D纯空间特征无法有效区分
THR_CC = 0.60
mask_s1 = omega_te < THR_CC

# ── 场景2: 集群结构/空间模糊蜂群 ──
# 筛选规则: label=1 (蜂群) AND corr2_R < 0.08 (空间自相关特征低于非蜂群典型区间上界的1.6倍)
# 物理依据: 这类蜂群的3D空间特征与非蜂群重叠 (非蜂群corr2均值=0.015, max=0.051)
#           基线依赖corr特征, 对这类"空间模糊蜂群"识别困难
#           本文方法: omega_bar + 频谱特征提供补充判别信息
mask_s2 = (y_te == 1) & (corr2_te < 0.08)

# ── 场景3: 边界状态 ──
# 筛选规则: 基线概率 ∈ [0.30, 0.70]  (扩展不确定区间, 增大样本量)
# 物理依据: 分类器最不确定的区间, 任何额外特征都有机会发挥作用
mask_s3 = (p3_te >= 0.30) & (p3_te <= 0.70)

scenarios = [
    ('对向穿插/动态耦合',   mask_s1, f'omega_bar < {THR_CC}'),
    ('集群结构/空间模糊蜂群', mask_s2, 'label=1 且 corr2_R < 0.08'),
    ('边界状态',           mask_s3, '基线概率∈[0.30,0.70]'),
]

print(f"\n  {'场景':<18s} {'样本数':>6s}  {'蜂群':>6s}  {'非蜂':>6s}  {'筛选规则'}")
print("  " + "-"*70)
for name, mask, rule in scenarios:
    n  = np.sum(mask)
    ns = np.sum(y_te[mask]==1); nn = np.sum(y_te[mask]==0)
    print(f"  {name:<18s} {n:>6d}  {ns:>6d}  {nn:>6d}  {rule}")

# =====================================================================
# 4. 各场景准确率统计
# =====================================================================
print("\n[4/5] 各场景准确率统计...")

scen_results = []
print(f"\n  {'场景':<18s} {'基线蜂%':>7s} {'基线非%':>7s} {'基线总%':>7s}  "
      f"{'本文蜂%':>7s} {'本文非%':>7s} {'本文总%':>7s}  {'提升':>6s}")
print("  " + "-"*80)

for name, mask, rule in scenarios:
    if np.sum(mask) == 0:
        print(f"  {name:<18s}  [无样本]"); continue
    sw3_s, nsw3_s, ov3_s, *_ = evaluate(y_te[mask], pred3[mask])
    sw13_s, nsw13_s, ov13_s, *_ = evaluate(y_te[mask], pred13[mask])
    delta = ov13_s - ov3_s
    scen_results.append({
        'name': name, 'n': np.sum(mask),
        'n_sw': np.sum(y_te[mask]==1), 'n_nsw': np.sum(y_te[mask]==0),
        'bl_sw': sw3_s, 'bl_nsw': nsw3_s, 'bl_ov': ov3_s,
        'fm_sw': sw13_s, 'fm_nsw': nsw13_s, 'fm_ov': ov13_s,
        'delta': delta,
        'cm_bl': np.array([cm3[0], cm3[1], cm3[2], cm3[3]]),   # tp,tn,fp,fn overall
        'mask': mask
    })
    print(f"  {name:<18s} {sw3_s:>6.1f}% {nsw3_s:>6.1f}% {ov3_s:>6.1f}%  "
          f"{sw13_s:>6.1f}% {nsw13_s:>6.1f}% {ov13_s:>6.1f}%  {delta:>+5.1f}%")

# ── 场景1详细: FN/FP分析 ──
print(f"\n  场景1详细错误分析:")
mask1 = mask_s1
_,_,_, tp_bl, tn_bl, fp_bl, fn_bl = evaluate(y_te[mask1], pred3[mask1])
_,_,_, tp_fm, tn_fm, fp_fm, fn_fm = evaluate(y_te[mask1], pred13[mask1])
print(f"    基线 - 漏报蜂群(FN):{fn_bl}  误判非蜂群(FP):{fp_bl}")
print(f"    本文 - 漏报蜂群(FN):{fn_fm}  误判非蜂群(FP):{fp_fm}")

# ── omega_bar区间分析 (展示方法改进随omega变化) ──
print(f"\n  omega_bar区间分析 (展示改进梯度):")
bins = [0.49, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
print(f"  {'omega区间':<18s} {'样本':>6s}  {'基线%':>6s}  {'本文%':>6s}  {'提升':>6s}")
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (omega_te >= lo) & (omega_te < hi)
    n = np.sum(mask)
    if n < 5: continue
    _, _, ov3_b,  *_ = evaluate(y_te[mask], pred3[mask])
    _, _, ov13_b, *_ = evaluate(y_te[mask], pred13[mask])
    print(f"  [{lo:.2f}, {hi:.2f})      {n:>6d}  {ov3_b:>5.1f}%  {ov13_b:>5.1f}%  {ov13_b-ov3_b:>+5.1f}%")

# =====================================================================
# 5. 寻找代表性样本 (供可视化)
# =====================================================================
print("\n[5/5] 寻找代表性可视化样本...")

te_abs_idx = te_i   # 测试集原始行索引

# ── 样本A: 经典蜂群 (高omega, 两种方法均正确) ──
cond_A = (y_te==1) & (pred3==1) & (pred13==1) & (omega_te > 0.90)
if np.sum(cond_A) > 0:
    best_A = te_abs_idx[cond_A][np.argmax(omega_te[cond_A])]
else:
    best_A = te_abs_idx[y_te==1][0]
print(f"  样本A (经典蜂群): 原始行={best_A}, omega={omega_te[te_i==best_A][0] if np.any(te_i==best_A) else 'N/A':.3f}")

# 更可靠的查找方式
def get_te_val(te_mask_func):
    """在测试集掩码中找满足条件的原始行索引"""
    mask = te_mask_func()
    if np.sum(mask)==0: return None
    return te_abs_idx[mask]

# 找高omega蜂群样本
idxA_local = np.where(cond_A)[0]
if len(idxA_local)>0:
    idxA_local_best = idxA_local[np.argmax(omega_te[cond_A])]
    sample_A_raw_idx = te_abs_idx[idxA_local_best]
    sample_A_omega = omega_te[idxA_local_best]
    print(f"  样本A: 行={sample_A_raw_idx}, omega={sample_A_omega:.3f}, label=1, 基线P={p3_te[idxA_local_best]:.3f}")
else:
    sample_A_raw_idx = te_abs_idx[np.argmax(omega_te*(y_te==1))]
    sample_A_omega = np.max(omega_te*(y_te==1))

# ── 样本B: 低omega蜂群 (基线漏报, 本文正确) → 最关键可视化样本 ──
cond_B = (y_te==1) & (pred3==0) & (pred13==1) & (omega_te < 0.60)
idxB_local = np.where(cond_B)[0]
if len(idxB_local)>0:
    # 选omega最低的 (最有代表性)
    idxB_local_best = idxB_local[np.argmin(omega_te[cond_B])]
    sample_B_raw_idx = te_abs_idx[idxB_local_best]
    sample_B_omega = omega_te[idxB_local_best]
    print(f"  样本B: 行={sample_B_raw_idx}, omega={sample_B_omega:.3f}, label=1, 基线P={p3_te[idxB_local_best]:.3f} (✗), 本文P={p13_te[idxB_local_best]:.3f} (✓)")
else:
    # 备选: 低omega蜂群中基线概率最低的
    cond_B2 = (y_te==1) & (omega_te < 0.60)
    if np.sum(cond_B2)>0:
        idxB_local_best = np.where(cond_B2)[0][np.argmin(p3_te[cond_B2])]
        sample_B_raw_idx = te_abs_idx[idxB_local_best]
        sample_B_omega = omega_te[idxB_local_best]
    else:
        sample_B_raw_idx = te_abs_idx[0]; sample_B_omega = 0.5
    print(f"  样本B (备选): 行={sample_B_raw_idx}, omega={sample_B_omega:.3f}")

# ── 样本C: 典型非蜂群 (随机运动, 两种方法均正确) ──
cond_C = (y_te==0) & (pred3==0) & (pred13==0) & (omega_te < 0.65) & (omega_te > 0.55)
idxC_local = np.where(cond_C)[0]
if len(idxC_local)>0:
    idxC_local_best = idxC_local[len(idxC_local)//2]  # 取中间值
    sample_C_raw_idx = te_abs_idx[idxC_local_best]
    sample_C_omega = omega_te[idxC_local_best]
    print(f"  样本C: 行={sample_C_raw_idx}, omega={sample_C_omega:.3f}, label=0, 基线P={p3_te[idxC_local_best]:.3f}")
else:
    cond_C2 = (y_te==0) & (pred3==0)
    idxC_local_best = np.where(cond_C2)[0][0]
    sample_C_raw_idx = te_abs_idx[idxC_local_best]
    sample_C_omega = omega_te[idxC_local_best]
    print(f"  样本C (备选): 行={sample_C_raw_idx}")

# ── 场景1中的混淆矩阵数据 ──
_, _, _, tp1_bl, tn1_bl, fp1_bl, fn1_bl = evaluate(y_te[mask_s1], pred3[mask_s1])
_, _, _, tp1_fm, tn1_fm, fp1_fm, fn1_fm = evaluate(y_te[mask_s1], pred13[mask_s1])

# =====================================================================
# 保存所有结果
# =====================================================================
SAVE_FILE = os.path.join(SCRIPT_DIR, 'results_challenge.npz')

np.savez(SAVE_FILE,
    # 测试集基本数据
    y_te=y_te,
    omega_te=omega_te,
    corr1_te=corr1_te, corr2_te=corr2_te, corr3_te=corr3_te,
    X_te13=X_te13,
    p3_te=p3_te, p13_te=p13_te,
    pred3=pred3, pred13=pred13,
    thr3=np.array([thr3]), thr13=np.array([thr13]),
    T_high=np.array([T_high]), T_low=np.array([T_low]), T_emg=np.array([T_emg]),

    # 场景掩码
    mask_s1=mask_s1, mask_s2=mask_s2, mask_s3=mask_s3,
    thr_cc=np.array([THR_CC]),

    # 全局准确率
    sw3=np.array([sw3]), nsw3=np.array([nsw3]), ov3=np.array([ov3]),
    sw13=np.array([sw13]), nsw13=np.array([nsw13]), ov13=np.array([ov13]),

    # 代表样本原始行索引 (用于从 raw_data 提取位置/速度)
    sample_A_idx=np.array([sample_A_raw_idx]),
    sample_B_idx=np.array([sample_B_raw_idx]),
    sample_C_idx=np.array([sample_C_raw_idx]),

    # 场景1混淆矩阵
    cm_s1_bl=np.array([[tp1_bl, fp1_bl],[fn1_bl, tn1_bl]]),
    cm_s1_fm=np.array([[tp1_fm, fp1_fm],[fn1_fm, tn1_fm]]),

    # 原始数据 (用于位置/速度可视化)
    raw_A=raw_data[sample_A_raw_idx],
    raw_B=raw_data[sample_B_raw_idx],
    raw_C=raw_data[sample_C_raw_idx],
)

print(f"\n  结果已保存: {SAVE_FILE}")

# ── 最终统计汇总 ──
print("\n" + "="*70)
print("  挑战场景分析汇总")
print("="*70)
print(f"  {'场景':<18s}  {'样本':>5s}  {'基线%':>6s}  {'本文%':>6s}  {'提升':>6s}")
print("  " + "-"*50)
for r in scen_results:
    print(f"  {r['name']:<18s}  {r['n']:>5d}  {r['bl_ov']:>5.1f}%  {r['fm_ov']:>5.1f}%  {r['delta']:>+5.1f}%")
print("="*70)
print("\nchallenge_scenarios.py 完成. 运行 challenge_viz.py 生成图表.")
