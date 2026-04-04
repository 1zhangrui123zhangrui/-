"""
Table5_Full_Simulation.py
==============================================================================
表5 完整仿真 (综合修复版)

修复内容:
  1. 变体1-3: 用验证集最优阈值替代固定0.45
     - 变体1/2: 在[min,max]范围全局搜索最优单阈值
     - 变体3: 双阈值准则 (corr3 <= 0.25 OR corr3 >= 0.45)
  2. 阈值搜索范围扩展至 [0.03, 0.97] 以防错过低阈值最优点
  3. +SCDAM行: 使用 [corr1, corr2, corr3, omega_bar] 4维特征
     - omega_bar = 全局速度对齐均值, 是SCDAM框架的核心输出
     - 对编队飞行(flocking)蜂群: omega_bar ≈ 0.8-1.0 (高速度一致性)
     - 对非蜂群(随机运动): omega_bar ≈ 0.5 (方向随机)
     - 解决了3D特征对flocking蜂群区分力不足的根本问题
  4. 迟滞决策: T_high/T_low 参数基于验证集自适应设置

关键逻辑说明:
  - 3D特征的局限: corr2对flocking蜂群偏低(空间分布均匀→修正距离偏差小→相关低)
  - omega_bar的作用: 直接捕获速度方向一致性, 弥补空间特征的盲区
  - 13D特征: spectral(9D) + omega_bar + SCDAM-corr(3D) = 全面描述时空规律性
==============================================================================
"""

import sys, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), 'csv')
sys.path.insert(0, SCRIPT_DIR)

from paper2_algorithms import extract_features_single, apply_hysteresis_static


# =====================================================================
# 工具函数
# =====================================================================

def evaluate(y_true, y_pred):
    """返回 (蜂群召回%, 非蜂群召回%, 综合准确率%)"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sw  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    nsw = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    ov  = (tp + tn) / len(y_true) * 100
    return sw, nsw, ov


def find_best_threshold_prob(y_true, probs):
    """
    在概率输出上搜索最优阈值 (最大化综合准确率)
    搜索范围扩展至 [0.03, 0.97] 以覆盖低概率最优点
    """
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        _, _, ov = evaluate(y_true, (probs >= thr).astype(int))
        if ov > best_ov:
            best_ov = ov
            best_thr = thr
    return best_thr


def find_best_threshold_feature(y_true, values, n_steps=600):
    """
    在特征值范围上搜索最优阈值 (用于变体单阈值决策)
    """
    v_min, v_max = float(values.min()), float(values.max())
    best_thr, best_ov = (v_min + v_max) / 2, 0.0
    for thr in np.linspace(v_min, v_max, n_steps):
        _, _, ov = evaluate(y_true, (values >= thr).astype(int))
        if ov > best_ov:
            best_ov = ov
            best_thr = thr
    return best_thr


# =====================================================================
# 1. 加载数据
# =====================================================================
print("=" * 72)
print("  Table 5 Full Simulation (Comprehensive Fix)")
print("=" * 72)

print("\n[1/7] Loading data...")
df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'),
                     low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
data   = df_raw.iloc[:, :-1].values.astype(float)
labels = df_raw.iloc[:, -1].values.astype(int)

# 加载预提取的3D基线特征 (MATLAB Table5_Step1_Feature_Extraction.m 输出)
bl_df = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n = min(len(labels), len(bl_df))
data   = data[:min_n]
labels = labels[:min_n]
features_baseline = bl_df[['corr1', 'corr2', 'corr3']].values[:min_n]
num_samples = min_n

print(f"  Samples: {num_samples} | Swarm: {np.sum(labels==1)} | Non-swarm: {np.sum(labels==0)}")

# =====================================================================
# 2. 数据划分 7:1.5:1.5
# =====================================================================
print("\n[2/7] Data split (7:1.5:1.5)...")
rng = np.random.RandomState(42)
idx  = rng.permutation(num_samples)
n_tr = int(num_samples * 0.70)
n_va = int(num_samples * 0.15)
train_idx = idx[:n_tr]
val_idx   = idx[n_tr:n_tr + n_va]
test_idx  = idx[n_tr + n_va:]
y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
print(f"  Train: {n_tr} | Val: {n_va} | Test: {len(test_idx)}")

# =====================================================================
# 3. 变体1-3: 最优阈值决策 (在验证集上搜索, 在测试集上评估)
# =====================================================================
print("\n[3/7] Variants 1-3 (optimal threshold from validation set)...")
f_val_bl  = features_baseline[val_idx]
f_test_bl = features_baseline[test_idx]

variant_results = []

# 变体1 (corr1): 单阈值, 在验证集最优
thr_v1 = find_best_threshold_feature(y_val, f_val_bl[:, 0])
r_v1 = evaluate(y_test, (f_test_bl[:, 0] >= thr_v1).astype(int))
variant_results.append(r_v1)
print(f"  Variant1 (thr={thr_v1:.4f}): {r_v1[0]:.1f}% / {r_v1[1]:.1f}% / {r_v1[2]:.1f}%")

# 变体2 (corr2): 单阈值, 在验证集最优
thr_v2 = find_best_threshold_feature(y_val, f_val_bl[:, 1])
r_v2 = evaluate(y_test, (f_test_bl[:, 1] >= thr_v2).astype(int))
variant_results.append(r_v2)
print(f"  Variant2 (thr={thr_v2:.4f}): {r_v2[0]:.1f}% / {r_v2[1]:.1f}% / {r_v2[2]:.1f}%")

# 变体3 (corr3): 双阈值准则 (论文1: corr3 <= 0.25 OR corr3 >= 0.45 判为蜂群)
s3_test = f_test_bl[:, 2]
pred_v3 = ((s3_test <= 0.25) | (s3_test >= 0.45)).astype(int)
r_v3 = evaluate(y_test, pred_v3)
variant_results.append(r_v3)
print(f"  Variant3 (bimodal <=0.25 or >=0.45): {r_v3[0]:.1f}% / {r_v3[1]:.1f}% / {r_v3[2]:.1f}%")

# =====================================================================
# 4. CatBoost-3D 基准 (论文1: depth=6, iter=200)
# =====================================================================
print("\n[4/7] CatBoost-3D (Paper-1 baseline)...")
X_tr_3d = features_baseline[train_idx]
X_va_3d = features_baseline[val_idx]
X_te_3d = features_baseline[test_idx]

model_3d = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.05,
    loss_function='Logloss', random_seed=42, verbose=0)
model_3d.fit(X_tr_3d, y_train)

p3_val  = model_3d.predict_proba(X_va_3d)[:, 1]
p3_test = model_3d.predict_proba(X_te_3d)[:, 1]
thr_3d  = find_best_threshold_prob(y_val, p3_val)
r4      = evaluate(y_test, (p3_test >= thr_3d).astype(int))
print(f"  CatBoost-3D: {r4[0]:.1f}% / {r4[1]:.1f}% / {r4[2]:.1f}%  (thr={thr_3d:.3f})")

# =====================================================================
# 5-6. 批量提取 SCDAM特征 (omega_bar) 和 13D 完整特征
# =====================================================================
print("\n[5-6/7] Extracting SCDAM features (omega_bar) + 13D features...")
m   = 8
lam = 1.0
fpd = 12   # fields_per_drone


def extract_all(indices, tag=""):
    """
    返回:
      omega_arr : (n,)   全局速度对齐均值 (SCDAM核心输出)
      feat_13d  : (n,13) 完整13维特征
    """
    n = len(indices)
    omega_arr = np.zeros(n)
    feat_13d  = np.zeros((n, 13))
    t0 = time.time()
    for i, ix in enumerate(indices):
        row = data[ix]
        X  = row[0::fpd];  Y  = row[1::fpd]
        VX = row[2::fpd];  VY = row[3::fpd]
        res = extract_features_single(X, Y, VX, VY, m, lam)
        omega_arr[i] = res['omega_bar']
        feat_13d[i]  = res['feat_13d']
        if (i + 1) % 2000 == 0:
            print(f"    {tag} {i+1}/{n}  ({time.time()-t0:.0f}s)")
    print(f"    {tag} done  ({time.time()-t0:.0f}s)")
    return omega_arr, feat_13d


omega_train, X_tr_13 = extract_all(train_idx, "Train")
omega_val,   X_va_13 = extract_all(val_idx,   "Val")
omega_test,  X_te_13 = extract_all(test_idx,  "Test")

# =====================================================================
# 5. +SCDAM (4D = 原始3D空间特征 + omega_bar)
#
#    核心逻辑:
#    SCDAM框架通过计算omega矩阵得到omega_bar (全局速度对齐度):
#      - 编队飞行蜂群: 所有无人机同向运动 -> omega_bar ≈ 0.8~1.0
#      - 非蜂群(随机运动): 方向随机 -> omega_bar ≈ 0.5
#    这弥补了纯空间特征对flocking蜂群(空间不规则但速度一致)的识别盲区
# =====================================================================
print("\n[5/7] +SCDAM (4D: 3D-spatial + omega_bar)...")

X_tr_4d = np.hstack([X_tr_3d, omega_train.reshape(-1, 1)])
X_va_4d = np.hstack([X_va_3d, omega_val.reshape(-1, 1)])
X_te_4d = np.hstack([X_te_3d, omega_test.reshape(-1, 1)])

model_sc = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.05,
    loss_function='Logloss', random_seed=42, verbose=0)
model_sc.fit(X_tr_4d, y_train)

psc_val  = model_sc.predict_proba(X_va_4d)[:, 1]
psc_test = model_sc.predict_proba(X_te_4d)[:, 1]
thr_sc   = find_best_threshold_prob(y_val, psc_val)
r5       = evaluate(y_test, (psc_test >= thr_sc).astype(int))
print(f"  +SCDAM(4D): {r5[0]:.1f}% / {r5[1]:.1f}% / {r5[2]:.1f}%  (thr={thr_sc:.3f})")

# =====================================================================
# 6. +频谱特征 13D (无迟滞)
#    feat_13d = [Ed,Pd,Hd, Ef,Pf,Hf, Eb,Pb,Hb, omega_bar, corr1_M, corr2_M, corr3_M]
#    depth=8, iter=300 (论文2 Section 7.1)
# =====================================================================
print("\n[6/7] +Spectral Features 13D (no hysteresis)...")

model_13 = CatBoostClassifier(
    iterations=300, depth=8, learning_rate=0.03,
    l2_leaf_reg=5, subsample=0.8,
    loss_function='Logloss', random_seed=42, verbose=0)
model_13.fit(X_tr_13, y_train,
             eval_set=Pool(X_va_13, y_val),
             early_stopping_rounds=50, verbose=0)

p13_val  = model_13.predict_proba(X_va_13)[:, 1]
p13_test = model_13.predict_proba(X_te_13)[:, 1]
thr_13   = find_best_threshold_prob(y_val, p13_val)
pred_13  = (p13_test >= thr_13).astype(int)
r6       = evaluate(y_test, pred_13)
print(f"  +13D: {r6[0]:.1f}% / {r6[1]:.1f}% / {r6[2]:.1f}%  (thr={thr_13:.3f})")

# =====================================================================
# 7. 本文方法 (完整) = 13D + 迟滞决策
#    T_high/T_low 以验证集最优阈值为中心自适应设置
# =====================================================================
print("\n[7/7] Full Method (13D + Hysteresis)...")

# 以验证集最优阈值为基准设置迟滞区间
T_high = min(thr_13 + 0.05, 0.90)
T_low  = max(thr_13 - 0.10, 0.10)
T_emg  = max(thr_13 - 0.30, 0.05)

pred_hyst = apply_hysteresis_static(p13_test, T_high=T_high, T_low=T_low, T_emg=T_emg)
r7 = evaluate(y_test, pred_hyst)
print(f"  Full(T_high={T_high:.3f}, T_low={T_low:.3f}): {r7[0]:.1f}% / {r7[1]:.1f}% / {r7[2]:.1f}%")

# =====================================================================
# 8. 汇总输出
# =====================================================================
all_rows = [
    ('Variant1 (baseline) [1]',       variant_results[0]),
    ('Variant2 (baseline) [1]',       variant_results[1]),
    ('Variant3 (baseline) [1]',       variant_results[2]),
    ('CatBoost-3D (baseline) [1]',    r4),
    ('+SCDAM (4D: +omega_bar)',        r5),
    ('+Spectral Features 13D',        r6),
    ('Full Method (13D+Hysteresis)',   r7),
]

print("\n")
print("+" + "-"*72 + "+")
print("|  Table 5: Static Classification Accuracy on Test Set (%)" + " "*15 + "|")
print("+" + "-"*38 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*12 + "+")
print(f"| {'Method':<36s} | {'Swarm':>8s} | {'Non-sw':>8s} | {'Overall':>10s} |")
print("+" + "-"*38 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*12 + "+")
for name, (sw, nsw, ov) in all_rows:
    print(f"| {name:<36s} | {sw:>7.1f}% | {nsw:>7.1f}% | {ov:>9.1f}%  |")
print("+" + "-"*38 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*12 + "+")

print(f"\nIncremental gain:")
print(f"  3D -> +SCDAM:  {r4[2]:.1f}% -> {r5[2]:.1f}%  ({r5[2]-r4[2]:+.1f}%)")
print(f"  +SCDAM -> 13D: {r5[2]:.1f}% -> {r6[2]:.1f}%  ({r6[2]-r5[2]:+.1f}%)")
print(f"  13D -> Full:   {r6[2]:.1f}% -> {r7[2]:.1f}%  ({r7[2]-r6[2]:+.1f}%)")

# 特征重要性
fn = ['Ed','Pd','Hd','Ef','Pf','Hf','Eb','Pb','Hb','omega','corr1_M','corr2_M','corr3_M']
imp = model_13.get_feature_importance()
print(f"\nFeature importance (13D model):")
for i in np.argsort(imp)[::-1]:
    bar = '#' * int(imp[i] / 1.5)
    print(f"  {fn[i]:<8s}: {imp[i]:5.1f}%  {bar}")

# omega_bar 分布验证
print(f"\nomega_bar distribution (key discriminator):")
om_tr = omega_train
ym_tr = y_train
print(f"  Swarm     omega_bar: mean={np.mean(om_tr[ym_tr==1]):.4f}  "
      f"std={np.std(om_tr[ym_tr==1]):.4f}  "
      f"min={np.min(om_tr[ym_tr==1]):.4f}  max={np.max(om_tr[ym_tr==1]):.4f}")
print(f"  Non-swarm omega_bar: mean={np.mean(om_tr[ym_tr==0]):.4f}  "
      f"std={np.std(om_tr[ym_tr==0]):.4f}  "
      f"min={np.min(om_tr[ym_tr==0]):.4f}  max={np.max(om_tr[ym_tr==0]):.4f}")

# 保存模型和结果
model_3d.save_model(os.path.join(SCRIPT_DIR, "model_3d.cbm"))
model_sc.save_model(os.path.join(SCRIPT_DIR, "model_scdam.cbm"))
model_13.save_model(os.path.join(SCRIPT_DIR, "model_13d.cbm"))
np.savez(os.path.join(SCRIPT_DIR, 'table5_results.npz'),
         prob_13d_test=p13_test, y_test=y_test,
         results=np.array([list(r4), list(r5), list(r6), list(r7)]))

print(f"\nSimulation complete. Models saved.")
