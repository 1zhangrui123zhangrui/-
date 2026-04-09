# -*- coding: utf-8 -*-
"""
section_7_3_feature_importance.py
==============================================================================
第 7.3 节补充  特征重要性分析  — 回应审稿人意见 2.7
==============================================================================

回应三条审稿意见：

① 没有 CatBoost 内置特征重要性得分排序图
   → 使用 PredictionValuesChange（模型内置）和 LossFunctionChange（数据驱动）
     两种度量标准，对 13 个特征排序，绘制水平条形图

② ω̄ 被描述为"误报抑制的关键"但无可解释性分析支撑
   → 分别计算全测试集和对向穿插子集（omega_bar < 0.60）的特征重要性
   → 展示 ω̄ 在对向穿插场景下重要性是否更突出

③ SHAP 值（如有 shap 库则自动启用）
   → 全局平均 |SHAP| 排序图
   → 对向穿插子集平均 |SHAP| 排序图
   → 如 shap 未安装则回退到 LossFunctionChange

13 维特征布局：
  [0] Ed  [1] Pd  [2] Hd  — 距离谱  {能量, 峰均比, 熵}
  [3] Ef  [4] Pf  [5] Hf  — 微分谱  ← Hf 通常最重要
  [6] Eb  [7] Pb  [8] Hb  — 偏差谱
  [9]  ω̄                  — 全局速度对齐均值（核心创新）
  [10] corr1_M  [11] corr2_M  [12] corr3_M  — SCDAM 自相关变体

输出文件：
  figures/fig_feature_importance_global.png   — 全测试集特征重要性（双度量）
  figures/fig_feature_importance_scene1.png   — 对向穿插子集特征重要性
  figures/fig_feature_importance_shap.png     — SHAP 分析（如可用）
  python/results_feature_importance.npz       — 完整数值结果
==============================================================================
"""

import sys, os
import numpy as np
import pandas as pd
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
plt.rcParams.update({'font.family': _chosen, 'axes.unicode_minus': False, 'font.size': 11})

SEED        = 42
OMEGA_CC_THR = 0.60   # 对向穿插/动态耦合场景阈值

# =====================================================================
# 特征名称（含物理含义）
# =====================================================================
FEAT_NAMES_CN = [
    'E_d  距离谱能量',
    'P_d  距离谱峰均比',
    'H_d  距离谱熵',
    'E_f  微分谱能量',
    'P_f  微分谱峰均比',
    'H_f  微分谱熵★',       # ★ 通常最重要
    'E_b  偏差谱能量',
    'P_b  偏差谱峰均比',
    'H_b  偏差谱熵',
    'ω̄   全局速度对齐★',    # ★ SCDAM核心创新
    'corr1_M  SCDAM变体1',
    'corr2_M  SCDAM变体2',
    'corr3_M  SCDAM变体3',
]
FEAT_NAMES_SHORT = [
    'E_d', 'P_d', 'H_d',
    'E_f', 'P_f', 'H_f★',
    'E_b', 'P_b', 'H_b',
    'ω̄★',
    'corr1_M', 'corr2_M', 'corr3_M',
]
N_FEAT = 13

# 特征分组颜色（按谱类型）
FEAT_COLORS = (
    ['#4C72B0'] * 3 +   # 距离谱 (蓝)
    ['#DD8452'] * 3 +   # 微分谱 (橙)
    ['#55A868'] * 3 +   # 偏差谱 (绿)
    ['#C44E52']     +   # ω̄ (红，核心)
    ['#8172B2'] * 3     # SCDAM相关 (紫)
)


# =====================================================================
# 工具函数
# =====================================================================

def evaluate(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return (tp + tn) / len(y_true) * 100


def find_best_thr(y_true, probs):
    best_thr, best_ov = 0.5, 0.0
    for thr in np.arange(0.03, 0.97, 0.005):
        ov = evaluate(y_true, (probs >= thr).astype(int))
        if ov > best_ov: best_ov = ov; best_thr = thr
    return best_thr


def plot_importance_bar(importances, title, save_path,
                        highlight_idx=None, ref_importances=None,
                        ref_label='参考', extra_text=''):
    """
    绘制特征重要性水平条形图（降序排列）。
    highlight_idx: 高亮特定特征（如 ω̄ 和 H_f）
    ref_importances: 可选的参考重要性（同图对比，如全集 vs 子集）
    """
    order = np.argsort(importances)[::-1]   # 从大到小
    imp_sorted   = importances[order]
    names_sorted = [FEAT_NAMES_CN[i] for i in order]
    colors_sorted= [FEAT_COLORS[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(N_FEAT)

    bars = ax.barh(y_pos, imp_sorted, color=colors_sorted,
                   alpha=0.80, edgecolor='white', height=0.65, zorder=3)

    # 高亮特征
    if highlight_idx is not None:
        for rank, orig_idx in enumerate(order):
            if orig_idx in highlight_idx:
                bars[rank].set_edgecolor('gold')
                bars[rank].set_linewidth(2.5)
                bars[rank].set_alpha(1.0)

    # 叠加参考值（虚线框）
    if ref_importances is not None:
        ref_sorted = ref_importances[order]
        ax.barh(y_pos, ref_sorted, color='none',
                edgecolor='black', height=0.65,
                linewidth=1.2, linestyle='--', zorder=4,
                label=ref_label)

    # 数值标注
    max_val = imp_sorted[0]
    for rank, (bar, val) in enumerate(zip(bars, imp_sorted)):
        ax.text(val + max_val * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}',
                va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=9.5)
    ax.set_xlabel('特征重要性得分', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max_val * 1.22)
    ax.invert_yaxis()
    ax.grid(True, axis='x', ls=':', alpha=0.5)

    # 图例（按谱类型）
    legend_items = [
        plt.Rectangle((0,0), 1, 1, color='#4C72B0', alpha=0.8, label='距离谱  Φ_dist'),
        plt.Rectangle((0,0), 1, 1, color='#DD8452', alpha=0.8, label='微分谱  Φ_diff'),
        plt.Rectangle((0,0), 1, 1, color='#55A868', alpha=0.8, label='偏差谱  Φ_bias'),
        plt.Rectangle((0,0), 1, 1, color='#C44E52', alpha=0.8, label='ω̄ (SCDAM核心)'),
        plt.Rectangle((0,0), 1, 1, color='#8172B2', alpha=0.8, label='SCDAM相关变体'),
    ]
    if ref_importances is not None:
        legend_items.append(
            plt.Rectangle((0,0), 1, 1, fill=False,
                           edgecolor='black', lw=1.5, ls='--', label=ref_label))
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    if extra_text:
        fig.text(0.01, -0.02, extra_text, fontsize=8, color='gray', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")


# =====================================================================
# 1. 加载数据与模型
# =====================================================================
print("=" * 70)
print("  Section 7.3 补充  特征重要性分析（回应审稿人意见 2.7）")
print("=" * 70)

print("\n[1/4] 加载数据、特征与模型...")

df_raw = pd.read_csv(os.path.join(CSV_DIR, 'Swarm_Behaviour_Data.csv'), low_memory=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
labels = df_raw.iloc[:, -1].values.astype(int)
bl_df  = pd.read_csv(os.path.join(CSV_DIR, 'catboost_features_table5.csv'))
min_n  = min(len(labels), len(bl_df))
labels = labels[:min_n]

# 与论文相同的 70/15/15 划分（seed=42）
rng       = np.random.RandomState(SEED)
idx_all   = rng.permutation(min_n)
n_tr      = int(min_n * 0.70);  n_va = int(min_n * 0.15)
train_idx = idx_all[:n_tr];     val_idx  = idx_all[n_tr:n_tr+n_va]
test_idx  = idx_all[n_tr+n_va:]

CACHE = os.path.join(SCRIPT_DIR, 'ablation_features_cache.npz')
if not os.path.exists(CACHE):
    print("  [错误] 缺少 ablation_features_cache.npz，请先运行 ablation_7_3.py")
    sys.exit(1)

cache   = np.load(CACHE)
X_tr_13 = cache['X_tr_13']
X_va_13 = cache['X_va_13']
X_te_13 = cache['X_te_13']
y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]

# omega_bar（特征索引 9）用于定义对向穿插子集（λ 无关）
omega_bar_te = X_te_13[:, 9]
mask_cc      = omega_bar_te < OMEGA_CC_THR
n_cc         = int(np.sum(mask_cc))
X_te_cc      = X_te_13[mask_cc]
y_te_cc      = y_te[mask_cc]

print(f"  全测试集:      {len(y_te)} 样本")
print(f"  对向穿插子集:  {n_cc} 样本 (omega_bar < {OMEGA_CC_THR})")
print(f"    蜂群={np.sum(y_te_cc==1)}  非蜂群={np.sum(y_te_cc==0)}")

# 加载或训练模型
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_13d.cbm')
if os.path.exists(MODEL_PATH):
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"  已加载模型: {MODEL_PATH}")
else:
    print("  训练 CatBoost-13D...")
    model = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.03,
        l2_leaf_reg=5, subsample=0.8,
        loss_function='Logloss', random_seed=SEED, verbose=0)
    model.fit(X_tr_13, y_tr,
              eval_set=Pool(X_va_13, y_va),
              early_stopping_rounds=50, verbose=0)
    model.save_model(MODEL_PATH)
    print("  训练完成")

# 验证模型准确率（与论文数值对比）
thr = find_best_thr(y_va, model.predict_proba(X_va_13)[:, 1])
ov  = evaluate(y_te, (model.predict_proba(X_te_13)[:, 1] >= thr).astype(int))
ov_cc = evaluate(y_te_cc, (model.predict_proba(X_te_cc)[:, 1] >= thr).astype(int))
print(f"  模型准确率: 全测试集={ov:.2f}%  对向穿插子集={ov_cc:.2f}%")


# =====================================================================
# 2. CatBoost 内置特征重要性（两种度量）
# =====================================================================
print("\n[2/4] 计算 CatBoost 特征重要性...")

# ── PredictionValuesChange: 预测值变化量（模型内置，无需数据）
imp_pvc = model.get_feature_importance(type='PredictionValuesChange')

# ── LossFunctionChange: 损失函数变化量（数据驱动，更可靠）
#    全测试集
pool_te    = Pool(X_te_13, y_te)
imp_lfc_te = model.get_feature_importance(data=pool_te,
                                           type='LossFunctionChange')

# ── LossFunctionChange: 对向穿插子集
if n_cc >= 10:
    pool_cc    = Pool(X_te_cc, y_te_cc)
    imp_lfc_cc = model.get_feature_importance(data=pool_cc,
                                               type='LossFunctionChange')
else:
    imp_lfc_cc = np.zeros(N_FEAT)
    print("  [警告] 对向穿插子集样本不足 10 个，跳过 LossFunctionChange")

# 归一化到百分比（便于比较）
def normalize(x):
    s = x.sum()
    return x / s * 100 if s > 0 else x

imp_pvc_pct    = normalize(imp_pvc)
imp_lfc_te_pct = normalize(imp_lfc_te)
imp_lfc_cc_pct = normalize(imp_lfc_cc)

# ── 打印排名表格 ──────────────────────────────────────────────────────────
print(f"\n  特征重要性排名（%）")
print(f"  {'排名':>4s}  {'特征':>20s}  {'PVC%':>7s}  {'LFC全集%':>9s}  {'LFC子集%':>9s}")
print("  " + "-" * 60)
order_lfc = np.argsort(imp_lfc_te_pct)[::-1]
for rank, i in enumerate(order_lfc):
    marker = ' ★' if i in [5, 9] else ''    # Hf 和 ω̄
    print(f"  {rank+1:>4d}  {FEAT_NAMES_CN[i]:>20s}  "
          f"{imp_pvc_pct[i]:>7.2f}  {imp_lfc_te_pct[i]:>9.2f}  "
          f"{imp_lfc_cc_pct[i]:>9.2f}{marker}")

# ω̄ 在两个场景的排名对比
rank_omega_te = int(np.where(order_lfc == 9)[0][0]) + 1
order_lfc_cc  = np.argsort(imp_lfc_cc_pct)[::-1]
rank_omega_cc = int(np.where(order_lfc_cc == 9)[0][0]) + 1
print(f"\n  ω̄ 特征排名:  全测试集第 {rank_omega_te} 位  |  对向穿插子集第 {rank_omega_cc} 位")
if rank_omega_cc < rank_omega_te:
    print(f"  → 对向穿插场景下 ω̄ 重要性排名更靠前，支持[误报抑制核心]的论文论断")
else:
    print(f"  → ω̄ 在两场景排名相近，建议在论文中如实描述")


# =====================================================================
# 3. SHAP 分析（如 shap 已安装）
# =====================================================================
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("\n[3/4] SHAP 值计算（检测到 shap 库）...")
except ImportError:
    print("\n[3/4] shap 库未安装，跳过 SHAP 分析")
    print("      安装方式: pip install shap")
    print("      → 改用 LossFunctionChange 代替 SHAP 分析")

shap_vals_te = None
shap_vals_cc = None

if SHAP_AVAILABLE:
    try:
        # CatBoost SHAP（Tree SHAP，快速）
        shap_vals_te = model.get_feature_importance(
            data=pool_te, type='ShapValues')[:, :N_FEAT]   # (n, 13)
        if n_cc >= 10:
            shap_vals_cc = model.get_feature_importance(
                data=pool_cc, type='ShapValues')[:, :N_FEAT]
        print(f"  全测试集 SHAP 计算完成: shape={shap_vals_te.shape}")
    except Exception as e:
        print(f"  SHAP 计算失败: {e}，回退到 LossFunctionChange")
        shap_vals_te = None


# =====================================================================
# 4. 生成图表
# =====================================================================
print("\n[4/4] 生成特征重要性图表...")

highlight = [5, 9]   # H_f 和 ω̄（高亮显示）

# ── 图 1: 全测试集特征重要性（PVC + LFC 对比）───────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

def _subplot_bar(ax, importances, title, colors, highlight_idx):
    order  = np.argsort(importances)[::-1]
    y_pos  = np.arange(N_FEAT)
    imp_s  = importances[order]
    name_s = [FEAT_NAMES_SHORT[i] for i in order]
    col_s  = [colors[i] for i in order]
    bars = ax.barh(y_pos, imp_s, color=col_s, alpha=0.80,
                   edgecolor='white', height=0.65, zorder=3)
    for rank, (orig_idx, bar, val) in enumerate(zip(order, bars, imp_s)):
        if orig_idx in highlight_idx:
            bar.set_edgecolor('gold'); bar.set_linewidth(2.5); bar.set_alpha(1.0)
        ax.text(val + imp_s[0] * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_yticks(y_pos); ax.set_yticklabels(name_s, fontsize=9.5)
    ax.set_xlabel('重要性得分 (%)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, imp_s[0] * 1.25)
    ax.invert_yaxis()
    ax.grid(True, axis='x', ls=':', alpha=0.5)

_subplot_bar(ax1, imp_pvc_pct, 'PredictionValuesChange\n（模型内置，无需数据）',
             FEAT_COLORS, highlight)
_subplot_bar(ax2, imp_lfc_te_pct,
             f'LossFunctionChange\n（数据驱动，全测试集 n={len(y_te)}）',
             FEAT_COLORS, highlight)

# 图例
legend_items = [
    plt.Rectangle((0,0),1,1,color='#4C72B0',alpha=0.8,label='距离谱 Φ_dist'),
    plt.Rectangle((0,0),1,1,color='#DD8452',alpha=0.8,label='微分谱 Φ_diff'),
    plt.Rectangle((0,0),1,1,color='#55A868',alpha=0.8,label='偏差谱 Φ_bias'),
    plt.Rectangle((0,0),1,1,color='#C44E52',alpha=0.8,label='ω̄ (SCDAM核心)★'),
    plt.Rectangle((0,0),1,1,color='#8172B2',alpha=0.8,label='SCDAM 相关变体'),
]
ax2.legend(handles=legend_items, loc='lower right', fontsize=8.5)
fig1.suptitle('CatBoost-13D 特征重要性分析（全测试集）\n★ 金色边框 = 论文重点特征 H_f 和 ω̄',
              fontsize=13, y=1.02)
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, 'fig_feature_importance_global.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig1)
print(f"  图1已保存: {p1}")


# ── 图 2: 对向穿插子集特征重要性 vs 全集对比 ─────────────────────────────
if n_cc >= 10:
    fig2, ax = plt.subplots(figsize=(11, 6.5))
    y_pos = np.arange(N_FEAT)

    order_cc = np.argsort(imp_lfc_cc_pct)[::-1]
    imp_cc_s = imp_lfc_cc_pct[order_cc]
    imp_te_s = imp_lfc_te_pct[order_cc]  # 全集（按子集排序对齐）
    name_s   = [FEAT_NAMES_SHORT[i] for i in order_cc]
    col_s    = [FEAT_COLORS[i] for i in order_cc]

    bar_width = 0.38
    bars_cc = ax.barh(y_pos + bar_width/2, imp_cc_s, bar_width,
                      color=col_s, alpha=0.85, edgecolor='white',
                      label=f'对向穿插子集 (n={n_cc})', zorder=3)
    bars_te = ax.barh(y_pos - bar_width/2, imp_te_s, bar_width,
                      color=col_s, alpha=0.40, edgecolor='white',
                      label=f'全测试集 (n={len(y_te)})', zorder=2)

    # 高亮 ω̄
    omega_rank_cc = int(np.where(order_cc == 9)[0][0])
    bars_cc[omega_rank_cc].set_edgecolor('gold')
    bars_cc[omega_rank_cc].set_linewidth(3.0)
    bars_te[omega_rank_cc].set_edgecolor('gold')
    bars_te[omega_rank_cc].set_linewidth(2.5)

    # 标注 ω̄ 排名变化
    ax.text(max(imp_cc_s) * 0.6,
            omega_rank_cc + bar_width/2,
            f' ← ω̄ 全集第{rank_omega_te}位 vs 子集第{rank_omega_cc}位',
            va='center', fontsize=9.5, color='#C44E52', fontweight='bold')

    ax.set_yticks(y_pos); ax.set_yticklabels(name_s, fontsize=9.5)
    ax.set_xlabel('LossFunctionChange 重要性 (%)', fontsize=12)
    ax.set_title(
        f'对向穿插场景 vs 全集 — 特征重要性对比\n'
        f'（对向穿插: omega_bar < {OMEGA_CC_THR}, n={n_cc}; 全集: n={len(y_te)}）',
        fontsize=12)
    ax.set_xlim(0, max(imp_cc_s) * 1.45)
    ax.invert_yaxis()
    ax.grid(True, axis='x', ls=':', alpha=0.5)

    legend_items2 = [
        plt.Rectangle((0,0),1,1,color='gray',alpha=0.85,label=f'对向穿插子集 (深色, n={n_cc})'),
        plt.Rectangle((0,0),1,1,color='gray',alpha=0.40,label=f'全测试集 (浅色, n={len(y_te)})'),
        plt.Rectangle((0,0),1,1,color='#4C72B0',alpha=0.8,label='距离谱'),
        plt.Rectangle((0,0),1,1,color='#DD8452',alpha=0.8,label='微分谱'),
        plt.Rectangle((0,0),1,1,color='#55A868',alpha=0.8,label='偏差谱'),
        plt.Rectangle((0,0),1,1,color='#C44E52',alpha=0.8,label='ω̄★'),
        plt.Rectangle((0,0),1,1,color='#8172B2',alpha=0.8,label='SCDAM相关'),
    ]
    ax.legend(handles=legend_items2, loc='lower right', fontsize=9, ncol=2)
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, 'fig_feature_importance_scene1.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig2)
    print(f"  图2已保存: {p2}")
else:
    print("  图2跳过（对向穿插子集样本不足）")


# ── 图 3: SHAP 摘要图（如可用）或 LFC 双列对比 ──────────────────────────
if shap_vals_te is not None:
    # SHAP: 全集平均 |SHAP|
    mean_shap_te = np.mean(np.abs(shap_vals_te), axis=0)
    mean_shap_cc = np.mean(np.abs(shap_vals_cc), axis=0) if shap_vals_cc is not None else np.zeros(N_FEAT)
    mean_shap_te_pct = normalize(mean_shap_te)
    mean_shap_cc_pct = normalize(mean_shap_cc)

    plot_importance_bar(
        mean_shap_te_pct,
        title=f'SHAP 特征重要性 (平均 |SHAP|，全测试集 n={len(y_te)})',
        save_path=os.path.join(FIGURES_DIR, 'fig_feature_importance_shap.png'),
        highlight_idx=highlight,
        ref_importances=mean_shap_cc_pct if shap_vals_cc is not None else None,
        ref_label=f'对向穿插子集 (n={n_cc})',
        extra_text='SHAP (SHapley Additive exPlanations) 基于博弈论, 保证特征贡献之和等于预测值变化。'
    )

    print(f"\n  SHAP Top-3 特征（全测试集）:")
    shap_order = np.argsort(mean_shap_te_pct)[::-1]
    for i in range(3):
        idx = shap_order[i]
        print(f"    {i+1}. {FEAT_NAMES_CN[idx]}  平均|SHAP|={mean_shap_te_pct[idx]:.2f}%")
else:
    # 回退：LFC 全集 vs 子集并列图
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6.5))
    _subplot_bar(ax3a, imp_lfc_te_pct,
                 f'全测试集 LossFunctionChange\n(n={len(y_te)})', FEAT_COLORS, highlight)
    if n_cc >= 10:
        _subplot_bar(ax3b, imp_lfc_cc_pct,
                     f'对向穿插子集 LossFunctionChange\n(n={n_cc}, omega_bar<{OMEGA_CC_THR})',
                     FEAT_COLORS, highlight)
    else:
        ax3b.text(0.5, 0.5, f'对向穿插子集\n样本不足\nn={n_cc}',
                  ha='center', va='center', transform=ax3b.transAxes, fontsize=14)
    fig3.suptitle('特征重要性对比（全集 vs 对向穿插场景）\n★ 金色边框 = 论文重点特征',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, 'fig_feature_importance_shap.png')
    fig3.savefig(p3, dpi=150, bbox_inches='tight'); plt.close(fig3)
    print(f"  图3（SHAP替代）已保存: {p3}")


# =====================================================================
# 5. 保存完整数值结果
# =====================================================================
save_dict = dict(
    feat_names   = np.array(FEAT_NAMES_CN),
    imp_pvc      = imp_pvc,
    imp_pvc_pct  = imp_pvc_pct,
    imp_lfc_te   = imp_lfc_te,
    imp_lfc_te_pct = imp_lfc_te_pct,
    imp_lfc_cc   = imp_lfc_cc,
    imp_lfc_cc_pct = imp_lfc_cc_pct,
    rank_omega_te = np.array([rank_omega_te]),
    rank_omega_cc = np.array([rank_omega_cc]),
    n_test       = np.array([len(y_te)]),
    n_cc         = np.array([n_cc]),
    omega_cc_thr = np.array([OMEGA_CC_THR]),
)
if shap_vals_te is not None:
    save_dict['shap_mean_te_pct'] = normalize(np.mean(np.abs(shap_vals_te), axis=0))
    if shap_vals_cc is not None:
        save_dict['shap_mean_cc_pct'] = normalize(np.mean(np.abs(shap_vals_cc), axis=0))

np.savez(os.path.join(SCRIPT_DIR, 'results_feature_importance.npz'), **save_dict)

# =====================================================================
# 最终总结
# =====================================================================
print("\n" + "=" * 70)
print("  特征重要性分析结论")
print("=" * 70)

top3_lfc = np.argsort(imp_lfc_te_pct)[::-1][:3]
print(f"  LFC 全集 Top-3:")
for rank, i in enumerate(top3_lfc):
    print(f"    {rank+1}. {FEAT_NAMES_CN[i]}  = {imp_lfc_te_pct[i]:.2f}%")

print(f"\n  ω̄ (全局速度对齐) 重要性:")
print(f"    全测试集 LFC 排名:  第 {rank_omega_te} 位  ({imp_lfc_te_pct[9]:.2f}%)")
print(f"    对向穿插子集 LFC 排名: 第 {rank_omega_cc} 位  ({imp_lfc_cc_pct[9]:.2f}%)")

if rank_omega_cc < rank_omega_te:
    print(f"  ✓ ω̄ 在对向穿插子集排名更靠前（第{rank_omega_cc}位 < 第{rank_omega_te}位）")
    print(f"    直接支持论文 [omega_bar 在低速度一致性场景(对向穿插)中起关键作用] 的论断")
else:
    diff = imp_lfc_cc_pct[9] - imp_lfc_te_pct[9]
    print(f"  ℹ ω̄ 子集排名（{rank_omega_cc}）与全集排名（{rank_omega_te}）相近")
    print(f"    绝对得分变化: {diff:+.2f}%，建议在论文中客观描述")

print(f"\n  SHAP 分析: {'已完成（见 fig_feature_importance_shap.png）' if shap_vals_te is not None else '未完成（pip install shap 后重新运行）'}")
print("=" * 70)
print("\nsection_7_3_feature_importance.py 完成.")
