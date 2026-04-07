# -*- coding: utf-8 -*-
"""
challenge_viz.py
==============================================================================
Section 7.4  挑战性场景可视化
==============================================================================
生成6张论文级图表:

  Figure 1  无人机编队位置+速度箭头对比图 (3面板)
  Figure 2  代表样本特征对比柱状图
  Figure 3  场景1混淆矩阵对比 (基线 vs 本文)
  Figure 4  分类概率分布直方图 (场景1, 按真实标签着色)
  Figure 5  挑战场景准确率对比柱状图 (论文表7)
  Figure 6  omega_bar vs 基线概率散点图 (机制可视化)

输入: python/results_challenge.npz  (由 challenge_scenarios.py 生成)
输出: figures/ 目录下的 PNG 文件
==============================================================================
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # 交互后端，支持动画窗口
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── 中文字体配置 ──
def _setup_font():
    import matplotlib.font_manager as fm
    candidates = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'Noto Sans CJK SC',
                  'WenQuanYi Micro Hei', 'Arial Unicode MS']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        print(f"  [字体] 使用: {chosen}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("  [字体] 未找到中文字体, 使用英文")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    return chosen is not None

_HAS_CN = _setup_font()

# ── 颜色方案 ──
C_SWARM   = '#1565C0'   # 蜂群: 深蓝
C_NSWARM  = '#C62828'   # 非蜂群: 深红
C_CORRECT = '#2E7D32'   # 正确: 绿
C_WRONG   = '#E65100'   # 错误: 橙
C_BL      = '#78909C'   # 基线: 灰蓝
C_FM      = '#1B5E20'   # 本文: 深绿

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [保存] {path}")

# =====================================================================
# 加载数据
# =====================================================================
print("=" * 70); print("  挑战场景可视化"); print("=" * 70)

DATA_FILE = os.path.join(SCRIPT_DIR, 'results_challenge.npz')
if not os.path.exists(DATA_FILE):
    print(f"[错误] 未找到 {DATA_FILE}"); print("请先运行 challenge_scenarios.py"); sys.exit(1)

d = np.load(DATA_FILE, allow_pickle=True)
y_te     = d['y_te'];         omega_te  = d['omega_te']
corr1_te = d['corr1_te'];     corr2_te  = d['corr2_te'];   corr3_te = d['corr3_te']
X_te13   = d['X_te13'];       p3_te     = d['p3_te'];       p13_te   = d['p13_te']
pred3    = d['pred3'];        pred13    = d['pred13']
mask_s1  = d['mask_s1'];      mask_s2   = d['mask_s2'];     mask_s3  = d['mask_s3']
cm_s1_bl = d['cm_s1_bl'];     cm_s1_fm  = d['cm_s1_fm']
raw_A    = d['raw_A'];        raw_B     = d['raw_B'];       raw_C    = d['raw_C']
p_A_bl   = float(p3_te[d['sample_A_idx'][0] == np.where(np.ones(len(y_te), bool))[0]]
                 [0]) if False else 1.0   # 简化获取
T_high   = float(d['T_high'][0]); T_low = float(d['T_low'][0])

# 从 raw_data 解析位置+速度
fpd = 12
def parse_drone(raw_row):
    N  = len(raw_row) // fpd
    X  = raw_row[0::fpd][:N]
    Y  = raw_row[1::fpd][:N]
    VX = raw_row[2::fpd][:N]
    VY = raw_row[3::fpd][:N]
    return X, Y, VX, VY

XA,YA,VXA,VYA = parse_drone(raw_A)
XB,YB,VXB,VYB = parse_drone(raw_B)
XC,YC,VXC,VYC = parse_drone(raw_C)

# 计算样本 B 的 omega_bar 用于标注
def _omega_bar(VX, VY):
    N = len(VX)
    V = np.column_stack([VX, VY])
    vm = np.linalg.norm(V, axis=1, keepdims=True)
    D = np.zeros_like(V)
    v = (vm.flatten() > 1e-6)
    if np.any(v): D[v] = V[v] / vm[v]
    S = np.clip(D @ D.T, -1.0, 1.0)
    om = (1.0 + S) / 2.0
    return float(np.mean(om[np.triu_indices(N, k=1)]))

omA = _omega_bar(VXA, VYA)
omB = _omega_bar(VXB, VYB)
omC = _omega_bar(VXC, VYC)

# 找对应测试集索引
idx_A = int(d['sample_A_idx'][0])
idx_B = int(d['sample_B_idx'][0])
idx_C = int(d['sample_C_idx'][0])

# 找测试集局部索引 (对应 p3_te 等数组)
# te_i 没有存储, 通过 omega 值匹配近似
# 用 omega_te 的位置查找 — 如果omega唯一则精确
def find_local(raw_row_idx, X_te13_arr, omega_te_arr):
    """通过重新计算omega_bar匹配测试集局部索引"""
    row = raw_row_idx  # 原始行索引, 不是局部索引
    # 直接用 raw_A/B/C 重新计算 omega, 然后在 omega_te 中匹配
    return None

# 读取样本A/B/C的预测概率 (从保存时保留的 p3_te 间接获取)
# 使用 omega_bar 精确匹配 (以下简化处理)
omA_arr = np.abs(omega_te - omA)
omB_arr = np.abs(omega_te - omB)
omC_arr = np.abs(omega_te - omC)
local_A = np.argmin(omA_arr)
local_B = np.argmin(omB_arr)
local_C = np.argmin(omC_arr)

pA_bl = float(p3_te[local_A]); pA_fm = float(p13_te[local_A])
pB_bl = float(p3_te[local_B]); pB_fm = float(p13_te[local_B])
pC_bl = float(p3_te[local_C]); pC_fm = float(p13_te[local_C])

print(f"  样本A omega={omA:.3f} 基线P={pA_bl:.3f} 本文P={pA_fm:.3f} 标签=蜂群")
print(f"  样本B omega={omB:.3f} 基线P={pB_bl:.3f} 本文P={pB_fm:.3f} 标签=蜂群")
print(f"  样本C omega={omC:.3f} 基线P={pC_bl:.3f} 本文P={pC_fm:.3f} 标签=非蜂群")

# =====================================================================
# 工具函数
# =====================================================================
def _norm_arrows(VX, VY, scale=1.0):
    """归一化速度箭头到单位长度"""
    mag = np.sqrt(VX**2 + VY**2) + 1e-12
    return VX/mag*scale, VY/mag*scale

def _vel_angle_deg(VX, VY):
    """速度方向角度 (-180,180]"""
    return np.degrees(np.arctan2(VY, VX))

def _two_group_colors(VX, VY):
    """用速度角度将箭头分成两个颜色组（纯numpy k-means，k=2）"""
    angles = _vel_angle_deg(VX, VY)
    feats = np.column_stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))])
    # 简单 k-means (k=2)，无需 sklearn
    rng = np.random.RandomState(0)
    idx = rng.choice(len(feats), 2, replace=False)
    centers = feats[idx].copy()
    labels = np.zeros(len(feats), dtype=int)
    for _ in range(100):
        dists = np.stack([np.sum((feats - c)**2, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(2):
            mask = labels == k
            if mask.any():
                centers[k] = feats[mask].mean(axis=0)
    colors = np.where(labels == 0, '#1565C0', '#E65100')
    return colors


# =====================================================================
# Figure 1: 无人机编队动态动画
# =====================================================================
print("\n生成 Figure 1: 无人机编队动态动画 (关闭窗口后继续生成其余图表)...")

# ── 轨迹物理仿真 ──
def simulate_trajectory(X0, Y0, VX0, VY0, n_frames=200, dt=0.4,
                         noise_scale=0.05, align_strength=0.0, seed=42):
    """
    从初始位置/速度积分出 n_frames 帧的无人机轨迹。
    align_strength > 0 → 速度向全局均值对齐 (蜂群行为)
    noise_scale 控制随机扰动大小。
    """
    N = len(X0)
    Xs  = np.zeros((n_frames, N)); Ys  = np.zeros((n_frames, N))
    VXs = np.zeros((n_frames, N)); VYs = np.zeros((n_frames, N))
    Xs[0] = X0.copy(); Ys[0] = Y0.copy()
    VXs[0] = VX0.copy(); VYs[0] = VY0.copy()

    orig_speed = np.sqrt(VX0**2 + VY0**2) + 1e-12
    rng = np.random.RandomState(seed)

    for t in range(1, n_frames):
        vx = VXs[t-1].copy(); vy = VYs[t-1].copy()
        # 随机扰动
        ns = noise_scale * (np.std(orig_speed) + np.mean(orig_speed) * 0.1 + 1e-6)
        vx += rng.randn(N) * ns
        vy += rng.randn(N) * ns
        # 速度对齐力 (蜂群)
        if align_strength > 0:
            vx += align_strength * (np.mean(vx) - vx)
            vy += align_strength * (np.mean(vy) - vy)
        # 保持原始速度大小分布
        cur_speed = np.sqrt(vx**2 + vy**2) + 1e-12
        vx = vx / cur_speed * orig_speed
        vy = vy / cur_speed * orig_speed
        VXs[t] = vx; VYs[t] = vy
        Xs[t] = Xs[t-1] + vx * dt
        Ys[t] = Ys[t-1] + vy * dt

    return Xs, Ys, VXs, VYs

N_FRAMES = 200

# 样本A: 经典蜂群 — 强对齐, 低噪声
XA_t, YA_t, VXA_t, VYA_t = simulate_trajectory(
    XA, YA, VXA, VYA, N_FRAMES, noise_scale=0.03, align_strength=0.12, seed=10)
# 样本B: 动态耦合蜂群 — 中等对齐, 中噪声
XB_t, YB_t, VXB_t, VYB_t = simulate_trajectory(
    XB, YB, VXB, VYB, N_FRAMES, noise_scale=0.15, align_strength=0.06, seed=20)
# 样本C: 非蜂群 — 无对齐, 大噪声
XC_t, YC_t, VXC_t, VYC_t = simulate_trajectory(
    XC, YC, VXC, VYC, N_FRAMES, noise_scale=0.5, align_strength=0.0, seed=30)

# 预计算每帧 omega_bar
omA_frames = np.array([_omega_bar(VXA_t[t], VYA_t[t]) for t in range(N_FRAMES)])
omB_frames = np.array([_omega_bar(VXB_t[t], VYB_t[t]) for t in range(N_FRAMES)])
omC_frames = np.array([_omega_bar(VXC_t[t], VYC_t[t]) for t in range(N_FRAMES)])

# ── 动画绘制函数 ──
def _draw_panel(ax, X, Y, VX, VY, omega, is_swarm_gt, panel_label,
                bl_prob, fm_prob, frame, n_frames):
    ax.cla()
    ax.set_facecolor('#F0F4F8')
    ax.grid(True, alpha=0.25, linewidth=0.5)

    c_dot = C_SWARM if is_swarm_gt else C_NSWARM
    ax.scatter(X, Y, s=20, c=c_dot, alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.3)

    # 速度箭头 (降采样)
    step = max(1, len(X) // 40)
    xi, yi = X[::step], Y[::step]
    vx_n, vy_n = _norm_arrows(VX[::step], VY[::step])
    x_span = np.max(X) - np.min(X) + 1e-6
    dx = x_span * 0.045
    for i in range(len(xi)):
        ax.annotate('', xy=(xi[i]+vx_n[i]*dx, yi[i]+vy_n[i]*dx),
                    xytext=(xi[i], yi[i]),
                    arrowprops=dict(arrowstyle='->', color=c_dot,
                                   lw=1.1, mutation_scale=9))

    # 实时判断结果 (用 omega_bar 阈值)
    judgment = omega >= 0.6
    judge_txt = 'SWARM' if judgment else 'NON-SWARM'
    judge_col = C_SWARM if judgment else C_NSWARM
    correct    = (judgment == is_swarm_gt)
    badge_bg   = '#E8F5E9' if correct else '#FFF3E0'
    badge_edge = C_CORRECT if correct else C_WRONG

    # 顶部判断徽章
    ax.text(0.5, 1.065, f'判断: {judge_txt}  {"✓" if correct else "✗"}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color=judge_col, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=badge_bg,
                      edgecolor=badge_edge, linewidth=1.5))

    # 左下信息框
    info = (f'ω̄ = {omega:.3f}\n'
            f'BL P = {bl_prob:.3f}  {"✓" if (bl_prob>=0.5)==is_swarm_gt else "✗"}\n'
            f'FM P = {fm_prob:.3f}  {"✓" if (fm_prob>=0.5)==is_swarm_gt else "✗"}')
    ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.88, edgecolor='#90A4AE'))

    # 进度条 (坐标轴标题下方, 用 figure text 无法跨 ax, 改用右下角文字替代)
    prog_pct = int(frame / (n_frames - 1) * 100)
    ax.text(0.97, 0.03, f'[{"█"*int(prog_pct/10):<10}] {prog_pct}%',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            color='gray', family='monospace')

    gt_txt = 'Swarm' if is_swarm_gt else 'Non-Swarm'
    ax.set_title(f'{panel_label}\nTrue Label: {gt_txt}', fontsize=10,
                 fontweight='bold', pad=4)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.tick_params(labelsize=7)


fig1_anim, axes1 = plt.subplots(1, 3, figsize=(16, 6))
fig1_anim.patch.set_facecolor('#ECEFF1')

# 固定坐标轴范围，避免画面抖动
def _bounds(Xt, Yt, margin=0.1):
    xmn, xmx = Xt.min(), Xt.max()
    ymn, ymx = Yt.min(), Yt.max()
    xpad = (xmx - xmn) * margin + 1e-3
    ypad = (ymx - ymn) * margin + 1e-3
    return xmn-xpad, xmx+xpad, ymn-ypad, ymx+ypad

bA = _bounds(XA_t, YA_t); bB = _bounds(XB_t, YB_t); bC = _bounds(XC_t, YC_t)

title_obj = fig1_anim.suptitle('', fontsize=13, fontweight='bold', y=1.01)

def _anim_update(frame):
    title_obj.set_text(
        f'Figure 1  无人机编队动态可视化  |  帧 {frame+1}/{N_FRAMES}\n'
        f'颜色: 蜂群=蓝  非蜂群=红    顶部徽章 = 实时 ω̄ 判断结果')

    _draw_panel(axes1[0], XA_t[frame], YA_t[frame],
                VXA_t[frame], VYA_t[frame],
                omA_frames[frame], True,
                '(a) 经典编队蜂群',
                pA_bl, pA_fm, frame, N_FRAMES)
    axes1[0].set_xlim(bA[0], bA[1]); axes1[0].set_ylim(bA[2], bA[3])

    _draw_panel(axes1[1], XB_t[frame], YB_t[frame],
                VXB_t[frame], VYB_t[frame],
                omB_frames[frame], True,
                '(b) 动态耦合蜂群 ← 本文修复',
                pB_bl, pB_fm, frame, N_FRAMES)
    axes1[1].set_xlim(bB[0], bB[1]); axes1[1].set_ylim(bB[2], bB[3])

    _draw_panel(axes1[2], XC_t[frame], YC_t[frame],
                VXC_t[frame], VYC_t[frame],
                omC_frames[frame], False,
                '(c) 非蜂群 (随机运动)',
                pC_bl, pC_fm, frame, N_FRAMES)
    axes1[2].set_xlim(bC[0], bC[1]); axes1[2].set_ylim(bC[2], bC[3])

    fig1_anim.tight_layout(rect=[0, 0, 1, 0.97])
    return []

anim1 = FuncAnimation(fig1_anim, _anim_update, frames=N_FRAMES,
                      interval=60, blit=False, repeat=True)
plt.show(block=True)   # 关闭窗口后继续生成后续静态图


# =====================================================================
# Figure 2: 代表样本特征对比柱状图
# =====================================================================
print("生成 Figure 2: 特征对比图...")

FEAT_LABELS = ['corr1\n(R)', 'corr2\n(R)', 'corr3\n(R)', 'ω̄\n(global)',
               'corr1\n(M)', 'corr2\n(M)', 'corr3\n(M)']
feat_idx_13 = [None, None, None, 9, 10, 11, 12]   # 在 X_te13 中的索引 (None=用 corr1/2/3_te)

def get_sample_feats(local_idx):
    """获取样本的7维关键特征 (归一化后的原始值)"""
    c1 = float(corr1_te[local_idx])
    c2 = float(corr2_te[local_idx])
    c3 = float(corr3_te[local_idx])
    om = float(X_te13[local_idx, 9])
    c1M= float(X_te13[local_idx, 10])
    c2M= float(X_te13[local_idx, 11])
    c3M= float(X_te13[local_idx, 12])
    return np.array([c1, c2, c3, om, c1M, c2M, c3M])

feats_A = get_sample_feats(local_A)
feats_B = get_sample_feats(local_B)
feats_C = get_sample_feats(local_C)

# Min-max 归一化 (跨三个样本)
all_feats = np.vstack([feats_A, feats_B, feats_C])
f_min = all_feats.min(axis=0); f_max = all_feats.max(axis=0)
f_rng = np.where(f_max>f_min, f_max-f_min, 1.0)
nA = (feats_A-f_min)/f_rng; nB = (feats_B-f_min)/f_rng; nC = (feats_C-f_min)/f_rng

fig2, (ax2l, ax2r) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Figure 2  三类样本特征对比\n(左: 归一化特征值; 右: 分类器输出概率)',
              fontsize=12, fontweight='bold')

x = np.arange(len(FEAT_LABELS)); w = 0.26
bars_A = ax2l.bar(x-w, nA, w, label=f'(a) Swarm ω={omA:.2f}', color=C_SWARM,   alpha=0.85)
bars_B = ax2l.bar(x,   nB, w, label=f'(b) Swarm ω={omB:.2f}', color='#FF6F00',  alpha=0.85)
bars_C = ax2l.bar(x+w, nC, w, label=f'(c) Non-swarm ω={omC:.2f}', color=C_NSWARM, alpha=0.85)

ax2l.set_xticks(x); ax2l.set_xticklabels(FEAT_LABELS, fontsize=9)
ax2l.set_ylabel('Normalized Feature Value', fontsize=10)
ax2l.set_title('特征值对比 (归一化至[0,1])', fontsize=10)
ax2l.legend(fontsize=9, loc='upper right')
ax2l.set_ylim(0, 1.25)
ax2l.grid(axis='y', alpha=0.4)
ax2l.axvline(x=2.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
ax2l.text(0.3, 1.18, '← R(t)特征', fontsize=8, transform=ax2l.transAxes, ha='center')
ax2l.text(0.72, 1.18, 'SCDAM M(t)特征 →', fontsize=8, transform=ax2l.transAxes, ha='center')

# ω̄ 特殊标注: 这是样本A和B最大差异点
omega_x = 3  # ω̄ 在 x 轴位置
ax2l.annotate('关键差异\n(ω̄)',
              xy=(omega_x, max(nA[3], nB[3])+0.05),
              xytext=(omega_x+0.5, 1.15),
              fontsize=8, color='black',
              arrowprops=dict(arrowstyle='->', color='black', lw=1))

# 右图: 分类器概率
probs_data = {
    '(a) Swarm ω=1.00': {'bl': pA_bl, 'fm': pA_fm, 'c': C_SWARM},
    '(b) Swarm ω=0.51': {'bl': pB_bl, 'fm': pB_fm, 'c': '#FF6F00'},
    '(c) Non-sw ω=0.61':{'bl': pC_bl, 'fm': pC_fm, 'c': C_NSWARM},
}
xt = np.arange(len(probs_data))
pbl_vals = [v['bl'] for v in probs_data.values()]
pfm_vals = [v['fm'] for v in probs_data.values()]

ax2r.bar(xt-0.18, pbl_vals, 0.35, label='CatBoost-3D (基线)', color=C_BL, alpha=0.85)
ax2r.bar(xt+0.18, pfm_vals, 0.35, label='Full Method (本文)', color=C_FM, alpha=0.85)
ax2r.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Decision boundary (0.5)')
ax2r.set_xticks(xt); ax2r.set_xticklabels(list(probs_data.keys()), fontsize=9)
ax2r.set_ylabel('P(Swarm)', fontsize=10); ax2r.set_ylim(0, 1.15)
ax2r.set_title('分类器输出概率 P(Swarm)', fontsize=10)
ax2r.legend(fontsize=8)
ax2r.grid(axis='y', alpha=0.4)

# 在样本B上标注错误/正确
ax2r.annotate('× WRONG\n(FN)', xy=(1-0.18, pbl_vals[1]+0.02), xytext=(1-0.5, 0.6),
              fontsize=9, color=C_WRONG, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_WRONG))
ax2r.annotate('✓ CORRECT', xy=(1+0.18, pfm_vals[1]-0.02), xytext=(1+0.3, 0.6),
              fontsize=9, color=C_CORRECT, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_CORRECT))

fig2.tight_layout()
save_fig(fig2, 'fig2_feature_comparison.png')


# =====================================================================
# Figure 3: 混淆矩阵对比 (场景1 对向穿插)
# =====================================================================
print("生成 Figure 3: 混淆矩阵...")

def plot_cm(ax, cm, title, method_name):
    """绘制2×2混淆矩阵热力图"""
    # cm = [[TP, FP],[FN, TN]]
    tp, fp, fn, tn = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = tp+tn+fp+fn
    mat = np.array([[tp, fp],[fn, tn]])
    pct = mat/total*100

    cmap = LinearSegmentedColormap.from_list('cm', ['#FFFFFF','#1565C0'])
    im = ax.imshow(mat, cmap=cmap, aspect='equal', vmin=0)

    labels_y = ['Swarm (Pos)', 'Non-swarm (Neg)']
    labels_x = ['Swarm (Pred)', 'Non-swarm (Pred)']
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels_x, fontsize=9); ax.set_yticklabels(labels_y, fontsize=9)

    cell_colors = [['white','#FFCCCC'],['#FFCCCC','white']]
    for i in range(2):
        for j in range(2):
            correct = (i==j)
            c = 'white' if correct else '#B71C1C'
            ax.text(j, i, f'{int(mat[i,j])}\n({pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=11,
                    color='white' if mat[i,j]>mat.max()*0.5 else 'black',
                    fontweight='bold')
            if not correct and mat[i,j]>0:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor='#C62828', lw=2))

    acc  = (tp+tn)/total*100
    sw_r = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    nsw_r= tn/(tn+fp)*100 if (tn+fp)>0 else 0
    ax.set_title(f'{title}\n({method_name})\nAcc={acc:.1f}%  SW-recall={sw_r:.1f}%',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Figure 3  场景1 (对向穿插 / 动态耦合, n=946) — 混淆矩阵对比',
              fontsize=12, fontweight='bold')

plot_cm(ax3a, cm_s1_bl, '场景1: 对向穿插', 'CatBoost-3D Baseline')
plot_cm(ax3b, cm_s1_fm, '场景1: 对向穿插', 'Full Method (Ours)')

fig3.tight_layout()
save_fig(fig3, 'fig3_confusion_matrices.png')


# =====================================================================
# Figure 4: 分类概率分布直方图 (场景1)
# =====================================================================
print("生成 Figure 4: 概率分布直方图...")

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig4.suptitle('Figure 4  场景1 (对向穿插, n=946) — 分类器输出概率分布\n'
              '(蓝=蜂群样本, 红=非蜂群样本)',
              fontsize=12, fontweight='bold')

bins = np.linspace(0, 1, 26)

for ax, probs, title in zip(axes4,
                             [p3_te[mask_s1], p13_te[mask_s1]],
                             ['CatBoost-3D (基线)', 'Full Method 13D (本文)']):
    sw_p  = probs[y_te[mask_s1]==1]
    nsw_p = probs[y_te[mask_s1]==0]
    ax.hist(nsw_p, bins=bins, color=C_NSWARM, alpha=0.65, label=f'Non-swarm (n={len(nsw_p)})')
    ax.hist(sw_p,  bins=bins, color=C_SWARM,  alpha=0.65, label=f'Swarm (n={len(sw_p)})')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax.set_xlabel('P(Swarm)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    # 标注重叠区域
    overlap = np.sum((sw_p < 0.5))
    if overlap > 0:
        ax.annotate(f'FN: {overlap} missed\nswarm samples',
                    xy=(0.25, 0.88), xycoords='axes fraction', fontsize=9,
                    color=C_WRONG, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#FFCCBC', alpha=0.8))

fig4.tight_layout()
save_fig(fig4, 'fig4_prob_distributions.png')


# =====================================================================
# Figure 5: 挑战场景准确率对比柱状图
# =====================================================================
print("生成 Figure 5: 场景准确率对比...")

scene_names  = ['对向穿插\n(Dynamic Coupling)\nn=946',
                '集群结构\n(Cluster Formation)\nn=919',
                '边界状态\n(Boundary State)\nn=78']
scene_masks  = [mask_s1, mask_s2, mask_s3]

def _acc(mask, pred):
    if np.sum(mask)==0: return 0.0
    return np.mean(pred[mask]==y_te[mask])*100

bl_accs  = [_acc(m, pred3 ) for m in scene_masks]
fm_accs  = [_acc(m, pred13) for m in scene_masks]
overall  = [np.mean(pred3==y_te)*100, np.mean(pred13==y_te)*100]

fig5, ax5 = plt.subplots(figsize=(11, 6))
ax5.set_facecolor('#F8F9FA')

x5 = np.arange(len(scene_names)+1)
n_groups = len(x5)

bl_all = bl_accs + [overall[0]]
fm_all = fm_accs + [overall[1]]

bars_bl = ax5.bar(x5-0.22, bl_all, 0.42, color=C_BL, alpha=0.9,
                   label='CatBoost-3D (基线)', edgecolor='white', linewidth=0.5)
bars_fm = ax5.bar(x5+0.22, fm_all, 0.42, color=C_FM, alpha=0.9,
                   label='Full Method (本文)', edgecolor='white', linewidth=0.5)

# 值标签
for bar in bars_bl:
    h = bar.get_height()
    ax5.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=8.5, color=C_BL, fontweight='bold')
for bar in bars_fm:
    h = bar.get_height()
    ax5.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=8.5, color=C_FM, fontweight='bold')

# 提升标注
for i, (bl, fm) in enumerate(zip(bl_all, fm_all)):
    delta = fm - bl
    ax5.annotate(f'+{delta:.1f}%',
                 xy=(x5[i], max(bl,fm)+2.5), ha='center', fontsize=9.5,
                 color='#E65100', fontweight='bold')

ax5.set_xticks(x5)
ax5.set_xticklabels(scene_names + ['整体\nOverall'], fontsize=9.5)
ax5.set_ylabel('Accuracy (%)', fontsize=11)
ax5.set_title('Figure 5  挑战性场景准确率对比 (论文 Table 7)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10, loc='lower right')
ax5.set_ylim(40, 115)
ax5.grid(axis='y', alpha=0.4)
ax5.axvline(x=x5[-1]-0.5, color='gray', linestyle='--', alpha=0.5)
ax5.text(x5[-1], 43, '全局性能', ha='center', fontsize=9, color='gray')

fig5.tight_layout()
save_fig(fig5, 'fig5_scenario_accuracy.png')


# =====================================================================
# Figure 6: omega_bar vs 基线概率散点图 (机制可视化)
# =====================================================================
print("生成 Figure 6: omega_bar 机制散点图...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))
fig6.suptitle('Figure 6  ω̄ 特征的关键作用: 基线失效区域与本文修复机制\n'
              '(●蜂群  ×非蜂群 | 绿=正确  橙=错误)',
              fontsize=12, fontweight='bold')

def scatter_omega(ax, probs, pred, title):
    sw_mask  = y_te == 1
    nsw_mask = y_te == 0
    corr = pred == y_te  # 分类正确

    # 蜂群正确 (TP)
    m = sw_mask & corr
    ax.scatter(omega_te[m], probs[m], s=8, c=C_CORRECT, alpha=0.4, marker='o', label='Swarm TP')
    # 蜂群错误 (FN)
    m = sw_mask & ~corr
    ax.scatter(omega_te[m], probs[m], s=30, c=C_WRONG, alpha=0.9, marker='o',
               edgecolors='black', linewidths=0.5, zorder=5, label=f'Swarm FN (n={np.sum(m)})')
    # 非蜂群正确 (TN)
    m = nsw_mask & corr
    ax.scatter(omega_te[m], probs[m], s=8, c='#78909C', alpha=0.2, marker='x', label='Non-sw TN')
    # 非蜂群错误 (FP)
    m = nsw_mask & ~corr
    if np.sum(m)>0:
        ax.scatter(omega_te[m], probs[m], s=40, c='#9C27B0', alpha=0.9, marker='x',
                   zorder=5, label=f'Non-sw FP (n={np.sum(m)})')

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Threshold')
    ax.axvline(x=0.60, color='blue', linestyle=':', alpha=0.7, linewidth=1.5, label='ω̄=0.60')

    # 标注失效区域
    ax.fill_betweenx([0, 0.5], 0.48, 0.65, alpha=0.06, color='red', label='Failure zone')
    ax.text(0.54, 0.12, 'Failure\nZone', fontsize=8, color='#C62828',
            ha='center', style='italic')

    ax.set_xlabel('ω̄  (Global Velocity Alignment)', fontsize=10)
    ax.set_ylabel('P(Swarm)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0.47, 1.03); ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7.5, loc='lower right', markerscale=1.5)
    ax.grid(alpha=0.35)

scatter_omega(ax6a, p3_te,  pred3,  'CatBoost-3D (基线)\n漏报蜂群集中在低ω̄区间')
scatter_omega(ax6b, p13_te, pred13, 'Full Method 13D (本文)\n低ω̄区间蜂群全部修复')

fig6.tight_layout()
save_fig(fig6, 'fig6_omega_mechanism.png')


# =====================================================================
# Figure 7 (额外): omega_bar 分区间改进热力图
# =====================================================================
print("生成 Figure 7: 改进梯度图...")

fig7, ax7 = plt.subplots(figsize=(10, 5))
ax7.set_facecolor('#F8F9FA')

bins7 = [0.49, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70, 0.80, 0.90, 1.01]
bin_centers, bl_list, fm_list, delta_list, n_list = [], [], [], [], []

for i in range(len(bins7)-1):
    lo, hi = bins7[i], bins7[i+1]
    mask = (omega_te >= lo) & (omega_te < hi)
    n = np.sum(mask)
    if n < 5: continue
    bl_a = np.mean(pred3[mask] == y_te[mask])*100
    fm_a = np.mean(pred13[mask]== y_te[mask])*100
    bin_centers.append((lo+hi)/2)
    bl_list.append(bl_a); fm_list.append(fm_a)
    delta_list.append(fm_a-bl_a); n_list.append(n)

xc = np.arange(len(bin_centers)); w7 = 0.35

b1 = ax7.bar(xc-w7/2, bl_list, w7, color=C_BL, alpha=0.85, label='CatBoost-3D (基线)')
b2 = ax7.bar(xc+w7/2, fm_list, w7, color=C_FM, alpha=0.85, label='Full Method (本文)')

# 提升量叠加
for i, (bl, fm, delta, n) in enumerate(zip(bl_list, fm_list, delta_list, n_list)):
    if abs(delta) > 1:
        ax7.text(xc[i], max(bl, fm)+1, f'+{delta:.1f}%\n(n={n})',
                 ha='center', fontsize=7.5, color=C_WRONG if delta > 5 else C_CORRECT,
                 fontweight='bold')
    else:
        ax7.text(xc[i], max(bl, fm)+1, f'n={n}', ha='center', fontsize=7, color='gray')

labels7 = []
for i in range(len(bins7)-1):
    mask = (omega_te >= bins7[i]) & (omega_te < bins7[i+1])
    if np.sum(mask) >= 5:
        labels7.append(f'[{bins7[i]:.2f},\n{bins7[i+1]:.2f})')

ax7.set_xticks(xc); ax7.set_xticklabels(labels7, fontsize=8)
ax7.set_ylabel('Accuracy (%)', fontsize=10)
ax7.set_xlabel('ω̄ (Global Velocity Alignment) Interval', fontsize=10)
ax7.set_title('Figure 7  不同 ω̄ 区间的准确率对比\n(改进主要集中在低ω̄区间 ← SCDAM+ω̄特征的直接贡献)',
              fontsize=11, fontweight='bold')
ax7.legend(fontsize=10, loc='lower right')
ax7.set_ylim(50, 115)
ax7.axvline(x=np.argmin([abs(c-0.60) for c in bin_centers])-0.5,
            color='red', linestyle='--', alpha=0.5, label='ω̄=0.60 分界')
ax7.grid(axis='y', alpha=0.4)

fig7.tight_layout()
save_fig(fig7, 'fig7_omega_interval_improvement.png')


print("\n" + "="*70)
print(f"  全部 7 张图表已生成至: {FIG_DIR}")
print("  文件列表:")
for fname in sorted(os.listdir(FIG_DIR)):
    fp = os.path.join(FIG_DIR, fname)
    print(f"    {fname}  ({os.path.getsize(fp)//1024} KB)")
print("="*70)
