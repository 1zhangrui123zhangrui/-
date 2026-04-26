# -*- coding: utf-8 -*-
"""
fig_challenge_redraw.py
==============================================================================
Publication-quality redraw of Figures 1-7 (Section 7.4 challenge scenarios).
Loads pre-computed results_challenge.npz and generates static PNG files.

Design standards (IEEE / high-impact journal):
  - Font : Times New Roman (serif), 9 pt body / 8 pt ticks
  - DPI  : 300
  - Width: single-column 3.5 in  |  double-column 7.16 in
  - English labels only
  - Panel labels (a)(b)(c) in top-left, bold
  - Colorblind-friendly palette
  - Tick direction: inward; minimal grid (alpha 0.35, dashed)
  - No verbose titles inside panels (captions belong in the paper)
==============================================================================
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Global paper style
# ─────────────────────────────────────────────────────────────────────────────
PAPER_W1 = 3.5    # single-column width (inches)
PAPER_W2 = 7.16   # double-column width (inches)

plt.rcParams.update({
    'font.family':         'serif',
    'font.serif':          ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size':           9,
    'axes.labelsize':      9,
    'axes.titlesize':      9,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'legend.framealpha':   0.85,
    'legend.edgecolor':    '0.8',
    'legend.handlelength': 1.8,
    'figure.dpi':          300,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.04,
    'lines.linewidth':     1.5,
    'lines.markersize':    5,
    'axes.linewidth':      0.8,
    'axes.unicode_minus':  False,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.minor.width':   0.5,
    'ytick.minor.width':   0.5,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'grid.linewidth':      0.5,
    'grid.alpha':          0.35,
    'grid.linestyle':      '--',
    'axes.grid':           True,
})

# ── Colorblind-friendly palette ───────────────────────────────────────────────
C_SWARM   = '#1f77b4'   # blue   — swarm (ground truth)
C_NSWARM  = '#d62728'   # red    — non-swarm (ground truth)
C_BL      = '#7f7f7f'   # grey   — CatBoost-3D baseline
C_FM      = '#2ca02c'   # green  — proposed full method
C_CORRECT = '#2ca02c'   # green  — correct classification
C_WRONG   = '#ff7f0e'   # orange — wrong classification
C_THR     = '#555555'   # dark grey — threshold line
C_OMEGA   = '#9467bd'   # purple — omega_bar feature highlight
C_3RD     = '#ff7f0e'   # orange — third sample (swarm w/ low omega)

def panel_label(ax, label, x=0.02, y=0.97, fontsize=9):
    """Place bold panel label (a)/(b)/(c) in top-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'  Saved: {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed data
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(SCRIPT_DIR, 'results_challenge.npz')
if not os.path.exists(DATA_FILE):
    print(f'[ERROR] Data file not found: {DATA_FILE}')
    print('Please run challenge_scenarios.py first.')
    sys.exit(1)

print('Loading results_challenge.npz ...')
d = np.load(DATA_FILE, allow_pickle=True)

y_te     = d['y_te']
omega_te = d['omega_te']
corr1_te = d['corr1_te']
corr2_te = d['corr2_te']
corr3_te = d['corr3_te']
X_te13   = d['X_te13']
p3_te    = d['p3_te']
p13_te   = d['p13_te']
pred3    = d['pred3']
pred13   = d['pred13']
mask_s1  = d['mask_s1']
mask_s2  = d['mask_s2']
mask_s3  = d['mask_s3']
cm_s1_bl = d['cm_s1_bl']
cm_s1_fm = d['cm_s1_fm']
raw_A    = d['raw_A']
raw_B    = d['raw_B']
raw_C    = d['raw_C']
T_high   = float(d['T_high'][0])
T_low    = float(d['T_low'][0])

# Parse positions and velocities from flat raw rows
fpd = 12
def parse_drone(raw_row):
    N  = len(raw_row) // fpd
    X  = raw_row[0::fpd][:N]
    Y  = raw_row[1::fpd][:N]
    VX = raw_row[2::fpd][:N]
    VY = raw_row[3::fpd][:N]
    return X, Y, VX, VY

XA, YA, VXA, VYA = parse_drone(raw_A)
XB, YB, VXB, VYB = parse_drone(raw_B)
XC, YC, VXC, VYC = parse_drone(raw_C)

def _omega_bar(VX, VY):
    N = len(VX)
    V  = np.column_stack([VX, VY])
    vm = np.linalg.norm(V, axis=1, keepdims=True)
    D  = np.zeros_like(V)
    v  = vm.flatten() > 1e-6
    if np.any(v):
        D[v] = V[v] / vm[v]
    S  = np.clip(D @ D.T, -1.0, 1.0)
    om = (1.0 + S) / 2.0
    return float(np.mean(om[np.triu_indices(N, k=1)]))

omA = _omega_bar(VXA, VYA)
omB = _omega_bar(VXB, VYB)
omC = _omega_bar(VXC, VYC)

# Match samples in test set by closest omega_bar value
local_A = int(np.argmin(np.abs(omega_te - omA)))
local_B = int(np.argmin(np.abs(omega_te - omB)))
local_C = int(np.argmin(np.abs(omega_te - omC)))

pA_bl = float(p3_te[local_A]);  pA_fm = float(p13_te[local_A])
pB_bl = float(p3_te[local_B]);  pB_fm = float(p13_te[local_B])
pC_bl = float(p3_te[local_C]);  pC_fm = float(p13_te[local_C])

print(f'  Sample A: omega={omA:.3f}  BL={pA_bl:.3f}  FM={pA_fm:.3f}  GT=Swarm')
print(f'  Sample B: omega={omB:.3f}  BL={pB_bl:.3f}  FM={pB_fm:.3f}  GT=Swarm')
print(f'  Sample C: omega={omC:.3f}  BL={pC_bl:.3f}  FM={pC_fm:.3f}  GT=Non-swarm')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _norm_arrows(VX, VY):
    """Return unit-direction vectors for velocity arrows."""
    mag = np.sqrt(VX**2 + VY**2) + 1e-12
    return VX / mag, VY / mag


# =============================================================================
# FIGURE 1 — Drone formation snapshots (3-panel static)
# =============================================================================
print('[1/7] Figure 1: Drone formation snapshots ...')

def draw_formation_panel(ax, X, Y, VX, VY, omega, is_swarm,
                         bl_prob, fm_prob, panel_tag, dot_color):
    """Draw one drone-formation panel in IEEE style."""
    ax.set_facecolor('white')

    # Scatter: drone positions
    ax.scatter(X, Y, s=10, c=dot_color, alpha=0.65,
               edgecolors='none', zorder=3)

    # Velocity arrows (subsample ~30 agents to avoid clutter)
    step = max(1, len(X) // 30)
    xi, yi = X[::step], Y[::step]
    ux, uy = _norm_arrows(VX[::step], VY[::step])
    x_span = max(float(np.ptp(X)), 1.0)
    arrow_len = x_span * 0.045
    ax.quiver(xi, yi, ux * arrow_len, uy * arrow_len,
              color=dot_color, scale=1, scale_units='xy',
              width=0.004, headwidth=4, headlength=5,
              alpha=0.85, zorder=4)

    # Info box: omega, GT, probabilities
    gt_str  = 'Swarm' if is_swarm else 'Non-swarm'
    bl_tick = u'\u2713' if (bl_prob >= 0.5) == is_swarm else u'\u2717'
    fm_tick = u'\u2713' if (fm_prob >= 0.5) == is_swarm else u'\u2717'
    info = (f'$\\bar{{\\omega}}={omega:.3f}$\n'
            f'GT: {gt_str}\n'
            f'BL: {bl_prob:.2f} {bl_tick}  '
            f'FM: {fm_prob:.2f} {fm_tick}')
    ax.text(0.97, 0.97, info,
            transform=ax.transAxes, fontsize=7,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='0.7', alpha=0.92))

    panel_label(ax, panel_tag)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.tick_params(labelsize=7)

    xpad = float(np.ptp(X)) * 0.08 + 1.0
    ypad = float(np.ptp(Y)) * 0.08 + 1.0
    ax.set_xlim(X.min() - xpad, X.max() + xpad)
    ax.set_ylim(Y.min() - ypad, Y.max() + ypad)


fig1, axes1 = plt.subplots(1, 3, figsize=(PAPER_W2, 2.7))

draw_formation_panel(axes1[0], XA, YA, VXA, VYA, omA,
                     True,  pA_bl, pA_fm, '(a)', C_SWARM)
draw_formation_panel(axes1[1], XB, YB, VXB, VYB, omB,
                     True,  pB_bl, pB_fm, '(b)', C_3RD)
draw_formation_panel(axes1[2], XC, YC, VXC, VYC, omC,
                     False, pC_bl, pC_fm, '(c)', C_NSWARM)

# Subtitles below panels (IEEE style: caption-style, not inside axes)
for ax, sub in zip(axes1, ['(a) Classic formation swarm',
                             '(b) Dynamic-coupled swarm',
                             '(c) Non-swarm (random motion)']):
    ax.set_title(sub, fontsize=8, pad=3)

# Shared figure-level legend
swarm_p  = mpatches.Patch(color=C_SWARM,  label='Swarm')
dyn_p    = mpatches.Patch(color=C_3RD,    label='Dynamic-coupled swarm')
nswarm_p = mpatches.Patch(color=C_NSWARM, label='Non-swarm')
fig1.legend(handles=[swarm_p, dyn_p, nswarm_p],
            loc='upper center', ncol=3, fontsize=7.5,
            framealpha=0.9, bbox_to_anchor=(0.5, 1.03))

plt.tight_layout(w_pad=1.0)
save(fig1, 'fig1_drone_formation.png')


# =============================================================================
# FIGURE 2 — Feature comparison & classifier output probability
# =============================================================================
print('[2/7] Figure 2: Feature comparison ...')

FEAT_LABELS_7 = [
    r'$\rho_1(R)$', r'$\rho_2(R)$', r'$\rho_3(R)$',
    r'$\bar{\omega}$',
    r'$\rho_1(M)$', r'$\rho_2(M)$', r'$\rho_3(M)$',
]

def get_sample_feats(local_idx):
    return np.array([
        float(corr1_te[local_idx]),
        float(corr2_te[local_idx]),
        float(corr3_te[local_idx]),
        float(X_te13[local_idx, 9]),    # omega_bar
        float(X_te13[local_idx, 10]),   # corr1_M
        float(X_te13[local_idx, 11]),   # corr2_M
        float(X_te13[local_idx, 12]),   # corr3_M
    ])

feats_A = get_sample_feats(local_A)
feats_B = get_sample_feats(local_B)
feats_C = get_sample_feats(local_C)

all_f = np.vstack([feats_A, feats_B, feats_C])
f_min = all_f.min(axis=0)
f_rng = np.where(all_f.max(axis=0) > f_min,
                 all_f.max(axis=0) - f_min, 1.0)
nA = (feats_A - f_min) / f_rng
nB = (feats_B - f_min) / f_rng
nC = (feats_C - f_min) / f_rng

fig2, (ax2l, ax2r) = plt.subplots(1, 2, figsize=(PAPER_W2, 2.6))

x  = np.arange(len(FEAT_LABELS_7))
w  = 0.26
ax2l.bar(x - w, nA, w, color=C_SWARM, alpha=0.85,
         label=f'(a) Swarm, $\\bar{{\\omega}}$={omA:.2f}')
ax2l.bar(x,     nB, w, color=C_3RD,   alpha=0.85,
         label=f'(b) Swarm, $\\bar{{\\omega}}$={omB:.2f}')
ax2l.bar(x + w, nC, w, color=C_NSWARM, alpha=0.85,
         label=f'(c) Non-sw., $\\bar{{\\omega}}$={omC:.2f}')

ax2l.set_xticks(x)
ax2l.set_xticklabels(FEAT_LABELS_7, fontsize=8)
ax2l.set_ylabel('Normalized feature value')
ax2l.set_ylim(0, 1.38)
ax2l.legend(fontsize=7, loc='upper right')
# Vertical divider between R(t) and M(t) feature groups
ax2l.axvline(x=2.5, color='0.5', ls=':', lw=0.8)
ax2l.text(1.0,  1.31, r'$R(t)$ features', ha='center', fontsize=7, color='0.4')
ax2l.text(5.0,  1.31, r'$M(t)$ features', ha='center', fontsize=7, color='0.4')
# Highlight omega_bar column
ax2l.axvspan(2.6, 3.4, color=C_OMEGA, alpha=0.10, zorder=0)
panel_label(ax2l, '(a)')

# Right sub-panel: classifier output probability
sample_labels = [
    f'(a) Swarm\n$\\bar{{\\omega}}$={omA:.2f}',
    f'(b) Swarm\n$\\bar{{\\omega}}$={omB:.2f}',
    f'(c) Non-sw.\n$\\bar{{\\omega}}$={omC:.2f}',
]
pbl = [pA_bl, pB_bl, pC_bl]
pfm = [pA_fm, pB_fm, pC_fm]
xt  = np.arange(3)

ax2r.bar(xt - 0.2, pbl, 0.38, color=C_BL, alpha=0.85,
         label='CatBoost-3D (baseline)', edgecolor='white')
ax2r.bar(xt + 0.2, pfm, 0.38, color=C_FM, alpha=0.85,
         label='Proposed (13D)', edgecolor='white')
ax2r.axhline(y=0.5, color=C_THR, ls='--', lw=1.2,
             label='Decision boundary (0.5)')
ax2r.set_xticks(xt)
ax2r.set_xticklabels(sample_labels, fontsize=8)
ax2r.set_ylabel('$P$(Swarm)')
ax2r.set_ylim(0, 1.22)
ax2r.legend(fontsize=7, loc='upper right')

# Annotate FN/TP on sample B
ax2r.annotate('FN', xy=(1 - 0.2, pbl[1] + 0.02),
              xytext=(1 - 0.6, 0.65), fontsize=8,
              color=C_WRONG, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_WRONG, lw=0.8))
ax2r.annotate('TP', xy=(1 + 0.2, pfm[1] + 0.02),
              xytext=(1 + 0.38, 0.65), fontsize=8,
              color=C_CORRECT, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_CORRECT, lw=0.8))
panel_label(ax2r, '(b)')

plt.tight_layout(w_pad=1.2)
save(fig2, 'fig2_feature_comparison.png')


# =============================================================================
# FIGURE 3 — Confusion matrices (Scenario 1: counter-insertion)
# =============================================================================
print('[3/7] Figure 3: Confusion matrices ...')

def draw_cm(ax, cm, title_tag):
    tp, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tn = int(cm[1, 0]), int(cm[1, 1])
    total  = tp + fp + fn + tn
    mat    = np.array([[tp, fp], [fn, tn]], dtype=float)
    pct    = mat / total * 100

    cmap = LinearSegmentedColormap.from_list(
        'cm_blue', ['#f7fbff', C_SWARM], N=256)
    ax.imshow(mat, cmap=cmap, aspect='equal', vmin=0, vmax=mat.max())

    col_labels = ['Pred: Swarm', 'Pred: Non-sw.']
    row_labels = ['Swarm (pos.)', 'Non-sw. (neg.)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.tick_params(length=0)   # hide tick marks — not meaningful for discrete matrix

    # Cell borders: white separator lines + orange highlight on off-diagonal errors
    ax.axhline(0.5, color='white', lw=2.0, zorder=4)
    ax.axvline(0.5, color='white', lw=2.0, zorder=4)

    for i in range(2):
        for j in range(2):
            txt_c = 'white' if mat[i, j] > mat.max() * 0.55 else '#1a1a1a'
            ax.text(j, i, f'{int(mat[i, j])}\n({pct[i, j]:.1f}%)',
                    ha='center', va='center',
                    fontsize=10, color=txt_c, fontweight='bold')
            # Orange border on off-diagonal cells that contain errors
            if i != j and mat[i, j] > 0:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor=C_WRONG,
                                     lw=2.0, zorder=5)
                ax.add_patch(rect)

    acc   = (tp + tn) / total * 100
    sw_r  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    nsw_r = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    ax.set_title(
        f'{title_tag}\n'
        f'Acc={acc:.1f}%  SW-rec.={sw_r:.1f}%  NSW-rec.={nsw_r:.1f}%',
        fontsize=8.5, pad=6)
    ax.set_xlabel('Predicted label', fontsize=9)
    ax.set_ylabel('True label', fontsize=9)
    ax.grid(False)
    # Clean outer spine
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('0.4')

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(PAPER_W2, 3.0))
draw_cm(ax3a, cm_s1_bl, 'CatBoost-3D (baseline)')
draw_cm(ax3b, cm_s1_fm, 'Proposed method (13D)')
plt.tight_layout(w_pad=2.5)

# (a)(b) labels centred below each subplot's x-axis label — no overlap
fig3.canvas.draw()
for ax_i, label in zip([ax3a, ax3b], ['(a)', '(b)']):
    xlabel_bbox = ax_i.xaxis.label.get_window_extent().transformed(
        fig3.transFigure.inverted())
    pos = ax_i.get_position()
    fig3.text(pos.x0 + pos.width / 2, xlabel_bbox.y0 - 0.012,
              label, ha='center', va='top',
              fontsize=9, fontweight='bold',
              transform=fig3.transFigure)

save(fig3, 'fig3_confusion_matrices.png')


# =============================================================================
# FIGURE 4 — Probability distribution histograms (Scenario 1)
# =============================================================================
print('[4/7] Figure 4: Probability distributions ...')

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(PAPER_W2, 2.6), sharey=True)
bins = np.linspace(0, 1, 26)

def draw_prob_hist(ax, probs, mask, title_tag, panel_tag):
    sw_p  = probs[mask & (y_te == 1)]
    nsw_p = probs[mask & (y_te == 0)]
    ax.hist(nsw_p, bins=bins, color=C_NSWARM, alpha=0.65,
            label=f'Non-swarm ($n$={len(nsw_p)})', zorder=2)
    ax.hist(sw_p,  bins=bins, color=C_SWARM,  alpha=0.65,
            label=f'Swarm ($n$={len(sw_p)})', zorder=2)
    ax.axvline(x=0.5, color=C_THR, ls='--', lw=1.2,
               label='Threshold (0.5)', zorder=3)
    ax.set_xlabel('$P$(Swarm)')
    ax.set_ylabel('Count')
    ax.set_title(title_tag, fontsize=8)
    ax.legend(fontsize=7, loc='upper center')
    fn = int(np.sum(sw_p < 0.5))
    if fn > 0:
        ax.text(0.05, 0.84, f'FN: {fn}',
                transform=ax.transAxes, fontsize=8,
                color=C_WRONG, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec=C_WRONG, alpha=0.92))
    panel_label(ax, panel_tag)

draw_prob_hist(ax4a, p3_te,  mask_s1, 'CatBoost-3D (baseline)', '(a)')
draw_prob_hist(ax4b, p13_te, mask_s1, 'Proposed method (13D)',  '(b)')
ax4b.set_ylabel('')

plt.tight_layout(w_pad=1.0)
save(fig4, 'fig4_prob_distributions.png')


# =============================================================================
# FIGURE 5 — Challenge scenario accuracy comparison (bar chart)
# =============================================================================
print('[5/7] Figure 5: Scenario accuracy comparison ...')

scene_labels = [
    'Counter-\ninsertion\n($n$=946)',
    'Cluster\nformation\n($n$=919)',
    'Boundary\nstate\n($n$=78)',
    'Overall\n($n$=3603)',
]
scene_masks = [mask_s1, mask_s2, mask_s3,
               np.ones(len(y_te), dtype=bool)]

def _acc(mask, pred):
    n = int(np.sum(mask))
    return float(np.mean(pred[mask] == y_te[mask])) * 100 if n > 0 else 0.0

bl_accs = [_acc(m, pred3)  for m in scene_masks]
fm_accs = [_acc(m, pred13) for m in scene_masks]

fig5, ax5 = plt.subplots(figsize=(PAPER_W2, 2.8))

x5 = np.arange(len(scene_labels))
W  = 0.35
bars_bl = ax5.bar(x5 - W/2, bl_accs, W,
                  color=C_BL, alpha=0.85, edgecolor='white', linewidth=0.4,
                  label='CatBoost-3D (baseline)')
bars_fm = ax5.bar(x5 + W/2, fm_accs, W,
                  color=C_FM, alpha=0.85, edgecolor='white', linewidth=0.4,
                  label='Proposed (13D + hysteresis)')

# Value labels on top of bars
for bar in list(bars_bl) + list(bars_fm):
    h = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.35,
             f'{h:.1f}', ha='center', va='bottom', fontsize=7)

# Delta annotations
for i, (bl, fm) in enumerate(zip(bl_accs, fm_accs)):
    delta = fm - bl
    if abs(delta) > 0.05:
        sign = '+' if delta >= 0 else ''
        ax5.text(x5[i], max(bl, fm) + 2.2,
                 f'{sign}{delta:.1f}pp',
                 ha='center', fontsize=7.5, fontweight='bold',
                 color=C_WRONG if delta > 3 else C_CORRECT)

# Separator before 'Overall' column
ax5.axvline(x=x5[-1] - 0.6, color='0.7', ls=':', lw=0.8)

ax5.set_xticks(x5)
ax5.set_xticklabels(scene_labels, fontsize=8)
ax5.set_ylabel('Accuracy (%)')
ax5.set_ylim(40, 112)
ax5.legend(fontsize=8, loc='lower right')

plt.tight_layout()
save(fig5, 'fig5_scenario_accuracy.png')


# =============================================================================
# FIGURE 6 — omega_bar vs P(Swarm) scatter (mechanism visualization)
# =============================================================================
print('[6/7] Figure 6: omega_bar mechanism scatter ...')

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(PAPER_W2, 2.8), sharey=True)

def draw_scatter(ax, probs, pred, title_tag, panel_tag, show_ylabel=True):
    sw_mask  = y_te == 1
    nsw_mask = y_te == 0
    corr     = pred == y_te

    # TN (correct non-swarm) — light background
    m = nsw_mask & corr
    ax.scatter(omega_te[m], probs[m], s=4, c='0.75',
               alpha=0.25, marker='x', zorder=1,
               label=f'Non-sw. TN ($n$={int(m.sum())})')

    # TP (correct swarm)
    m = sw_mask & corr
    ax.scatter(omega_te[m], probs[m], s=5, c=C_SWARM,
               alpha=0.35, marker='o', zorder=2,
               label=f'Swarm TP ($n$={int(m.sum())})')

    # FP (false alarm non-swarm)
    m = nsw_mask & ~corr
    if int(m.sum()) > 0:
        ax.scatter(omega_te[m], probs[m], s=28, c=C_3RD,
                   alpha=0.9, marker='D', zorder=5,
                   edgecolors='k', linewidths=0.4,
                   label=f'Non-sw. FP ($n$={int(m.sum())})')

    # FN (missed swarm) — most critical
    m = sw_mask & ~corr
    ax.scatter(omega_te[m], probs[m], s=28, c=C_NSWARM,
               alpha=0.9, marker='o', zorder=5,
               edgecolors='k', linewidths=0.4,
               label=f'Swarm FN ($n$={int(m.sum())})')

    # Reference lines
    ax.axhline(y=0.5,  color=C_THR,   ls='--', lw=1.0, label='Decision boundary')
    ax.axvline(x=0.60, color=C_OMEGA, ls=':',  lw=1.0,
               label=r'$\bar{\omega}=0.60$')

    # Failure zone shading
    ax.fill_between([0.48, 0.65], 0, 0.5,
                    alpha=0.07, color=C_NSWARM, zorder=0)
    ax.text(0.565, 0.10, 'Failure\nzone', ha='center', fontsize=7,
            color=C_NSWARM, style='italic', transform=ax.get_xaxis_transform())

    ax.set_xlabel(r'$\bar{\omega}$ (velocity alignment)')
    if show_ylabel:
        ax.set_ylabel('$P$(Swarm)')
    ax.set_xlim(0.47, 1.03)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(title_tag, fontsize=8)
    ax.legend(fontsize=6.5, loc='lower right', markerscale=1.5)
    panel_label(ax, panel_tag)

draw_scatter(ax6a, p3_te,  pred3,  'CatBoost-3D (baseline)', '(a)', show_ylabel=True)
draw_scatter(ax6b, p13_te, pred13, 'Proposed method (13D)',  '(b)', show_ylabel=False)

plt.tight_layout(w_pad=1.0)
save(fig6, 'fig6_omega_mechanism.png')


# =============================================================================
# FIGURE 7 — Accuracy by omega_bar interval (gradient bar chart)
# =============================================================================
print('[7/7] Figure 7: omega_bar interval accuracy ...')

bins7 = [0.49, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70, 0.80, 0.90, 1.01]
bin_centers, bl_list, fm_list, delta_list, n_list, xlabels = \
    [], [], [], [], [], []

for i in range(len(bins7) - 1):
    lo, hi = bins7[i], bins7[i + 1]
    mask   = (omega_te >= lo) & (omega_te < hi)
    n      = int(np.sum(mask))
    if n < 5:
        continue
    bl_a = float(np.mean(pred3[mask]  == y_te[mask])) * 100
    fm_a = float(np.mean(pred13[mask] == y_te[mask])) * 100
    bin_centers.append((lo + hi) / 2)
    bl_list.append(bl_a)
    fm_list.append(fm_a)
    delta_list.append(fm_a - bl_a)
    n_list.append(n)
    xlabels.append(f'[{lo:.2f},\n{hi:.2f})')

xc = np.arange(len(bin_centers))
W7 = 0.35

fig7, ax7 = plt.subplots(figsize=(PAPER_W2, 2.8))

ax7.bar(xc - W7/2, bl_list, W7, color=C_BL, alpha=0.85,
        edgecolor='white', linewidth=0.4,
        label='CatBoost-3D (baseline)')
ax7.bar(xc + W7/2, fm_list, W7, color=C_FM, alpha=0.85,
        edgecolor='white', linewidth=0.4,
        label='Proposed (13D)')

for i, (bl, fm, delta, n) in enumerate(
        zip(bl_list, fm_list, delta_list, n_list)):
    top  = max(bl, fm)
    sign = '+' if delta >= 0 else ''
    if abs(delta) > 1.0:
        ax7.text(xc[i], top + 0.9,
                 f'{sign}{delta:.1f}pp',
                 ha='center', fontsize=7, fontweight='bold',
                 color=C_WRONG if delta > 5 else C_CORRECT)
    # Sample count below bars
    ax7.text(xc[i], 51.2, f'$n$={n}',
             ha='center', fontsize=6, color='0.45')

# Mark omega_bar = 0.60 boundary
if len(bin_centers) > 0:
    boundary_idx = int(np.argmin([abs(c - 0.60) for c in bin_centers]))
    ax7.axvline(x=boundary_idx - 0.5,
                color=C_OMEGA, ls='--', lw=0.9, alpha=0.8,
                label=r'$\bar{\omega}=0.60$ boundary')

ax7.set_xticks(xc)
ax7.set_xticklabels(xlabels, fontsize=7)
ax7.set_xlabel(r'$\bar{\omega}$ (velocity alignment) interval')
ax7.set_ylabel('Accuracy (%)')
ax7.set_ylim(50, 117)
ax7.legend(fontsize=8, loc='lower right')

plt.tight_layout()
save(fig7, 'fig7_omega_interval_improvement.png')


print(f'\nAll 7 figures saved to: {FIGURES_DIR}')
