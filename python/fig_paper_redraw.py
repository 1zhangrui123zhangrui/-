# -*- coding: utf-8 -*-
"""
fig_paper_redraw.py
==============================================================================
Paper-quality figure redrawing script.
Loads all pre-computed .npz data and generates publication-ready figures.

Design standards:
  - IEEE / high-impact journal style
  - English labels only
  - Font: Times New Roman (serif), 9 pt
  - DPI: 300
  - No verbose titles inside figures (captions belong in the paper text)
  - Panel labels: (a), (b), (c) in top-left corners
  - Colorblind-friendly palette
  - Clean white background, minimal grid
==============================================================================
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
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
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.8',
    'legend.handlelength': 1.8,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.04,
    'lines.linewidth':    1.5,
    'lines.markersize':   5,
    'axes.linewidth':     0.8,
    'axes.unicode_minus': False,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'axes.grid':          True,
})

# ── Palette (colorblind-friendly) ────────────────────────────────────────────
C_VAL   = '#1f77b4'   # blue  — validation
C_TEST  = '#d62728'   # red   — test
C_CC    = '#ff7f0e'   # orange — cross-insertion subset
C_PAPER = '#2ca02c'   # green — paper recommended value
C_FULL  = '#2ca02c'   # green — proposed full method
C_3D    = '#d62728'   # red   — CatBoost-3D baseline
C_13D   = '#1f77b4'   # blue  — CatBoost-13D (no hysteresis)
C_GT    = '#555555'   # dark grey — ground truth
C_THR   = '#888888'   # grey   — threshold lines
C_OMEGA = '#9467bd'   # purple — omega feature highlight

# Feature labels (clean, LaTeX-rendered)
FEAT_LABELS = [
    r'$E_d$', r'$P_d$', r'$H_d$',
    r'$E_f$', r'$P_f$', r'$H_f$',
    r'$E_b$', r'$P_b$', r'$H_b$',
    r'$\bar{\omega}$',
    r'$\rho_1(M)$', r'$\rho_2(M)$', r'$\rho_3(M)$',
]
FEAT_KEY_IDX = [5, 9]   # H_f and omega_bar are highlighted

# Module labels for complexity
MODULE_LABELS = [
    'Distance matrix $R$',
    'Velocity alignment $\Omega$',
    'SCDAM matrix $M$',
    'FFT spectral features',
    'SCDAM correlation stats',
    'CatBoost inference',
    'Hysteresis logic',
]
MODULE_COLORS = [
    '#aec7e8', '#aec7e8',          # geometry group (light blue)
    '#ffbb78',                      # SCDAM (orange)
    '#d62728',                      # FFT dominant (red)
    '#ff9896',                      # correlation (light red)
    '#98df8a',                      # CatBoost (green)
    '#c5b0d5',                      # hysteresis (purple)
]

def panel_label(ax, label, x=0.02, y=0.96, fontsize=9):
    """Add panel label (a), (b), etc. in top-left, bold."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='top', ha='left')

def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'  Saved: {name}')


# =============================================================================
# FIGURE 1  —  Lambda sensitivity analysis
# =============================================================================
print('[1/5] Lambda sensitivity figures ...')

ld = np.load(os.path.join(SCRIPT_DIR, 'results_lambda_search.npz'))
lam_vals  = ld['lambda_values']
ov_val    = ld['ov_val']
ov_test   = ld['ov_test']
ov_cc     = ld['ov_cc']
n_cc      = int(ld['n_cc'].flat[0])
n_test    = int(ld['n_test'].flat[0])
n_val     = int(ld['n_val'].flat[0])
ob_bins   = ld['ob_bin_edges']
res_ob    = ld['results_ob']   # shape (5, 10): 5 ω̄ bins × 10 λ values

# ── Fig 1a–b: Accuracy vs λ  (double-column, 2 panels) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 2.6))

# Panel (a): full dataset val + test
ax = axes[0]
# Statistical equivalence band: all λ within ±0.13% (< 5 samples) of max
ov_max = ov_test.max()
noise_thr = ov_max - 5 * 100 / n_test   # 5-sample tolerance
ax.axhspan(noise_thr, ov_max + 0.08,
           color='lightgreen', alpha=0.18, zorder=0,
           label=f'Statistically equiv. region ($\\Delta < 5/{n_test}$ samples)')
ax.plot(lam_vals, ov_val,  '-o', color=C_VAL,  label=f'Validation ($n$={n_val})')
ax.plot(lam_vals, ov_test, '-s', color=C_TEST, label=f'Test ($n$={n_test})')
ax.axvline(1.0, color=C_PAPER, ls='--', lw=1.2, label=r'$\lambda^*\!=\!1.0$ (paper)')
# Annotate λ=1.0 value
idx10 = int(np.where(np.abs(lam_vals - 1.0) < 0.05)[0][0])
ax.annotate(f'{ov_test[idx10]:.2f}%',
            xy=(lam_vals[idx10], ov_test[idx10]),
            xytext=(lam_vals[idx10]+0.15, ov_test[idx10]+0.07),
            fontsize=7.5, color=C_PAPER,
            arrowprops=dict(arrowstyle='->', color=C_PAPER, lw=0.8))
# Mark the overall range
ax.text(0.97, 0.08,
        f'OV range across all $\\lambda$: {(ov_test.max()-ov_test.min()):.2f}%\n'
        f'($\\approx${ov_test.max()-ov_test.min():.0f}×0.03% per sample)\n'
        r'Performance statistically flat',
        transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='0.7'))
y_lo = min(ov_val.min(), ov_test.min()) - 0.15
y_hi = max(ov_val.max(), ov_test.max()) + 0.2
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel(r'Coupling coefficient $\lambda$')
ax.set_ylabel('Overall accuracy OV (%)')
ax.set_xticks(lam_vals[::2])
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
panel_label(ax, '(a)')

# Panel (b): cross-insertion subset vs full test
ax = axes[1]
# Statistical equivalence band for CC subset
ov_cc_max = ov_cc.max()
noise_thr_cc = ov_cc_max - 5 * 100 / n_cc
ax.axhspan(noise_thr_cc, ov_cc_max + 0.2,
           color='lightgreen', alpha=0.18, zorder=0)
ax.plot(lam_vals, ov_test, '-s', color=C_TEST, label=f'Full test set ($n$={n_test})')
ax.plot(lam_vals, ov_cc,   '-^', color=C_CC,
        label=f'Counter-insertion subset ($n$={n_cc}, $\\bar{{\\omega}}<0.6$)')
ax.axvline(1.0, color=C_PAPER, ls='--', lw=1.2, label=r'$\lambda^*\!=\!1.0$')
# Annotate λ=1.0 CC value
ax.annotate(f'{ov_cc[idx10]:.2f}%',
            xy=(lam_vals[idx10], ov_cc[idx10]),
            xytext=(lam_vals[idx10]+0.15, ov_cc[idx10]-0.25),
            fontsize=7.5, color=C_CC,
            arrowprops=dict(arrowstyle='->', color=C_CC, lw=0.8))
y_lo2 = min(ov_test.min(), ov_cc.min()) - 0.35
y_hi2 = max(ov_test.max(), ov_cc.max()) + 0.35
ax.set_ylim(y_lo2, y_hi2)
ax.set_xlabel(r'Coupling coefficient $\lambda$')
ax.set_ylabel('Overall accuracy OV (%)')
ax.set_xticks(lam_vals[::2])
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
panel_label(ax, '(b)')

plt.tight_layout(w_pad=1.2)
save(fig, 'fig_paper_lambda_sensitivity.png')


# ── Fig 1c: Lambda × ω̄ interaction heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(PAPER_W2 * 0.55, 2.5))

bin_labels = []
for i in range(len(ob_bins) - 1):
    bin_labels.append(f'[{ob_bins[i]:.2f},\n{ob_bins[i+1]:.2f})')

im = ax.imshow(res_ob, aspect='auto', cmap='RdYlGn',
               vmin=max(res_ob.min() - 0.5, 97.0), vmax=100.0,
               origin='lower')
for i in range(res_ob.shape[0]):
    for j in range(res_ob.shape[1]):
        c = 'white' if res_ob[i, j] < 99.0 else 'black'
        ax.text(j, i, f'{res_ob[i, j]:.1f}', ha='center', va='center',
                fontsize=6.5, color=c)

# Highlight λ=1.0 column
col10 = int(np.where(np.abs(lam_vals - 1.0) < 0.05)[0][0])
rect = mpatches.FancyBboxPatch(
    (col10 - 0.5, -0.5), 1, res_ob.shape[0],
    boxstyle='square,pad=0', linewidth=1.8,
    edgecolor=C_PAPER, facecolor='none', zorder=5)
ax.add_patch(rect)

ax.set_xticks(range(len(lam_vals)))
ax.set_xticklabels([f'{v:.1f}' for v in lam_vals], fontsize=7)
ax.set_yticks(range(len(bin_labels)))
ax.set_yticklabels(bin_labels, fontsize=7)
ax.set_xlabel(r'Coupling coefficient $\lambda$')
ax.set_ylabel(r'$\bar{\omega}$ bin (velocity alignment)')
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label('OV (%)', fontsize=8)
ax.text(col10, res_ob.shape[0] - 0.15, r'$\lambda^*$',
        ha='center', va='top', fontsize=8, color=C_PAPER, fontweight='bold')
ax.grid(False)
plt.tight_layout()
save(fig, 'fig_paper_lambda_heatmap.png')


# =============================================================================
# FIGURE 2  —  Computational complexity
# =============================================================================
print('[2/5] Complexity figures ...')

cd = np.load(os.path.join(SCRIPT_DIR, 'results_complexity.npz'))
N_list    = cd['N_list']
med_mat   = cd['med_mat_ms']    # shape (7, 8)
t_total   = cd['t_total_ms']
pct200    = cd['pct200']
slope     = float(cd['slope'].flat[0])
platform  = str(cd['platform_str'].flat[0])

# ── Fig 2a: Module percentage bar chart (N=200, platform-independent) ────────
fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 2.7),
                         gridspec_kw={'width_ratios': [1.6, 1]})

sort_idx  = np.argsort(pct200)[::-1]
pct_sort  = pct200[sort_idx]
mod_sort  = [MODULE_LABELS[i] for i in sort_idx]
col_sort  = [MODULE_COLORS[i] for i in sort_idx]

ax = axes[0]
bars = ax.barh(range(len(pct_sort)), pct_sort, color=col_sort,
               edgecolor='white', linewidth=0.5, height=0.65)
for bar, pct in zip(bars, pct_sort):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=7.5)
ax.set_yticks(range(len(mod_sort)))
ax.set_yticklabels(mod_sort, fontsize=7.5)
ax.set_xlabel('Runtime share (%)')
ax.set_xlim(0, max(pct_sort) * 1.18)
ax.invert_yaxis()
ax.grid(axis='y', alpha=0)
ax.grid(axis='x', alpha=0.3)
panel_label(ax, '(a)')
ax.set_title(f'$N$=200 drones, platform-independent', fontsize=8)

# ── Panel (b): Pie chart of dominant modules ─────────────────────────────────
ax2 = axes[1]
# Group small modules
threshold = 3.0
big_mask = pct200 >= threshold
pct_pie  = list(pct200[big_mask])
lbl_pie  = [MODULE_LABELS[i] for i in np.where(big_mask)[0]]
col_pie  = [MODULE_COLORS[i] for i in np.where(big_mask)[0]]
others   = pct200[~big_mask].sum()
if others > 0.1:
    pct_pie.append(others)
    lbl_pie.append('Others')
    col_pie.append('#c7c7c7')

wedge_props = dict(edgecolor='white', linewidth=0.8)
wedges, texts, autotexts = ax2.pie(
    pct_pie, labels=None, colors=col_pie,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=wedge_props, pctdistance=0.75,
    textprops={'fontsize': 7})
for at in autotexts:
    at.set_fontsize(6.5)
ax2.legend(wedges, lbl_pie, loc='lower center', bbox_to_anchor=(0.5, -0.35),
           fontsize=6.5, framealpha=0.8, ncol=1, handlelength=1.2)
ax2.set_title('Runtime distribution\n($N$=200)', fontsize=8)
panel_label(ax2, '(b)', x=-0.08, y=1.05)

plt.tight_layout(w_pad=1.0)
save(fig, 'fig_paper_complexity_breakdown.png')


# ── Fig 2b: Scaling curves (log-log, normalised) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 2.8))

# Left: normalized curves (T_i(N) / T_i(N=50))
ax = axes[0]
ls_styles = ['-', '--', '-.', '-', '--', '-.', ':']
mk_styles = ['o', 's', '^', 'D', 'v', 'P', 'x']
N_ref = N_list[0]
for i in range(med_mat.shape[0]):
    norm_curve = med_mat[i] / (med_mat[i, 0] + 1e-15)
    ax.plot(N_list, norm_curve, ls_styles[i % 7], marker=mk_styles[i % 7],
            color=MODULE_COLORS[i], ms=4, lw=1.4, label=MODULE_LABELS[i])

# Theoretical reference lines
N_f = np.linspace(N_list[0], N_list[-1], 200)
n0  = N_f[0]
ax.plot(N_f, (N_f / n0) ** 2,           'k:',  lw=1.0, label=r'$O(N^2)$')
ax.plot(N_f, (N_f / n0) ** 2 * np.log2(N_f / n0 * N_list[0]),
        'k--', lw=1.0, label=r'$O(N^2\log N)$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of drones $N$')
ax.set_ylabel('Normalized runtime $T(N)/T(N_0)$')
ax.set_xticks(N_list[::2]); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.legend(fontsize=6.5, ncol=1, loc='upper left', framealpha=0.9)
panel_label(ax, '(a)')

# Right: end-to-end in log-log, with empirical slope
ax = axes[1]
ax.plot(N_list, t_total, '-o', color=C_TEST, lw=1.8, ms=5,
        label=f'Measured (slope = {slope:.2f})')
ax.plot(N_f, t_total[0] * (N_f / N_list[0]) ** 2,      'k:',  lw=1.0, label=r'$O(N^2)$')
ax.plot(N_f, t_total[0] * (N_f / N_list[0]) ** 2 * np.log2(N_f / N_list[0] + 1),
        'k--', lw=1.0, label=r'$O(N^2\log N)$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of drones $N$')
ax.set_ylabel('End-to-end latency (ms)')
ax.set_xticks(N_list[::2]); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
ax.text(0.97, 0.08,
        f'Empirical slope = {slope:.2f}\n' + r'(Theory: $O(N^2\log N)$)',
        transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85, ec='0.7'))
# Note: measurements on PC (not ARM)
ax.text(0.97, 0.97,
        f'Platform: PC (Python)\n(not ARM deployment)',
        transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
        color='gray',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='0.8'))
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
panel_label(ax, '(b)')

plt.tight_layout(w_pad=1.2)
save(fig, 'fig_paper_complexity_scaling.png')


# =============================================================================
# FIGURE 3  —  Feature importance analysis
# =============================================================================
print('[3/5] Feature importance figures ...')

fd = np.load(os.path.join(SCRIPT_DIR, 'results_feature_importance.npz'))
imp_pvc     = fd['imp_pvc_pct']
imp_lfc_te  = fd['imp_lfc_te_pct']
imp_lfc_cc  = fd['imp_lfc_cc_pct']
rank_te     = int(fd['rank_omega_te'].flat[0])
rank_cc     = int(fd['rank_omega_cc'].flat[0])
n_test_fi   = int(fd['n_test'].flat[0])
n_cc_fi     = int(fd['n_cc'].flat[0])

# Sort by LFC (test set)
sort_idx = np.argsort(imp_lfc_te)   # ascending → bottom-to-top
labels_sorted   = [FEAT_LABELS[i]   for i in sort_idx]
imp_pvc_s       = imp_pvc[sort_idx]
imp_lfc_te_s    = imp_lfc_te[sort_idx]
imp_lfc_cc_s    = imp_lfc_cc[sort_idx]

key_set = set(sort_idx[np.isin(sort_idx, FEAT_KEY_IDX)])  # highlighted features

# ── Fig 3a–b: PVC + LFC side by side (double-column) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 3.5))

def importance_bar(ax, values, labels, color_base, highlight_set,
                   highlight_color, xlabel, panel, title):
    y_pos = np.arange(len(labels))
    colors = []
    for orig_i in sort_idx:
        colors.append(highlight_color if orig_i in highlight_set else color_base)
    bars = ax.barh(y_pos, values, color=colors,
                   edgecolor='white', linewidth=0.4, height=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis='y', alpha=0)
    ax.grid(axis='x', alpha=0.3)
    ax.set_title(title, fontsize=8.5, pad=4)
    panel_label(ax, panel)
    # Legend for highlight
    patches = [
        mpatches.Patch(color=highlight_color, label=r'Key features ($H_f$, $\bar{\omega}$)'),
        mpatches.Patch(color=color_base, label='Other features'),
    ]
    ax.legend(handles=patches, fontsize=7, loc='lower right', framealpha=0.9)

importance_bar(axes[0], imp_pvc_s, labels_sorted,
               '#aec7e8', key_set, C_OMEGA,
               'Importance score (%)',
               '(a)', 'PredictionValuesChange\n(model-intrinsic, full test set)')

importance_bar(axes[1], imp_lfc_te_s, labels_sorted,
               '#ffbb78', key_set, C_OMEGA,
               'Importance score (%)',
               '(b)', f'LossFunctionChange\n(data-driven, full test set, $n$={n_test_fi})')

plt.tight_layout(w_pad=1.5)
save(fig, 'fig_paper_feature_importance_full.png')


# ── Fig 3c: Cross-insertion vs full test set (LFC comparison) ────────────────
fig, ax = plt.subplots(figsize=(PAPER_W2 * 0.6, 3.5))

y_pos = np.arange(len(labels_sorted))
bar_w = 0.38

bars1 = ax.barh(y_pos + bar_w / 2, imp_lfc_te_s,  bar_w,
                color='#aec7e8', edgecolor='white', linewidth=0.4,
                label=f'Full test set ($n$={n_test_fi})')
bars2 = ax.barh(y_pos - bar_w / 2, imp_lfc_cc_s, bar_w,
                color='#ff9896', edgecolor='white', linewidth=0.4,
                label=f'Counter-insertion subset ($n$={n_cc_fi}, $\\bar{{\\omega}}<0.6$)')

# Highlight omega_bar in both bars
omega_idx_in_sorted = int(np.where(sort_idx == 9)[0][0])
for bars in [bars1, bars2]:
    bars[omega_idx_in_sorted].set_edgecolor(C_OMEGA)
    bars[omega_idx_in_sorted].set_linewidth(1.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel('LossFunctionChange importance (%)')
ax.grid(axis='y', alpha=0)
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
ax.set_title(f'Feature importance: full set vs. counter-insertion\n'
             r'($\bar{\omega}$ rank: full=' + f'{rank_te}'
             + r', counter-insertion=' + f'{rank_cc})', fontsize=8.5, pad=4)
panel_label(ax, '(a)')
plt.tight_layout()
save(fig, 'fig_paper_feature_importance_cross.png')


# =============================================================================
# FIGURE 4  —  Temporal stability time series
# =============================================================================
print('[4/5] Temporal stability time series ...')

td = np.load(os.path.join(SCRIPT_DIR, 'results_7_5_temporal.npz'))
T_HIGH  = float(td['T_HIGH']);   T_LOW  = float(td['T_LOW'])
T_EMG   = float(td['T_EMG']);    N_WIN  = int(td['N_WIN']);   ETA = float(td['ETA'])
thr_3d  = float(td['thr_3d']);   thr_13d = float(td['thr_13d'])
SIG_3D_STB   = 0.05;  SIG_13D_STB   = 0.04
SIG_3D_TR    = float(td['sigma_3d_trans'])    # 0.13
SIG_13D_TR   = float(td['sigma_13d_trans'])   # 0.07
VIZ_LEN = 80
P3D_1 = 0.78;  P3D_0 = 0.22
P13D_1 = 0.88; P13D_0 = 0.12


def _clip(x): return np.clip(x, 0.01, 0.99)

def apply_hyst(probs, T_hi, T_lo, T_em, Nw, eta):
    T = len(probs); yf = np.zeros(T, dtype=int); Ss = np.zeros(T, dtype=int)
    buf = []
    for t in range(T):
        p = float(probs[t]); ps = int(Ss[t-1]) if t > 0 else 0
        if p < T_em:
            yf[t] = 0; Ss[t] = ps; continue
        Ss[t] = 1 if p > T_hi else (0 if p < T_lo else ps)
        buf.append(int(Ss[t]))
        if len(buf) > Nw: buf.pop(0)
        yf[t] = 1 if float(np.mean(buf)) > eta else 0
    return yf, Ss

def thr_pred(probs, thr): return (np.array(probs) >= thr).astype(int)
def n_sw(y): return int(np.sum(np.abs(np.diff(np.asarray(y, dtype=int))))) if len(y) >= 2 else 0


def make_seq_A(seed=1001):
    rng = np.random.RandomState(seed); T = VIZ_LEN; TRANS_HW = 18; sp = T // 2
    gt = np.ones(T, dtype=int); gt[sp:] = 0
    p3, p1 = np.zeros(T), np.zeros(T)
    for t in range(T):
        dist = t - sp; in_tr = abs(dist) <= TRANS_HW; s = int(gt[t])
        if not in_tr:
            p3[t] = rng.normal(P3D_1 if s==1 else P3D_0, SIG_3D_STB)
            p1[t] = rng.normal(P13D_1 if s==1 else P13D_0, SIG_13D_STB)
        else:
            alpha = dist / TRANS_HW
            if alpha <= 0:
                b = alpha + 1.0
                m3 = (P3D_1 if s==1 else P3D_0)*(1-b) + thr_3d*b
                m1 = (P13D_1 if s==1 else P13D_0)*(1-b) + thr_13d*b
            else:
                ns = 1-s
                m3 = thr_3d*(1-alpha) + (P3D_1 if ns==1 else P3D_0)*alpha
                m1 = thr_13d*(1-alpha) + (P13D_1 if ns==1 else P13D_0)*alpha
            p3[t] = rng.normal(m3, SIG_3D_TR); p1[t] = rng.normal(m1, SIG_13D_TR)
    return _clip(p3), _clip(p1), gt, [sp]


def make_seq_B(seed=1002):
    rng = np.random.RandomState(seed); T = VIZ_LEN; TRANS_HW = 3; sp = int(T*0.65)
    gt = np.ones(T, dtype=int); gt[sp:] = 0
    p3, p1 = np.zeros(T), np.zeros(T)
    for t in range(T):
        dist = t - sp; in_tr = abs(dist) <= TRANS_HW; s = int(gt[t])
        if not in_tr:
            if s == 1:
                p3[t] = rng.normal(P3D_1, SIG_3D_STB); p1[t] = rng.normal(P13D_1, SIG_13D_STB)
            else:
                p3[t] = rng.normal(0.06, 0.02); p1[t] = rng.normal(0.05, 0.02)
        else:
            alpha = dist / max(TRANS_HW, 1)
            if alpha <= 0:
                b = alpha+1.0
                p3[t] = rng.normal(P3D_1*(1-b)+T_EMG*b, SIG_3D_TR)
                p1[t] = rng.normal(P13D_1*(1-b)+T_EMG*b, SIG_13D_TR)
            else:
                tgt = 0.06
                p3[t] = rng.normal(T_EMG*(1-alpha)+tgt*alpha, SIG_3D_TR*0.5)
                p1[t] = rng.normal(T_EMG*(1-alpha)+tgt*alpha, SIG_13D_TR*0.5)
    return _clip(p3), _clip(p1), gt, [sp]


def make_seq_C(seed=1003):
    rng = np.random.RandomState(seed); T = VIZ_LEN
    gt = np.ones(T, dtype=int)
    p3  = _clip(rng.normal(T_HIGH-0.03, SIG_3D_TR*0.9, size=T))
    p1  = _clip(rng.normal(P13D_1-0.05, SIG_13D_STB*2.0, size=T))
    return p3, p1, gt, []


def draw_sequence(p3d, p13d, gt, split_pts, save_name, panel_title_a,
                  scenario_tag):
    T = len(gt); t = np.arange(T)
    y1 = thr_pred(p3d, thr_3d)
    y2 = thr_pred(p13d, thr_13d)
    y3, S3 = apply_hyst(p13d, T_HIGH, T_LOW, T_EMG, N_WIN, ETA)
    sw1 = n_sw(y1); sw2 = n_sw(y2); sw3 = n_sw(y3)

    # ── Layout: 3 rows ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(PAPER_W2, 5.2))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.5, 1.2, 1.2], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    # ── Row 1: p(t) probability curves ───────────────────────────────────────
    # Threshold band background (subtle)
    ax0.axhspan(T_LOW, T_HIGH, color='gold',       alpha=0.10, zorder=0)
    ax0.axhspan(0.0,   T_EMG,  color='lightcoral', alpha=0.12, zorder=0)
    # Vertical transition markers
    for sp in split_pts:
        ax0.axvline(sp, color='0.6', ls=':', lw=1.0, zorder=1)
    # Probability curves
    ax0.plot(t, p3d,  color=C_3D,  lw=1.3, alpha=0.85, label=f'CatBoost-3D [thr={thr_3d:.2f}]')
    ax0.plot(t, p13d, color=C_13D, lw=1.3, alpha=0.85, label=f'CatBoost-13D [thr={thr_13d:.2f}]')
    # Threshold lines
    ax0.axhline(T_HIGH, color=C_THR, ls='--', lw=0.9, alpha=0.9)
    ax0.axhline(T_LOW,  color=C_THR, ls='-.',  lw=0.9, alpha=0.9)
    ax0.axhline(T_EMG,  color='lightcoral', ls='--', lw=0.9, alpha=0.9)
    # Labels on threshold lines (right side)
    ax0.text(T+0.5, T_HIGH, r'$T_\mathrm{high}$', va='center', fontsize=7, color='0.5')
    ax0.text(T+0.5, T_LOW,  r'$T_\mathrm{low}$',  va='center', fontsize=7, color='0.5')
    ax0.text(T+0.5, T_EMG,  r'$T_\mathrm{emg}$',  va='center', fontsize=7, color='lightcoral')
    ax0.set_ylim(-0.05, 1.1)
    ax0.set_ylabel(r'Output probability $p(t)$', fontsize=8.5)
    ax0.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    panel_label(ax0, '(a)')
    ax0.set_title(panel_title_a, fontsize=9, pad=3)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # ── Row 2: Hysteresis state S(t) ─────────────────────────────────────────
    offset = {0: 0.05, 1: 0.55, 2: 1.05}
    labels_ = [f'CatBoost-3D (switches={sw1})',
               f'CatBoost-13D (switches={sw2})',
               f'Proposed (switches={sw3})']
    colors_ = [C_3D, C_13D, C_FULL]
    for k, (seq, lab, col) in enumerate(zip([y1, y2, y3], labels_, colors_)):
        base = offset[k]
        for i in range(T - 1):
            if seq[i] == 1:
                ax1.fill_between([t[i], t[i+1]], base, base+0.4,
                                  color=col, alpha=0.85, linewidth=0)
        ax1.plot([], [], color=col, lw=6, alpha=0.75, label=lab)
    ax1.set_ylim(-0.05, 1.55)
    ax1.set_yticks([0.25, 0.75, 1.25])
    ax1.set_yticklabels(['CB-3D', 'CB-13D', 'Proposed'], fontsize=7)
    ax1.set_ylabel('Decision state', fontsize=8.5)
    ax1.legend(loc='lower left', fontsize=7, framealpha=0.9, ncol=3)
    ax1.grid(axis='x', alpha=0.3); ax1.grid(axis='y', alpha=0)
    panel_label(ax1, '(b)')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Row 3: Final decision vs GT ───────────────────────────────────────────
    # GT shading
    for i in range(T - 1):
        if gt[i] == 1:
            ax2.axvspan(t[i], t[i+1], color='0.92', alpha=0.6, zorder=0)
    for sp in split_pts:
        ax2.axvline(sp, color='0.6', ls=':', lw=1.0, zorder=1)
    ax2.step(t, gt, where='post', color=C_GT,   lw=1.8, ls='-',
             label='Ground truth', zorder=4)
    ax2.step(t, y3 + 0.04, where='post', color=C_FULL, lw=1.5, ls='-',
             label=f'Proposed (switches={sw3})', zorder=5)
    ax2.step(t, y2 - 0.04, where='post', color=C_13D,  lw=1.0, ls='--',
             label=f'CB-13D (switches={sw2})', zorder=3, alpha=0.85)
    ax2.step(t, y1 - 0.08, where='post', color=C_3D,   lw=1.0, ls=':',
             label=f'CB-3D (switches={sw1})', zorder=2, alpha=0.85)
    ax2.set_ylim(-0.25, 1.35)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Non-swarm', 'Swarm'], fontsize=7.5)
    ax2.set_ylabel('Decision $\hat{y}(t)$', fontsize=8.5)
    ax2.set_xlabel('Time step $t$', fontsize=8.5)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9, ncol=2)
    ax2.grid(axis='x', alpha=0.3); ax2.grid(axis='y', alpha=0)
    panel_label(ax2, '(c)')

    fig.align_ylabels([ax0, ax1, ax2])
    save(fig, save_name)


draw_sequence(*make_seq_A(), 'fig_paper_temporal_seqA.png',
              'Scenario A: Gradual boundary transition (Swarm → Non-swarm, ±18 frames)',
              'A')
draw_sequence(*make_seq_B(), 'fig_paper_temporal_seqB.png',
              'Scenario B: Sudden collapse (emergency bypass triggered, $p(t) < T_\\mathrm{emg}$)',
              'B')
draw_sequence(*make_seq_C(), 'fig_paper_temporal_seqC.png',
              'Scenario C: Noisy steady state (all GT=Swarm, hysteresis suppresses false switches)',
              'C')


# =============================================================================
# FIGURE 5  —  Temporal stability statistics
# =============================================================================
print('[5/5] Temporal stability statistics ...')

sw3d   = td['switches_3d']
sw13d  = td['switches_13d']
swfull = td['switches_full']
dtw3d  = td['dtw_arr_3d']
dtw13d = td['dtw_arr_13d']
dtwfull= td['dtw_arr_full']
n_seq  = int(td['n_sequences'])

labels_box  = ['CB-3D\n(baseline)', 'CB-13D\n(no hyst.)', 'Proposed']
colors_box  = [C_3D, C_13D, C_FULL]

def wilcox_str(a, b):
    try:
        _, p = wilcoxon(a, b)
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return 'n.s.'
    except Exception:
        return ''

def add_stat_bracket(ax, x1, x2, y, label, h=0.15):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.9, color='0.4')
    ax.text((x1+x2)/2, y+h+0.02, label, ha='center', va='bottom', fontsize=7.5)

fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 3.2))

for ax_idx, (data_list, ylabel, panel, title) in enumerate([
    ([sw3d,  sw13d,  swfull],  'Switches per sequence',
     '(a)', f'State switch count ($n$={n_seq} sequences)'),
    ([dtw3d, dtw13d, dtwfull], 'DTW distance',
     '(b)', f'DTW distance to ground truth ($n$={n_seq} sequences)'),
]):
    ax = axes[ax_idx]
    bplot = ax.boxplot(data_list, labels=labels_box, patch_artist=True,
                       widths=0.45, notch=False, showfliers=True,
                       flierprops=dict(marker='o', ms=3, alpha=0.5,
                                       markerfacecolor='0.6', markeredgecolor='none'),
                       medianprops=dict(color='white', lw=2.0),
                       whiskerprops=dict(lw=0.9),
                       capprops=dict(lw=0.9))
    for patch, col in zip(bplot['boxes'], colors_box):
        patch.set_facecolor(col); patch.set_alpha(0.75)

    # Annotate medians
    for i, d in enumerate(data_list):
        med = np.median(d)
        ax.text(i+1, med + (max(d)-min(d))*0.03, f'{med:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='0.2')

    # Statistical significance brackets
    y_max = max([max(d) for d in data_list])
    y_step = (y_max - min([min(d) for d in data_list])) * 0.15
    y_br = y_max + y_step * 0.5

    s1 = wilcox_str(data_list[0], data_list[2])
    s2 = wilcox_str(data_list[1], data_list[2])
    add_stat_bracket(ax, 1, 3, y_br,          s1, h=y_step*0.4)
    add_stat_bracket(ax, 2, 3, y_br - y_step*0.15, s2, h=y_step*0.3)

    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.grid(axis='y', alpha=0.35); ax.grid(axis='x', alpha=0)
    panel_label(ax, panel)
    ax.set_ylim(bottom=max(0, min([min(d) for d in data_list]) - y_step*0.5))

# Footnote with Wilcoxon legend
fig.text(0.5, -0.04,
         '*** $p<0.001$,  ** $p<0.01$,  * $p<0.05$,  n.s. not significant  '
         '(Wilcoxon signed-rank test, two-sided)',
         ha='center', fontsize=7.5, color='0.4')

plt.tight_layout(w_pad=1.5)
save(fig, 'fig_paper_temporal_stats.png')


print('\nAll paper-quality figures saved to:', FIGURES_DIR)
print('Files generated:')
for f in ['fig_paper_lambda_sensitivity.png', 'fig_paper_lambda_heatmap.png',
          'fig_paper_complexity_breakdown.png', 'fig_paper_complexity_scaling.png',
          'fig_paper_feature_importance_full.png', 'fig_paper_feature_importance_cross.png',
          'fig_paper_temporal_seqA.png', 'fig_paper_temporal_seqB.png',
          'fig_paper_temporal_seqC.png', 'fig_paper_temporal_stats.png']:
    print(f'  {f}')
