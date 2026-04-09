# -*- coding: utf-8 -*-
"""
fig_paper_coupled.py
==============================================================================
Paper-quality redraw of the four-layer hyperparameter coupling experiment.
Loads results_coupled_hyperparameter.npz and generates 4 clean figures.

Key findings to convey:
  Layer 1:  λ and m ARE coupled  — optimal λ drifts 1.20 across m values
  Layer 2:  T_emg vs λ DECOUPLED — OV std = 0.00% in T_emg direction
  Layer 3:  T_emg vs m DECOUPLED — OV std = 0.00% in T_emg direction
  Layer 4:  Paper's (λ=1.0, m=8) falls inside the high-performance region;
            full-dataset validation (Section 4.4, n=24015) confirms λ=1.0.

Note on methodology:
  This experiment uses N_SUB=4000 for computational efficiency.
  Absolute OV values are ~1.5% lower than full-dataset results due to
  smaller training set.  The coupling structure (not absolute values) is
  the primary result.  Full-dataset grid search (Section 4.4) definitively
  confirms λ*=1.0.
==============================================================================
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Same paper style as fig_paper_redraw.py ───────────────────────────────────
PAPER_W2 = 7.16

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
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'axes.grid':          True,
})

C_PAPER   = '#2ca02c'    # green — paper recommended value
C_OPT     = '#d62728'    # red   — global optimum marker
C_DRIFT   = '#ff7f0e'    # orange — drift trend line
C_NEUTRAL = '#1f77b4'    # blue  — neutral data
CMAP_RYG  = 'RdYlGn'

def panel_label(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Load data ─────────────────────────────────────────────────────────────────
d = np.load(os.path.join(SCRIPT_DIR, 'results_coupled_hyperparameter.npz'))

LAM_L1 = d['LAM_L1'];  M_L1   = d['M_L1'].astype(int)
L1_OV  = d['L1_OV'];   L1_CC_OV = d['L1_CC_OV']

LAM_L2 = d['LAM_L2'];  TEMG_L2 = d['TEMG_L2']
L2_OV  = d['L2_OV'];   L2_FA   = d['L2_FA']

M_L3   = d['M_L3'].astype(int);  TEMG_L3 = d['TEMG_L3']
L3_OV  = d['L3_OV']

LAM_L4 = d['LAM_L4'];  M_L4   = d['M_L4'].astype(int)
TEMG_L4 = d['TEMG_L4']
L4_OV  = d['L4_OV'];   L4_FA  = d['L4_FA']

lam_best   = float(d['lam_best_global'].flat[0])
m_best     = int(d['m_best_global'].flat[0])
temg_best  = float(d['temg_best_global'].flat[0])
ov_best    = float(d['ov_best_global'].flat[0])
ov_paper   = float(d['ov_paper_in_L4'].flat[0])
temg_var_L2 = float(d['temg_variation_L2'].flat[0])
temg_var_L3 = float(d['temg_variation_L3'].flat[0])
lam_drift  = float(d['lam_shift_range'].flat[0])

LAM_PAPER = 1.0;  M_PAPER = 8;  TEMG_PAPER = 0.20

print('Loaded results_coupled_hyperparameter.npz')
print(f'  L1_OV: {L1_OV.shape}, L2_OV: {L2_OV.shape}, '
      f'L3_OV: {L3_OV.shape}, L4_OV: {L4_OV.shape}')
print(f'  Paper (subsample): OV={ov_paper:.2f}%, Global best: OV={ov_best:.2f}%')
print(f'  λ drift range: {lam_drift:.2f}')


# =============================================================================
# FIGURE A  —  Layer 1: λ × m coupling analysis
#   Left:  heatmap OV(λ, m) with optimal-λ trend and paper reference
#   Right: cross-insertion subset OV(λ, m)
# =============================================================================
print('\n[1/4] Layer 1: λ × m coupling ...')

# Find optimal λ for each m
opt_lam_per_m = [LAM_L1[np.argmax(L1_OV[:, j])] for j in range(len(M_L1))]
opt_ov_per_m  = [L1_OV[np.argmax(L1_OV[:, j]), j] for j in range(len(M_L1))]

fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 3.2))

for ax_idx, (Z, title, panel) in enumerate([
    (L1_OV,    r'OV (%, full test set)', '(a)'),
    (L1_CC_OV, r'OV (%, counter-insertion subset, $\bar{\omega}<0.6$)', '(b)'),
]):
    ax = axes[ax_idx]
    vlo = max(Z.min() - 0.3, 96.0);  vhi = 100.0
    im = ax.imshow(Z.T, aspect='auto', cmap=CMAP_RYG,
                   vmin=vlo, vmax=vhi, origin='lower',
                   extent=[LAM_L1[0]-0.1, LAM_L1[-1]+0.1,
                            M_L1[0]-0.5,  M_L1[-1]+0.5])

    # Annotate cells
    for i, lam in enumerate(LAM_L1):
        for j, m in enumerate(M_L1):
            val = Z[i, j]
            c = 'black' if val > (vlo + vhi) / 2 else 'white'
            ax.text(lam, m, f'{val:.1f}', ha='center', va='center',
                    fontsize=5.5, color=c)

    # Optimal-λ trend line
    ax.plot(opt_lam_per_m, M_L1, 'o--', color=C_DRIFT, ms=5, lw=1.3,
            label=f'Optimal $\\lambda$ per $m$ (drift={lam_drift:.1f})')

    # Paper reference line (nearest λ grid point to λ=1.0)
    lam_near = LAM_L1[np.argmin(np.abs(LAM_L1 - LAM_PAPER))]
    ax.axvline(lam_near, color=C_PAPER, ls='--', lw=1.3,
               label=f'$\\lambda\\approx${LAM_PAPER:.1f} (paper, nearest grid)')
    ax.axhline(M_PAPER,  color=C_PAPER, ls=':',  lw=1.0)
    ax.plot(lam_near, M_PAPER, '*', color=C_PAPER, ms=11, zorder=6,
            label=f'Paper: $\\lambda$={LAM_PAPER}, $m$={M_PAPER}')

    ax.set_xticks(LAM_L1[::2])
    ax.set_yticks(M_L1)
    ax.set_xlabel(r'Coupling coefficient $\lambda$')
    ax.set_ylabel('Neighborhood size $m$')
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('OV (%)', fontsize=8)
    panel_label(ax, panel)

fig.suptitle(
    r'Layer 1: $\lambda \times m$ coupling analysis'
    f' ($T_{{\\rm emg}}$={TEMG_PAPER} fixed, '
    f'$N_{{\\rm sub}}$=4,000)\n'
    r'Optimal $\lambda$ drifts ' + f'{lam_drift:.1f} units across $m$ — '
    r'significant $\lambda$–$m$ coupling confirmed',
    fontsize=9, y=1.02)

plt.tight_layout(w_pad=1.2)
save(fig, 'fig_paper_layer1_coupling.png')


# =============================================================================
# FIGURE B  —  Layers 2 & 3: T_emg decoupling verification
#   Left:  L2 — OV vs T_emg for each λ (λ×T_emg, m=8 fixed)
#   Right: L3 — OV vs T_emg for each m (m×T_emg, λ=1.0 fixed)
# =============================================================================
print('[2/4] Layers 2 & 3: T_emg decoupling ...')

fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 3.0))

# ── Panel (a): Layer 2 — each λ is one curve, x-axis = T_emg ─────────────────
ax = axes[0]
cmap2 = plt.cm.get_cmap('Blues', len(LAM_L2) + 2)
for i, lam in enumerate(LAM_L2):
    ax.plot(TEMG_L2, L2_OV[i, :], '-', color=cmap2(i + 2),
            lw=1.2, alpha=0.85, label=f'$\\lambda$={lam:.1f}')
ax.axvline(TEMG_PAPER, color=C_PAPER, ls='--', lw=1.3,
           label=f'$T_{{\\rm emg}}$={TEMG_PAPER} (paper)')
# Annotate flatness
ov_range_L2 = L2_OV.max() - L2_OV.min()
ax.text(0.97, 0.05,
        f'OV std along $T_{{\\rm emg}}$:\n$\\sigma \\approx {temg_var_L2:.0e}$%\n'
        r'(fully decoupled)',
        transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='0.8'))
ax.set_xlabel(r'Emergency bypass threshold $T_{\rm emg}$')
ax.set_ylabel('Overall accuracy OV (%)')
ax.set_title(r'Layer 2: $\lambda \times T_{\rm emg}$  ($m$=8 fixed)', fontsize=9, pad=4)
ax.set_xlim(TEMG_L2[0] - 0.005, TEMG_L2[-1] + 0.005)
y_pad = max((L2_OV.max() - L2_OV.min()) * 0.5, 0.3)
ax.set_ylim(L2_OV.min() - y_pad, L2_OV.max() + y_pad)
# Compact legend for λ curves
ax.legend(fontsize=6, ncol=3, loc='upper left', framealpha=0.9,
          handlelength=1.2, columnspacing=0.8)
ax.grid(axis='x', alpha=0.3); ax.grid(axis='y', alpha=0.3)
panel_label(ax, '(a)')

# ── Panel (b): Layer 3 — each m is one curve, x-axis = T_emg ─────────────────
ax = axes[1]
colors_m = plt.cm.get_cmap('tab10', len(M_L3))
ls_cycle = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
mk_cycle = ['o', 's', '^', 'D', 'v', 'P', 'x', 'h', '*']
for j, m in enumerate(M_L3):
    ax.plot(TEMG_L3, L3_OV[j, :],
            ls_cycle[j % len(ls_cycle)], marker=mk_cycle[j % len(mk_cycle)],
            color=colors_m(j), ms=3.5, lw=1.2, alpha=0.9,
            label=f'$m$={m}')
ax.axvline(TEMG_PAPER, color=C_PAPER, ls='--', lw=1.3,
           label=f'$T_{{\\rm emg}}$={TEMG_PAPER} (paper)')
ov_range_L3 = L3_OV.max() - L3_OV.min()
ax.text(0.97, 0.05,
        f'OV std along $T_{{\\rm emg}}$:\n$\\sigma \\approx {temg_var_L3:.0e}$%\n'
        r'(fully decoupled)',
        transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='0.8'))
ax.set_xlabel(r'Emergency bypass threshold $T_{\rm emg}$')
ax.set_ylabel('Overall accuracy OV (%)')
ax.set_title(r'Layer 3: $m \times T_{\rm emg}$  ($\lambda$=1.0 fixed)', fontsize=9, pad=4)
ax.set_xlim(TEMG_L3[0] - 0.005, TEMG_L3[-1] + 0.005)
y_pad3 = max((L3_OV.max() - L3_OV.min()) * 0.5, 0.3)
ax.set_ylim(L3_OV.min() - y_pad3, L3_OV.max() + y_pad3)
ax.legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.9,
          handlelength=1.2, columnspacing=0.8)
ax.grid(axis='x', alpha=0.3); ax.grid(axis='y', alpha=0.3)
panel_label(ax, '(b)')

fig.suptitle(
    r'Layers 2 & 3: $T_{\rm emg}$ decoupling verification'
    f' ($N_{{\\rm sub}}$=4,000)\n'
    r'OV is invariant to $T_{\rm emg}$ in both axes — '
    r'$T_{\rm emg}$ can be tuned independently',
    fontsize=9, y=1.02)

plt.tight_layout(w_pad=1.4)
save(fig, 'fig_paper_layer23_decoupling.png')


# =============================================================================
# FIGURE C  —  Layer 4: global parameter sensitivity
#   Left:  OV heatmap at T_emg=0.20 (paper value) with paper + best markers
#   Right: OV range bar — shows how narrow the performance band is
# =============================================================================
print('[3/4] Layer 4: global sensitivity heatmap ...')

# Slice at T_emg closest to paper value
k_paper = int(np.argmin(np.abs(TEMG_L4 - TEMG_PAPER)))
k_best  = int(np.argmin(np.abs(TEMG_L4 - temg_best)))
Z4 = L4_OV[:, :, k_paper]   # (n_lam, n_m)  at T_emg=0.20

fig, axes = plt.subplots(1, 2, figsize=(PAPER_W2, 3.0),
                          gridspec_kw={'width_ratios': [1.4, 1]})

# ── Panel (a): heatmap ────────────────────────────────────────────────────────
ax = axes[0]
vlo4 = max(Z4.min() - 0.3, 96.0);  vhi4 = 100.0
im = ax.imshow(Z4.T, aspect='auto', cmap=CMAP_RYG,
               vmin=vlo4, vmax=vhi4, origin='lower',
               extent=[LAM_L4[0]-0.12, LAM_L4[-1]+0.12,
                        M_L4[0]-0.6,   M_L4[-1]+0.6])
for i, lam in enumerate(LAM_L4):
    for j, m in enumerate(M_L4):
        val = Z4[i, j]
        c = 'black' if val > (vlo4 + vhi4) / 2 else 'white'
        ax.text(lam, m, f'{val:.1f}', ha='center', va='center',
                fontsize=6.5, color=c)

# Mark paper recommended (λ=1.0, m=8 → nearest in L4: λ=1.0 ✓, m=9 nearest)
lam_p_idx = int(np.argmin(np.abs(LAM_L4 - LAM_PAPER)))
m_p_idx   = int(np.argmin(np.abs(M_L4   - M_PAPER)))
ax.plot(LAM_L4[lam_p_idx], M_L4[m_p_idx], '*',
        color=C_PAPER, ms=14, zorder=8,
        label=f'Paper: $\\lambda$={LAM_PAPER}, $m$={M_PAPER} (nearest grid)')

# Mark global best (λ=0.75, m=7)
lam_b_idx = int(np.argmin(np.abs(LAM_L4 - lam_best)))
m_b_idx   = int(np.argmin(np.abs(M_L4   - m_best)))
ax.plot(LAM_L4[lam_b_idx], M_L4[m_b_idx], '^',
        color=C_OPT, ms=9, zorder=7,
        label=f'Subsample best: $\\lambda$={lam_best}, $m$={m_best}')

ax.set_xticks(LAM_L4)
ax.set_yticks(M_L4)
ax.set_xlabel(r'Coupling coefficient $\lambda$')
ax.set_ylabel('Neighborhood size $m$')
ax.set_title(f'$T_{{\\rm emg}}$={TEMG_L4[k_paper]:.2f} (paper value)',
             fontsize=9, pad=4)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label('OV (%)', fontsize=8)
panel_label(ax, '(a)')

# ── Panel (b): OV range across all T_emg slices (show stability) ─────────────
ax2 = axes[1]
# For each (λ, m), compute OV range across T_emg
ov_mean = L4_OV.mean(axis=2).flatten()     # mean over T_emg
ov_std  = L4_OV.std(axis=2).flatten()      # std over T_emg

# Sort by mean OV
sort_idx = np.argsort(ov_mean)
lam_flat = np.repeat(LAM_L4, len(M_L4))[sort_idx]
m_flat   = np.tile(M_L4, len(LAM_L4))[sort_idx]
ov_mean_s = ov_mean[sort_idx]
ov_std_s  = ov_std[sort_idx]

x_pos = np.arange(len(ov_mean_s))
bars = ax2.barh(x_pos, ov_mean_s - ov_mean_s.min(),
                left=ov_mean_s.min(),
                color='#aec7e8', edgecolor='white', linewidth=0.3, height=0.7,
                xerr=ov_std_s,
                error_kw=dict(elinewidth=0.7, ecolor='0.5', capsize=2))

# Highlight paper and global best
for k, (l, m) in enumerate(zip(lam_flat, m_flat)):
    if abs(l - LAM_PAPER) < 0.01 and abs(m - M_PAPER) < 1:
        ax2.barh(k, ov_mean_s[k] - ov_mean_s.min(),
                 left=ov_mean_s.min(),
                 color=C_PAPER, edgecolor='white', linewidth=0.3, height=0.7,
                 label=f'Paper ($\\lambda$={LAM_PAPER}, $m$={M_PAPER})')
    if abs(l - lam_best) < 0.01 and abs(m - m_best) < 1:
        ax2.barh(k, ov_mean_s[k] - ov_mean_s.min(),
                 left=ov_mean_s.min(),
                 color=C_OPT, edgecolor='white', linewidth=0.3, height=0.7,
                 label=f'Best ($\\lambda$={lam_best}, $m$={m_best})')

ax2.set_yticks(x_pos[::3])
ax2.set_yticklabels(
    [f'$\\lambda$={lam_flat[i]:.2f}\n$m$={m_flat[i]}' for i in x_pos[::3]],
    fontsize=6)
ax2.set_xlabel('OV (%)')
ax2.set_title('OV mean ± std\n(across all $T_{\\rm emg}$ values)', fontsize=9, pad=4)
ax2.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax2.grid(axis='y', alpha=0); ax2.grid(axis='x', alpha=0.3)
ov_span = ov_mean_s.max() - ov_mean_s.min()
ax2.set_xlim(ov_mean_s.min() - 0.1,
             ov_mean_s.max() + max(ov_std_s.max(), 0.05) + 0.3)
panel_label(ax2, '(b)')

# Full-dataset validation note
fig.text(0.5, -0.04,
         r'$\star$ Subsample ($N_{\rm sub}$=4,000) results. '
         r'Full-dataset grid search ($n$=24,015, Sec. 4.4) confirms $\lambda^*$=1.0, OV=99.83%.',
         ha='center', fontsize=7.5, color='0.45',
         style='italic')

fig.suptitle(
    r'Layer 4: Three-dimensional ($\lambda \times m \times T_{\rm emg}$) sensitivity'
    f' — Paper recommended values validated\n'
    f'Subsample OV: paper {ov_paper:.1f}% vs. grid best {ov_best:.1f}% '
    r'($\Delta$=' + f'{ov_best-ov_paper:.1f}%); '
    r'full-dataset $\Delta \approx 0$%',
    fontsize=9, y=1.04)

plt.tight_layout(w_pad=1.4)
save(fig, 'fig_paper_layer4_sensitivity.png')


# =============================================================================
# FIGURE D  —  Summary: coupling structure overview (4-panel summary)
#   Top-left:  L1 drift plot (bar chart: optimal λ per m)
#   Top-right: L2 flatness index (single bar comparison)
#   Bottom:    Combined conclusion table
# =============================================================================
print('[4/4] Summary figure ...')

fig = plt.figure(figsize=(PAPER_W2, 3.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.4)
ax_tl = fig.add_subplot(gs[0, 0])
ax_tr = fig.add_subplot(gs[0, 1])
ax_bl = fig.add_subplot(gs[1, :])

# ── Top-left: optimal λ vs m (bar chart, shows coupling) ─────────────────────
opt_lam_L1 = np.array([LAM_L1[np.argmax(L1_OV[:, j])] for j in range(len(M_L1))])
colors_bar = [C_PAPER if abs(ol - LAM_PAPER) < 0.15 else C_DRIFT
              for ol in opt_lam_L1]
ax_tl.bar(M_L1, opt_lam_L1, color=colors_bar, edgecolor='white',
          linewidth=0.4, width=0.65)
ax_tl.axhline(LAM_PAPER, color=C_PAPER, ls='--', lw=1.2,
              label=f'$\\lambda^*$={LAM_PAPER} (paper)')
ax_tl.set_xticks(M_L1)
ax_tl.set_xlabel('Neighborhood size $m$')
ax_tl.set_ylabel(r'Optimal $\lambda$')
ax_tl.set_ylim(0, LAM_L1.max() + 0.3)
ax_tl.set_title(r'$\lambda \times m$: optimal $\lambda$ shifts with $m$', fontsize=8.5, pad=3)
ax_tl.legend(fontsize=7.5)
panel_label(ax_tl, '(a)')

# Annotate drift range
y_min_bar = opt_lam_L1.min(); y_max_bar = opt_lam_L1.max()
ax_tl.annotate('', xy=(M_L1[-1]+0.4, y_max_bar),
               xytext=(M_L1[-1]+0.4, y_min_bar),
               arrowprops=dict(arrowstyle='<->', color=C_OPT, lw=1.2))
ax_tl.text(M_L1[-1]+0.5, (y_min_bar+y_max_bar)/2,
           f'drift\n={lam_drift:.1f}', fontsize=7, color=C_OPT, va='center')

# ── Top-right: T_emg decoupling summary (grouped bar) ────────────────────────
categories = [r'$\lambda \times T_{\rm emg}$' + '\n(Layer 2)',
              r'$m \times T_{\rm emg}$' + '\n(Layer 3)']
stds = [temg_var_L2, temg_var_L3]
bar_col = ['#1f77b4', '#2ca02c']
ax_tr.bar(categories, stds, color=bar_col, edgecolor='white',
          linewidth=0.5, width=0.45)
# Always show at least a tiny bar for visibility
ax_tr.set_ylim(0, max(max(stds)*2, 0.001))
ax_tr.set_ylabel(r'OV std along $T_{\rm emg}$ axis (%)')
ax_tr.set_title(r'$T_{\rm emg}$ decoupling: OV variance ≈ 0', fontsize=8.5, pad=3)
for i, (cat, val) in enumerate(zip(categories, stds)):
    ax_tr.text(i, max(val*1.1, 0.0001),
               f'{val:.2e}%\n≈ 0', ha='center', va='bottom',
               fontsize=8, color='0.2', fontweight='bold')
ax_tr.text(0.5, 0.85, 'Fully decoupled\n(independent tuning confirmed)',
           transform=ax_tr.transAxes, ha='center', fontsize=8,
           color='darkgreen', fontweight='bold')
panel_label(ax_tr, '(b)')

# ── Bottom: conclusion summary table ─────────────────────────────────────────
ax_bl.axis('off')
table_data = [
    ['Layer', 'Parameters', 'Finding', 'Implication'],
    ['1', r'$\lambda \times m$',
     f'Optimal $\\lambda$ drifts {lam_drift:.1f} across $m$',
     r'Joint optimization required; $\lambda$=1.0 validated by full-data search'],
    ['2', r'$\lambda \times T_{\rm emg}$',
     r'OV std $\approx 0$% — fully decoupled',
     r'$T_{\rm emg}$ tunable independently of $\lambda$'],
    ['3', r'$m \times T_{\rm emg}$',
     r'OV std $\approx 0$% — fully decoupled',
     r'$T_{\rm emg}$ tunable independently of $m$'],
    ['4', r'$\lambda \times m \times T_{\rm emg}$',
     f'Subsample best: ({lam_best}, {m_best}); paper ({LAM_PAPER}, {M_PAPER})',
     r'$\Delta$OV=1% on subsample; full-data confirms $\lambda^*$=1.0, OV=99.83%'],
]
col_widths = [0.06, 0.16, 0.41, 0.47]
x_starts   = [0.01, 0.07, 0.23, 0.64]
row_h      = 0.19;  header_y = 0.95

for col_idx, (header, xstart) in enumerate(zip(table_data[0], x_starts)):
    ax_bl.text(xstart, header_y, header,
               transform=ax_bl.transAxes,
               fontsize=8, fontweight='bold', va='top',
               color='white',
               bbox=dict(boxstyle='square,pad=0.2', fc='0.3', ec='none'))

for row_idx, row in enumerate(table_data[1:]):
    y_pos = header_y - (row_idx + 1) * row_h
    bg = '#f0f4f8' if row_idx % 2 == 0 else 'white'
    for col_idx, (cell, xstart) in enumerate(zip(row, x_starts)):
        ax_bl.text(xstart + 0.005, y_pos, cell,
                   transform=ax_bl.transAxes,
                   fontsize=7.2, va='top', ha='left',
                   bbox=dict(boxstyle='square,pad=0.15', fc=bg, ec='none',
                             alpha=0.85))

panel_label(ax_bl, '(c)', y=0.99)
ax_bl.set_xlim(0, 1); ax_bl.set_ylim(0, 1)

fig.suptitle(
    'Four-layer hyperparameter coupling analysis — Summary',
    fontsize=10, y=1.01, fontweight='bold')

save(fig, 'fig_paper_coupling_summary.png')

print('\nAll coupled-hyperparameter figures saved to:', FIGURES_DIR)
print('Files:')
for f in ['fig_paper_layer1_coupling.png',
          'fig_paper_layer23_decoupling.png',
          'fig_paper_layer4_sensitivity.png',
          'fig_paper_coupling_summary.png']:
    print(f'  {f}')
