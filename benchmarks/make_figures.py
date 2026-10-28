"""Figures for BENCHMARKS.md, rebuilt from the JSON files in results/ (no training here)."""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RESULTS, load

# reference categorical slots (light mode), fixed order; text stays in neutral ink
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
                     'figure.facecolor': 'white', 'savefig.dpi': 160, 'legend.frameon': False})
FIG = os.path.join(os.path.dirname(RESULTS), 'figures')
os.makedirs(FIG, exist_ok=True)


def have(name): return os.path.exists(os.path.join(RESULTS, name + '.json'))


def fig_trace():
    if not have('b1_couette_aequa'): return
    tr = load('b1_couette_aequa')['trace']
    it = np.array([max(r['it'], 1) for r in tr])
    true = np.sqrt([r['eps_err_sq'] for r in tr])
    bound = np.array([r['cert_eps_err_bound'] for r in tr])
    G = np.array([r['G_up'] for r in tr]); EE = np.array([r['bregman_E'] + r['bregman_Estar'] for r in tr])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    a = ax[0]
    a.loglog(it, bound, '-o', color=C[1], ms=5, label='certified upper bound (computed without $u^*$)')
    a.loglog(it, true, '-o', color=C[0], ms=5, label='true error (needs the exact solution)')
    a.fill_between(it, true, bound, color=C[1], alpha=0.10, lw=0)
    a.set_xlabel('training iteration (iteration 0 drawn at 1; last point after L-BFGS)')
    a.set_ylabel(r'$\|\varepsilon(u^*-v)\|_{L^2}$')
    a.set_title(r'Energy error: bound $\sqrt{G/\nu_*}$ vs truth', color=INK, fontsize=10)
    a.legend(loc='lower left', fontsize=8.5)
    a = ax[1]
    a.loglog(it, G, '-o', color=C[1], ms=5, label=r'certified $G_{\rm up}$')
    a.loglog(it, EE, '--', color=C[0], label=r'true $E+E^*$')
    a.set_xlabel('training iteration'); a.set_ylabel('energy units')
    rel = np.max(np.abs(G - EE)/EE)
    a.set_title(f'Gap identity $G=E+E^*$ (max rel. diff. {rel:.1e})', color=INK, fontsize=10)
    a.legend(loc='lower left', fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'b1_certificate_vs_true_error.png')); plt.close(fig)


def fig_methods():
    if not have('b1_couette_fem'): return
    fem = load('b1_couette_fem')['rows']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for k, key in enumerate(('rel_energy_seminorm', 'rel_L2_velocity')):
        a = ax[k]
        a.loglog([r['time_s'] for r in fem], [r[key] for r in fem], '-s', color=C[4], ms=6,
                 label='Taylor-Hood FEM (5 meshes), no bound')
        for (name, lab, col, cert) in (('aequa', 'Aequa (gap loss), certified', C[0], True),
                                        ('deepritz', 'Deep Ritz, sampled loss, no bound', C[1], False),
                                        ('deepritz_quad', 'Deep Ritz, quadrature L-BFGS, no bound', C[5], False),
                                        ('pinn_hard', 'PINN, exact $v$ + residual loss, no bound', C[2], False),
                                        ('pinn', 'PINN, soft constraints, no bound', C[3], False)):
            if not have('b1_couette_' + name): continue
            d = load('b1_couette_' + name)
            f = d['final']
            a.loglog([d['train_time_s']], [f[key]], 'o', ms=10, color=col, mfc=col if cert else 'white', mew=2, label=lab)
            if cert and key == 'rel_energy_seminorm':
                b = d['final']['cert_rel_energy_bound']
                a.loglog([d['train_time_s']], [b], 'v', ms=9, color=col, label='Aequa certified upper bound')
                a.vlines(d['train_time_s'], f[key], b, color=col, lw=1.2)
        a.set_xlabel('wall time on 2 CPU cores [s]')
        a.set_ylabel('relative energy-seminorm error' if k == 0 else r'relative $L^2$ velocity error')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5)
    fig.suptitle('Carreau Couette flow: accuracy against cost', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(os.path.join(FIG, 'b1_methods_accuracy_vs_time.png')); plt.close(fig)


def fig_cylinder():
    if not (have('b2_cylinder_fem') and have('b2_cylinder_aequa')): return
    fem = load('b2_cylinder_fem')['rows']; aq = load('b2_cylinder_aequa')
    n = [r['ndofs'] for r in fem]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, key, lo, hi, lab in ((ax[0], 'J', aq['J_lo'], aq['J_hi'], r'minimal dissipation $J^*$'),
                                (ax[1], 'W', aq['W_lo'], aq['W_hi'], r'power $W^*=Q\,\Delta p$')):
        a.axhspan(lo, hi, color=C[0], alpha=0.15, lw=0)
        a.axhline(lo, color=C[0], lw=1.2); a.axhline(hi, color=C[0], lw=1.2)
        a.semilogx(n, [r[key] for r in fem], '-s', color=C[4], ms=6, label='FEM on refined meshes')
        a.text(n[0], hi, '  Aequa certified enclosure', va='bottom', color=INK2, fontsize=8.5)
        a.set_xlabel('FEM unknowns'); a.set_title(lab, color=INK, fontsize=10)
        a.legend(fontsize=8.5, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'b2_cylinder_enclosures.png')); plt.close(fig)


if __name__ == '__main__':
    fig_trace(); fig_methods(); fig_cylinder(); print('figures in', FIG)
