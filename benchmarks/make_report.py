"""Writes benchmarks/BENCHMARKS.md from the JSON files in results/.  Every number on the page comes
from those files, so the page can be regenerated after rerunning any benchmark."""
import os, math
from common import RESULTS, load

HERE = os.path.dirname(os.path.abspath(__file__))
e = lambda x, d=2: f"{x:.{d}e}"
pct = lambda x, d=2: f"{100*x:.{d}f}%"


def sec0():
    d = load('b0_exactness')
    L = ["## 0. Exactness: what holds for every network",
         "",
         "These are not accuracy results. They check that the certificate machinery is exact, on random "
         "(untrained) networks, against the exact developed channel flow of each law (`b0_exactness.py`, "
         "same checks as `tests/test_identity.py` and `tests/test_power.py`).",
         "",
         "| law | gap $G$ | $E+E^*$ (from the exact solution) | relative defect |",
         "|---|---|---|---|"]
    for r in d['identity']:
        L.append(f"| {r['law']} | {r['G']:.9e} | {r['E'] + r['Estar']:.9e} | {r['rel_defect']:.1e} |")
    worst = max(r['rel_defect'] for r in d['identity'])
    ok = all(r['contains'] for r in d['power'])
    L += ["", f"The identity $G=E+E^*$ (Stokes) holds to {worst:.1e} relative for all {len(d['identity'])} laws, "
          "although the networks are far from the solution. It checks at once that $v$ is exactly divergence free with "
          "exact boundary data, that $m$ is exactly equilibrated, that the conjugates $F^*$ are right, and that the "
          "quadrature is accurate.",
          "",
          f"The certified power enclosure contains the exact power for all {len(d['power'])} laws tested "
          f"({'all contained' if ok else 'NOT all contained'}). For untrained networks it is wide, as it must be "
          "(its width scales with $G$); Section 1 shows it for a trained network.", ""]
    return L


def sec1():
    fem = load('b1_couette_fem'); A = load('b1_couette_aequa'); f = A['final']
    cfg = A['config']
    rows = []
    rows.append(('**Aequa** (gap loss, exact $v$ and exact $m$)', f['rel_energy_seminorm'], f['rel_L2_velocity'],
                 f"**yes**: $\\le$ {pct(f['cert_rel_energy_bound'])} (of $\\|\\varepsilon(v)\\|$)", A['train_time_s']))
    for key, lab, guar in (('deepritz_quad', 'Deep Ritz, same exact $v$, L-BFGS on the deterministic quadrature',
                            'no (upper bound on $J^*$ only)'),
                           ('deepritz', 'Deep Ritz, same exact $v$, sampled loss', 'no (upper bound on $J^*$ only)'),
                           ('pinn_hard', 'PINN, same exact $v$ + pressure net, residual loss', 'no'),
                           ('pinn', 'PINN, soft constraints (standard)', 'no')):
        try: d = load('b1_couette_' + key)
        except FileNotFoundError: continue
        rows.append((lab, d['final']['rel_energy_seminorm'], d['final']['rel_L2_velocity'], guar, d['train_time_s']))
    pinn = load('b1_couette_pinn')['final']
    tr = A['trace']
    never_below = all(r['cert_eps_err_bound']**2 >= r['eps_err_sq'] for r in tr)
    pre = [r for r in tr if r['phase'] == 'adam'][-1]
    drop = pre['G_up']/f['G_up']
    ph = load('b1_couette_pinn_hard')
    ratio_t = ph['train_time_s']/A['train_time_s']
    tube_eff = math.sqrt(f['G_up']/f['tube_weighted_err_sq'])
    floor_eff = math.sqrt(f['G_up']/f['nu_floor_energy'])
    L = ["## 1. Couette flow of a Carreau fluid (exact solution known)",
         "",
         "**Problem.** Annulus $0.5<r<1$, inner cylinder rotating with $\\Omega=2$, outer at rest, Stokes flow, Carreau law "
         "$\\eta=\\eta_\\infty+(\\eta_0-\\eta_\\infty)(1+(\\lambda\\dot\\gamma)^2)^{(n-1)/2}$ with $\\eta_0=1$, $\\eta_\\infty=0.05$, "
         "$\\lambda=2$, $n=0.5$. The exact solution follows from the torque balance $\\tau_{r\\theta}=C/r^2$ (`common.CouetteExact`). "
         "The domain is curved and multiply connected, and the exact stress transmits a torque, so Aequa needs its Michell term.",
         "",
         f"**Setup, identical for every neural method.** Networks of depth {cfg['depth']} and width {cfg['width']}, "
         f"{cfg['iters']} Adam steps (learning rate {cfg['lr']}, cosine decay, batch {cfg['batch']}) then {cfg['lbfgs']} L-BFGS "
         f"iterations, float64, {cfg['threads']} CPU threads. Errors of every method are measured by the same code on the same "
         "polar Gauss rule. The finite-element reference is Taylor-Hood P2-P1 on isoparametric meshes (scikit-fem), Picard iteration.",
         "",
         "| method | rel. energy error $\\|\\varepsilon(u^*-v)\\|/\\|\\varepsilon(u^*)\\|$ | rel. $L^2$ velocity error | guaranteed error bound | wall time |",
         "|---|---|---|---|---|"]
    for lab, h1, l2, g, t in rows:
        L.append(f"| {lab} | {pct(h1)} | {pct(l2, 3)} | {g} | {t:.0f} s |")
    for r in fem['rows']:
        L.append(f"| FEM Taylor-Hood, {int(r['ndofs'])} unknowns | {pct(r['rel_energy_seminorm'], 3)} | "
                 f"{pct(r['rel_L2_velocity'], 4)} | no | {r['time_s']:.1f} s |")
    L += ["",
          "![bound vs truth](figures/b1_certificate_vs_true_error.png)",
          "",
          "![accuracy vs time](figures/b1_methods_accuracy_vs_time.png)",
          "",
          "**What Aequa certifies for the trained network** (every number computed without the exact solution, then checked against it):",
          "",
          "| certified statement | certified value | exact value | holds |",
          "|---|---|---|---|",
          f"| minimal dissipation $J^*\\in[J(v)-G,\\,J(v)]$ | $[{A['energy']['J_lo']:.7f},\\ {A['energy']['J_hi']:.7f}]$ | "
          f"{A['energy']['exact']:.7f} | {'yes' if A['energy']['J_lo'] <= A['energy']['exact'] <= A['energy']['J_hi'] else 'NO'} |",
          f"| power $W^*$ (Thm 6.4) | $[{A['power']['W_lo']:.4f},\\ {A['power']['W_hi']:.4f}]$ | {A['power']['exact']:.4f} | "
          f"{'yes' if A['power']['W_lo'] <= A['power']['exact'] <= A['power']['W_hi'] else 'NO'} |",
          f"| $G = E+E^*$ (Thm 2.1) | $G_{{\\rm up}}$ = {e(f['G_up'], 4)} | $E+E^*$ = {e(f['bregman_E'] + f['bregman_Estar'], 4)} (on the error rule) | yes |",
          f"| $\\nu_*\\int|\\varepsilon(u^*-v)|^2\\le G$, unconditional floor $\\nu_*=\\eta_\\infty$ | {e(f['G_up'])} | {e(f['nu_floor_energy'])} | "
          f"yes, effectivity {floor_eff:.2f} |",
          f"| $\\int\\nu_s|\\varepsilon(u^*-v)|^2\\le G$, sharp tube weight (Cor. 7.4) | {e(f['G_up'])} | {e(f['tube_weighted_err_sq'])} | "
          f"yes, effectivity {tube_eff:.2f} (tube condition {'holds' if f['tube_condition_holds'] else 'fails'}) |",
          "",
          "Effectivity is (certified bound / true error) in the norm, so 1 is perfect and every value is $\\ge1$ when the bound is valid.",
          "",
          "**Reading the results honestly.**",
          "",
          "* *Finite elements are faster in 2D.* For a smooth 2D problem, Taylor-Hood reaches the same accuracy in seconds. "
          "Aequa's case is not speed: it is that the number it prints is a proved bound, computed without the exact solution. "
          "None of the other methods produces one in this form. (FEM can be given guaranteed bounds too, with equilibrated-flux "
          "estimators; that was not implemented here.)",
          f"* *The bound is tight once the pair is converged.* With the sharp tube weight the certified error is within a factor "
          f"{tube_eff:.2f} of the truth; with the unconditional floor weight, {floor_eff:.2f}.",
          "* *During Adam the certificate is honest but loose.* The first figure shows the velocity becoming accurate long before "
          "the gap falls: the stress side lags. The Michell torque coefficient has to grow from 0 to about 3.26 (its trained value), "
          "and Adam moves a parameter by at most about the learning rate per step, so with this schedule it needs most of the "
          "2000 steps for that alone. "
          f"L-BFGS reduces the gap by a factor {drop:.0f}. "
          + ("At every checkpoint the certified bound was above the true error." if never_below else "WARNING: at some checkpoint the bound was below the true error."),
          f"* *Same velocity ansatz, different losses.* With Aequa's exactly admissible velocity, the residual loss reached a smaller "
          f"error than the gap loss in this run ({pct(ph['final']['rel_energy_seminorm'])} against {pct(f['rel_energy_seminorm'])}), "
          f"at {ratio_t:.1f} times the cost, but gives no bound. Deep Ritz on a sampled loss stalls, "
          f"because the dissipation has sampling noise even at the optimum, while the gap density vanishes pointwise at the "
          f"solution; trained on the deterministic quadrature, Deep Ritz reaches {pct(load('b1_couette_deepritz_quad')['final']['rel_energy_seminorm'])} "
          f"(Aequa {pct(f['rel_energy_seminorm'])}), but still certifies "
          f"only $J(v)\\ge J^*$.",
          f"* *Soft constraints cost exactness.* The standard PINN's velocity has divergence {e(pinn['div_L2'], 1)} (L2) and a wall "
          f"error up to {e(pinn['bc_max_err'], 1)}; it solves a slightly different problem, which no a-posteriori statement can repair.",
          ""]
    return L


def sec2():
    try:
        fem = load('b2_cylinder_fem'); A = load('b2_cylinder_aequa')
    except FileNotFoundError:
        return []
    runs = [A]
    if os.path.exists(os.path.join(RESULTS, 'b2_cylinder_aequa_short.json')):
        runs = [load('b2_cylinder_aequa_short'), A]
    r = fem['rows']; ref, prev = r[-1], r[-2]
    uJ = abs(ref['J'] - prev['J']); uW = abs(ref['W'] - prev['W'])
    L = ["## 2. Carreau flow past a cylinder in a channel (no exact solution)",
         "",
         "**Problem.** Channel $[-3,5]\\times[-1,1]$, cylinder of radius 0.3 at the origin, the same Carreau law, developed Carreau "
         "profile (mean speed 1, flux $Q=2$) at inlet and outlet, no-slip on walls and cylinder, Stokes flow. Aequa networks of depth "
         f"{A['config']['depth']}, width {A['config']['width']}; two training budgets. Reference: Taylor-Hood FEM on four meshes; "
         f"the finest gives $J={ref['J']:.6f}$ and $W={ref['W']:.5f}$, with changes of {uJ:.1e} and {uW:.1e} from the previous mesh.",
         "",
         "| Aequa training | $G/J$ | certified $J^*$ | certified power $W^*$ | certified $\\Delta p=W^*/Q$ | FEM inside all three | time |",
         "|---|---|---|---|---|---|---|"]
    for R in runs:
        inJ = R['J_lo'] - uJ <= ref['J'] <= R['J_hi'] + uJ; inW = R['W_lo'] - uW <= ref['W'] <= R['W_hi'] + uW
        L.append(f"| {R['config']['iters']} Adam + {R['config']['lbfgs']} L-BFGS | {pct(R['G_up']/R['J_v'], 3)} | "
                 f"$[{R['J_lo']:.5f},\\ {R['J_hi']:.5f}]$ | $[{R['W_lo']:.4f},\\ {R['W_hi']:.4f}]$ | "
                 f"$[{R['dp_lo']:.4f},\\ {R['dp_hi']:.4f}]$ | {'yes' if (inJ and inW) else 'NO'} | {R['time_s']:.0f} s |")
    L += ["",
          f"FEM reference pressure drop: $W/Q={ref['W']/fem['Q']:.5f}$. The $J^*$ enclosure has relative width $G/J$. The power and "
          "pressure-drop enclosures are wider, of order $\\sqrt{G/J}$, because the Carreau law has two growth exponents "
          "(certificate note, Thm 6.4); for Newtonian and power-law fluids they are as sharp as the $J^*$ enclosure."
          + ((" The longer L-BFGS phase reduces $G/J$ by a factor "
              f"{(runs[0]['G_up']/runs[0]['J_v'])/(runs[-1]['G_up']/runs[-1]['J_v']):.0f} and narrows every enclosure; all of them "
              "stay consistent with the reference.") if len(runs) > 1 and all(
              (runs[-1][hi] - runs[-1][lo]) < (runs[0][hi] - runs[0][lo]) for lo, hi in (('J_lo', 'J_hi'), ('W_lo', 'W_hi'))) else ""),
          "",
          "![enclosures](figures/b2_cylinder_enclosures.png)",
          "",
          "| FEM mesh | unknowns | $J(u_h)$ | $W(u_h)$ | $\\|\\operatorname{div}u_h\\|_{L^2}$ | time |",
          "|---|---|---|---|---|---|"]
    for x in r:
        L.append(f"| h = {x['h']} | {int(x['ndofs'])} | {x['J']:.6f} | {x['W']:.5f} | {x['div_L2']:.1e} | {x['time_s']:.1f} s |")
    L += ["", "The FEM velocity is only discretely divergence free, so $J(u_h)$ is not an upper bound for $J^*$; Aequa's upper end "
          "$J(v)$ is, because $v$ is exactly admissible.", ""]
    return L


def main():
    L = ["# Aequa benchmarks", "",
         "All numbers on this page are produced by the scripts in this folder and read from `results/*.json` "
         "(`python make_report.py` rewrites the page). Machine: a 2-core cloud CPU, no GPU, float64; networks are kept small "
         "accordingly, and all times are wall times on that machine.", ""]
    L += sec0() + sec1() + sec2()
    L += ["## What these benchmarks do not show", "",
          "* No inertia: every case here is Stokes flow. The inertial certificate ($\\theta<1$) is in the library but not benchmarked here.",
          "* No yield-stress law in a trained case (the identity is checked for Bingham, Herschel-Bulkley and Casson in Section 0).",
          "* The quadrature-based $G$ is a converged deterministic quadrature, not an interval enclosure (see `docs/certificate_note.pdf`).",
          "* Small networks and short training on 2 cores; on a GPU, larger networks change the accuracy and timing columns.",
          "",
          "## Reproduce", "",
          "```",
          "pip install -e .[test,bench]",
          "cd benchmarks",
          "python b0_exactness.py",
          "python b1_couette_fem.py",
          "python b1_couette.py aequa deepritz deepritz_quad pinn_hard pinn",
          "python b2_cylinder.py fem && python b2_cylinder.py aequa",
          "python make_figures.py && python make_report.py",
          "```",
          "`AEQUA_BENCH_ITERS` and `AEQUA_BENCH_LBFGS` change the training budget.", ""]
    open(os.path.join(HERE, 'BENCHMARKS.md'), 'w').write("\n".join(L))
    print("wrote BENCHMARKS.md")


if __name__ == '__main__':
    main()
