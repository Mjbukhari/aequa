"""Benchmark 1: Couette flow of a Carreau fluid between a rotating inner cylinder and a fixed outer one.

Exact solution known (common.CouetteExact), curved and multiply connected domain (the exact stress
carries a torque, so Aequa needs its Michell term).  Four neural methods with the same network size,
sampling, optimiser and iteration budget:

  aequa      gap loss, exactly admissible pair  -> certified error bound
  deepritz   energy loss, same exact velocity   -> upper bound on J* only
  pinn_hard  residual loss, same exact velocity -> no bound
  pinn       residual loss, soft constraints    -> no bound

Every method is measured by the same code (common.error_report) on the same polar Gauss rule.
For Aequa the certificate is also recorded at checkpoints during training, next to the true error.

usage:  python b1_couette.py [method ...]      (default: all four)
"""
import math, sys, time, os
import numpy as np
import torch
import aequa as aq
from aequa.solver import DT
from common import CouetteExact, CARREAU, annulus_rule, error_report, save

ITERS = int(os.environ.get('AEQUA_BENCH_ITERS', 2000))
LBFGS = int(os.environ.get('AEQUA_BENCH_LBFGS', 200))
WIDTH, DEPTH, BATCH, LR = 32, 3, 1024, 3e-3
CHECKPOINTS = sorted(set([0, 25, 50, 100, 200, 400, 800, 1200, 1600, ITERS]))
torch.set_num_threads(os.cpu_count())

ex = CouetteExact()
law = aq.Carreau(**CARREAU)
P, W = annulus_rule(ex.Ri, ex.Ro, nr=32, nt=128)


def make_solver(seed=0):
    dom = aq.annulus(ex.Ro, ex.Ri)
    dom.set_bc({'outer': aq.NoSlip(), 'inner': aq.Rotating((0.0, 0.0), ex.Om)})
    return aq.Solver(dom, law, rho=0.0, width=WIDTH, depth=DEPTH, seed=seed)


def true_errors_aequa(S):
    f = S.evaluate(P, keys=('u', 'v', 'e11', 'e12', 'm11', 'm12', 'taum'))
    rep = error_report(P, W, ex, f['u'], f['v'], f['e11'], f['e12'])
    # dual Bregman distance E* = int F*(|m|/sqrt2) - F*(|S|/sqrt2) - eps(u):(m - S)
    S11, S12 = ex.stress(P[:, 0], P[:, 1]); ae, be = ex.strain(P[:, 0], P[:, 1])
    Fs = lambda t: law.Fstar(torch.tensor(t, dtype=DT)).numpy()
    Es = Fs(f['taum']) - Fs(np.sqrt(S11**2 + S12**2)) - 2*(ae*(f['m11'] - S11) + be*(f['m12'] - S12))
    rep['bregman_Estar'] = float((Es*W).sum())
    # sharp tube weight (Thm 3.4 / Cor 7.4): int nu_s |eps(u - v)|^2 <= G when |eps(u-v)| <= s on the domain
    s = 0.1*S.Us/S.Ls
    nev = np.sqrt(2*f['e11']**2 + 2*f['e12']**2)
    nu_s = law.nu0(torch.tensor(math.sqrt(2)*(nev + s), dtype=DT)).numpy()
    de = 2*((f['e11'] - ae)**2 + (f['e12'] - be)**2)
    rep['tube_weighted_err_sq'] = float((nu_s*de*W).sum())
    rep['tube_condition_holds'] = bool(np.sqrt(de).max() <= s)
    rep['tube_s'] = s
    rep['eps_v_sq'] = float((nev**2*W).sum())
    return rep


def run_aequa():
    S = make_solver()
    opt = torch.optim.Adam(S.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS, eta_min=LR*0.02)
    trace, t_train, t0 = [], 0.0, time.time()
    def checkpoint(it, phase):
        tc = time.time()
        gap = S.certify_gap()
        rep = true_errors_aequa(S)
        nu = law.floor
        row = dict(it=it, phase=phase, train_time_s=t_train, G_up=gap['G_up'], G_fine=gap['fine']['G'],
                   G_coarse=gap['coarse']['G'], quad_diff=gap['quad_diff'], n_quad=gap['fine']['n_points'],
                   J_v=gap['fine']['diss'],
                   cert_eps_err_bound=math.sqrt(gap['G_up']/nu),           # >= ||eps(u - v)|| (Lemma 2.4)
                   cert_rel_energy_bound=math.sqrt(gap['G_up']/nu/rep['eps_v_sq']),
                   **rep, certify_time_s=time.time() - tc)
        trace.append(row)
        print(f"[aequa] it {it:5d} ({phase}) G_up {row['G_up']:.3e}  E+E* {rep['bregman_E'] + rep['bregman_Estar']:.3e}  "
              f"relL2 {rep['rel_L2_velocity']:.2e} relH1 {rep['rel_energy_seminorm']:.2e} "
              f"bound(H1) {row['cert_rel_energy_bound']:.2e}  [train {t_train:.0f}s, cert {row['certify_time_s']:.0f}s]", flush=True)
        return row
    for it in range(ITERS + 1):
        if it in CHECKPOINTS: checkpoint(it, 'adam')
        if it == ITERS: break
        ts = time.time()
        X, w = S.sample(BATCH); X.requires_grad_(True)
        loss = (w*S.fields(X, smooth=S.smooth)['gap']).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        t_train += time.time() - ts
    if LBFGS:
        ts = time.time()
        X, wL = S.sample(8*BATCH); X.requires_grad_(True)
        opt2 = torch.optim.LBFGS(S.parameters(), lr=0.5, max_iter=LBFGS, history_size=50, line_search_fn='strong_wolfe')
        def closure():
            opt2.zero_grad(); l = (wL*S.fields(X, smooth=S.smooth)['gap']).sum(); l.backward(); return l
        opt2.step(closure)
        t_train += time.time() - ts
        checkpoint(ITERS + LBFGS, 'lbfgs')
    gap = S.certify_gap()
    pb = S.power_bounds(gap)
    res = aq.Result(S, gap, None, 0.0, 0.1)
    summary = res.summary()
    print(summary)
    torch.save(S.state_dict(), os.path.join(os.path.dirname(__file__), 'results', 'b1_aequa_state.pt'))
    return dict(trace=trace, final=trace[-1], power=dict(W_lo=pb['W_lo'], W_hi=pb['W_hi'], exact=ex.power()),
                energy=dict(J_lo=gap['fine']['diss'] - gap['G_up'], J_hi=gap['fine']['diss'], exact=ex.dissipation()),
                summary=summary, train_time_s=t_train)


def run_deepritz():
    from baselines import DeepRitz
    S = make_solver()
    m = DeepRitz(S).train(ITERS, lr=LR, lbfgs=LBFGS, n=BATCH)
    f = S.evaluate(P, keys=('u', 'v', 'e11', 'e12'))
    rep = error_report(P, W, ex, f['u'], f['v'], f['e11'], f['e12'])
    J = float((law.F(torch.tensor(2*np.sqrt(f['e11']**2 + f['e12']**2))).numpy()*W).sum())
    print(f"[deepritz] relL2 {rep['rel_L2_velocity']:.2e} relH1 {rep['rel_energy_seminorm']:.2e} "
          f"J(v)-J* {J - ex.dissipation():.2e}  [{m.time:.0f}s]", flush=True)
    return dict(final=rep, J_v=J, exact_J=ex.dissipation(), train_time_s=m.time, log=m.log)


def run_deepritz_quad():
    """Deep Ritz given every advantage: the same Adam phase, then L-BFGS on the deterministic quadrature
    rule of the library (coarse level), so the dissipation it minimises has no sampling noise"""
    from baselines import DeepRitz
    S = make_solver()
    m = DeepRitz(S).train(ITERS, lr=LR, lbfgs=0, n=BATCH)
    R = S.rule(0.1, 12)
    X = torch.tensor(R.P, dtype=DT).requires_grad_(True); Wq = torch.tensor(R.W, dtype=DT)
    params = [p for name, p in S.named_parameters() if not any(k in name for k in ('chi', 'poly', 'michell', 'corner_beta'))]
    opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=LBFGS, history_size=50, line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad(); l = (Wq*S.law.F(S.fields(X, smooth=S.smooth, need_dual=False)['gdot'])).sum(); l.backward(); return l
    t0 = time.time(); opt2.step(closure); t_l = time.time() - t0
    f = S.evaluate(P, keys=('u', 'v', 'e11', 'e12'))
    rep = error_report(P, W, ex, f['u'], f['v'], f['e11'], f['e12'])
    J = float((law.F(torch.tensor(2*np.sqrt(f['e11']**2 + f['e12']**2))).numpy()*W).sum())
    print(f"[deepritz_quad] relL2 {rep['rel_L2_velocity']:.2e} relH1 {rep['rel_energy_seminorm']:.2e} "
          f"J(v)-J* {J - ex.dissipation():.2e}  [{m.time + t_l:.0f}s, {len(R.W)} quadrature points]", flush=True)
    return dict(final=rep, J_v=J, exact_J=ex.dissipation(), train_time_s=m.time + t_l, n_quad=len(R.W), log=m.log)


def run_pinn_hard():
    from baselines import HardPINN
    S = make_solver()
    m = HardPINN(S, width=WIDTH, depth=DEPTH).train(ITERS, lr=LR, lbfgs=LBFGS, n=BATCH)
    f = S.evaluate(P, keys=('u', 'v', 'e11', 'e12'))
    rep = error_report(P, W, ex, f['u'], f['v'], f['e11'], f['e12'])
    print(f"[pinn_hard] relL2 {rep['rel_L2_velocity']:.2e} relH1 {rep['rel_energy_seminorm']:.2e} "
          f"final loss {m.log[-1][1]:.2e}  [{m.time:.0f}s]", flush=True)
    return dict(final=rep, final_loss=m.log[-1][1], train_time_s=m.time, log=m.log)


def run_pinn():
    from baselines import PINN
    Ri, Ro, Om = ex.Ri, ex.Ro, ex.Om
    def si(n):
        r = np.sqrt(np.random.uniform(Ri**2, Ro**2, n)); t = np.random.uniform(-math.pi, math.pi, n)
        return torch.tensor(np.c_[r*np.cos(t), r*np.sin(t)], dtype=DT)
    def sb(n):
        t = np.random.uniform(-math.pi, math.pi, n); t2 = np.random.uniform(-math.pi, math.pi, n)
        X = np.r_[np.c_[Ri*np.cos(t), Ri*np.sin(t)], np.c_[Ro*np.cos(t2), Ro*np.sin(t2)]]
        U = np.r_[np.c_[-Om*Ri*np.sin(t), Om*Ri*np.cos(t)], np.zeros((n, 2))]
        return torch.tensor(X, dtype=DT), torch.tensor(U, dtype=DT)
    m = PINN(law, si, sb, scale_L=Ro, width=WIDTH, depth=DEPTH).train(ITERS, lr=LR, lbfgs=LBFGS, n=BATCH)
    f = m.evaluate(P)
    rep = error_report(P, W, ex, f['u'], f['v'], f['e11'], f['e12'])
    # the PINN velocity is not exactly divergence free: record the divergence, and the boundary error
    rep['div_L2'] = float(math.sqrt((((f['e11'] + f['e22'])**2)*W).sum()))
    Xb, Ub = sb(2000); fb = m.evaluate(Xb.numpy())
    rep['bc_max_err'] = float(np.max(np.hypot(fb['u'] - Ub[:, 0].numpy(), fb['v'] - Ub[:, 1].numpy())))
    print(f"[pinn] relL2 {rep['rel_L2_velocity']:.2e} relH1 {rep['rel_energy_seminorm']:.2e} div {rep['div_L2']:.1e} "
          f"bc {rep['bc_max_err']:.1e} final loss {m.log[-1][1]:.2e}  [{m.time:.0f}s]", flush=True)
    return dict(final=rep, final_loss=m.log[-1][1], train_time_s=m.time, log=m.log)


if __name__ == '__main__':
    which = sys.argv[1:] or ['aequa', 'deepritz', 'pinn_hard', 'pinn']
    cfg = dict(iters=ITERS, lbfgs=LBFGS, width=WIDTH, depth=DEPTH, batch=BATCH, lr=LR, law=CARREAU,
               Ri=ex.Ri, Ro=ex.Ro, Om=ex.Om, threads=torch.get_num_threads())
    for name in which:
        out = dict(config=cfg, **globals()['run_' + name]())
        save('b1_couette_' + name, out)
