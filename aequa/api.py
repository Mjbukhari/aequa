"""One-call interface:  res = aequa.solve(domain, law, bcs, rho=...)

The summary states, for the trained pair, exactly which theorem applies and with which constants
(certificate note, Sec. 6), and labels every number by its guarantee level:
  [exact]       holds for every value of the weights (algebra of the ansatz, Lemma 3.1, Lemma 2.3)
  [quadrature]  deterministic quadrature with a two-resolution difference and sector tails
                (Sec. 5); not interval-verified
  [sampled]     constants computed from sampled network values with a Lipschitz margin (theta);
                not interval-verified
"""
import math
import numpy as np
import torch
from .solver import Solver, DT
from .theta import theta_bound


class Result:
    def __init__(self, solver, gap, theta, lam, s0):
        self.solver, self.gapinfo, self.thetainfo, self.lam, self.s0 = solver, gap, theta, lam, s0

    @property
    def G(self):
        """the gap value used in all bounds: fine-rule value + |fine - coarse| + tails (+ curve term)"""
        return self.gapinfo['G_up']

    @property
    def theta(self): return None if self.thetainfo is None else self.thetainfo['theta']

    @property
    def tube(self):
        """absolute tube width s (shear-rate units): s0 U/L"""
        S = self.solver
        return self.s0*S.Us/S.Ls

    def weights(self):
        """quadrature points/weights and the two energy weights at the points:
        nu_floor (unconditional) and nu_tube (Thm 3.4, valid in the strain tube)"""
        S = self.solver
        P, W = S.quadrature()
        f = S.evaluate(P, keys=('e11', 'e12', 'gdot'))
        ne = np.sqrt(2*f['e11']**2 + 2*f['e12']**2)                    # |eps(v)|
        gmax = math.sqrt(2)*((1 + self.lam)*ne + self.tube)        # Cor. 3.5 with s = s0 U/L + lam |eps v|
        law = S.law
        nu_tube = law.nu0(torch.tensor(gmax, dtype=DT), torch.tensor(f['gdot'], dtype=DT)).numpy()
        nu_floor = np.full_like(ne, law.floor)
        return P, W, ne, nu_floor, nu_tube

    def summary(self):
        S, g, law = self.solver, self.gapinfo, self.solver.law
        lam_c, beta_c = S.exponents()
        L = ["AEQUA certificate (v0.3)",
             f"  law          : {law}   (growth p = {law.p:.3g}, conjugate p' = {law.pconj:.3g}, floor nu_* = {law.floor:.4g})",
             f"  inertia      : rho = {S.rho}",
             f"  [exact]      admissible for every weight (Lemma 3.1): div v = 0, v = v_D, div Airy(chi) = 0;"]
        if len(lam_c):
            L.append(f"               corner exponents lambda = {np.round(lam_c, 4).tolist()} > {law.lam_min:.4f}, "
                     f"beta = {np.round(beta_c, 4).tolist()} > {law.beta_min:.4f}")
        L.append(f"               boundary data imposed = user data up to {S.dom.data_mismatch:.1e} (Remark 4.4)")
        if S.vertex_jumps:
            L.append(f"               {len(S.vertex_jumps)} vertices with a velocity jump (allowed since p < 2, Lemma 4.2)")
        fi, co = g['fine'], g['coarse']
        L += [f"  [quadrature] gap G: fine {fi['G']:.6e} ({fi['n_points']} points), coarse {co['G']:.6e}, "
              f"sector tails {fi['tail_sum']:.1e}" + (f", curve term {g['curve_term']:.1e}" if g['curve_term'] else ""),
              f"               G used below = {self.G:.6e}   (min pointwise gap {fi['min_gap']:.1e}; >= 0 up to rounding)",
              f"               dissipation J(v) = int F(gd(v)) = {fi['diss']:.6e}"]
        J = fi['diss']
        P, W, ne, nu_f, nu_t = self.weights()
        Ef = float((nu_f*ne**2*W).sum()); Et = float((nu_t*ne**2*W).sum())
        cond = law.nu0_is_conditional()
        thick = (not law.shear_thinning) and hasattr(law, 'n') and law.floor == 0.0

        def norm_lines(B):
            out = []
            if law.floor > 0:
                out.append(f"               nu_* int |eps(u - v)|^2 <= {B:.4e}   "
                           f"(= {100*math.sqrt(B/Ef):.3g} % of nu_* int |eps(v)|^2)   [unconditional, Lemma 2.4 / Prop. 4.1(i)]")
            if thick:
                out.append(f"               int nu0 |eps(u - v)|^2 <= {B:.4e}, nu0 = 2 K c_q gd(v)^(n-1)   "
                           f"(= {100*math.sqrt(B/Et):.3g} %)   [unconditional, Lemma 3.8]")
            elif law.shear_thinning and (Et > Ef*(1 + 1e-12) or law.floor == 0):
                out.append(f"               int nu_s |eps(u - v)|^2 <= {B:.4e}, nu_s = F''(sqrt2((1+lam)|eps v| + s)) "
                           f"(= {100*math.sqrt(B/Et):.3g} %)")
                out.append(f"               [tube statement, Cor. 7.4: holds unless |eps(u - v)| > s0 U/L + lam |eps(v)| "
                           f"on a set of positive measure; s = {self.tube:.3g}, lam = {self.lam}]")
            return out

        if S.rho == 0:
            if law.p > 1:
                L.append(f"  STATUS       : Stokes regime (rho = 0): G = E + E* exactly (Thm 2.1)")
            else:
                L.append(f"  STATUS       : Stokes regime (rho = 0): G = E + E* for every solution in W^(1,1) with bounded "
                         f"stress (Thm 2.1); the enclosure below holds unconditionally (weak duality, Prop. 6.1)")
            L.append(f"               energy enclosure  J(v) - G <= J* <= J(v):  [{J - self.G:.6e}, {J:.6e}]"
                     f"  (half-width {50*self.G/max(J, 1e-300):.3g} % of J)")
            L += norm_lines(self.G)
            try:
                pb = S.power_bounds(g)
                L.append(f"  [quadrature] power  W* = int S*:eps(u*) in [{pb['W_lo']:.6e}, {pb['W_hi']:.6e}]   (Thm 6.4)")
                if pb['Q_in'] > 0 and not pb['moving_walls']:
                    L.append(f"               pressure drop W*/Q_in in [{pb['W_lo']/pb['Q_in']:.6e}, {pb['W_hi']/pb['Q_in']:.6e}]"
                             f"  (flux-weighted drop of the normal stress, Q_in = {pb['Q_in']:.6g}; Cor. 6.5)")
            except Exception as ex:
                L.append(f"  power enclosure not computed: {ex}")
        elif S.jump_vertices:
            L.append("  STATUS       : NOT certified with inertia: the theta enclosure has no corner cells at vertices "
                     "with a data jump, where |eps(v)| ~ 1/r (Lemma 3.2); regularise the data or set rho = 0")
        elif law.p <= 1:
            L.append("  STATUS       : NOT certified: the inertial theorems need growth p > 1 (dissipative solutions "
                     "in W^{1,p}, p > 2d/(d+2)); this law saturates (p = 1)")
        else:
            th = self.theta
            if th is None:
                L.append("  STATUS       : theta not computed: no statement for the inertial problem")
            else:
                ti = self.thetainfo
                L.append(f"  [sampled]    theta <= {th:.4f} = 2 (lambda_D {ti['lambda_D']:.4f} + eta^2 {ti['eta2']:.4f})  "
                         f"(Thm 20.2; majorants and floor from sampled values)")
                if th < 1:
                    B = self.G/(1 - th)
                    L.append(f"  STATUS       : gap regime theta < 1: E + E* <= G/(1 - theta) = {B:.4e}   (Thm 6.1 / Cor. 7.4)")
                    L += norm_lines(B)
                elif th < 2:
                    L.append("  STATUS       : hybrid regime 1 <= theta < 2: existence and uniqueness near v hold "
                             "(Thm 9.1/17.1), but the bound needs the residual r = |m - S(eps v)|, which v0.3 does not "
                             "evaluate: no numerical bound is reported")
                else:
                    L.append("  STATUS       : NOT certified (theta >= 2): the energy method does not apply; the "
                             "preconditioned check (Sec. 19) is not implemented")
        return "\n".join(L)

    def __call__(self, x, y):
        P = np.c_[np.ravel(x), np.ravel(y)]
        return self.solver.evaluate(P, keys=('u', 'v', 'p'))


def solve(domain, law, bcs, rho=0.0, iters=4000, width=48, depth=4, batch=None, lr=3e-3, lbfgs=300,
          theta=True, lam=0.0, s0=0.1, theta_target=0.1, verbose=True, seed=0, warm_start=None, h=None,
          device='auto', precision='auto', lbfgs_points=None):
    """Train a gap-certified network for steady generalised-Newtonian flow and certify it.
    s0: tube width in units of U/L (only used for floorless shear-thinning laws and for theta).
    device: 'auto' (GPU if PyTorch sees one, else CPU), 'cuda' or 'cpu'.
    precision: training precision; 'auto' = float32 on a GPU, float64 on the CPU; 'mixed' or 'double'
               to force it. The certificate is always computed in float64, whatever the training precision.
    batch: training points per Adam step; default 1536 on the CPU, 4096 on a GPU.
    lbfgs_points: size of the fixed L-BFGS sample set (default 8*batch)."""
    domain.set_bc(bcs)
    S = Solver(domain, law, rho=rho, width=width, depth=depth, seed=seed, device=device, precision=precision)
    if batch is None: batch = 4096 if S.device.type == 'cuda' else 1536
    if warm_start is not None:            # continuation: start from a solved neighbouring problem
        S.load_state_dict(warm_start.solver.state_dict() if isinstance(warm_start, Result) else warm_start)
    if verbose: print(domain.describe() + "\n" + S.device_info(), flush=True)
    S.train(iters=iters, n=batch, lr=lr, verbose=max(1, iters//10) if verbose else 0, lbfgs=lbfgs,
            lbfgs_points=lbfgs_points)
    gap = S.certify_gap(h=h)
    th = theta_bound(S, lam=lam, s=s0*S.Us/S.Ls, target=theta_target, verbose=verbose) if (theta and rho > 0 and law.p > 1 and not S.jump_vertices) else None
    res = Result(S, gap, th, lam, s0)
    if verbose: print(res.summary())
    return res
