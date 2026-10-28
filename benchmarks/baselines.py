"""Neural baselines, trained on the same problem, with the same network sizes and optimiser schedule as Aequa.

PINN        standard physics-informed network: (u, v, p) from one MLP, loss = mean squared momentum
            residual + mean squared divergence + weighted squared boundary mismatch (soft constraints).
PINN-hard   the exactly admissible velocity of Aequa (v = curl psi, exact boundary data) plus a pressure
            network, trained on the same strong-form momentum residual.  Isolates the effect of the loss.
DeepRitz    the exactly admissible velocity of Aequa, trained on the dissipation J(v) = int F(gd(v))
            alone (the primal half of the gap).  J(v) is an upper bound for J*, but there is no lower
            bound, hence no computable error bound.
None of the three produces a guaranteed error bound; that is the point of the comparison.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
from aequa.solver import MLP, d1, DT


def _adam_lbfgs(params, loss_fn, sample, iters, lr, lbfgs, n, log, every):
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr*0.02)
    t0 = time.time()
    for it in range(iters):
        batch = sample(n)
        loss = loss_fn(*batch)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % every == 0 or it == iters - 1:
            log.append((it, float(loss.detach()), time.time() - t0))
    if lbfgs:
        batch = sample(8*n)
        opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, history_size=50, line_search_fn='strong_wolfe')
        def closure():
            opt2.zero_grad(); l = loss_fn(*batch); l.backward(); return l
        opt2.step(closure)
        log.append((iters + lbfgs, float(closure().detach()), time.time() - t0))
    return time.time() - t0


class PINN(nn.Module):
    """soft-constraint PINN for Stokes flow of a generalised-Newtonian fluid"""
    def __init__(self, law, sample_interior, sample_boundary, scale_L=1.0, width=48, depth=4, w_bc=100.0, seed=0):
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed)
        self.law, self.si, self.sb, self.L, self.w_bc = law, sample_interior, sample_boundary, scale_L, w_bc
        self.net = nn.ModuleList([MLP(width, depth, nin=2, gain=1.0) for _ in range(3)])

    def uvp(self, X):
        z = X/self.L
        return [n(z) for n in self.net]

    def residual(self, X):
        u, v, p = self.uvp(X)
        gu = d1(u, X); gv = d1(v, X); gp = d1(p, X)
        e11, e22, e12 = gu[:, 0], gv[:, 1], 0.5*(gu[:, 1] + gv[:, 0])
        gd = torch.sqrt(2*(e11**2 + e22**2 + 2*e12**2) + 1e-12)
        eta = self.law.dF(gd)/gd
        t11, t12, t22 = 2*eta*e11, 2*eta*e12, 2*eta*e22
        r1 = d1(t11, X)[:, 0] + d1(t12, X)[:, 1] - gp[:, 0]
        r2 = d1(t12, X)[:, 0] + d1(t22, X)[:, 1] - gp[:, 1]
        return r1, r2, e11 + e22

    def loss(self, Xi, Xb, Ub):
        r1, r2, dv = self.residual(Xi)
        u, v, _ = self.uvp(Xb)
        return (r1**2 + r2**2).mean() + (dv**2).mean() + self.w_bc*((u - Ub[:, 0])**2 + (v - Ub[:, 1])**2).mean()

    def train(self, iters, lr=3e-3, lbfgs=300, n=1536, every=50):
        self.log = []
        def sample(n):
            Xi = self.si(n).requires_grad_(True); Xb, Ub = self.sb(max(256, n//4))
            return Xi, Xb, Ub
        self.time = _adam_lbfgs(list(self.parameters()), self.loss, sample, iters, lr, lbfgs, n, self.log, every)
        return self

    def evaluate(self, P, chunk=4096):
        out = {k: [] for k in ('u', 'v', 'e11', 'e12', 'e22')}
        for i in range(0, len(P), chunk):
            X = torch.tensor(P[i:i + chunk], dtype=DT).requires_grad_(True)
            u, v, _ = self.uvp(X); gu = d1(u, X); gv = d1(v, X)
            for k, val in (('u', u), ('v', v), ('e11', gu[:, 0]), ('e22', gv[:, 1]), ('e12', 0.5*(gu[:, 1] + gv[:, 0]))):
                out[k].append(val.detach().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


class HardPINN(nn.Module):
    """Aequa's exactly admissible velocity (from an aequa Solver S) + a pressure MLP, residual loss"""
    def __init__(self, S, width=48, depth=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.S = S; self.pnet = MLP(width, depth, nin=2, gain=1.0)

    def loss(self, X, w):
        S = self.S
        f = S.fields(X, smooth=S.smooth, need_dual=False)
        a, b = f['e11'], f['e12']                       # eps22 = -a exactly
        gd = 2*torch.sqrt(a*a + b*b + (S.smooth*S.Us/S.Ls)**2)
        eta = S.law.dF(gd)/gd
        t11, t12 = 2*eta*a, 2*eta*b
        p = S.Ss*self.pnet(torch.stack([(X[:, 0] - S.cx)/S.Ls, (X[:, 1] - S.cy)/S.Ls], -1))
        gp = d1(p, X)
        g11 = d1(t11, X); g12 = d1(t12, X)
        r1 = g11[:, 0] + g12[:, 1] - gp[:, 0]
        r2 = g12[:, 0] - g11[:, 1] - gp[:, 1]           # t22 = -t11
        return (w*(r1**2 + r2**2)).sum()/w.sum()

    def train(self, iters, lr=3e-3, lbfgs=300, n=1536, every=50):
        self.log = []
        def sample(n):
            X, w = self.S.sample(n); return X.requires_grad_(True), w
        self.time = _adam_lbfgs(list(self.parameters()), self.loss, sample, iters, lr, lbfgs, n, self.log, every)
        return self


class DeepRitz:
    """dissipation minimisation over Aequa's exactly admissible velocity (Stokes only)"""
    def __init__(self, S):
        self.S = S

    def loss(self, X, w):
        f = self.S.fields(X, smooth=self.S.smooth, need_dual=False)
        return (w*self.S.law.F(f['gdot'])).sum()

    def train(self, iters, lr=3e-3, lbfgs=300, n=1536, every=50):
        self.log = []
        S = self.S
        params = [p for name, p in S.named_parameters() if not any(k in name for k in ('chi', 'poly', 'michell', 'corner_beta'))]
        def sample(n):
            X, w = S.sample(n); return X.requires_grad_(True), w
        self.time = _adam_lbfgs(params, self.loss, sample, iters, lr, lbfgs, n, self.log, every)
        return self
