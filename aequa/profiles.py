"""
Fully developed (unidirectional) channel profiles for any law of rheology.py.

For plane Poiseuille flow between y0 and y1 driven by a pressure gradient -G, the shear stress is
tau = G |y - yc| and the shear rate is gd = (F')^{-1}(tau) (zero where tau <= tau_y), so
    u(y) = int_{|y - yc|}^{h} gd(G s) ds,          h = (y1 - y0)/2,
and G is fixed by the mean speed.  These are exact solutions of the steady problem (with or
without inertia) in a straight channel, and the natural inflow/outflow data for a long channel.
They are used as boundary data (Profile) and as exact references in the tests.
"""
import math
import numpy as np
from scipy.optimize import brentq
import torch

DT = torch.float64


def inverse_flow_curve(law, tau):
    """gd with F'(gd) = tau (0 where tau <= F'(0)), vectorised bisection to machine precision"""
    tau = np.asarray(tau, float)
    t = torch.tensor(tau, dtype=DT)
    base = float(law.dF(torch.zeros(1, dtype=DT))[0])
    hi = torch.ones_like(t)
    for _ in range(2000):
        need = law.dF(hi) < t
        if not bool(need.any()): break
        hi = torch.where(need, 2*hi, hi)
    lo = torch.zeros_like(t)
    for _ in range(200):
        mid = 0.5*(lo + hi)
        up = law.dF(mid) > t
        hi = torch.where(up, mid, hi); lo = torch.where(up, lo, mid)
    g = 0.5*(lo + hi)
    g = torch.where(t <= base, torch.zeros_like(g), g)
    return g.numpy()


class Developed:
    """developed channel flow of `law` across [y0, y1] with the given mean speed.
    Attributes: G (pressure gradient), plug (half-width of the unyielded core), and the callables
    speed(y), dudy(y), shear_stress(y) (the xy component of the deviatoric stress for flow in +x)."""
    def __init__(self, law, y0, y1, mean_speed, n=4001):
        from scipy.interpolate import CubicHermiteSpline
        self.law, self.y0, self.y1, self.U = law, float(y0), float(y1), float(mean_speed)
        self.h = 0.5*(self.y1 - self.y0); self.yc = 0.5*(self.y0 + self.y1)
        ty = law.yield_stress
        xg, wg = np.polynomial.legendre.leggauss(10)
        def grid(G):
            s = np.linspace(0.0, self.h, n)
            pl = ty/G if ty > 0 else 0.0
            if 0 < pl < self.h: s = np.unique(np.r_[s, pl])
            return s
        def panel_integrals(G, s, weight_s):
            a_, b_ = s[:-1], s[1:]
            X = 0.5*(b_ - a_)[:, None]*(xg[None, :] + 1) + a_[:, None]
            W = 0.5*(b_ - a_)[:, None]*wg[None, :]
            gd = inverse_flow_curve(law, G*X.ravel()).reshape(X.shape)
            return ((X if weight_s else 1.0)*gd*W).sum(1)
        def flux(G):                       # 2 int_0^h s gd(G s) ds (exact up to Gauss error ~1e-15)
            s = grid(G); return 2*panel_integrals(G, s, True).sum()
        target = self.U*2*self.h
        Ghi = 1.0
        while flux(Ghi) < target: Ghi *= 2
        Glo = ty/self.h if ty > 0 else 0.0
        self.G = brentq(lambda G: flux(G) - target, Glo, Ghi, xtol=1e-15, rtol=1e-15)
        s = grid(self.G)
        seg = panel_integrals(self.G, s, False)                    # int over each panel of gd
        u = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]                 # u(s_i) = int_{s_i}^h gd
        gd = inverse_flow_curve(law, self.G*s)
        self._ui = CubicHermiteSpline(s, u, -gd)                   # exact derivative data: error O(h^4)
        self.plug = min(self.h, ty/self.G) if ty > 0 else 0.0

    def speed(self, y):
        a = np.clip(np.abs(np.asarray(y, float) - self.yc), 0.0, self.h)
        return self._ui(a)

    def dudy(self, y):
        y = np.asarray(y, float); a = np.abs(y - self.yc)
        gd = inverse_flow_curve(self.law, self.G*np.minimum(a, self.h))
        return -np.sign(y - self.yc)*gd

    def shear_stress(self, y):
        return -self.G*(np.asarray(y, float) - self.yc)


def developed_profile(law, y0, y1, mean_speed, coord='y'):
    """speed function for aequa.Profile: the developed profile of `law` across [y0, y1]
    (coordinate 'y' for flow along x, 'x' for flow along y)"""
    D = Developed(law, y0, y1, mean_speed)
    if coord == 'y':
        f = lambda x, y: D.speed(y)
    else:
        f = lambda x, y: D.speed(x)
    f.developed = D
    return f
