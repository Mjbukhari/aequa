"""
End-to-end check of the certificate identity  G = E + E*  (Stokes, rho = 0) against exact solutions.

For an exactly admissible pair (v, m) and the exact solution (u*, S*),
    G - E - E* = - int (m - S*) : eps(v - u*) dx = 0
by the divergence theorem (certificate note, Thm 2.1 with rho = 0).  The left side is evaluated with
the deterministic rule of quadrature.py, so this test checks at once: exact incompressibility and
boundary data of v, exact equilibrium of m (including Michell terms on a multiply connected domain),
the conjugates F*, and the quadrature.  The networks are random (untrained), so v and m are far from
u*, S*: the identity has to hold for every value of the weights.
"""
import math
import numpy as np
import torch
import aequa as aq
from aequa.profiles import Developed, developed_profile

DT = torch.float64


def _randomise(S, scale=0.3, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in S.parameters():
            p.add_(scale*torch.randn(p.shape, generator=g, dtype=DT))


def _bregman_parts(S, P, W, a_s, b_s, S11, S12):
    """returns G, E, E* by the rule (P, W), given the exact strain (a_s, b_s) and stress (S11, S12)"""
    f = S.evaluate(P, keys=('e11', 'e12', 'm11', 'm12', 'gdot', 'taum', 'gap'))
    law = S.law
    F = lambda g: law.F(torch.tensor(g, dtype=DT)).numpy()
    Fs = lambda t: law.Fstar(torch.tensor(t, dtype=DT)).numpy()
    gd_s = 2*np.sqrt(a_s**2 + b_s**2); t_s = np.sqrt(S11**2 + S12**2)
    a, b, m11, m12 = f['e11'], f['e12'], f['m11'], f['m12']
    E = F(f['gdot']) - F(gd_s) - 2*(S11*(a - a_s) + S12*(b - b_s))
    Es = Fs(f['taum']) - Fs(t_s) - 2*(a_s*(m11 - S11) + b_s*(m12 - S12))
    scale = (2*np.abs(a*m11 + b*m12)*W).sum() + (np.abs(f['gap'])*W).sum()
    return (f['gap']*W).sum(), (E*W).sum(), (Es*W).sum(), scale


def _rect_rule(x0, x1, y0, y1, nx, ny, ybreaks=(), n=8):
    """tensor Gauss rule on a rectangle with extra breakpoints in y (the yield lines of the exact
    solution, where its strain has a kink): spectrally accurate for the piecewise smooth integrand"""
    g, w = np.polynomial.legendre.leggauss(n)
    xe = np.linspace(x0, x1, nx + 1); ye = np.unique(np.r_[np.linspace(y0, y1, ny + 1), list(ybreaks)])
    X = (0.5*np.diff(xe)[:, None]*(g + 1) + xe[:-1, None]).ravel(); WX = (0.5*np.diff(xe)[:, None]*w).ravel()
    Y = (0.5*np.diff(ye)[:, None]*(g + 1) + ye[:-1, None]).ravel(); WY = (0.5*np.diff(ye)[:, None]*w).ravel()
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    return np.c_[XX.ravel(), YY.ravel()], np.outer(WX, WY).ravel()


def _channel_case(law):
    dom = aq.channel(2.0, 2.0, x0=0.0)
    prof = developed_profile(law, -1.0, 1.0, 1.0)
    D = prof.developed
    dom.set_bc({'inlet': aq.Profile(prof, (1, 0)), 'outlet': aq.Profile(prof, (1, 0)), 'wall': aq.NoSlip()})
    S = aq.Solver(dom, law, rho=0.0, width=16, depth=2)
    _randomise(S)
    # library rule, split along the yield lines of the exact solution (where its strain has a kink)
    from aequa.quadrature import build_rule
    lines = [[(-1.0, s), (3.0, s)] for s in ((-D.plug, D.plug) if D.plug > 0 else ())]
    R = build_rule(dom, h=0.05, breaklines=lines)
    P, W = R.P, R.W
    y = P[:, 1]
    a_s = np.zeros_like(y); b_s = 0.5*D.dudy(y)
    S11 = np.zeros_like(y); S12 = D.shear_stress(y)
    G, E, Es, sc = _bregman_parts(S, P, W, a_s, b_s, S11, S12)
    # an independent check of the rule: the network gap (smooth away from the vertices) on a finer rule
    PL, WL = S.quadrature(h=0.025)
    G_lib = (S.evaluate(PL)['gap']*WL).sum()
    return G, E, Es, sc, G_lib


def test_identity_channel_all_laws():
    laws = [aq.Newtonian(1.0), aq.PowerLaw(1.0, 0.5), aq.Bingham(1.0, 0.5),
            aq.HerschelBulkley(3.72, 0.41, 3.31), aq.Carreau(1.0, 0.05, 2.0, 0.5),
            aq.CarreauYasuda(1.0, 0.05, 2.0, 0.4, 1.5), aq.Casson(0.5, 0.3), aq.Cross(1.0, 0.05, 2.0, 0.7)]
    for law in laws:
        G, E, Es, sc, G_lib = _channel_case(law)
        rel = abs(G - E - Es)/sc
        print(f"{law.name:55s} G={G:.9e} E+E*={E + Es:.9e} |G-E-E*|/scale={rel:.1e}  G_lib/G-1={G_lib/G - 1:+.1e}")
        assert rel < 1e-9
        assert E >= -1e-10*sc and Es >= -1e-10*sc
        # the gap density of a yield-stress or power law is only Lipschitz where the network strain
        # vanishes or the network stress crosses the yield value, so plain quadrature converges at an
        # algebraic rate there; the rule's own two-resolution estimate reports this (certify_gap)
        assert abs(G_lib/G - 1) < (1e-6 if law.yield_stress > 0 or law.p < 2 else 1e-8)


def test_identity_couette_michell():
    """concentric Couette flow (Newtonian): multiply connected, the exact stress carries a torque,
    so the Michell term is essential for admissibility"""
    Ri, Ro, om, mu = 0.4, 1.0, 1.5, 0.7
    dom = aq.annulus(Ro, Ri)
    dom.set_bc({'outer': aq.NoSlip(), 'inner': aq.Rotating((0.0, 0.0), om)})
    S = aq.Solver(dom, aq.Newtonian(mu), rho=0.0, width=16, depth=2)
    _randomise(S)
    P, W = S.quadrature(h=0.05)
    B = om*Ri**2*Ro**2/(Ro**2 - Ri**2)
    r = np.hypot(P[:, 0], P[:, 1]); c, s = P[:, 0]/r, P[:, 1]/r
    erth = -B/r**2
    a_s = erth*(-2*s*c); b_s = erth*(c*c - s*s)
    G, E, Es, sc = _bregman_parts(S, P, W, a_s, b_s, 2*mu*a_s, 2*mu*b_s)
    rel = abs(G - E - Es)/sc
    print(f"Couette: G={G:.6e} E+E*={E + Es:.6e} rel={rel:.1e}")
    # quadrature-limited (the defect decays like h^4 under refinement, checked separately)
    assert rel < 1e-8


if __name__ == "__main__":
    test_identity_channel_all_laws(); test_identity_couette_michell()
