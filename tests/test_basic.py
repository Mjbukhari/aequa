import numpy as np, torch
import aequa as aq

def _dom():
    dom = aq.channel(6.0, 2.0, x0=-2.0, obstacles=[((0.0, 0.0), 0.3, 'cyl')])
    prof = aq.poiseuille_profile(-1, 1, 1.0)
    dom.set_bc({'inlet': aq.Profile(prof, (1, 0)), 'outlet': aq.Profile(prof, (1, 0)),
                'wall': aq.NoSlip(), 'cyl': aq.Rotating((0, 0), 2.0)})
    return dom

def test_exact_boundary_values():
    dom = _dom(); S = aq.Solver(dom, aq.Newtonian(1.0), rho=1.0, width=16, depth=2)
    for pc in dom.pieces_all:
        lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 7)[1:-1]
        P = pc.point(p) + 1e-8*pc.inward_normal(p)
        f = S.evaluate(P, keys=('u', 'v'))
        ub = dom.bcs[pc.tag](pc.point(p))
        assert np.abs(np.c_[f['u'], f['v']] - ub).max() < 1e-5

def test_gap_nonnegative():
    dom = _dom(); S = aq.Solver(dom, aq.Carreau(1.0, 0.05, 1.0, 0.5), rho=1.0, width=16, depth=2)
    P, W = S.quadrature(h=0.1)
    assert S.evaluate(P)['gap'].min() > -1e-9

def test_conjugates():
    for L in [aq.Newtonian(1.0), aq.PowerLaw(1.0, 0.5), aq.Bingham(1.0, 2.0), aq.Carreau(1.0, 0.01, 1.0, 0.4),
              aq.Casson(0.5, 1.5), aq.Cross(1.0, 0.01, 2.0, 0.7), aq.HerschelBulkley(3.72, 0.41, 3.31)]:
        mono, fy = aq.check_convex(L)
        assert mono and fy < 1e-8

def test_flux_check():
    dom = aq.channel(4.0, 2.0)
    try:
        dom.set_bc({'inlet': aq.Profile(lambda x, y: 1 + 0*x, (1, 0)), 'outlet': aq.NoSlip(), 'wall': aq.NoSlip()})
        assert False, "unbalanced flux must be rejected"
    except ValueError:
        pass
