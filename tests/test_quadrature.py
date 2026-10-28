"""
Checks of the deterministic rule (certificate note, Sec. 5):
  * exact geometry: the rule integrates 1 to the exact area of polygons and circular holes
    (curved non-circular walls: up to the reported polyline error);
  * positivity of all weights;
  * a singular density r^-kappa at a re-entrant corner (kappa < 2) is integrated with the
    sector tail estimate of Lemma 5.2, against an independent 1D polar integral.
"""
import math
import numpy as np
from scipy.integrate import quad
import aequa as aq
from aequa.quadrature import build_rule

par = lambda a, b, U: (lambda x, y: 6*U*(y - a)*(b - y)/(b - a)**2)
parx = lambda a, b, U: (lambda x, y: 6*U*(x - a)*(b - x)/(b - a)**2)


def test_areas_and_weights():
    cases = [
        (aq.channel(6., 2., x0=-2., obstacles=[aq.circle((0, 0), .3, 'cyl')]),
         {'inlet': aq.Profile(par(-1, 1, 1.), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.), (1, 0)),
          'wall': aq.NoSlip(), 'cyl': aq.Rotating((0, 0), 2.)}, 12 - math.pi*.09),
        (aq.contraction(10, 4, 8, 1), {'inlet': aq.Profile(par(-4, 4, .25), (1, 0)),
          'outlet': aq.Profile(par(-1, 1, 1.), (1, 0)), 'wall': aq.NoSlip()}, 96.0),
        (aq.annulus(1.0, 0.4, inner_center=(0.2, 0.)), {'outer': aq.NoSlip(), 'inner': aq.Rotating((0.2, 0.), 1.)},
         math.pi*(1 - .16)),
        (aq.channel(8., 2., x0=-2., obstacles=[aq.square((0, 0), .4, 'sq', angle=.3)]),
         {'inlet': aq.Profile(par(-1, 1, 1.), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.), (1, 0)),
          'wall': aq.NoSlip(), 'sq': aq.NoSlip()}, 16 - .16),
        (aq.cross_slot(3., 1.), {'inlet1': aq.Profile(par(-.5, .5, 1.), (1, 0)), 'inlet2': aq.Profile(par(-.5, .5, 1.), (-1, 0)),
          'outlet1': aq.Profile(parx(-.5, .5, 1.), (0, 1)), 'outlet2': aq.Profile(parx(-.5, .5, 1.), (0, -1)),
          'wall': aq.NoSlip()}, 11.0),
    ]
    for dom, bcs, exact in cases:
        dom.set_bc(bcs)
        R = build_rule(dom, h=0.1)
        print(f"area {R.area():.14f} exact {exact:.14f}  min weight {R.W.min():.1e}")
        assert abs(R.area() - exact) < 1e-10*exact
        assert R.W.min() > 0


def _corner_rect(a, b, kappa):
    """int over [0,a] x [0,b] of r^-kappa (corner at the origin), by polar 1D integrals"""
    tc = math.atan2(b, a)
    f1 = lambda t: (a/math.cos(t))**(2 - kappa)/(2 - kappa)
    f2 = lambda t: (b/math.sin(t))**(2 - kappa)/(2 - kappa)
    return quad(f1, 0, tc, epsabs=1e-14, epsrel=1e-13)[0] + quad(f2, tc, math.pi/2, epsabs=1e-14, epsrel=1e-13)[0]


def test_singular_corner_density():
    dom = aq.contraction(10.0, 4.0, 8.0, 1.0)
    dom.set_bc({'inlet': aq.Profile(par(-4, 4, .25), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.), (1, 0)), 'wall': aq.NoSlip()})
    iv = [i for i, v in enumerate(dom.vertices) if abs(v['V'][0]) < 1e-12 and abs(v['V'][1] - 1) < 1e-12][0]
    for kappa in (0.5, 1.2, 1.7):
        # exact: upstream [-10,0]x[1,4] and [-10,0]x[-4,1], downstream [0,8]x[-1,1]; corner (0,1) at a corner of each
        ex = _corner_rect(10, 3, kappa) + _corner_rect(10, 5, kappa) + _corner_rect(8, 2, kappa)
        R = build_rule(dom, h=0.1, singular_vertices=(iv,))
        r = np.hypot(R.P[:, 0], R.P[:, 1] - 1.0)
        out = R.integrate(r**-kappa, {iv: kappa})
        val = out['value'] + out['tail_sum']
        print(f"kappa={kappa}: rule {out['value']:.12f} + tail {out['tail_sum']:.3e} = {val:.12f}, exact {ex:.12f}, rel {val/ex - 1:+.1e}")
        assert abs(val/ex - 1) < 1e-8


if __name__ == "__main__":
    test_areas_and_weights(); test_singular_corner_density(); test_jump_vertex_density()


def test_jump_vertex_density():
    """Lemma 4.2 / Lemma 5.2: at a lid corner with a velocity jump (p < 2) the gap density behaves like
    r^-p; the solver grades the sector there and the rule with its tail matches the exact value"""
    dom = aq.cavity(1.0, 1.0)
    dom.set_bc({'lid': aq.MovingWall(1.0), 'wall': aq.NoSlip()})
    S = aq.Solver(dom, aq.PowerLaw(1.0, 0.5))
    assert len(S.jump_vertices) == 2 and set(S.jump_vertices) <= set(S.singular_vertices)
    kth = S.kappa_theory()
    assert all(abs(kth[i] - 1.5) < 1e-12 for i in S.jump_vertices)
    R = S.rule()
    for iv in S.jump_vertices:
        V = dom.vertices[iv]['V']
        r = np.hypot(R.P[:, 0] - V[0], R.P[:, 1] - V[1])
        out = R.integrate(r**-1.5, {iv: 1.5})
        val = out['value'] + [t['tail'] for t in out['tails'] if t['vertex'] == iv][0]
        # the other sectors see a smooth function: their tails are negligible
        ex = _corner_rect(1.0, 1.0, 1.5)
        print(f"jump vertex {V}: rule+tail {val:.12f}, exact {ex:.12f}, rel {val/ex - 1:+.1e}")
        assert abs(val/ex - 1) < 1e-8
