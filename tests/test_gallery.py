"""Exact boundary data on the standard geometries of the literature (random networks):
v = v_D on every piece (Lemma 3.1, Remark 4.3) and the imposed data equal the user data
up to the reported mismatch (Remark 4.4)."""
import numpy as np
import torch
import aequa as aq
from aequa.geometry import *

par = lambda a, b, U: (lambda x, y: 6*U*(y - a)*(b - y)/(b - a)**2)
parx = lambda a, b, U: (lambda x, y: 6*U*(x - a)*(b - x)/(b - a)**2)


def check(dom, bcs, name, tol=5e-5):
    dom.set_bc(bcs)
    S = aq.Solver(dom, aq.Newtonian(1.0), rho=1.0, width=16, depth=2)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in S.parameters(): p.add_(0.3*torch.randn(p.shape, generator=g, dtype=p.dtype))
    worst = 0.0
    for pc in dom.pieces_all:
        lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 9)[1:-1]
        B = pc.point(p); P = B + 1e-7*pc.inward_normal(p)
        f = S.evaluate(P, keys=('u', 'v'))
        worst = max(worst, float(np.abs(np.c_[f['u'], f['v']] - dom.bcs[pc.tag](B)).max()))
    print(f"{name:45s} pieces={len(dom.pieces_all):3d}  max |v - u_b| = {worst:.1e}  data mismatch {dom.data_mismatch:.1e}")
    assert worst < tol and dom.data_mismatch < 1e-9


def test_gallery():
    d = t_junction(3.0, 1.0, 4.0, 1.0, obstacles=[circle((0.0, 0.5), 0.2, 'rotor')])
    check(d, {'inlet': Profile(parx(-0.5, 0.5, 1.0), (0, 1)), 'outlet1': Profile(par(0, 1, 0.5), (1, 0)),
              'outlet2': Profile(par(0, 1, 0.5), (-1, 0)), 'wall': NoSlip(), 'rotor': Rotating((0.0, 0.5), 2.0)},
          'T-junction + rotating cylinder')
    d = channel(8.0, 2.0, x0=-2.0, obstacles=[square((0, 0), 0.4, 'sq', angle=0.3), triangle((2.5, 0.2), 0.5, 'tri')])
    check(d, {'inlet': Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': Profile(par(-1, 1, 1.0), (1, 0)),
              'wall': NoSlip(), 'sq': NoSlip(), 'tri': NoSlip()}, 'channel + tilted square + triangle')
    d = channel(6.0, 2.0, x0=-2.0, obstacles=[ellipse((0, 0), 0.4, 0.15, 'ell', angle=0.4)])
    check(d, {'inlet': Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': Profile(par(-1, 1, 1.0), (1, 0)),
              'wall': NoSlip(), 'ell': NoSlip()}, 'channel + inclined ellipse')
    top = lambda x: 1.0 + 0.3*np.sin(2*np.pi*x/4); bot = lambda x: -top(x)
    d = wavy_channel(8.0, top, bot, x0=0.0)
    prof = lambda x, y: 1.5*2.0/(2*top(x))*(1 - (y/top(x))**2)
    check(d, {'inlet': Profile(prof, (1, 0)), 'outlet': Profile(prof, (1, 0)), 'wall': NoSlip()}, 'peristaltic (wavy) channel')
    d = annulus(1.0, 0.4, inner_center=(0.2, 0.0))
    check(d, {'outer': NoSlip(), 'inner': Rotating((0.2, 0.0), 1.0)}, 'eccentric annulus, rotating inner cylinder')
    d = cavity(1.0, 1.0, obstacles=[circle((0.5, 0.4), 0.15, 'cyl')])
    check(d, {'lid': Velocity(lambda x, y: (16*x**2*(1 - x)**2, 0*x)), 'wall': NoSlip(), 'cyl': NoSlip()},
          'lid-driven cavity (regularised lid) + cylinder')
    d = backward_step(2.0, 0.5, 6.0, 1.0, obstacles=[square((1.5, 0.5), 0.2, 'sq')])
    check(d, {'inlet': Profile(par(0.5, 1.0, 1.0), (1, 0)), 'outlet': Profile(par(0.0, 1.0, 0.5), (1, 0)),
              'wall': NoSlip(), 'sq': NoSlip()}, 'backward-facing step + square')
    d = contraction(10.0, 4.0, 8.0, 1.0, obstacles=[circle((-3.0, 0.0), 0.5, 'cyl')])
    check(d, {'inlet': Profile(par(-4, 4, 0.25), (1, 0)), 'outlet': Profile(par(-1, 1, 1.0), (1, 0)),
              'wall': NoSlip(), 'cyl': NoSlip()}, '4:1 contraction + cylinder')
    d = cross_slot(3.0, 1.0)
    check(d, {'inlet1': Profile(par(-0.5, 0.5, 1.0), (1, 0)), 'inlet2': Profile(par(-0.5, 0.5, 1.0), (-1, 0)),
              'outlet1': Profile(parx(-0.5, 0.5, 1.0), (0, 1)), 'outlet2': Profile(parx(-0.5, 0.5, 1.0), (0, -1)),
              'wall': NoSlip()}, 'cross-slot')


if __name__ == "__main__":
    test_gallery()
