"""
Pointwise checks of exact admissibility (certificate note, Lemma 3.1) for random networks:
    div v = 0,   v = v_D on every boundary piece,   div(Airy chi) = 0,
on domains with holes (Michell terms), re-entrant corners (corner terms) and curved walls.
These hold identically in exact arithmetic; here they are checked to rounding level.
"""
import math
import numpy as np
import torch
import aequa as aq

DT = torch.float64
par = lambda a, b, U: (lambda x, y: 6*U*(y - a)*(b - y)/(b - a)**2)
parx = lambda a, b, U: (lambda x, y: 6*U*(x - a)*(b - x)/(b - a)**2)


def _cases():
    yield 'T-junction + rotor', aq.t_junction(3.0, 1.0, 4.0, 1.0, obstacles=[aq.circle((0.0, 0.5), 0.2, 'rotor')]), \
        {'inlet': aq.Profile(parx(-0.5, 0.5, 1.0), (0, 1)), 'outlet1': aq.Profile(par(0, 1, 0.5), (1, 0)),
         'outlet2': aq.Profile(par(0, 1, 0.5), (-1, 0)), 'wall': aq.NoSlip(), 'rotor': aq.Rotating((0.0, 0.5), 2.0)}
    yield '4:1 contraction', aq.contraction(10.0, 4.0, 8.0, 1.0), \
        {'inlet': aq.Profile(par(-4, 4, 0.25), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'wall': aq.NoSlip()}
    yield 'channel + tilted square + ellipse', aq.channel(8.0, 2.0, x0=-2.0, obstacles=[
        aq.square((0, 0), 0.4, 'sq', angle=0.3), aq.ellipse((2.5, 0.0), 0.4, 0.15, 'ell', angle=0.4)]), \
        {'inlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
         'wall': aq.NoSlip(), 'sq': aq.NoSlip(), 'ell': aq.NoSlip()}


def _random_solver(dom, bcs, law):
    dom.set_bc(bcs)
    S = aq.Solver(dom, law, rho=0.7, width=16, depth=2)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in S.parameters(): p.add_(0.3*torch.randn(p.shape, generator=g, dtype=DT))
    return S


def _interior_points(dom, n, rng):
    x0, x1, y0, y1 = dom.bbox
    Q = np.c_[rng.uniform(x0, x1, 8*n), rng.uniform(y0, y1, 8*n)]
    Q = Q[dom.inside(Q[:, 0], Q[:, 1])][:n]
    # add points close to every vertex (inside the fluid sector)
    extra = []
    for v in dom.vertices:
        for r in (1e-3, 1e-2)*np.array(v['rho']):
            for t in (0.25, 0.5, 0.75):
                a = t*v['angle']
                extra.append(v['V'] + r*(math.cos(a)*v['e1'] + math.sin(a)*v['e2']))
    return np.r_[Q, np.array(extra)]


def test_divergence_and_equilibrium():
    rng = np.random.default_rng(0)
    for name, dom, bcs in _cases():
        S = _random_solver(dom, bcs, aq.Newtonian(1.0))
        P = _interior_points(dom, 400, rng)
        X = torch.tensor(P, dtype=DT).requires_grad_(True)
        f = S.fields(X)
        d1 = lambda q: torch.autograd.grad(q.sum(), X, retain_graph=True, create_graph=True)[0]
        divv = d1(f['u'])[:, 0] + d1(f['v'])[:, 1]
        # deviatoric stress m minus rho v(x)v plus the recovered pressure must be divergence free:
        # sigma = m - rho dev(v(x)v) - p I - rho/2 |v|^2 I = Airy(chi)
        rho = S.rho; u, v = f['u'], f['v']
        s11 = f['m11'] - rho*u*u - f['p'] + 0*v
        s22 = -f['m11'] - rho*v*v - f['p']
        s12 = f['m12'] - rho*u*v
        r1 = d1(s11)[:, 0] + d1(s12)[:, 1]
        r2 = d1(s12)[:, 0] + d1(s22)[:, 1]
        scale_v = float(d1(u).detach().abs().max()); scale_s = float(d1(s11).detach().abs().max() + d1(s12).detach().abs().max())
        dv = float(divv.abs().max())/scale_v; ds = float((r1.abs() + r2.abs()).max())/scale_s
        print(f"{name:35s} max|div v|/|grad v| = {dv:.1e}   max|div sigma|/|grad sigma| = {ds:.1e}")
        # rounding only: near vertices the lift's second derivatives are sums of O(r^-2) terms
        assert dv < 1e-9 and ds < 1e-9


def test_boundary_values():
    for name, dom, bcs in _cases():
        S = _random_solver(dom, bcs, aq.Newtonian(1.0))
        worst = 0.0
        for pc in dom.pieces_all:
            lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 9)[1:-1]
            Bp = pc.point(p); Pp = Bp + 1e-7*pc.inward_normal(p)
            f = S.evaluate(Pp, keys=('u', 'v'))
            worst = max(worst, float(np.abs(np.c_[f['u'], f['v']] - dom.bcs[pc.tag](Bp)).max()))
        print(f"{name:35s} max|v - u_b| at distance 1e-7 from the boundary = {worst:.1e}; data mismatch {dom.data_mismatch:.1e}")
        assert worst < 5e-5 and dom.data_mismatch < 1e-9


def test_corner_exponent_floors():
    """Lemma 3.1: whatever the weights, lambda > lambda_min and beta > beta_min"""
    for law in [aq.Newtonian(1.0), aq.HerschelBulkley(3.72, 0.41, 3.31), aq.PowerLaw(1.0, 0.3), aq.PowerLaw(1.0, 0.2), aq.PowerLaw(1.0, 1.5)]:
        dom = aq.contraction(10.0, 4.0, 8.0, 1.0)
        S = _random_solver(dom, {'inlet': aq.Profile(par(-4, 4, 0.25), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
                                 'wall': aq.NoSlip()}, law)
        with torch.no_grad():
            S.corner_lam.fill_(-50.0); S.corner_beta.fill_(-50.0)      # push the parameters as low as possible
        lam, beta = S.exponents()
        # rho > 0 here, so the inertial floor of Lemma 3.2 applies (stronger for 1 < p < 4/3)
        assert (lam > law.lam_min_inertial).all() and (beta > law.beta_min).all()
        kth = S.kappa_theory()
        assert all(k < 2.0 for k in kth.values())


def test_vertex_data():
    """Lemma 4.2: a lid with a velocity jump is rejected for p >= 2 and accepted for p < 2"""
    dom = aq.cavity(1.0, 1.0)
    dom.set_bc({'lid': aq.MovingWall(1.0), 'wall': aq.NoSlip()})
    try:
        aq.Solver(dom, aq.Newtonian(1.0))
        assert False, "discontinuous lid must be rejected for a Newtonian fluid"
    except ValueError:
        pass
    S = aq.Solver(dom, aq.PowerLaw(1.0, 0.5))          # p = 1.5 < 2: finite dissipation
    assert len(S.vertex_jumps) == 2
    dom = aq.cavity(1.0, 1.0)
    dom.set_bc({'lid': aq.Velocity(lambda x, y: (16*x**2*(1 - x)**2, 0*x)), 'wall': aq.NoSlip()})
    aq.Solver(dom, aq.Newtonian(1.0))


def test_noslip_vertex_blending():
    """Remark 4.3: at a no-slip vertex the lift is exactly constant near the vertex (no cancellation
    error in its second derivatives), while the boundary data stay exact"""
    dom = aq.contraction(10.0, 4.0, 8.0, 1.0)
    dom.set_bc({'inlet': aq.Profile(par(-4, 4, 0.25), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'wall': aq.NoSlip()})
    v = [v for v in dom.vertices if abs(v['V'][0]) < 1e-12 and abs(v['V'][1] - 1) < 1e-12][0]
    assert v['rigid']
    rs = np.geomspace(1e-9, 0.2, 12)*v["rho"]
    P = np.array([v['V'] + r*(math.cos(0.4*v['angle'])*v['e1'] + math.sin(0.4*v['angle'])*v['e2']) for r in rs])
    X = torch.tensor(P, dtype=DT).requires_grad_(True)
    psi = dom.psi_lift(X[:, 0], X[:, 1])
    g = torch.autograd.grad(psi.sum(), X, create_graph=True)[0]
    H = torch.stack([torch.autograd.grad(g[:, i].sum(), X, retain_graph=True)[0] for i in range(2)], 1)
    assert float(H.abs().max()) == 0.0 and float(g.abs().max()) == 0.0


if __name__ == "__main__":
    test_divergence_and_equilibrium(); test_boundary_values(); test_corner_exponent_floors()
    test_vertex_data(); test_noslip_vertex_blending(); print("all admissibility tests passed")
