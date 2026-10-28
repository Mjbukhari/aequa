"""Benchmark 2: Stokes flow of a Carreau fluid past a cylinder in a channel (no exact solution).

Aequa returns certified enclosures of the minimal dissipation J*, of the power W* and of the pressure
drop W*/Q.  The reference values come from Taylor-Hood FEM on a sequence of meshes; the check is that
every certified enclosure contains the converged FEM value.

usage:  python b2_cylinder.py fem | aequa
"""
import math, os, sys, time
import numpy as np
from common import CARREAU, save

X0, X1, Y0, Y1 = -3.0, 5.0, -1.0, 1.0
C, R = (0.0, 0.0), 0.3
U_MEAN = 1.0
ITERS = int(os.environ.get('AEQUA_BENCH_ITERS', 2000))
LBFGS = int(os.environ.get('AEQUA_BENCH_LBFGS', 200))


def profile():
    import aequa as aq
    return aq.developed_profile(aq.Carreau(**CARREAU), Y0, Y1, U_MEAN)


def run_fem():
    from fem import channel_cylinder_mesh, solve_carreau_stokes, fem_energy_power
    prof = profile()
    rows = []
    for h, hw in ((0.2, 0.05), (0.1, 0.025), (0.05, 0.0125), (0.035, 0.008)):
        m = channel_cylinder_mesh(X0, X1, Y0, Y1, C, R, h, hw)
        tol = 1e-9
        inl = m.facets_satisfying(lambda x: np.abs(x[0] - X0) < tol, boundaries_only=True)
        out = m.facets_satisfying(lambda x: np.abs(x[0] - X1) < tol, boundaries_only=True)
        wal = m.facets_satisfying(lambda x: (np.abs(x[1] - Y0) < tol) | (np.abs(x[1] - Y1) < tol), boundaries_only=True)
        cyl = m.facets_satisfying(lambda x: np.hypot(x[0] - C[0], x[1] - C[1]) < R + 1e-6, boundaries_only=True)
        sp = lambda x, y: (prof(x, y), 0*x)
        sol = solve_carreau_stokes(m, [(inl, sp), (out, sp), (wal, lambda x, y: (0*x, 0*y)), (cyl, lambda x, y: (0*x, 0*y))])
        ew = fem_energy_power(sol)
        rows.append(dict(h=h, h_wall=hw, ndofs=sol['ndofs'], picard_iters=sol['iters'], time_s=sol['time'], **ew))
        print(f"FEM h={h} dofs={sol['ndofs']} J={ew['J']:.8f} W={ew['W']:.8f} div={ew['div_L2']:.1e} [{sol['time']:.1f}s]", flush=True)
    save('b2_cylinder_fem', dict(rows=rows, Q=2.0*U_MEAN))


def run_aequa():
    import torch, aequa as aq
    torch.set_num_threads(os.cpu_count())
    law = aq.Carreau(**CARREAU); prof = profile()
    dom = aq.channel(X1 - X0, Y1 - Y0, x0=X0, obstacles=[aq.circle(C, R, 'cylinder')])
    t0 = time.time()
    res = aq.solve(dom, law, {'inlet': aq.Profile(prof, (1, 0)), 'outlet': aq.Profile(prof, (1, 0)),
                              'wall': aq.NoSlip(), 'cylinder': aq.NoSlip()},
                   rho=0.0, iters=ITERS, lbfgs=LBFGS, width=32, depth=3, batch=1024, lr=3e-3)
    tt = time.time() - t0
    S, g = res.solver, res.gapinfo
    pb = S.power_bounds(g)
    J = g['fine']['diss']
    out = dict(G_up=res.G, G_fine=g['fine']['G'], G_coarse=g['coarse']['G'], quad_diff=g['quad_diff'],
               tails=g['fine']['tail_sum'], J_v=J, J_lo=J - res.G, J_hi=J, W_lo=pb['W_lo'], W_hi=pb['W_hi'],
               Q_in=pb['Q_in'], dp_lo=pb['W_lo']/pb['Q_in'], dp_hi=pb['W_hi']/pb['Q_in'],
               n_quad=g['fine']['n_points'], time_s=tt, summary=res.summary(),
               config=dict(iters=ITERS, lbfgs=LBFGS, width=32, depth=3, batch=1024))
    torch.save(S.state_dict(), os.path.join(os.path.dirname(__file__), 'results', 'b2_aequa_state.pt'))
    save('b2_cylinder_aequa', out)


if __name__ == '__main__':
    {'fem': run_fem, 'aequa': run_aequa}[sys.argv[1]]()
