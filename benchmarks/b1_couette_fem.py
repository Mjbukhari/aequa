"""Benchmark 1, reference method: Taylor-Hood FEM for Carreau Couette flow in the annulus, on a
sequence of meshes (accuracy against cost).  No guaranteed bound is produced by this method."""
import numpy as np, time
from common import CouetteExact, CARREAU, save
from fem import polar_annulus_mesh, solve_carreau_stokes, fem_errors

ex = CouetteExact()
rows = []
for nr in (2, 4, 8, 16, 32):
    nt = 8*nr
    m = polar_annulus_mesh(ex.Ri, ex.Ro, nr, nt)
    rmid = 0.5*(ex.Ri + ex.Ro)
    inner = m.facets_satisfying(lambda x: np.hypot(x[0], x[1]) < rmid, boundaries_only=True)
    outer = m.facets_satisfying(lambda x: np.hypot(x[0], x[1]) > rmid, boundaries_only=True)
    sol = solve_carreau_stokes(m, [(inner, lambda x, y: (-ex.Om*y, ex.Om*x)),
                                   (outer, lambda x, y: (0*x, 0*y))])
    er = fem_errors(sol, ex)
    row = dict(nr=nr, nt=nt, ndofs=sol['ndofs'], picard_iters=sol['iters'], time_s=sol['time'], **er,
               J_minus_Jstar=er['J'] - ex.dissipation())
    rows.append(row)
    print(f"nr={nr:3d} dofs={sol['ndofs']:7d} it={sol['iters']:3d} t={sol['time']:6.2f}s  "
          f"L2={er['rel_L2_velocity']:.2e} H1={er['rel_energy_seminorm']:.2e} E={er['bregman_E']:.2e} "
          f"J-J*={row['J_minus_Jstar']:+.2e} divL2={er['div_L2']:.1e} area-err={er['area']-np.pi*0.75:.1e}", flush=True)
save('b1_couette_fem', dict(problem='Carreau Couette annulus', exact_J=ex.dissipation(), exact_power=ex.power(),
                            rows=rows))
