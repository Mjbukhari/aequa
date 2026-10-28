"""Reference finite-element solver: Taylor-Hood P2-P1 on an isoparametric (quadratic) triangular mesh,
Picard iteration for a Carreau fluid, all-Dirichlet velocity data, Stokes (no inertia).

This is the standard method the comparison is made against.  It is accurate and fast in 2D, but its
velocity is only discretely divergence free and it carries no guaranteed error bound.
"""
import math, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from skfem import (MeshTri, MeshTri2, ElementTriP2, ElementTriP1, ElementVector, Basis, FacetBasis,
                   BilinearForm, Functional, asm, bmat, condense, solve)
from skfem.helpers import ddot, sym_grad, div
from common import carreau_eta, carreau_F, CARREAU


def polar_annulus_mesh(Ri, Ro, nr, nt):
    """structured (r, theta) triangulation of the annulus; every node and every edge midpoint is
    placed exactly on its polar position, so the circles are represented by quadratic arcs"""
    r = np.linspace(Ri, Ro, nr + 1); th = np.linspace(0, 2*math.pi, nt, endpoint=False)
    R, T = np.meshgrid(r, th, indexing='ij')
    idx = lambda i, j: i*nt + (j % nt)
    tris = []
    for i in range(nr):
        for j in range(nt):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if (i + j) % 2 == 0: tris += [(a, b, c), (a, c, d)]
            else: tris += [(a, b, d), (b, c, d)]
    pr = np.c_[R.ravel(), T.ravel()]                       # polar coordinates of the vertices
    p = np.c_[pr[:, 0]*np.cos(pr[:, 1]), pr[:, 0]*np.sin(pr[:, 1])].T
    m1 = MeshTri(p, np.array(tris).T)
    m2 = MeshTri2.from_mesh(m1)
    # move every edge midpoint to the polar midpoint of its edge
    f = m1.facets
    r_mid = 0.5*(pr[f[0], 0] + pr[f[1], 0])
    dth = np.angle(np.exp(1j*(pr[f[1], 1] - pr[f[0], 1])))
    t_mid = pr[f[0], 1] + 0.5*dth
    nv = m1.p.shape[1]
    D = m2.doflocs.copy()
    chord = 0.5*(m1.p[:, f[0]] + m1.p[:, f[1]])
    assert np.allclose(D[:, nv:], chord), "unexpected dof ordering in MeshTri2"
    D[:, nv:] = np.vstack([r_mid*np.cos(t_mid), r_mid*np.sin(t_mid)])
    return MeshTri2(D, m2.t)


def solve_carreau_stokes(mesh, dirichlet, law=CARREAU, tol=1e-10, maxit=200, verbose=False, intorder=6):
    """dirichlet: list of (facet indices, function (x, y) -> (ux, uy)).  Returns dict with the velocity
    and pressure coefficients, bases, iteration count and wall time."""
    t0 = time.time()
    eu = ElementVector(ElementTriP2()); ep = ElementTriP1()
    ub = Basis(mesh, eu, intorder=intorder); pb = Basis(mesh, ep, intorder=intorder)

    @BilinearForm
    def visc(u, v, w):
        return 2*w['eta']*ddot(sym_grad(u), sym_grad(v))

    @BilinearForm
    def bdiv(u, q, w):
        return -div(u)*q

    B = asm(bdiv, ub, pb)
    # boundary data by facet L2 projection (exact for data that are polynomial on each curved facet)
    x = np.zeros(ub.N + pb.N)
    D = []
    for facets, fn in dirichlet:
        fb = FacetBasis(mesh, eu, facets=facets, intorder=intorder)
        g = fb.project(lambda X: np.array(fn(X[0], X[1])))
        dofs = ub.get_dofs(facets).all()
        x[dofs] = g[dofs]; D.append(dofs)
    D = np.unique(np.concatenate(D + [np.array([ub.N])]))           # + one pressure dof (mean gauge)
    eta = law['eta0']*np.ones((mesh.t.shape[1], ub.X.shape[-1]))
    it_used = 0
    for it in range(maxit):
        A = asm(visc, ub, eta=eta)
        K = bmat([[A, B.T], [B, None]], 'csr')
        xn = solve(*condense(K, np.zeros(K.shape[0]), x=x, D=D))
        du = np.linalg.norm(xn[:ub.N] - x[:ub.N])/max(np.linalg.norm(xn[:ub.N]), 1e-300)
        x = xn; it_used = it + 1
        gu = ub.interpolate(x[:ub.N]).grad
        e = 0.5*(gu + np.transpose(gu, (1, 0, 2, 3)))
        gd = np.sqrt(2*(e*e).sum(axis=(0, 1)))
        eta = carreau_eta(gd, **law)
        if verbose: print(f"  picard {it}: du {du:.2e}")
        if du < tol and it > 0: break
    return dict(x=x, ub=ub, pb=pb, iters=it_used, time=time.time() - t0, ndofs=ub.N + pb.N)


def fem_errors(sol, ex, law=CARREAU):
    """errors against an exact solution object (velocity, strain, stress), at the FEM quadrature points"""
    ub = sol['ub']; uh = ub.interpolate(sol['x'][:ub.N])
    X = ub.global_coordinates().value                      # (2, nel, nq)
    xs, ys = X[0], X[1]
    W = ub.dx                                              # (nel, nq) weights incl. Jacobian
    u, v = uh.value[0], uh.value[1]
    gu = uh.grad; e11 = gu[0, 0]; e12 = 0.5*(gu[0, 1] + gu[1, 0]); e22 = gu[1, 1]
    ue, ve = ex.velocity(xs, ys); ae, be = ex.strain(xs, ys)
    L2 = math.sqrt(((((u - ue)**2 + (v - ve)**2))*W).sum()/(((ue**2 + ve**2))*W).sum())
    de = (e11 - ae)**2 + 2*(e12 - be)**2 + (e22 + ae)**2       # |eps(v - u)|^2, eps22(u*) = -eps11(u*)
    H1 = math.sqrt((de*W).sum()/((2*(ae**2 + be**2))*W).sum())
    gdv = np.sqrt(2*(e11**2 + 2*e12**2 + e22**2)); gdu = 2*np.sqrt(ae**2 + be**2)
    S11, S12 = ex.stress(xs, ys)
    E = (carreau_F(gdv, **law) - carreau_F(gdu, **law) - (S11*(e11 - ae) + 2*S12*(e12 - be) - S11*(e22 + ae)))
    divu = e11 + e22
    return dict(rel_L2_velocity=L2, rel_energy_seminorm=H1, eps_err_sq=float((de*W).sum()),
                bregman_E=float((E*W).sum()), J=float((carreau_F(gdv, **law)*W).sum()),
                div_L2=math.sqrt(float((divu**2*W).sum())), area=float(W.sum()))


def channel_cylinder_mesh(x0, x1, y0, y1, c, R, h, h_wall):
    """triangulation of a rectangular channel minus a disc: graded polar rings around the disc
    (first ring at spacing h_wall), a uniform grid of spacing h elsewhere, Delaunay; the edge
    midpoints on the disc boundary are moved onto the circle (isoparametric P2 geometry)"""
    from scipy.spatial import Delaunay
    nt = int(math.ceil(2*math.pi*R/h_wall))
    th = np.linspace(-math.pi, math.pi, nt, endpoint=False)
    rings, r = [], R
    while True:
        rings.append(r)
        dr = min(h, h_wall*(r/R)**1.6)
        if r + dr > R + 2.5*R: break
        r += dr
    pts = [np.c_[c[0] + rr*np.cos(th + 0.5*(k % 2)*(th[1] - th[0])), c[1] + rr*np.sin(th + 0.5*(k % 2)*(th[1] - th[0]))]
           for k, rr in enumerate(rings)]
    nx, ny = int(math.ceil((x1 - x0)/h)), int(math.ceil((y1 - y0)/h))
    X, Y = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1), indexing='ij')
    G = np.c_[X.ravel(), Y.ravel()]
    keep = np.hypot(G[:, 0] - c[0], G[:, 1] - c[1]) > rings[-1] + 0.6*h
    pts = np.r_[np.concatenate(pts), G[keep]]
    pts = pts[(pts[:, 0] >= x0 - 1e-12) & (pts[:, 0] <= x1 + 1e-12) & (pts[:, 1] >= y0 - 1e-12) & (pts[:, 1] <= y1 + 1e-12)]
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(1)
    tri = tri[np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1]) > R]
    used = np.unique(tri); remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    m1 = MeshTri(pts[used].T, remap[tri].T)
    m2 = MeshTri2.from_mesh(m1)
    D = m2.doflocs.copy(); nv = m1.p.shape[1]; f = m1.facets
    a, b = m1.p[:, f[0]], m1.p[:, f[1]]
    on = (np.abs(np.hypot(a[0] - c[0], a[1] - c[1]) - R) < 1e-9) & (np.abs(np.hypot(b[0] - c[0], b[1] - c[1]) - R) < 1e-9)
    mid = D[:, nv:][:, on]
    ang = np.arctan2(mid[1] - c[1], mid[0] - c[0])
    D[:, nv + np.where(on)[0]] = np.vstack([c[0] + R*np.cos(ang), c[1] + R*np.sin(ang)])
    return MeshTri2(D, m2.t)


def fem_energy_power(sol, law=CARREAU):
    """J(u_h) = int F(gd(u_h)) and W(u_h) = int S(u_h):eps(u_h) on the FEM quadrature"""
    ub = sol['ub']; uh = ub.interpolate(sol['x'][:ub.N]); W = ub.dx
    gu = uh.grad; e = 0.5*(gu + np.transpose(gu, (1, 0, 2, 3)))
    gd = np.sqrt(2*(e*e).sum(axis=(0, 1)))
    return dict(J=float((carreau_F(gd, **law)*W).sum()), W=float((carreau_eta(gd, **law)*gd**2*W).sum()),
                div_L2=math.sqrt(float(((e[0, 0] + e[1, 1])**2*W).sum())), area=float(W.sum()))
