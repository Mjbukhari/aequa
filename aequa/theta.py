"""
Automatic certified bound for the inertia number theta (dual enclosure theorem).

In the normalisation of the gap theorems (E >= int nu0 |eps(e)|^2):
    theta := sup_{w in V}  -rho int w.eps(v)w / int nu0 |eps(w)|^2
           <= 2 * ( lambda_max(D) + eta^2 )
where D is the Gram matrix (in the 1/(2 nu0) weighted dev-norm) of equilibrated stresses
for cellwise-constant forcings P_K^{1/2} e_i 1_{K cap Omega}/|K cap Omega|^{1/2}, and eta^2
is the explicit local remainder (Payne-Weinberger on cells, 1D Poincare on wall cells,
angular Hardy at re-entrant corners, weighted Korn with a radial weight).

Valid on any 2D domain with all-Dirichlet data (Thm 20.2 needs no boundary regularity): cells are
axis-parallel boxes, cut cells are mean cells (Payne-Weinberger on the convex box, zero extension).
Stresses: line-integral particular solutions along x- and y-lines, cut at the boundary
(each connected piece of a line carries its own force to its nearer end), plus a
C^1 cubic tensor-spline Airy correction optimised in one sparse solve.

Guarantee level [sampled] (certificate note, Sec. 6): the cell majorants P_K, the floor nu_* and the
Gram matrix D are computed from sampled network values (with a Lipschitz margin), not with interval
arithmetic.  Every equilibrated stress used is admissible exactly: the free ends of the line
stresses are moved a margin OUTSIDE the domain, so no spurious point force lies in the fluid.
"""
import math, time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh
from scipy.interpolate import BSpline
from scipy.linalg import eigh, sqrtm

from .solver import strain_with_grad, DT

DEBUG = False
GHOST = 1e-8
G3 = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)]); W3 = np.array([5, 8, 5])/9.0


# ------------------------------------------------------------------ helpers
from .geometry import reentrant_corners

def dist_to_other_pieces(dom, pt):
    """distance from a vertex to boundary pieces not touching it"""
    d = np.inf
    for pc in dom.pieces_all:
        S = pc.sample(400)
        if np.min(np.linalg.norm(S - pt[None, :], axis=1)) < 1e-9: continue
        d = min(d, float(np.min(np.linalg.norm(S - pt[None, :], axis=1))))
    return d


class RadialWeight:
    """nu_tilde = nu_* f(rho) with f = min(1, (rho/r1)^a1) (and a2 below r2) about each
       re-entrant corner; Korn factor kappa = 1 - (omega/pi) alpha_local"""
    def __init__(self, corners, nus=1.0, r1=0.5, r2=1e-3, a1=0.15, a2=None):
        self.corners = corners; self.nus = nus; self.r1, self.r2, self.a1 = r1, r2, a1
        self.a2 = a2
    def rho(self, P):
        if not self.corners: return np.full(len(P), np.inf), np.ones(len(P))
        D = np.stack([np.linalg.norm(P - c[None, :], axis=1) for c, _ in self.corners], 1)
        k = D.argmin(1)
        om = np.array([w for _, w in self.corners])[k]
        return D.min(1), om
    def f(self, rho, om):
        a2 = self.a2 if self.a2 is not None else np.maximum(self.a1, 1.0 - np.pi/om)
        rho = np.maximum(rho, 1e-300)
        f1 = np.minimum(1.0, (rho/self.r1)**self.a1)
        f2 = (self.r2/self.r1)**self.a1*(rho/self.r2)**a2
        return np.where(rho >= self.r2, f1, f2)
    def __call__(self, rho, om): return self.nus*self.f(rho, om)
    def kappa(self, rho, om):
        a2 = self.a2 if self.a2 is not None else np.maximum(self.a1, 1.0 - np.pi/om)
        return np.where(rho >= self.r1, 1.0, np.where(rho >= self.r2, 1 - (om/np.pi)*self.a1, 1 - (om/np.pi)*a2))


def bspl_mats(x, knots, k=3):
    t = np.r_[[knots[0]]*(k + 1), np.repeat(knots[1:-1], k - 1), [knots[-1]]*(k + 1)]
    n = len(t) - k - 1
    def dmat(kk):
        nk = len(t) - kk - 1; rows, cols, vals = [], [], []
        for j in range(1, nk):
            den = t[j + kk] - t[j]
            if den <= 0: continue
            rows += [j, j]; cols += [j, j - 1]; vals += [kk/den, -kk/den]
        return sp.csr_matrix((vals, (rows, cols)), shape=(nk + 1, nk))
    B0 = BSpline.design_matrix(x, t, k).tocsr()
    B1 = (BSpline.design_matrix(x, t, k - 1).tocsr() @ dmat(k)).tocsr()
    B2 = (BSpline.design_matrix(x, t, k - 2).tocsr() @ (dmat(k - 1) @ dmat(k))).tocsr()
    return B0, B1, B2, n

def kr(A, B, nb):
    def pad(A):
        A = A.tocsr(); A.sort_indices(); n = np.diff(A.indptr); m = max(1, n.max())
        ci = np.zeros((A.shape[0], m), np.int64); vi = np.zeros((A.shape[0], m))
        pos = np.arange(A.nnz) - np.repeat(A.indptr[:-1], n)
        ci[np.repeat(np.arange(A.shape[0]), n), pos] = A.indices
        vi[np.repeat(np.arange(A.shape[0]), n), pos] = A.data
        return ci, vi
    ca, va = pad(A); cb, vb = pad(B)
    C = (ca[:, :, None]*nb + cb[:, None, :]).reshape(len(ca), -1)
    V = (va[:, :, None]*vb[:, None, :]).reshape(len(ca), -1)
    R = np.repeat(np.arange(len(ca)), C.shape[1])
    return R, C.ravel(), V.ravel()

def thin_graded(Z, hf):
    out = [Z[0]]
    for z in Z[1:-1]:
        if z - out[-1] >= hf(z) - 1e-12: out.append(z)
    if len(out) > 1 and Z[-1] - out[-1] < 1e-12: out.pop()
    out.append(Z[-1]); return np.array(out)

def line_intervals(dom, const, along, lo, hi, n=4000):
    """inside-intervals of the line {y=const} (along='x') or {x=const} (along='y')"""
    s = np.linspace(lo, hi, n)
    ins = dom.inside(s, np.full_like(s, const)) if along == 'x' else dom.inside(np.full_like(s, const), s)
    d = np.diff(np.r_[0, ins.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0] - 1
    out = []
    def refine(i0, i1):          # boundary between s[i0] (inside?) and s[i1]
        a, b = s[i0], s[i1]; fa = True
        f = (lambda z: dom.inside(np.array([z]), np.array([const]))[0]) if along == 'x' else \
            (lambda z: dom.inside(np.array([const]), np.array([z]))[0])
        ia = f(a)
        for _ in range(50):
            m = 0.5*(a + b)
            if f(m) == ia: a = m
            else: b = m
        return 0.5*(a + b)
    margin = 1e-9*max(1.0, abs(hi - lo)) + 2*dom.max_sag
    for a_, b_ in zip(st, en):
        L = refine(a_ - 1, a_) if a_ > 0 else lo
        R = refine(b_, b_ + 1) if b_ < n - 1 else hi
        # move the free ends outside the fluid: a line stress jumps at its end, and that jump must not
        # lie inside the domain (it would be a point force in the fluid)
        L -= margin; R += margin
        if a_ > 0 and dom.inside(*(np.array([L]), np.array([const])) if along == 'x' else (np.array([const]), np.array([L])))[0]:
            raise RuntimeError("line stress end inside the domain")
        if b_ < n - 1 and dom.inside(*(np.array([R]), np.array([const])) if along == 'x' else (np.array([const]), np.array([R])))[0]:
            raise RuntimeError("line stress end inside the domain")
        out.append((L, R))
    return out


# ------------------------------------------------------------------ main routine
def theta_bound(S, lam=0.0, s=0.1, target=0.25, base_cells=16, NS=5, safe=1.5,
                rho_c=3e-3, HK=None, verbose=True, max_iter=30):
    t0 = time.time()
    dom, law, rho = S.dom, S.law, S.rho
    x0b, x1b, y0b, y1b = dom.bbox
    Lb = max(x1b - x0b, y1b - y0b)
    if HK is None: HK = Lb/256
    corners = reentrant_corners(dom)

    def strain(P):
        return strain_with_grad(S, P)

    def nu0_of(a, b):
        ne = np.sqrt(2*a*a + 2*b*b)                         # |eps(v)|
        gmax = torch.tensor(math.sqrt(2)*((1 + lam)*ne + s), dtype=DT)
        gd_v = torch.tensor(math.sqrt(2)*ne, dtype=DT)
        return law.nu0(gmax, gd_v).numpy()

    # ---- floor nu_* for nu_tilde <= nu0 (dense samples; + polar samples at re-entrant corners)
    h = Lb/300
    xs = np.arange(x0b + h/2, x1b, h); ys = np.arange(y0b + h/2, y1b, h)
    XX, YY = np.meshgrid(xs, ys, indexing='ij'); m = dom.inside(XX, YY)
    Ps = np.c_[XX[m], YY[m]]
    for c, om in corners:
        rs = np.geomspace(1e-7, 1.0, 60); ph = np.linspace(0, 2*np.pi, 181)
        R, PH = np.meshgrid(rs, ph, indexing='ij')
        Q = np.c_[c[0] + (R*np.cos(PH)).ravel(), c[1] + (R*np.sin(PH)).ravel()]
        Ps = np.r_[Ps, Q[dom.inside(Q[:, 0], Q[:, 1])]]
    Es = strain(Ps); nus = nu0_of(Es[:, 0], Es[:, 1])
    r1 = 0.5*min([dist_to_other_pieces(dom, c) for c, _ in corners] + [1.0]) if corners else 1.0
    Wt = RadialWeight(corners, 1.0, r1=r1)
    rs_, om_ = Wt.rho(Ps)
    nus_star = 0.95*float(np.min(nus/Wt.f(rs_, om_)))
    Wt.nus = nus_star
    mu_s = np.sqrt(Es[:, 0]**2 + Es[:, 1]**2)
    if verbose: print(f"[theta] nu_* = {nus_star:.4g}, re-entrant corners: {len(corners)}", flush=True)

    # corner Hardy term
    HC = 0.0
    for c, om in corners:
        sel = (np.linalg.norm(Ps - c[None, :], axis=1) <= rho_c)
        if sel.any():
            rr = np.linalg.norm(Ps[sel] - c[None, :], axis=1)
            HC = max(HC, (om/np.pi)**2*np.max(rho*mu_s[sel]*rr**2/Wt(rr, om)/Wt.kappa(rr, np.full_like(rr, om))))

    # ---- adaptive tiling
    nb = base_cells
    hx0, hy0 = (x1b - x0b)/nb, (y1b - y0b)/max(1, round(nb*(y1b - y0b)/(x1b - x0b)))
    hc = min(hx0, hy0)
    nx0, ny0 = int(round((x1b - x0b)/hc)), int(round((y1b - y0b)/hc))
    todo = [(x0b + i*(x1b - x0b)/nx0, x0b + (i + 1)*(x1b - x0b)/nx0,
             y0b + j*(y1b - y0b)/ny0, y0b + (j + 1)*(y1b - y0b)/ny0) for i in range(nx0) for j in range(ny0)]
    axis_edges = [pc for pc in dom.pieces_all if pc.kind == 'segment' and (abs(pc.a[0] - pc.b[0]) < 1e-12 or abs(pc.a[1] - pc.b[1]) < 1e-12)]
    final = []
    for it in range(max_iter):
        if not todo: break
        # samples
        Pc = []
        for (x0, x1, y0, y1) in todo:
            gx, gy = np.meshgrid(np.linspace(x0, x1, NS), np.linspace(y0, y1, NS), indexing='ij')
            Pc.append(np.c_[gx.ravel(), gy.ravel()])
        Pc = np.stack(Pc)                                        # (ncell, NS^2, 2)
        ins = dom.inside(Pc[..., 0], Pc[..., 1])
        # a finer inside test for cells with no inside sample but maybe a sliver
        keep = ins.any(1)
        for k in np.where(~keep)[0]:
            x0, x1, y0, y1 = todo[k]
            gx, gy = np.meshgrid(np.linspace(x0, x1, 21), np.linspace(y0, y1, 21), indexing='ij')
            if dom.inside(gx, gy).any(): keep[k] = True
        work = [todo[k] for k in np.where(keep)[0]]; Pw = Pc[keep]; insw = ins[keep]
        if not work: break
        # corner patch cells
        nxt = []; wk, Pk, ik = [], [], []
        for c_, P_, i_ in zip(work, Pw, insw):
            x0, x1, y0, y1 = c_
            touch = [cc for cc, om in corners if x0 - 1e-12 <= cc[0] <= x1 + 1e-12 and y0 - 1e-12 <= cc[1] <= y1 + 1e-12]
            if touch:
                cc = touch[0]
                rmax = max(np.hypot(px - cc[0], py - cc[1]) for px in (x0, x1) for py in (y0, y1))
                if rmax <= rho_c: final.append(dict(c=c_, kind='corner')); continue
                nxt += split4(c_); continue
            wk.append(c_); Pk.append(P_); ik.append(i_)
        if wk:
            Pk = np.stack(Pk)
            ik = np.stack(ik)
            F = strain(Pk.reshape(-1, 2)).reshape(len(wk), NS*NS, 6)
            F = np.where(ik[..., None], F, np.nan)              # only samples inside the domain
            a, b = F[..., 0], F[..., 1]
            ab, bb = np.nanmean(a, 1), np.nanmean(b, 1)
            mu = np.sqrt(a*a + b*b); dev = np.sqrt((a - ab[:, None])**2 + (b - bb[:, None])**2)
            lip = np.nanmax(np.sqrt((F[..., 2:]**2).sum(-1)), 1)*safe
            for k, c_ in enumerate(wk):
                x0, x1, y0, y1 = c_
                if ik[k].sum() < 3 and max(x1 - x0, y1 - y0) > 1e-3*Lb:
                    nxt += split4(c_); continue
                if ik[k].sum() == 0:
                    raise RuntimeError(f"cell {c_} has no interior sample")
                cut = not bool(ik[k].all())
                delta = (2.0 if cut else 1.0)*0.5*math.hypot((x1 - x0)/(NS - 1), (y1 - y0)/(NS - 1))
                Mk = np.nanmax(mu[k] + dev[k]) + 2*lip[k]*delta
                Pn = rho*(Mk + math.hypot(ab[k], bb[k]))/2
                corners_pts = np.array([[px, py] for px in (x0, x1) for py in (y0, y1)])
                rr, oo = Wt.rho(np.r_[corners_pts, [[0.5*(x0 + x1), 0.5*(y0 + y1)]]])
                # closest distance of the rectangle to a corner
                if corners:
                    dmin = min(math.hypot(max(x0 - cc[0], 0, cc[0] - x1), max(y0 - cc[1], 0, cc[1] - y1)) for cc, _ in corners)
                    omk = oo.min()
                else:
                    dmin, omk = np.inf, np.pi
                den = float(Wt(np.array([dmin]), np.array([omk]))[0]*Wt.kappa(np.array([dmin]), np.array([omk]))[0])
                d = math.hypot(x1 - x0, y1 - y0)
                ca = Pn*(d/math.pi)**2/den
                full = not cut
                cb, side = np.inf, None
                if full:
                    for pc in axis_edges:
                        for sname, seg in (('bot', ((x0, y0), (x1, y0))), ('top', ((x0, y1), (x1, y1))),
                                           ('left', ((x0, y0), (x0, y1))), ('right', ((x1, y0), (x1, y1)))):
                            if on_segment(seg, pc):
                                hperp = (y1 - y0) if sname in ('bot', 'top') else (x1 - x0)
                                v = Pn*(2*hperp/math.pi)**2/den
                                if v < cb: cb, side = v, sname
                rec = dict(c=c_, ab=ab[k], bb=bb[k], M=Mk, Pn=Pn, eta2a=ca, eta2b=cb, side=side)
                if cb <= target: rec['kind'] = 'wall'; final.append(rec)
                elif ca <= target: rec['kind'] = 'mean'; final.append(rec)
                elif side is not None and cb <= 4*target: nxt += split_perp(c_, side)
                else: nxt += split4(c_)
        todo = nxt
        if verbose:
            print(f"[theta] tiling iter {it}: {len(final)} cells done, {len(todo)} to refine [{time.time()-t0:.0f}s]", flush=True)
    if todo:
        raise RuntimeError("tiling did not finish; increase max_iter or target")
    eta2 = max([c['eta2a'] if c['kind'] == 'mean' else c['eta2b'] for c in final if c['kind'] != 'corner'] + [HC])
    cells = [c for c in final if c['kind'] == 'mean']
    if verbose:
        print(f"[theta] cells: {len(cells)} mean, {sum(c['kind']=='wall' for c in final)} wall, "
              f"{sum(c['kind']=='corner' for c in final)} corner; eta^2 = {eta2:.4f}", flush=True)
    lamD, info = dual_lambda(S, dom, final, cells, rho, lam, s, nu0_of, HK, corners, verbose, t0)
    theta = 2.0*(lamD + eta2)
    return dict(theta=theta, lambda_D=lamD, eta2=eta2, nu_star=nus_star, n_cells=len(final),
                n_dofs=2*len(cells), **info)


def on_segment(seg, pc, tol=1e-9):
    """is the axis-aligned cell side 'seg' contained in the polygon edge pc?"""
    (ax, ay), (bx, by) = seg
    A, B = pc.a, pc.b
    for (px, py) in ((ax, ay), (bx, by)):
        cross = (B[0] - A[0])*(py - A[1]) - (B[1] - A[1])*(px - A[0])
        if abs(cross) > tol*max(1, pc.L): return False
        t = ((px - A[0])*(B[0] - A[0]) + (py - A[1])*(B[1] - A[1]))/pc.L**2
        if t < -tol or t > 1 + tol: return False
    return True

def split4(c):
    x0, x1, y0, y1 = c; xm, ym = 0.5*(x0 + x1), 0.5*(y0 + y1)
    return [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]

def split_perp(c, side):
    x0, x1, y0, y1 = c
    if side in ('top', 'bot'):
        ym = 0.5*(y0 + y1); return [(x0, x1, y0, ym), (x0, x1, ym, y1)]
    xm = 0.5*(x0 + x1); return [(x0, xm, y0, y1), (xm, x1, y0, y1)]


# ------------------------------------------------------------------ dual part
def dual_lambda(S, dom, allc, cells, rho, lam, s, nu0_of, HK, corners, verbose, t0, chunk=400):
    X = np.unique(np.r_[[c['c'][0] for c in allc], [c['c'][1] for c in allc]])
    Y = np.unique(np.r_[[c['c'][2] for c in allc], [c['c'][3] for c in allc]])
    # quadrature: 3x3 Gauss on the tensor grid of all cell edges, points inside the domain
    xc, yc = 0.5*(X[1:] + X[:-1]), 0.5*(Y[1:] + Y[:-1])
    J_, K_ = np.meshgrid(np.arange(len(xc)), np.arange(len(yc)), indexing='ij'); J_, K_ = J_.ravel(), K_.ravel()
    hx, hy = X[J_ + 1] - X[J_], Y[K_ + 1] - Y[K_]
    px = (xc[J_][:, None, None] + 0.5*hx[:, None, None]*G3[None, :, None]) + 0*G3[None, None, :]
    py = (yc[K_][:, None, None] + 0.5*hy[:, None, None]*G3[None, None, :]) + 0*G3[None, :, None]
    wq = (0.25*hx*hy)[:, None, None]*W3[None, :, None]*W3[None, None, :]
    px, py, wq = px.ravel(), py.ravel(), wq.ravel()
    ins = dom.inside(px, py); px, py, wq = px[ins], py[ins], wq[ins]
    Nq = len(px)
    E = strain_with_grad(S, np.c_[px, py])
    nuq = nu0_of(E[:, 0], E[:, 1]); Wq = wq/nuq
    if verbose: print(f"[theta] quadrature points: {Nq} [{time.time()-t0:.0f}s]", flush=True)
    # line structure: distinct y (rows) and x (columns) of quadrature points
    ykey = np.round(py, 12); xkey = np.round(px, 12)
    uy, inv_y = np.unique(ykey, return_inverse=True); ux, inv_x = np.unique(xkey, return_inverse=True)
    x0b, x1b, y0b, y1b = dom.bbox
    rows = [None]*len(uy); cols = [None]*len(ux)
    order_y = np.argsort(inv_y, kind='stable'); ptr_y = np.r_[0, np.cumsum(np.bincount(inv_y, minlength=len(uy)))]
    order_x = np.argsort(inv_x, kind='stable'); ptr_x = np.r_[0, np.cumsum(np.bincount(inv_x, minlength=len(ux)))]
    def row(k):
        if rows[k] is None:
            rows[k] = (order_y[ptr_y[k]:ptr_y[k + 1]], line_intervals(dom, uy[k], 'x', x0b, x1b))
        return rows[k]
    def col(k):
        if cols[k] is None:
            cols[k] = (order_x[ptr_x[k]:ptr_x[k + 1]], line_intervals(dom, ux[k], 'y', y0b, y1b))
        return cols[k]
    rr, cc, vv = [], [], []
    col_id = 0
    for c in cells:
        x0, x1, y0, y1 = c['c']
        inK = (px > x0) & (px < x1) & (py > y0) & (py < y1)
        area = wq[inK].sum()
        if area <= 0:
            area = (x1 - x0)*(y1 - y0)
        P = rho*0.5*np.array([[c['M'] - c['ab'], -c['bb']], [-c['bb'], c['M'] + c['ab']]])
        Sm = np.real(sqrtm(P))
        ks = np.unique(inv_y[inK]); js = np.unique(inv_x[inK])
        contrib = [[], []]     # per component: (idx, value) pairs
        for k in ks:
            idx, ivs = row(k)
            xx = px[idx]
            for (a_, b_) in ivs:
                fl, fr = max(x0, a_), min(x1, b_)
                if fr <= fl: continue
                seg = (xx >= a_ - 1e-12) & (xx <= b_ + 1e-12)
                if (fl - a_) < (b_ - fr):     # support runs to the left end
                    L = np.clip(fr - np.maximum(xx, fl), 0, fr - fl); sg = 1.0
                else:
                    L = np.clip(np.minimum(xx, fr) - fl, 0, fr - fl); sg = -1.0
                m_ = seg & (L > 0)
                contrib[0].append((idx[m_], 0.5*sg*L[m_]))          # d1 += s11/2 * f1
        for j in js:
            idx, ivs = col(j)
            yy = py[idx]
            for (a_, b_) in ivs:
                fl, fr = max(y0, a_), min(y1, b_)
                if fr <= fl: continue
                seg = (yy >= a_ - 1e-12) & (yy <= b_ + 1e-12)
                if (fl - a_) < (b_ - fr):
                    L = np.clip(fr - np.maximum(yy, fl), 0, fr - fl); sg = 1.0
                else:
                    L = np.clip(np.minimum(yy, fr) - fl, 0, fr - fl); sg = -1.0
                m_ = seg & (L > 0)
                contrib[1].append((idx[m_], -0.5*sg*L[m_]))         # d1 -= s22/2 * f2
        for i in range(2):
            f = Sm[:, i]/math.sqrt(area)
            for comp in (0, 1):
                for idx, val in contrib[comp]:
                    rr.append(idx); cc.append(np.full(len(idx), col_id)); vv.append(val*f[comp])
            col_id += 1
    NG = col_id
    T1 = sp.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(Nq, NG))
    T1.sum_duplicates()
    del rr, cc, vv
    if verbose: print(f"[theta] particular stresses: {NG} columns, nnz {T1.nnz} [{time.time()-t0:.0f}s]", flush=True)
    # Airy space
    def hf(z, axis):
        if not corners: return HK
        d = min(abs(z - (c[0] if axis == 0 else c[1])) for c, _ in corners)
        return min(HK, max(HK/8, d/2))
    kx = thin_graded(np.unique(np.r_[x0b, X, x1b]), lambda z: hf(z, 0))
    ky = thin_graded(np.unique(np.r_[y0b, Y, y1b]), lambda z: hf(z, 1))
    Bx0, Bx1, Bx2, nx = bspl_mats(px, kx); By0, By1, By2, ny = bspl_mats(py, ky)
    mk = lambda A, B: sp.csr_matrix((lambda r, c, v: (v, (r, c)))(*kr(A, B, ny)), shape=(Nq, nx*ny))
    Cyy = mk(Bx0, By2); Cxx = mk(Bx2, By0); Cxy = mk(Bx1, By1)
    A1 = (0.5*(Cyy - Cxx)).tocsr(); A2 = (-Cxy).tocsr(); del Cyy, Cxx, Cxy
    A1.eliminate_zeros(); A2.eliminate_zeros()
    cn = np.asarray(abs(A1).power(2).T @ Wq).ravel() + np.asarray(abs(A2).power(2).T @ Wq).ravel()
    used = np.where(cn > 1e-14*cn.max())[0]
    A1, A2 = A1[:, used].tocsr(), A2[:, used].tocsr()
    Wd = sp.diags(Wq)
    K = (A1.T @ Wd @ A1 + A2.T @ Wd @ A2).tocsc()
    # gauge: 4 interior points, unisolvent for span{1, x, y, x^2+y^2}
    rng = np.random.default_rng(0)
    for _ in range(200):
        gi = rng.choice(Nq, 4, replace=False)
        Mg = np.c_[np.ones(4), px[gi], py[gi], px[gi]**2 + py[gi]**2]
        if abs(np.linalg.det(Mg)) > 1e-3*Lb_(dom)**4: break
    gx0 = bspl_mats(px[gi], kx)[0]; gy0 = bspl_mats(py[gi], ky)[0]
    r_, c_, v_ = kr(gx0, gy0, ny)
    Gp = sp.csr_matrix((v_, (r_, c_)), shape=(4, nx*ny))[:, used]
    # ghost regularisation: small penalty on the full Hessian of chi over the whole bounding box
    # (removes the local kernels of cut spline elements; any chi gives a valid bound, so this is safe)
    kxc, kyc = 0.5*(kx[1:] + kx[:-1]), 0.5*(ky[1:] + ky[:-1]); hkx, hky = np.diff(kx), np.diff(ky)
    GX = (kxc[:, None] + 0.5*hkx[:, None]*G3[None, :]).ravel(); WX = (0.5*hkx[:, None]*W3[None, :]).ravel()
    GY = (kyc[:, None] + 0.5*hky[:, None]*G3[None, :]).ravel(); WY = (0.5*hky[:, None]*W3[None, :]).ravel()
    gxx_, gyy_ = np.meshgrid(GX, GY, indexing='ij'); wg = np.outer(WX, WY).ravel()
    bx0, bx1, bx2, _ = bspl_mats(gxx_.ravel(), kx); by0, by1, by2, _ = bspl_mats(gyy_.ravel(), ky)
    mk2 = lambda A, B: sp.csr_matrix((lambda r, c, v: (v, (r, c)))(*kr(A, B, ny)), shape=(len(wg), nx*ny))[:, used]
    Hs = [mk2(bx2, by0), mk2(bx1, by1), mk2(bx0, by2)]
    Rg = sum(Hk.T @ sp.diags(wg) @ Hk for Hk in Hs)
    K = (K + GHOST*np.mean(Wq/wq)*Rg).tocsc()
    Kg = (K + (Gp.T @ Gp)*np.median(K.diagonal())).tocsc()
    if DEBUG:
        import scipy.linalg as sla
        dg = 1/np.sqrt(K.diagonal()); Ks = (sp.diags(dg) @ K @ sp.diags(dg)).toarray()
        ev = sla.eigvalsh(Ks); print('[debug] smallest scaled eigenvalues of K:', ev[:8], ' largest', ev[-1])
        evg = sla.eigvalsh((sp.diags(dg) @ Kg @ sp.diags(dg)).toarray()); print('[debug] with gauge:', evg[:5])
    sj = 1.0/np.sqrt(Kg.diagonal())
    lu0 = splu((sp.diags(sj) @ Kg @ sp.diags(sj)).tocsc(), permc_spec='COLAMD')
    def solve(B):
        Xs = sj[:, None]*lu0.solve(sj[:, None]*B)
        for _ in range(3):
            Xs = Xs + sj[:, None]*lu0.solve(sj[:, None]*(B - Kg @ Xs))
        return Xs
    tb = rng.standard_normal((Kg.shape[0], 1)); res = np.linalg.norm(Kg @ solve(tb) - tb)/np.linalg.norm(tb)
    if verbose: print(f"[theta] Airy space {len(used)} functions, solve residual {res:.1e} [{time.time()-t0:.0f}s]", flush=True)
    Yc = (A1.T @ (Wd @ T1)).tocsc()
    D = (T1.T @ (Wd @ T1)).toarray()
    for c0 in range(0, NG, chunk):
        Yb = Yc[:, c0:c0 + chunk].toarray()
        D[:, c0:c0 + chunk] -= Yc.T @ solve(Yb)
    D = 0.5*(D + D.T)
    if NG > 9000:
        lamD = float(eigsh(D, k=1, which='LA', tol=1e-10, return_eigenvectors=False)[0])
    else:
        lamD = float(eigh(D, eigvals_only=True, subset_by_index=[NG - 1, NG - 1])[0])
    if verbose: print(f"[theta] lambda_max(D) = {lamD:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return lamD, dict(airy_functions=len(used), quad_points=Nq, airy_residual=res)

def Lb_(dom):
    x0, x1, y0, y1 = dom.bbox; return max(x1 - x0, y1 - y0)
