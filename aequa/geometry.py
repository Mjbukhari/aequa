"""
Geometry engine (v2): 2D domains bounded by closed loops of pieces (straight segments,
circles, arcs, graphs y = h(x), and arbitrary smooth closed or open curves).  One outer
loop, any number of holes (obstacles) of any shape.

Velocity v = curl(psi), psi = psi_D + phi^2 N:  div v = 0 exactly and v equals the
prescribed boundary velocity exactly on every piece, for any network N.
psi_D is a transfinite Hermite lift built from normalised distance functions
(Sukumar & Srivastava, CMAME 389 (2022) 114333); each hole carries one learnable constant.
"""
import math
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from matplotlib.path import Path

DT = torch.float64
J = lambda v: np.stack([-v[..., 1], v[..., 0]], -1)          # rotation by +90 degrees


# ----------------------------------------------------------------- 1D splines (torch)
class TSpline:
    """C^2 cubic spline in torch, with derivatives."""
    def __init__(self, p, vals, periodic=False):
        cs = CubicSpline(p, vals, bc_type='periodic' if periodic else 'not-a-knot')
        self.x = torch.tensor(cs.x, dtype=DT)
        self.c = torch.tensor(cs.c, dtype=DT)       # (4, n-1)
        self.periodic = periodic
        self.lo, self.hi = float(cs.x[0]), float(cs.x[-1])
    def wrap(self, p):
        return self.lo + torch.remainder(p - self.lo, self.hi - self.lo) if self.periodic else p.clamp(self.lo, self.hi)
    def __call__(self, p, k=0):
        p = self.wrap(p)
        i = torch.searchsorted(self.x, p.detach().contiguous(), right=True) - 1
        i = i.clamp(0, self.c.shape[1] - 1)
        d = p - self.x[i]; c = self.c[:, i]
        if k == 0: return ((c[0]*d + c[1])*d + c[2])*d + c[3]
        if k == 1: return (3*c[0]*d + 2*c[1])*d + c[2]
        if k == 2: return 6*c[0]*d + 2*c[1]
        raise ValueError


# ----------------------------------------------------------------- boundary pieces
class Segment:
    """straight piece a -> b; side = +1: fluid on the left of a->b, -1: on the right"""
    kind = 'segment'
    def __init__(self, a, b, tag, side=1):
        self.a, self.b, self.tag, self.side = np.asarray(a, float), np.asarray(b, float), tag, side
        self.L = float(np.linalg.norm(self.b - self.a))
        self.tau = (self.b - self.a)/self.L
        self.nu = side*J(self.tau)                        # unit normal pointing into the fluid
    def point(self, p): return self.a[None, :] + p[:, None]*(self.b - self.a)[None, :]
    def dxdp(self, p): return np.broadcast_to(self.b - self.a, (len(p), 2))
    def inward_normal(self, p): return np.broadcast_to(self.nu, (len(p), 2))
    def prange(self): return (0.0, 1.0, False)
    def dist(self, x, y):
        ax, ay = self.a; bx, by = self.b; L = self.L
        f = ((x - ax)*(by - ay) - (y - ay)*(bx - ax))/L
        cx, cy = 0.5*(ax + bx), 0.5*(ay + by)
        t = ((L/2)**2 - ((x - cx)**2 + (y - cy)**2))/L
        vp = torch.sqrt(t*t + f**4)
        return torch.sqrt(f*f + ((vp - t)/2)**2 + 1e-300)
    def param(self, x, y):
        ax, ay = self.a
        return (((x - ax)*(self.b[0] - ax) + (y - ay)*(self.b[1] - ay))/self.L**2).clamp(0.0, 1.0)
    def sample(self, n=200): return self.point(np.linspace(0, 1, n))


class Circle:
    """full circle; fluid outside (hole) or inside (outer boundary, side=+1)"""
    kind = 'circle'
    def __init__(self, center, radius, tag, fluid_outside=True):
        self.c, self.R, self.tag = np.asarray(center, float), float(radius), tag
        self.out = fluid_outside
    def point(self, p): return self.c[None, :] + self.R*np.c_[np.cos(p), np.sin(p)]
    def dxdp(self, p): return self.R*np.c_[-np.sin(p), np.cos(p)]
    def inward_normal(self, p):
        n = np.c_[np.cos(p), np.sin(p)]; return n if self.out else -n
    def prange(self): return (-math.pi, math.pi, True)
    def dist(self, x, y):
        r = torch.sqrt((x - self.c[0])**2 + (y - self.c[1])**2 + 1e-300)
        return (r - self.R).abs() + 1e-300
    def param(self, x, y): return torch.atan2(y - self.c[1], x - self.c[0])
    def sample(self, n=400): return self.point(np.linspace(-math.pi, math.pi, n))


class Curve:
    """smooth parametric curve C(p), p in [p0, p1] (periodic if closed), represented by a
    C^2 spline through n samples of the user's function.  Distance by differentiable
    Newton projection.  side = +1: fluid on the left of increasing p."""
    kind = 'curve'
    def __init__(self, fn, p0, p1, tag, side=1, closed=False, n=3001):
        p = np.linspace(p0, p1, n)
        P = np.asarray(fn(p), float).reshape(2, -1).T
        if closed: P[-1] = P[0]
        self.X, self.Y = TSpline(p, P[:, 0], closed), TSpline(p, P[:, 1], closed)
        self.p0, self.p1, self.tag, self.side, self.closed = p0, p1, tag, side, closed
        self.tab_p = torch.tensor(np.linspace(p0, p1, 801), dtype=DT)
        with torch.no_grad():
            self.tab = torch.stack([self.X(self.tab_p), self.Y(self.tab_p)], -1)
    def _np(self, p, k=0):
        pt = torch.tensor(np.asarray(p, float), dtype=DT)
        with torch.no_grad():
            return np.c_[self.X(pt, k).numpy(), self.Y(pt, k).numpy()]
    def point(self, p): return self._np(p)
    def dxdp(self, p): return self._np(p, 1)
    def inward_normal(self, p):
        t = self._np(p, 1); t = t/np.linalg.norm(t, axis=1, keepdims=True); return self.side*J(t)
    def prange(self): return (self.p0, self.p1, self.closed)
    def param(self, x, y, iters=5):
        with torch.no_grad():
            P = torch.stack([x.detach(), y.detach()], -1)
            idx = torch.cdist(P, self.tab).argmin(1)
            p = self.tab_p[idx]
        for _ in range(iters):            # Newton on (C(p) - x) . C'(p) = 0, differentiable in x
            cx, cy = self.X(p), self.Y(p); dx, dy = self.X(p, 1), self.Y(p, 1); ex, ey = self.X(p, 2), self.Y(p, 2)
            f = (cx - x)*dx + (cy - y)*dy
            fp = dx*dx + dy*dy + (cx - x)*ex + (cy - y)*ey
            p = p - f/torch.where(fp.abs() > 1e-12, fp, torch.full_like(fp, 1e-12))
            p = self.X.wrap(p)
        return p
    def dist(self, x, y):
        p = self.param(x, y)
        return torch.sqrt((x - self.X(p))**2 + (y - self.Y(p))**2 + 1e-300)
    def sample(self, n=400): return self.point(np.linspace(self.p0, self.p1, n))


# ----------------------------------------------------------------- boundary conditions
class BC:
    """velocity boundary condition u_b(x, y) -> (ux, uy) (numpy)"""
    def __init__(self, fn, name=''):
        self.fn, self.name = fn, name
    def __call__(self, P): return np.asarray(self.fn(P[:, 0], P[:, 1]), float).reshape(2, -1).T

def NoSlip(): return BC(lambda x, y: (0*x, 0*y), 'no-slip')
def MovingWall(ux, uy=0.0): return BC(lambda x, y: (0*x + ux, 0*y + uy), 'moving wall')
def Rotating(center, omega):
    """rigid rotation about center (counter-clockwise > 0).  Steady only for bodies of
    revolution (circles, annuli); for other shapes it is a frozen (quasi-steady) snapshot."""
    cx, cy = center
    return BC(lambda x, y: (-omega*(y - cy), omega*(x - cx)), f'rotating omega={omega}')
def Profile(fn_speed, direction):
    """prescribed speed fn(x, y) along 'direction' (unit vector)"""
    dx, dy = direction
    return BC(lambda x, y: (fn_speed(x, y)*dx, fn_speed(x, y)*dy), 'profile')
def Velocity(fn):
    """general prescribed velocity fn(x, y) -> (ux, uy)"""
    return BC(fn, 'velocity')


# ----------------------------------------------------------------- loops and shapes
def polygon_loop(vertices, tags, hole=False):
    """closed polygon; vertices counter-clockwise.  tags: one per edge (or a single tag)."""
    V = np.asarray(vertices, float); n = len(V)
    if isinstance(tags, str): tags = [tags]*n
    area2 = np.sum(V[:, 0]*np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1)*V[:, 1])
    if area2 < 0:                                   # make counter-clockwise, keep tags on their edges
        tags = [list(tags)[(n - 2 - j) % n] for j in range(n)]; V = V[::-1]
    side = -1 if hole else 1
    return [Segment(V[k], V[(k + 1) % n], tags[k], side) for k in range(n)]

def circle(center, radius, tag='obstacle'):
    return [Circle(center, radius, tag)]

def regular_polygon(center, radius, n, tag='obstacle', angle=0.0):
    th = angle + 2*np.pi*np.arange(n)/n
    V = np.c_[center[0] + radius*np.cos(th), center[1] + radius*np.sin(th)]
    return polygon_loop(V, tag, hole=True)

def square(center, side, tag='obstacle', angle=0.0):
    return regular_polygon(center, side/math.sqrt(2), 4, tag, angle + math.pi/4)

def rectangle(center, width, height, tag='obstacle', angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    V = [(-width/2, -height/2), (width/2, -height/2), (width/2, height/2), (-width/2, height/2)]
    V = [(center[0] + c*x - s*y, center[1] + s*x + c*y) for x, y in V]
    return polygon_loop(V, tag, hole=True)

def triangle(center, side, tag='obstacle', angle=0.0):
    return regular_polygon(center, side/math.sqrt(3), 3, tag, angle + math.pi/2)

def ellipse(center, a, b, tag='obstacle', angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    fn = lambda p: (center[0] + c*a*np.cos(p) - s*b*np.sin(p), center[1] + s*a*np.cos(p) + c*b*np.sin(p))
    return [Curve(fn, -math.pi, math.pi, tag, side=-1, closed=True)]

def closed_curve(fn, tag='obstacle', hole=True, p0=0.0, p1=2*math.pi):
    """any smooth closed curve fn(p) -> (x, y), counter-clockwise in p"""
    return [Curve(fn, p0, p1, tag, side=-1 if hole else 1, closed=True)]

def graph(h, x0, x1, tag, fluid='below'):
    """wall y = h(x), x from x0 to x1 (use for curved channel walls)"""
    fn = lambda p: (p, h(p))
    return Curve(fn, x0, x1, tag, side=1 if fluid == 'above' else -1)


# ----------------------------------------------------------------- the domain
class Domain:
    """
    outer : list of pieces forming the outer loop, head to tail, fluid on the side
            indicated by each piece (counter-clockwise loops: side=+1), or
            (vertices, tags) for a polygon
    holes : list of loops (lists of pieces), or legacy tuples (center, radius, tag)
    """
    def __init__(self, outer, tags=None, holes=()):
        if tags is not None:
            outer = polygon_loop(outer, tags)
        self.outer_loop = list(outer)
        self.hole_loops = []
        for h in holes:
            if isinstance(h, tuple) and len(h) == 3 and np.ndim(h[1]) == 0:
                self.hole_loops.append(circle(*h))
            else:
                self.hole_loops.append(list(h))
        self.pieces = self.outer_loop
        self.pieces_all = self.outer_loop + [pc for L in self.hole_loops for pc in L]
        # polygonal approximations for membership tests
        self._outer_path = Path(self._loop_points(self.outer_loop))
        self._hole_paths = [Path(self._loop_points(L)) for L in self.hole_loops]
        pts = self._loop_points(self.outer_loop)
        lo, hi = pts.min(0), pts.max(0)
        self.bbox = (lo[0], hi[0], lo[1], hi[1])
        # hole descriptors: centre, circumradius, is-circle (used for features and quadrature)
        self.holes = []
        for L in self.hole_loops:
            P = self._loop_points(L)
            A = P[:-1, 0]*P[1:, 1] - P[1:, 0]*P[:-1, 1]
            area = 0.5*A.sum()
            cx = ((P[:-1, 0] + P[1:, 0])*A).sum()/(6*area); cy = ((P[:-1, 1] + P[1:, 1])*A).sum()/(6*area)
            c = np.array([cx, cy]); R = float(np.max(np.linalg.norm(P - c, axis=1)))
            self.holes.append(HoleInfo(c, R, L, len(L) == 1 and L[0].kind == 'circle', abs(area)))
        self.bcs = {}; self.ready = False
        # largest distance between a curved piece and its polyline (used to keep line-stress ends
        # outside the fluid in theta.py, and reported by the quadrature)
        self.max_sag = 0.0
        for pc in self.pieces_all:
            if pc.kind == 'segment': continue
            lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 4001)
            pts = pc.point(p); mid = pc.point(0.5*(p[1:] + p[:-1]))
            self.max_sag = max(self.max_sag, float(np.linalg.norm(mid - 0.5*(pts[1:] + pts[:-1]), axis=1).max()))

    @staticmethod
    def _loop_points(loop, n=4000):
        """closed polyline through the loop: segments are represented exactly by their start
        point; curved pieces by n points (sagitta <= kappa (L/n)^2/8)"""
        P = []
        for pc in loop:
            if pc.kind == 'segment':
                P.append(pc.a[None, :])
            else:
                lo, hi, per = pc.prange()
                pts = pc.point(np.linspace(lo, hi, n + 1))
                P.append(pts[:-1])
        P = np.concatenate(P)
        return np.r_[P, P[:1]]

    def inside(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        Q = np.c_[x.ravel(), y.ravel()]
        m = self._outer_path.contains_points(Q)
        for hp in self._hole_paths:
            m &= ~hp.contains_points(Q)
        return m.reshape(x.shape)

    def area(self):
        P = self._loop_points(self.outer_loop)
        A = 0.5*abs(np.sum(P[:-1, 0]*P[1:, 1] - P[1:, 0]*P[:-1, 1]))
        return A - sum(h.area for h in self.holes)

    # --- boundary conditions ---
    def set_bc(self, bcs):
        self.bcs = bcs
        for pc in self.pieces_all:
            if pc.tag not in bcs:
                raise ValueError(f"no boundary condition for boundary '{pc.tag}'")
        self._build_lift()
        self.ready = True

    def _piece_tables(self, pc, gstart, n=8001):
        """boundary data of the lift on one piece, as functions of the piece parameter:
        g = psi (integral of the normal flux, cumulative Simpson: error ~ h^4), a = d psi/d nu"""
        from scipy.integrate import cumulative_simpson
        lo, hi, per = pc.prange()
        p = np.linspace(lo, hi, n)
        ub = self.bcs[pc.tag](pc.point(p))
        dgdp = -np.einsum('ij,ij->i', ub, J(pc.dxdp(p)))               # d psi/dp = -u . J(dx/dp)
        g = gstart + cumulative_simpson(dgdp, x=p, initial=0.0)
        a = -np.einsum('ij,ij->i', ub, J(pc.inward_normal(p)))         # d psi/d nu = -u . J(nu)
        return p, g, a, per, dgdp

    def _build_lift(self):
        """Transfinite Hermite lift (Prop. 21.1 of the inertial notes) with three additions
        (certificate note, Sec. 4):
          * loop closure: a residual net flux below tolerance is removed by a uniform normal
            velocity (reported as a data change), so psi_D is single valued on every loop;
          * vertex data: the jump of the velocity data at every vertex is recorded (Lemma 4.2);
          * the data actually imposed (spline of g', spline of a) is compared with the user data,
            and the maximal mismatch is recorded (Remark 4.4)."""
        self.g, self.a, self.hole_params, self.piece_hole = [], [], [], []
        loops = [self.outer_loop] + self.hole_loops
        self.data_change = 0.0
        tabs = []
        for li, L in enumerate(loops):
            gcur = 0.0; rows = []
            for pc in L:
                p, g, a, per, dgdp = self._piece_tables(pc, gcur)
                rows.append([pc, p, g, a, per])
                if not per: gcur = g[-1]
            closed_single = len(L) == 1 and L[0].prange()[2]
            cl = (rows[0][2][-1] - rows[0][2][0]) if closed_single else gcur
            scale = max(1e-12, max(np.abs(r[2]).max() for r in rows),
                        max(np.abs(r[3]).max() for r in rows))
            if abs(cl) > 1e-6*max(1.0, scale):
                where = 'outer boundary' if li == 0 else f'hole {li}'
                raise ValueError(f"net flux through the {where} is {cl:.3e}, not zero: "
                                 "inflow and outflow must balance (incompressible flow)")
            # remove the residual flux by a uniform normal velocity -cl/perimeter
            lens = []
            for pc, p, g, a, per in rows:
                d = np.linalg.norm(np.diff(pc.point(p), axis=0), axis=1); lens.append(np.r_[0, np.cumsum(d)])
            perim = sum(l[-1] for l in lens); s0 = 0.0
            for r, l in zip(rows, lens):
                r[2] = r[2] - cl*(s0 + l)/perim; s0 += l[-1]
            self.data_change = max(self.data_change, abs(cl)/max(perim, 1e-300))
            for pc, p, g, a, per in rows:
                if per: g[-1] = g[0]; a[-1] = a[0]
                self.g.append(TSpline(p, g, per)); self.a.append(TSpline(p, a, per))
                self.piece_hole.append(li - 1)            # -1: outer loop
            if li > 0:
                self.hole_params.append(torch.zeros((), dtype=DT, requires_grad=True))
        self.flux_scale = max(1e-12, max(float(t.c[3].abs().max()) for t in self.g))
        self._analyse_vertices()
        self._data_mismatch()

    # --- vertices -------------------------------------------------------------------------
    def _analyse_vertices(self):
        """every junction V of two consecutive pieces A -> B of a loop: fluid angle, jump of the
        velocity data, whether both adjacent pieces carry zero velocity (no-slip), and a radius
        rho_V such that the disc B(V, rho_V) meets no other piece (used by the quadrature, the
        corner enrichment and the no-slip blending)."""
        self.vertices = []
        k0 = 0
        for li, L in enumerate([self.outer_loop] + self.hole_loops):
            n = len(L)
            for k in range(n):
                A, B = L[k], L[(k + 1) % n]
                if n == 1 and A.prange()[2]:
                    continue                                   # a single closed curve has no vertex
                loA, hiA, _ = A.prange(); loB, hiB, _ = B.prange()
                V = A.point(np.array([hiA]))[0]
                if np.linalg.norm(V - B.point(np.array([loB]))[0]) > 1e-9*max(1.0, np.abs(V).max()):
                    raise ValueError(f"loop {li}: piece '{A.tag}' does not end where '{B.tag}' starts")
                tA = A.dxdp(np.array([hiA]))[0]; tA = tA/np.linalg.norm(tA)
                tB = B.dxdp(np.array([loB]))[0]; tB = tB/np.linalg.norm(tB)
                turn = math.atan2(tA[0]*tB[1] - tA[1]*tB[0], tA @ tB)
                side = A.side if hasattr(A, 'side') else (1 if getattr(A, 'out', True) is False else -1)
                ang = math.pi - turn if side > 0 else math.pi + turn          # fluid opening angle
                uA = self.bcs[A.tag](V[None, :])[0]; uB = self.bcs[B.tag](V[None, :])[0]
                jump = float(np.linalg.norm(uA - uB))
                def zero_near(pc, end):
                    lo, hi, _ = pc.prange()
                    q = np.linspace(hi - 0.25*(hi - lo), hi, 50) if end else np.linspace(lo, lo + 0.25*(hi - lo), 50)
                    return float(np.abs(self.bcs[pc.tag](pc.point(q))).max()) == 0.0
                rigid = A.kind == 'segment' and B.kind == 'segment' and zero_near(A, True) and zero_near(B, False)
                others = [pc for pc in self.pieces_all if pc is not A and pc is not B]
                dmin = min((float(np.min(np.linalg.norm(pc.sample(2000) - V[None, :], axis=1))) for pc in others),
                           default=1.0)
                lenA = A.L if A.kind == 'segment' else 1e300
                lenB = B.L if B.kind == 'segment' else 1e300
                rho = 0.45*min(dmin, lenA, lenB)
                # frame: e1 along the outgoing piece, e2 into the fluid
                e1 = tB; e2 = (B.side if hasattr(B, 'side') else 1)*J(e1)
                self.vertices.append(dict(V=V, loop=li, A=A, B=B, angle=ang, jump=jump, rigid=rigid,
                                          rho=rho, e1=e1, e2=e2, straight=(A.kind == 'segment' and B.kind == 'segment'),
                                          gA=k0 + k))
            k0 += n
        # constant psi values at rigid vertices (index of the spline of piece A, evaluated at its end)
        for v in self.vertices:
            if v['rigid']:
                spl = self.g[v['gA']]
                v['psi'] = float(spl(torch.tensor([spl.hi], dtype=DT))[0])
                v['blend'] = 0.5*v['rho']

    def _data_mismatch(self, m=997):
        """max |v_D - u_b| on the boundary: the BVP that is certified is the one with the imposed
        data v_D|dOmega (Remark 4.4); here the two are compared at off-node points"""
        worst = 0.0
        for k, pc in enumerate(self.pieces_all):
            lo, hi, per = pc.prange(); p = np.linspace(lo, hi, m)[1:-1]
            pt = torch.tensor(p, dtype=DT)
            with torch.no_grad():
                dg = self.g[k](pt, 1).numpy(); a = self.a[k](pt).numpy()
            dxdp = pc.dxdp(p); sp = np.linalg.norm(dxdp, axis=1); tau = dxdp/sp[:, None]
            nu = pc.inward_normal(p)
            ub = self.bcs[pc.tag](pc.point(p))
            # psi_tau = dg/ds = -u.J(tau) = u.nu_left ... compare both components through psi derivatives
            un_imp = dg/sp; un_dat = -np.einsum('ij,ij->i', ub, J(tau))
            ut_imp = a;     ut_dat = -np.einsum('ij,ij->i', ub, J(nu))
            worst = max(worst, float(np.abs(un_imp - un_dat).max()), float(np.abs(ut_imp - ut_dat).max()))
        self.data_mismatch = max(worst, self.data_change)

    def check_vertex_data(self, p_growth):
        """Lemma 4.2: a jump of the velocity data at a vertex makes eps(v_D) ~ 1/r there; the
        problem then has finite dissipation only if F grows like gd^p with p < 2."""
        bad = [v for v in self.vertices if v['jump'] > 1e-9*max(1.0, self.flux_scale)]
        if bad and p_growth >= 2:
            s = ", ".join(f"({v['V'][0]:.4g}, {v['V'][1]:.4g}) jump {v['jump']:.3g}" for v in bad)
            raise ValueError("discontinuous boundary velocity at vertices " + s + ": for a law with "
                             "quadratic (or faster) growth the dissipation is infinite (certificate note, "
                             "Lemma 4.2). Use a continuous (regularised) profile.")
        return bad

    def _boundary_samples(self, n=2001):
        for pc in self.pieces_all:
            lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, n)
            B = pc.point(p); ub = np.asarray(self.bcs[pc.tag](B), float)
            if ub.shape[0] == 2 and ub.shape[-1] != 2: ub = ub.T
            ds = np.linalg.norm(np.gradient(B, p, axis=0), axis=1)
            w = np.full(n, 2.0); w[1::2] = 4.0; w[0] = w[-1] = 1.0; w *= (p[1] - p[0])/3
            yield pc, ub, pc.inward_normal(p), ds*w

    def inflow_flux(self):
        """Q_in = int max(u_b . n_in, 0) ds over the boundary (Simpson, 2001 points per piece)"""
        return float(sum(((np.maximum((ub*nin).sum(1), 0.0))*w).sum() for _, ub, nin, w in self._boundary_samples()))

    def has_moving_walls(self, tol=1e-9):
        """True if some piece carries a tangential velocity (rotating or sliding wall)"""
        for _, ub, nin, w in self._boundary_samples(201):
            un = (ub*nin).sum(1); ut = ub - un[:, None]*nin
            if np.abs(ut).max() > tol*max(1.0, self.flux_scale) and np.abs(un).max() <= tol*max(1.0, self.flux_scale):
                return True
        return False

    # --- distance functions (torch) ---
    def phi(self, x, y, m=2.0):
        s = 0.0
        for pc in self.pieces_all:
            s = s + pc.dist(x, y).clamp_min(1e-12)**(-m)
        return s**(-1.0/m)

    def psi_lift(self, x, y, mu=2.0):
        ds = [pc.dist(x, y).clamp_min(1e-12) for pc in self.pieces_all]
        ws = [d**(-mu) for d in ds]; W = sum(ws)
        out = 0.0
        for k, (pc, d, w) in enumerate(zip(self.pieces_all, ds, ws)):
            p = pc.param(x, y)
            val = self.g[k](p) + d*self.a[k](p)
            h = self.piece_hole[k]
            if h >= 0: val = val + self.hole_params[h]
            out = out + (w/W)*val
        # no-slip vertices: psi_D := c + (1 - cut)(psi_D - c), exact boundary data are kept and
        # psi_D is exactly constant near the vertex (no cancellation error, Remark 4.3)
        for v in getattr(self, 'vertices', []):
            if not v.get('rigid'): continue
            c = v['psi'] + (self.hole_params[v['loop'] - 1] if v['loop'] > 0 else 0.0)
            r = torch.sqrt((x - v['V'][0])**2 + (y - v['V'][1])**2)
            s = (2*r/v['blend'] - 1).clamp(0.0, 1.0)          # cut = 1 for r <= blend/2, 0 for r >= blend
            cut = 1 - s**4*(35 - 84*s + 70*s**2 - 20*s**3)     # C^3
            out = c + (1 - cut)*(out - c)
        return out

    def shapely_polygon(self):
        """the fluid domain as a shapely polygon: exact for straight pieces, curved pieces by
        polylines with 4000 points per piece (vertices on the curve)"""
        from shapely.geometry import Polygon
        outer = self._loop_points(self.outer_loop)
        holes = [self._loop_points(L) for L in self.hole_loops]
        return Polygon(outer, holes).buffer(0)

    def describe(self):
        s = [f"Domain: outer loop of {len(self.outer_loop)} piece(s), {len(self.hole_loops)} obstacle(s), area {self.area():.4g}"]
        for pc in self.pieces_all:
            s.append(f"  - {pc.kind:7s} tag='{pc.tag}'  BC: {self.bcs[pc.tag].name if pc.tag in self.bcs else '??'}")
        return "\n".join(s)


class HoleInfo:
    def __init__(self, c, R, loop, is_circle, area):
        self.c, self.R, self.loop, self.is_circle, self.area = c, R, loop, is_circle, area


def reentrant_corners(dom, with_frames=False):
    """vertices between consecutive straight pieces where the fluid angle exceeds pi.
    with_frames: also return (e1, e2, omega, r_c): e1 along the outgoing wall, e2 rotated into
    the fluid, so the fluid sector is {0 < theta < omega} in the frame (e1, e2)"""
    out = []
    for L in [dom.outer_loop] + dom.hole_loops:
        n = len(L)
        for k in range(n):
            A, B = L[k], L[(k + 1) % n]
            if A.kind != 'segment' or B.kind != 'segment' or n < 2: continue
            dA, dB = A.b - A.a, B.b - B.a
            turn = math.atan2(dA[0]*dB[1] - dA[1]*dB[0], dA @ dB)
            ang = math.pi - turn if A.side > 0 else math.pi + turn
            if ang > math.pi + 1e-9:
                if not with_frames:
                    out.append((A.b.copy(), ang)); continue
                e1 = dB/np.linalg.norm(dB); e2 = B.side*J(e1)
                S = [pc.sample(400) for pc in dom.pieces_all if pc is not A and pc is not B]
                dmin = min(np.min(np.linalg.norm(s - A.b[None, :], axis=1)) for s in S) if S else 1.0
                rc = 0.45*min(dmin, A.L, B.L)
                out.append((A.b.copy(), ang, e1, e2, rc))
    return out


# ----------------------------------------------------------------- standard geometries
def channel(length, height, x0=0.0, y0=None, inlet='inlet', outlet='outlet', wall='wall', obstacles=()):
    """rectangular channel [x0, x0+length] x [y0, y0+height]; obstacles: loops or (c, R, tag)"""
    if y0 is None: y0 = -height/2
    V = [(x0, y0), (x0 + length, y0), (x0 + length, y0 + height), (x0, y0 + height)]
    return Domain(V, [wall, outlet, wall, inlet], obstacles)

def contraction(L_up, H_up, L_dn, H_dn, obstacles=(), inlet='inlet', outlet='outlet', wall='wall'):
    """planar sudden contraction (H_up -> H_dn at x = 0), symmetric about y = 0"""
    V = [(-L_up, -H_up), (0, -H_up), (0, -H_dn), (L_dn, -H_dn), (L_dn, H_dn), (0, H_dn), (0, H_up), (-L_up, H_up)]
    return Domain(V, [wall, wall, wall, outlet, wall, wall, wall, inlet], obstacles)

def expansion(L_up, H_up, L_dn, H_dn, obstacles=(), inlet='inlet', outlet='outlet', wall='wall'):
    """planar sudden expansion (H_up -> H_dn > H_up at x = 0)"""
    V = [(-L_up, -H_up), (0, -H_up), (0, -H_dn), (L_dn, -H_dn), (L_dn, H_dn), (0, H_dn), (0, H_up), (-L_up, H_up)]
    return Domain(V, [wall, wall, wall, outlet, wall, wall, wall, inlet], obstacles)

def backward_step(L_up, h_in, L_dn, H, obstacles=(), inlet='inlet', outlet='outlet', wall='wall'):
    """backward-facing step: inlet channel [H-h_in, H] for x<0, full height [0, H] for x>0"""
    V = [(-L_up, H - h_in), (0, H - h_in), (0, 0), (L_dn, 0), (L_dn, H), (-L_up, H)]
    return Domain(V, [wall, wall, wall, outlet, wall, inlet], obstacles)

def t_junction(L_in, W_in, L_branch, W_branch, obstacles=(), inlet='inlet', out1='outlet1', out2='outlet2', wall='wall'):
    """T-junction: inlet channel from the bottom (width W_in, length L_in) joining a
    horizontal branch of width W_branch extending L_branch to each side"""
    w, B = W_in/2, W_branch
    V = [(-w, -L_in), (w, -L_in), (w, 0), (L_branch, 0), (L_branch, B), (-L_branch, B), (-L_branch, 0), (-w, 0)]
    return Domain(V, [inlet, wall, wall, out1, wall, out2, wall, wall], obstacles)

def cross_slot(L, W, obstacles=(), in1='inlet1', in2='inlet2', out1='outlet1', out2='outlet2', wall='wall'):
    """cross-slot: inlets left/right, outlets top/bottom, arms of width W and length L"""
    w = W/2
    V = [(-w, -L), (w, -L), (w, -w), (L, -w), (L, w), (w, w), (w, L), (-w, L), (-w, w), (-L, w), (-L, -w), (-w, -w)]
    return Domain(V, [out2, wall, wall, in2, wall, wall, out1, wall, wall, in1, wall, wall], obstacles)

def cavity(width, height, obstacles=(), lid='lid', wall='wall'):
    """lid-driven cavity [0, width] x [0, height], lid on top"""
    V = [(0, 0), (width, 0), (width, height), (0, height)]
    return Domain(V, [wall, wall, lid, wall], obstacles)

def wavy_channel(length, h_top, h_bot, x0=0.0, obstacles=(), inlet='inlet', outlet='outlet', wall_top='wall', wall_bot='wall'):
    """channel between curved walls y = h_bot(x) and y = h_top(x) (peristaltic, corrugated,
    converging-diverging); inlet at x0 and outlet at x0+length are straight vertical lines"""
    x1 = x0 + length
    bot = Curve(lambda p: (p, h_bot(p)), x0, x1, wall_bot, side=1)
    right = Segment((x1, float(h_bot(np.array([x1]))[0])), (x1, float(h_top(np.array([x1]))[0])), outlet, 1)
    top = Curve(lambda p: (x0 + x1 - p, h_top(x0 + x1 - p)), x0, x1, wall_top, side=1)
    left = Segment((x0, float(h_top(np.array([x0]))[0])), (x0, float(h_bot(np.array([x0]))[0])), inlet, 1)
    return Domain([bot, right, top, left], None, obstacles)

def annulus(R_out, R_in, center=(0.0, 0.0), inner_center=None, outer='outer', inner='inner', obstacles=()):
    """(eccentric) annulus: outer circle R_out (fluid inside), inner cylinder R_in"""
    ci = center if inner_center is None else inner_center
    outer_loop = [Circle(center, R_out, outer, fluid_outside=False)]
    return Domain(outer_loop, None, [circle(ci, R_in, inner)] + list(obstacles))

def poiseuille_profile(y0, y1, mean_speed):
    H = y1 - y0
    return lambda x, y: 6*mean_speed*(y - y0)*(y1 - y)/H**2

def powerlaw_profile(y0, y1, mean_speed, n, coord='y'):
    """developed power-law profile across [y0, y1] (coordinate 'y' or 'x') with given mean speed"""
    h = 0.5*(y1 - y0); yc = 0.5*(y0 + y1); m = 1.0 + 1.0/n
    umax = mean_speed*(m + 1)/m
    if coord == 'y':
        return lambda x, y: umax*(1 - np.clip(np.abs(y - yc)/h, 0, 1)**m)
    return lambda x, y: umax*(1 - np.clip(np.abs(x - yc)/h, 0, 1)**m)
