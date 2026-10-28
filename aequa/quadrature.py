"""
Deterministic quadrature for the certificate integrals (certificate note, Sec. 5).

The domain is split into three kinds of regions, each integrated with exact geometry:

  vertex sectors   at every vertex between two straight pieces, the polygonal sector
                   {V + t rho(theta) (cos theta e1 + sin theta e2)}, whose outer boundary is a chord
                   polygon; Gauss rules in theta (one panel per chord) and in t, graded
                   geometrically towards the vertex down to t_min; the part t < t_min is a tail
                   (bounded separately, Lemma 5.2)
  circle bands     next to every circular piece, the region between the circle and a chord polygon
                   at a slightly larger (or smaller) radius, in polar coordinates, graded towards
                   the wall
  remainder        the rest of the polygonal domain, cut by a background grid of square cells:
                   full cells get a tensor Gauss rule, cut cells are clipped exactly (shapely) and
                   triangulated, each triangle getting a positive 7-point rule of degree 5

Straight pieces and circles are represented exactly.  Other curved pieces are polylines with
4000 points per piece; the area of the symmetric difference is reported (Remark 5.3).
All weights are positive, so the rule integrates a nonnegative density to a nonnegative number.
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, box as sbox

GL = {n: np.polynomial.legendre.leggauss(n) for n in range(1, 13)}

# Dunavant degree-5 rule on the reference triangle (barycentric coordinates, weights sum to 1)
_TRI_B = np.array([[1/3, 1/3, 1/3],
                   [0.059715871789770, 0.470142064105115, 0.470142064105115],
                   [0.470142064105115, 0.059715871789770, 0.470142064105115],
                   [0.470142064105115, 0.470142064105115, 0.059715871789770],
                   [0.797426985353087, 0.101286507323456, 0.101286507323456],
                   [0.101286507323456, 0.797426985353087, 0.101286507323456],
                   [0.101286507323456, 0.101286507323456, 0.797426985353087]])
_TRI_W = np.array([0.225] + [0.132394152788506]*3 + [0.125939180544827]*3)


def _gauss_1d(edges, n):
    x, w = GL[n]
    a, b = edges[:-1], edges[1:]
    P = (0.5*(b - a)[:, None]*(x[None, :] + 1) + a[:, None]).ravel()
    W = (0.5*(b - a)[:, None]*w[None, :]).ravel()
    return P, W


def _graded(t_min, n_panels):
    """edges t_min = t_0 < ... < t_n = 1, geometric"""
    return np.geomspace(t_min, 1.0, n_panels + 1)


class Region:
    def __init__(self, kind, P, W, info):
        self.kind, self.P, self.W, self.info = kind, P, W, info


def _mixed_edges(t_min, h_rel, n_geo):
    """edges on [t_min, 1]: n_geo geometric panels up to t_g = min(0.1, h_rel), then uniform panels of
    width <= h_rel (the integrand is smooth away from the vertex)"""
    t_g = min(0.1, max(h_rel, 10*t_min))
    geo = np.geomspace(t_min, t_g, n_geo + 1)
    nu = max(1, int(math.ceil((1.0 - t_g)/h_rel)))
    return np.r_[geo, np.linspace(t_g, 1.0, nu + 1)[1:]]


def _sector(v, h, nG, t_min, n_geo):
    V, e1, e2, om, rho = v['V'], v['e1'], v['e2'], v['angle'], v['rho']
    nch = max(4, int(math.ceil(rho*om/h)), int(math.ceil(8*om/(math.pi/2))))
    th = np.linspace(0.0, om, nch + 1)
    poly_pts = [V] + [V + rho*(math.cos(t)*e1 + math.sin(t)*e2) for t in th]
    poly = Polygon(poly_pts)
    T, WT = _gauss_1d(th, nG)
    tE = _mixed_edges(t_min, h/rho, n_geo)
    S, WS = _gauss_1d(tE, nG)
    d = th[1] - th[0]
    jpan = np.minimum((T // d).astype(int), nch - 1)
    tmid = 0.5*(th[jpan] + th[jpan + 1])
    rmax = rho*math.cos(d/2)/np.cos(T - tmid)
    TT, SS = np.meshgrid(T, S, indexing='ij')
    RM = np.broadcast_to(rmax[:, None], TT.shape)
    R = SS*RM
    P = V[None, :] + R.reshape(-1, 1)*(np.cos(TT).reshape(-1, 1)*e1[None, :] + np.sin(TT).reshape(-1, 1)*e2[None, :])
    W = (np.outer(WT, WS)*SS*RM**2).ravel()
    # innermost two radial panels, for the tail estimate (Lemma 5.2)
    ip = np.repeat(np.arange(len(tE) - 1), nG)          # panel index of each radial node
    IP = np.broadcast_to(ip[None, :], TT.shape).ravel()
    return Region('sector', P, W, dict(vertex=v, poly=poly, panel=IP, q=tE[1]/tE[0], t_min=t_min,
                                       area_tail=0.5*om*(rho*t_min)**2)), poly


def _circle_band(pc, band, h, nG, n_geo, inward):
    """region between the circle (radius R) and a chord polygon at radius R +- band"""
    c, R = pc.c, pc.R
    Ro = R - band if inward else R + band
    n_chord = max(64, int(math.ceil(2*math.pi*max(R, Ro)/h)))
    th = np.linspace(-math.pi, math.pi, n_chord + 1)
    d = th[1] - th[0]
    chord = Polygon([c + Ro*np.array([math.cos(t), math.sin(t)]) for t in th[:-1]])
    T, WT = _gauss_1d(th, nG)
    jpan = np.minimum(((T + math.pi)//d).astype(int), n_chord - 1)
    tmid = -math.pi + (jpan + 0.5)*d
    rch = Ro*math.cos(d/2)/np.cos(T - tmid)
    # wall-graded panels (boundary layers) down to 1e-6, then uniform; the integrand is smooth up to the wall
    S, WS = _gauss_1d(np.r_[0.0, _mixed_edges(1e-6, h/band, n_geo)], nG)
    TT, SS = np.meshgrid(T, S, indexing='ij')
    RC = np.broadcast_to(rch[:, None], TT.shape)
    if inward:
        RR = R - SS*(R - RC); jac = RR*(R - RC)
    else:
        RR = R + SS*(RC - R); jac = RR*(RC - R)
    P = c[None, :] + np.c_[(RR*np.cos(TT)).ravel(), (RR*np.sin(TT)).ravel()]
    W = (np.outer(WT, WS)*jac).ravel()
    return Region('band', P, W, dict(piece=pc, band=band)), chord


def build_rule(dom, h=None, nG=5, n_geo=24, t_min_sing=1e-8, t_min_reg=1e-6, singular_vertices=(),
               breaklines=()):
    """Gauss points and positive weights for integrals over the fluid domain.
    singular_vertices: indices into dom.vertices that carry corner enrichment (graded to t_min_sing)."""
    from .geometry import Circle
    x0, x1, y0, y1 = dom.bbox
    if h is None: h = min(x1 - x0, y1 - y0)/40
    Om = dom.shapely_polygon()
    regions, removed, occupied = [], [], []
    # vertex sectors
    for iv, v in enumerate(dom.vertices):
        if not v['straight']: continue
        tm = t_min_sing if iv in singular_vertices else t_min_reg
        reg, poly = _sector(v, h, nG, tm, n_geo)
        reg.info['index'] = iv; reg.info['singular'] = iv in singular_vertices
        regions.append(reg); removed.append(poly); occupied.append(poly)
    # circle bands
    hole_polys = [Polygon(dom._loop_points(L)) for L in dom.hole_loops]
    outer_full = Polygon(dom._loop_points(dom.outer_loop))
    for pc in dom.pieces_all:
        if not isinstance(pc, Circle): continue
        inward = not pc.out
        band = 0.5*pc.R if not inward else 0.3*pc.R
        own = [hp for hp, L in zip(hole_polys, dom.hole_loops) if pc in L]
        others = [hp for hp, L in zip(hole_polys, dom.hole_loops) if pc not in L]
        for _ in range(60):
            reg, chord = _circle_band(pc, band, h, nG, n_geo, inward)
            if not inward:
                # the chord polygon (which contains the hole) must lie in the domain with this hole filled
                filled = outer_full.difference(shapely.union_all(others)) if others else outer_full
                ring = chord
                ok = filled.buffer(1e-12).contains(chord)
            else:
                # fluid inside the circle: the ring outside the chord polygon must contain no hole
                ring = outer_full.difference(chord)
                ok = chord.contains(shapely.union_all(hole_polys)) if hole_polys else True
            ok = ok and all(not ring.intersects(r) for r in occupied)
            if ok: break
            band *= 0.8
        else:
            raise RuntimeError(f"no admissible band around circle '{pc.tag}'")
        regions.append(reg)
        if inward:   # fluid inside the circle: the remainder is the part of the domain inside the chord polygon
            Om = Om.intersection(chord); occupied.append(ring)
        else:
            removed.append(chord); occupied.append(chord)
    rem = Om
    for r in removed:
        if r is not None: rem = rem.difference(r)
    rem = rem.buffer(0)
    # optional breaklines (curves across which the integrand is known to have a kink): the remainder
    # is split along them, so no quadrature cell straddles a kink
    parts = [rem]
    if breaklines:
        from shapely.ops import split
        from shapely.geometry import LineString
        for ln in breaklines:
            ln = LineString(ln); new = []
            for pg in parts:
                new.extend(shapely.get_parts(split(pg, ln)) if pg.intersects(ln) else [pg])
            parts = new
    # background cells
    nx, ny = int(math.ceil((x1 - x0)/h)), int(math.ceil((y1 - y0)/h))
    xs = np.linspace(x0, x0 + nx*h, nx + 1); ys = np.linspace(y0, y0 + ny*h, ny + 1)
    XI, YI = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij'); XI, YI = XI.ravel(), YI.ravel()
    cells = shapely.box(xs[XI], ys[YI], xs[XI + 1], ys[YI + 1])
    gx, gw = GL[nG]
    Pf, Wf, Pc, Wc = [], [], [], []
    n_full = n_cut = 0
    for part in parts:
        shapely.prepare(part)
        full = shapely.contains_properly(part, cells)
        touch = shapely.intersects(part, cells) & ~full
        n_full += int(full.sum()); n_cut += int(touch.sum())
        # full cells: tensor Gauss
        cx0, cy0 = xs[XI[full]], ys[YI[full]]
        PX = (cx0[:, None, None] + 0.5*h*(gx[None, :, None] + 1)) + 0*gx[None, None, :]
        PY = (cy0[:, None, None] + 0.5*h*(gx[None, None, :] + 1)) + 0*gx[None, :, None]
        WW = np.broadcast_to(0.25*h*h*gw[:, None]*gw[None, :], PX.shape)
        Pf.append(np.c_[PX.ravel(), PY.ravel()]); Wf.append(WW.ravel().copy())
        # cut cells: exact clipping + triangles
        for cell in cells[touch]:
            inter = part.intersection(cell)
            if inter.is_empty or inter.area <= 0: continue
            tris = shapely.constrained_delaunay_triangles(inter)
            for t in shapely.get_parts(tris):
                xy = np.asarray(t.exterior.coords)[:3]
                ar = 0.5*abs((xy[1, 0] - xy[0, 0])*(xy[2, 1] - xy[0, 1]) - (xy[2, 0] - xy[0, 0])*(xy[1, 1] - xy[0, 1]))
                if ar <= 0: continue
                Pc.append(_TRI_B @ xy); Wc.append(ar*_TRI_W)
    Pf = np.concatenate(Pf); Wf = np.concatenate(Wf)
    Pc = np.concatenate(Pc) if Pc else np.zeros((0, 2)); Wc = np.concatenate(Wc) if Wc else np.zeros(0)
    regions.append(Region('cells', np.r_[Pf, Pc], np.r_[Wf, Wc], dict(n_full=n_full, n_cut=n_cut)))
    # geometric representation error of non-circular curved pieces (Remark 5.3)
    geom_err = 0.0
    for pc in dom.pieces_all:
        if pc.kind == 'curve':
            lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 4001)
            pts = pc.point(p); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mid = pc.point(0.5*(p[1:] + p[:-1])); chordmid = 0.5*(pts[1:] + pts[:-1])
            sag = np.linalg.norm(mid - chordmid, axis=1)
            geom_err += float((seg*sag).sum())
    return Rule(regions, geom_err, rem.area)


def _circle_pts(pc, n):
    th = np.linspace(-math.pi, math.pi, n + 1)[:-1]
    return pc.c[None, :] + pc.R*np.c_[np.cos(th), np.sin(th)]


class Rule:
    """a quadrature rule made of regions; integrate() returns the integral, a two-sided tail estimate
    for every vertex sector, and region-wise contributions."""
    def __init__(self, regions, geom_err, rem_area):
        self.regions, self.geom_err, self.rem_area = regions, geom_err, rem_area
        self.P = np.concatenate([r.P for r in regions])
        self.W = np.concatenate([r.W for r in regions])
        self.slices, i = [], 0
        for r in regions:
            self.slices.append(slice(i, i + len(r.W))); i += len(r.W)

    def area(self):
        """total area of the rule (sectors including their tails)"""
        return float(self.W.sum()) + sum(r.info['area_tail'] for r in self.regions if r.kind == 'sector')

    def integrate(self, f, kappa_theory=None):
        """f: values at self.P (nonnegative for a gap density).  Returns dict with the integral, the
        tail estimates of all sectors (Lemma 5.2) and their sum.  kappa_theory: optional map
        vertex index -> worst singular exponent kappa < 2 (|f| <~ r^-kappa) from the theory."""
        f = np.asarray(f, float)
        total = float((f*self.W).sum())
        tails = []
        for r, sl in zip(self.regions, self.slices):
            if r.kind != 'sector': continue
            fw = f[sl]*r.W
            ip = r.info['panel']; q = r.info['q']
            I1 = float(fw[ip == 0].sum()); I2 = float(fw[ip == 1].sum())
            kth = 0.0 if kappa_theory is None else kappa_theory.get(r.info['index'], 0.0)
            # if |f| ~ r^-k, the panel integrals scale like q^(2-k); the tail below t_min is I1/(q^(2-k)-1)
            if I1 > 0 and I2 > 0:
                kfit = 2.0 - math.log(I2/I1)/math.log(q)
            else:
                kfit = 0.0
            k = min(max(kfit, kth), 1.999)
            tail = abs(I1)/(q**(2.0 - k) - 1.0) if I1 != 0 else 0.0
            tails.append(dict(vertex=r.info['index'], I1=I1, kfit=kfit, ktheory=kth, tail=tail))
        return dict(value=total, tails=tails, tail_sum=sum(t['tail'] for t in tails))
