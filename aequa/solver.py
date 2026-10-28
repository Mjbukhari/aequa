"""
Gap-certified neural solver for steady generalised-Newtonian Navier-Stokes flow
with all-Dirichlet boundary data on a general 2D domain.

primal:  v = curl(psi_D + phi^2 N_psi + corner terms)       div v = 0, v = v_D on dOmega
dual:    m = dev( Airy(chi) + rho v (x) v ),  chi = polynomial + networks + Michell + corner terms
gap:     G = int [ F(gd) + F*(|m|/sqrt2) - m:eps(v) ] dx   >= 0   (Fenchel-Young)

Certificate note (v0.3), what this module guarantees and where:
  Lemma 3.1   admissibility for every value of the weights: the corner exponents are
              parametrised as lambda = lambda_min + delta + softplus(.), beta = beta_min + delta +
              softplus(.), with the law-dependent floors of Lemma 3.1, so v in W^{1,p}, Airy(chi) in
              L^{p'} and G < infinity hold by construction
  Lemma 4.2   vertex data are checked (discontinuous data rejected when p >= 2)
  Sec. 5      G is evaluated by the deterministic rule of quadrature.py, on two resolutions,
              with tail estimates at every vertex sector
"""
import math, time
import numpy as np
import torch
import torch.nn as nn

DT = torch.float64
torch.set_default_dtype(DT)


def d1(f, X):
    return torch.autograd.grad(f.sum(), X, create_graph=True)[0]


class MLP(nn.Module):
    def __init__(self, width=64, depth=4, nin=2, gain=0.1):
        super().__init__()
        L, k = [], nin
        for _ in range(depth):
            L += [nn.Linear(k, width), nn.Tanh()]; k = width
        self.body = nn.Sequential(*L); self.head = nn.Linear(k, 1)
        nn.init.normal_(self.head.weight, std=gain/math.sqrt(k)); nn.init.zeros_(self.head.bias)
    def forward(self, z): return self.head(self.body(z)).squeeze(-1)


class PolyAiry(nn.Module):
    """global polynomial Airy function (degree <= deg), minus the kernel {1,x,y,x^2+y^2}"""
    def __init__(self, deg=5):
        super().__init__()
        self.pw = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i) if i + j >= 2 and (i, j) != (0, 2)]
        self.c = nn.Parameter(torch.zeros(len(self.pw), dtype=DT))
    def forward(self, xs, ys):
        return sum(self.c[k]*xs**i*ys**j for k, (i, j) in enumerate(self.pw))


def stokes_corner_exponent(omega):
    """1 + z, z the smallest positive root (z != 1) of sin(z omega) = +- z sin(omega): the stream function
    of Stokes flow in a wedge of opening omega with no-slip walls behaves like r^(1+z) (Dean-Montagnon,
    Moffatt).  Used only to initialise the learnable exponent."""
    from scipy.optimize import brentq
    best = None
    for sgn in (1.0, -1.0):
        f = lambda z: math.sin(z*omega) + sgn*z*math.sin(omega)
        zs = np.linspace(1e-3, 4.0, 4001); fv = np.array([f(z) for z in zs])
        for i in np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]:
            z = brentq(f, zs[i], zs[i + 1])
            if abs(z - 1.0) > 1e-6 and (best is None or z < best): best = z
    return 1.0 + (best if best is not None else 1.0)


def _inv_softplus(y):
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


EXP_MARGIN = 0.02          # strict margin above the integrability floors of Lemma 3.1


def resolve_device(device=None):
    """None or 'auto': the first CUDA GPU if PyTorch sees one, else the CPU.
    'cuda', 'cuda:1', 'cpu' or a torch.device: that device (an error if CUDA is asked for but absent)."""
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = torch.device(device)
    if d.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but PyTorch sees no GPU (torch.cuda.is_available() "
                               "is False). Install a CUDA build of PyTorch, or on Colab choose Runtime > Change "
                               "runtime type > GPU; or pass device='cpu'.")
        if d.index is None: d = torch.device('cuda', torch.cuda.current_device())
    elif d.type != 'cpu':
        raise ValueError(f"unsupported device {device!r}: use 'auto', 'cpu' or 'cuda'")
    return d


def resolve_precision(precision, device):
    """dtype used for TRAINING. The certificate is always computed in float64.
    'auto': float32 on a GPU (consumer and T4/L4 GPUs run float64 about 30x slower), float64 on the CPU;
    'mixed': float32 training; 'double': float64 training."""
    if precision == 'auto':
        return torch.float32 if device.type == 'cuda' else torch.float64
    if precision == 'mixed': return torch.float32
    if precision == 'double': return torch.float64
    raise ValueError("precision must be 'auto', 'mixed' or 'double'")


class Solver(nn.Module):
    def __init__(self, domain, law, rho=0.0, width=64, depth=4, smooth=1e-3, seed=0, device='cpu', precision='auto'):
        """device: 'cpu' (default for this low-level class, so code that passes CPU tensors to fields()
        keeps working), 'cuda' or 'auto'. aequa.solve() uses 'auto'. precision: see resolve_precision."""
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed)
        if not domain.ready:
            raise ValueError("call domain.set_bc(...) first")
        self.dom, self.law, self.rho, self.smooth = domain, law, float(rho), smooth
        self.vertex_jumps = domain.check_vertex_data(law.p)          # Lemma 4.2
        x0, x1, y0, y1 = domain.bbox
        self.cx, self.cy = 0.5*(x0 + x1), 0.5*(y0 + y1)
        self.Ls = 0.5*max(x1 - x0, y1 - y0)
        # scales: velocity from boundary flux, stress from the law at the typical shear rate
        self.Us = max(domain.flux_scale, 1e-12)/(y1 - y0 if (y1 - y0) > 0 else 1.0)
        Ubs = []
        for pc in domain.pieces_all:
            lo, hi, _ = pc.prange(); p = np.linspace(lo, hi, 50)
            Ubs.append(np.abs(domain.bcs[pc.tag](pc.point(p))).max())
        self.Us = max(self.Us, max(Ubs), 1e-12)
        gtyp = torch.tensor([self.Us/self.Ls], dtype=DT)
        self.Ss = float(law.dF(gtyp)) + self.rho*self.Us**2 + 1e-12
        nin = 2 + 3*len(domain.holes)
        self.net_psi = MLP(width, depth, nin=nin); self.net_chi = MLP(width, depth, nin=nin); self.poly = PolyAiry(5)
        # local (multi-scale) corrections around each hole, scaled with the hole radius
        self.loc_psi = nn.ModuleList([MLP(width//2, depth, nin=5) for _ in domain.holes])
        self.loc_chi = nn.ModuleList([MLP(width//2, depth, nin=5) for _ in domain.holes])
        # multivalued Airy terms (Michell): net force (2) and torque (1) transmitted by each hole.
        # In a multiply connected domain these are NOT representable by a single-valued chi.
        self.michell = nn.Parameter(torch.zeros(3*len(domain.holes), dtype=DT))
        # automatic corner enrichment at every re-entrant corner: psi ~ r^lam g(theta), chi ~ r^beta h(theta),
        # with learnable exponents; the angular factor (1-t^2)^2 keeps the boundary data exact
        self.corner_vertices = [i for i, v in enumerate(domain.vertices) if v['straight'] and v['angle'] > math.pi + 1e-9]
        # vertices where the data jump (allowed for p < 2, Lemma 4.2): eps(v) ~ 1/r there, so the gap
        # density behaves like r^-p; they get the graded sector rule and the tail bound of Lemma 5.2
        self.jump_vertices = [i for i, v in enumerate(domain.vertices) if any(v is b for b in self.vertex_jumps)]
        if any(not domain.vertices[i]['straight'] for i in self.jump_vertices):
            raise ValueError("a velocity jump at a vertex between curved pieces is not supported (no sector rule)")
        self.singular_vertices = sorted(set(self.corner_vertices) | set(self.jump_vertices))
        self.corners = [(domain.vertices[i]['V'], domain.vertices[i]['angle'], domain.vertices[i]['e1'],
                         domain.vertices[i]['e2'], domain.vertices[i]['rho']) for i in self.corner_vertices]
        nc = len(self.corners)
        self.lam_floor = (law.lam_min_inertial if rho > 0 else law.lam_min) + EXP_MARGIN     # Lemmas 3.1, 3.2
        self.beta_floor = law.beta_min + EXP_MARGIN
        lam0, beta0 = [], []
        for (_, om, _, _, _) in self.corners:
            l = stokes_corner_exponent(om)                        # Newtonian value; learned afterwards
            b = 2.0 + (l - 2.0)*(law.p - 1.0)                     # stress ~ strain^(p-1)
            lam0.append(_inv_softplus(max(l - self.lam_floor, 0.05)))
            beta0.append(_inv_softplus(max(b - self.beta_floor, 0.05)))
        self.corner_lam = nn.Parameter(torch.tensor(lam0, dtype=DT).reshape(nc))
        self.corner_beta = nn.Parameter(torch.tensor(beta0, dtype=DT).reshape(nc))
        self.corner_psi = nn.ModuleList([MLP(16, 2, nin=1, gain=0.1) for _ in range(nc)])
        self.corner_chi = nn.ModuleList([MLP(16, 2, nin=1, gain=0.1) for _ in range(nc)])
        self.holes = nn.ParameterList([nn.Parameter(h.detach().clone()) for h in domain.hole_params])
        domain.hole_params = list(self.holes)          # the lift reads these trainable constants
        self.history = []
        self.skipped_steps = 0
        self.device = resolve_device(device)
        self.train_dtype = resolve_precision(precision, self.device)
        self.dtype = DT
        self._place(DT)

    # ---------------------------------------------------------------- device / precision
    def _place(self, dtype):
        """move every parameter to self.device in `dtype`. The geometry and rheology tables follow the
        input points automatically (geometry._like, rheology._qrule)."""
        if dtype != self.dtype or any(p.device != self.device for p in self.parameters()):
            self.to(device=self.device, dtype=dtype)
            self.dtype = dtype
        self.dom.hole_params = list(self.holes)        # keep the lift pointing at the live parameters
        return self

    def _points(self, P, requires_grad=False):
        X = torch.as_tensor(np.ascontiguousarray(P), dtype=self.dtype).to(self.device, non_blocking=True)
        return X.requires_grad_(True) if requires_grad else X

    def device_info(self):
        name = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'
        prec = 'float32' if self.train_dtype == torch.float32 else 'float64'
        return f"device: {self.device} ({name}), training in {prec}, certificate in float64"

    # ---------------------------------------------------------------- fields
    def _z(self, x, y):
        feats = [(x - self.cx)/self.Ls, (y - self.cy)/self.Ls]
        for h in self.dom.holes:          # local polar features: resolve boundary layers on obstacles
            dx, dy = x - h.c[0], y - h.c[1]; r = torch.sqrt(dx*dx + dy*dy + 1e-30)
            feats += [torch.log(r/h.R), dx/r, dy/r]
        return torch.stack(feats, -1)

    def fields(self, X, smooth=0.0, need_dual=True):
        x, y = X[:, 0], X[:, 1]
        z = self._z(x, y)
        corr = self.Us*self.Ls*self.net_psi(z)
        locs = self._local(x, y)
        for k, (zl, win, h) in enumerate(locs):
            corr = corr + self.Us/h.R*win*self.loc_psi[k](zl)
        psi = self.dom.psi_lift(x, y) + self.dom.phi(x, y)**2*corr
        cws = self._corner_terms(x, y)
        for k, (r, t, cut, rc) in enumerate(cws):
            lam = self.lam_floor + torch.nn.functional.softplus(self.corner_lam[k])
            psi = psi + self.Us*rc*cut*(r/rc)**lam*(1 - t*t)**2*self.corner_psi[k](t[:, None])
        g = d1(psi, X); gx = d1(g[:, 0], X); gy = d1(g[:, 1], X)
        v1, v2 = g[:, 1], -g[:, 0]
        a = gx[:, 1]                                  # eps11 = psi_xy
        b = 0.5*(gy[:, 1] - gx[:, 0])                 # eps12 = (psi_yy - psi_xx)/2
        out = dict(u=v1, v=v2, e11=a, e12=b, psi=psi)
        if smooth > 0:                                # training only; shear-rate unit U/L
            se = smooth*self.Us/self.Ls
            gd = 2*(torch.sqrt(a*a + b*b + se**2) - se)
        else:
            gd = 2*torch.sqrt(a*a + b*b)
        out['gdot'] = gd
        if not need_dual: return out
        chi = self.Ss*self.Ls**2*(self.poly(z[:, 0], z[:, 1]) + self.net_chi(z))
        for k, (zl, win, h) in enumerate(locs):
            chi = chi + self.Ss*h.R**2*win*self.loc_chi[k](zl)
            dx, dy = x - h.c[0], y - h.c[1]; r = torch.sqrt(dx*dx + dy*dy); th = torch.atan2(dy, dx)
            c = self.michell[3*k:3*k + 3]
            chi = chi + self.Ss*h.R*(c[0]*r*th*torch.sin(th) + c[1]*r*th*torch.cos(th)) + self.Ss*h.R**2*c[2]*th
        for k, (r, t, cut, rc) in enumerate(cws):
            beta = self.beta_floor + torch.nn.functional.softplus(self.corner_beta[k])
            chi = chi + self.Ss*rc**2*cut*(r/rc)**beta*self.corner_chi[k](t[:, None])
        c1 = d1(chi, X); cxx = d1(c1[:, 0], X); cyy = d1(c1[:, 1], X)
        m11 = 0.5*(cyy[:, 1] - cxx[:, 0]) + self.rho*0.5*(v1*v1 - v2*v2)
        m12 = -cxx[:, 1] + self.rho*v1*v2
        if smooth > 0:
            tm = torch.sqrt(m11*m11 + m12*m12 + (smooth*self.Ss)**2) - smooth*self.Ss
        else:
            tm = torch.sqrt(m11*m11 + m12*m12)
        gap = self.law.F(gd) + self.law.Fstar(tm) - 2*(a*m11 + b*m12)
        # pressure (up to a constant): p = -tr(Airy chi + rho v(x)v)/2
        out.update(m11=m11, m12=m12, taum=tm, gap=gap,
                   p=-0.5*(cxx[:, 0] + cyy[:, 1]) - 0.5*self.rho*(v1*v1 + v2*v2))
        return out

    def _corner_terms(self, x, y):
        out = []
        for (c, om, e1, e2, rc) in self.corners:
            dx, dy = x - c[0], y - c[1]
            r = torch.sqrt(dx*dx + dy*dy).clamp_min(1e-300)
            th = torch.remainder(torch.atan2(dx*e2[0] + dy*e2[1], dx*e1[0] + dy*e1[1]), 2*math.pi)
            t = (2*th/om - 1).clamp(-1.0, 1.0)
            s = (r/rc).clamp(0.0, 1.0)
            cut = 1 - s**4*(35 - 84*s + 70*s**2 - 20*s**3)            # C^3 cutoff, 1 near the corner, 0 for r >= rc
            out.append((r, t, cut, rc))
        return out

    def _local(self, x, y):
        out = []
        for h in self.dom.holes:
            dx, dy = x - h.c[0], y - h.c[1]; r = torch.sqrt(dx*dx + dy*dy + 1e-30)
            zl = torch.stack([dx/h.R, dy/h.R, torch.log(r/h.R), dx/r, dy/r], -1)
            win = 1.0/(1.0 + ((r - h.R)/(2*h.R))**4)
            out.append((zl, win, h))
        return out

    # ---------------------------------------------------------------- sampling
    def sample(self, n, gen=None):
        """mixture sampling: uniform over the domain + bands around holes.
        returns points and importance weights w_i with sum_i w_i f(x_i) ~ int f"""
        x0, x1, y0, y1 = self.dom.bbox
        A = self.dom.area()
        nh = len(self.dom.holes); alpha = 0.5 if nh else 1.0
        nu_ = int(round(alpha*n)); P = []
        while sum(len(p) for p in P) < nu_:
            Q = np.c_[np.random.uniform(x0, x1, 2*n), np.random.uniform(y0, y1, 2*n)]
            P.append(Q[self.dom.inside(Q[:, 0], Q[:, 1])])
        P = [np.concatenate(P)[:nu_]]
        bands = []
        for h in self.dom.holes:
            bw = 1.5*h.R; r_in = 0.5*h.R
            nb = (n - nu_)//nh
            u = np.random.uniform(0, 1, 4*nb); th = np.random.uniform(-np.pi, np.pi, 4*nb)
            r = np.sqrt(r_in**2 + u*((h.R + bw)**2 - r_in**2))
            Q = np.c_[h.c[0] + r*np.cos(th), h.c[1] + r*np.sin(th)]
            Q = Q[self.dom.inside(Q[:, 0], Q[:, 1])][:nb]
            P.append(Q)
            bands.append((h, bw, self._band_area(h, bw, r_in)))
        P = np.concatenate(P)
        # mixture density
        q = np.full(len(P), alpha/A)
        for (h, bw, Ab) in bands:
            rr = np.hypot(P[:, 0] - h.c[0], P[:, 1] - h.c[1]); inb = (rr <= h.R + bw) & (rr >= 0.5*h.R)
            q += inb*(1 - alpha)/nh/Ab
        w = 1.0/(q*len(P))
        return self._points(P), self._points(w)

    def _band_area(self, h, bw, r_in):
        """area of {r_in <= |x - c| <= R + bw} intersected with the fluid domain (the support of the band
        samples).  v0.3 used the area of the full ring minus the hole, which overweights the band when it
        leaves the domain (e.g. an annulus): the loss estimate was then biased.  Computed once, with
        shapely, to ~1e-5 relative."""
        if not hasattr(self, '_band_areas'): self._band_areas = {}
        key = id(h)
        if key not in self._band_areas:
            from shapely.geometry import Point
            ring = Point(*h.c).buffer(h.R + bw, quad_segs=512).difference(Point(*h.c).buffer(r_in, quad_segs=512))
            self._band_areas[key] = float(self.dom.shapely_polygon().intersection(ring).area)
        return self._band_areas[key]

    # ---------------------------------------------------------------- training
    def train(self, iters=3000, n=2048, lr=2e-3, verbose=200, lbfgs=0, lbfgs_points=None):
        """Adam on fresh samples, then (optionally) L-BFGS on one fixed sample set of lbfgs_points
        points (default 8n). Runs on self.device in self.train_dtype; the parameters are returned to
        float64 at the end, so everything after training (certificate, evaluation) is float64.
        A step whose loss or gradient is not finite is skipped (counted in self.skipped_steps)."""
        self._place(self.train_dtype)
        params = list(self.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr*0.02)
        t0 = time.time()
        for it in range(iters):
            X, w = self.sample(n); X.requires_grad_(True)
            f = self.fields(X, smooth=self.smooth)
            loss = (w*f['gap']).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
            lv, gv = torch.stack([loss.detach(), gnorm.detach().to(loss.dtype)]).tolist()   # one device sync
            if math.isfinite(lv) and math.isfinite(gv):
                opt.step(); self.history.append(lv)
            else:
                self.skipped_steps += 1
            sched.step()
            if verbose and (it % verbose == 0 or it == iters - 1):
                print(f"  it {it:5d}  gap(MC) {lv:.4e}   [{time.time()-t0:.0f}s]", flush=True)
        if lbfgs:
            X, wL = self.sample(lbfgs_points or 8*n); X.requires_grad_(True)
            opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, history_size=50,
                                     line_search_fn='strong_wolfe')
            def closure():
                opt2.zero_grad(); l = (wL*self.fields(X, smooth=self.smooth)['gap']).sum(); l.backward(); return l
            opt2.step(closure)
            if verbose:
                print(f"  L-BFGS: gap(MC, fixed set) {float(closure().detach()):.4e}   [{time.time()-t0:.0f}s]", flush=True)
            del X, wL
        if self.skipped_steps and verbose:
            print(f"  note: {self.skipped_steps} Adam step(s) skipped (non-finite loss or gradient)", flush=True)
        self._place(DT)
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return self

    # ---------------------------------------------------------------- certificate
    def exponents(self):
        """learned corner exponents (lambda_k, beta_k), all above the floors of Lemma 3.1"""
        sp = torch.nn.functional.softplus
        lam = (self.lam_floor + sp(self.corner_lam.double())).detach().cpu().numpy()
        beta = (self.beta_floor + sp(self.corner_beta.double())).detach().cpu().numpy()
        return lam, beta

    def kappa_theory(self):
        """vertex index -> singular exponent kappa of the gap density near a singular vertex:
        enriched re-entrant corner: kappa = -min((lam-2)p, (beta-2)p', lam+beta-4) < 2 (Lemma 3.1);
        vertex with a data jump: kappa = max(p, 1) < 2 (Lemma 4.2)"""
        lam, beta = self.exponents(); p, pc = self.law.p, self.law.pconj
        out = {}
        for k, iv in enumerate(self.corner_vertices):
            ks = [-(lam[k] - 2)*p, -(lam[k] + beta[k] - 4)]
            if math.isfinite(pc): ks.append(-(beta[k] - 2)*pc)
            out[iv] = max(0.0, max(ks))
        for iv in self.jump_vertices:      # F(gd) ~ r^-p and m:eps ~ r^-1 (m bounded there)
            out[iv] = max(out.get(iv, 0.0), self.law.p, 1.0)
        return out

    def rule(self, h=None, n_geo=24):
        from .quadrature import build_rule
        key = ('rule', h, n_geo)
        if not hasattr(self, '_rules'): self._rules = {}
        if key not in self._rules:
            self._rules[key] = build_rule(self.dom, h=h, n_geo=n_geo, singular_vertices=tuple(self.singular_vertices))
        return self._rules[key]

    def quadrature(self, h=None):
        """points and positive weights of the deterministic rule (quadrature.py)"""
        R = self.rule(h); return R.P, R.W

    def evaluate(self, P, keys=('gap',), chunk=None):
        """fields at points P (numpy, (N, 2)), always in float64, on self.device in chunks;
        returns numpy arrays"""
        self._place(DT)
        if chunk is None: chunk = 8192 if self.device.type == 'cuda' else 4096
        P = np.asarray(P, float)
        out = {k: [] for k in keys}
        need_dual = any(k in ('gap', 'm11', 'm12', 'taum', 'p') for k in keys)
        for i in range(0, len(P), chunk):
            X = self._points(P[i:i + chunk], requires_grad=True)
            f = self.fields(X, smooth=0.0, need_dual=need_dual)
            for k in keys: out[k].append(f[k].detach().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def certify_gap(self, h=None):
        """G on two resolutions of the deterministic rule (Sec. 5 of the certificate note).
        Returned upper value:  G_up = G_fine + |G_fine - G_coarse| + sum of sector tails + curve term,
        and the dissipation J(v) = int F(gd(v)) needed for the energy enclosure."""
        x0, x1, y0, y1 = self.dom.bbox
        if h is None: h = min(x1 - x0, y1 - y0)/40
        kth = self.kappa_theory()
        res = {}
        for hh, ng, tag in ((h, 24, 'fine'), (2*h, 12, 'coarse')):
            R = self.rule(hh, ng)
            f = self.evaluate(R.P, keys=('gap', 'gdot'))
            gap = f['gap']
            if not np.all(np.isfinite(gap)):
                raise ValueError("the gap density is +inf somewhere (the network stress exceeds the "
                                 "saturation stress of the law): G = +inf, no certificate")
            Ig = R.integrate(gap, kth)
            Fv = self.law.F(torch.tensor(f['gdot'])).numpy()
            Id = R.integrate(Fv, kth)
            if not hasattr(self, '_gd_cache'): self._gd_cache = {}
            self._gd_cache[tag] = (R, f['gdot'])
            res[tag] = dict(G=Ig['value'], tails=Ig['tails'], tail_sum=Ig['tail_sum'],
                            min_gap=float(gap.min()), max_gap=float(gap.max()), diss=Id['value'] + Id['tail_sum'],
                            n_points=len(R.W), geom_err=R.geom_err)
        fine, coarse = res['fine'], res['coarse']
        # a negative pointwise gap can only come from rounding; it is reported, never subtracted
        neg = max(0.0, -fine['min_gap'])
        Rf = self.rule(h)
        curve = Rf.geom_err*fine['max_gap'] if Rf.geom_err > 0 else 0.0
        res['G_up'] = fine['G'] + abs(fine['G'] - coarse['G']) + fine['tail_sum'] + curve
        res['quad_diff'] = abs(fine['G'] - coarse['G'])
        res['curve_term'] = curve
        res['data_mismatch'] = self.dom.data_mismatch
        res['rounding_neg'] = neg
        return res

    def power_bounds(self, gapinfo, hs=None):
        """Certified enclosure of the power W* = int S*:eps(u*) (rho = 0; certificate note, Thm 6.4):
            W* is a subgradient at t = 1 of the convex value function J*(t) (data t u_b), hence
            (J*(1) - J*(1-h))/h <= W* <= (J*(1+h) - J*(1))/h, with J*(t) <= int F(t gd(v)) and
            J*(1) >= J(v) - G; and p_lo J* <= W* <= p_hi J* (Lemma 6.3)."""
        if self.rho != 0:
            raise ValueError("the power enclosure is proved for rho = 0 only")
        kth = self.kappa_theory(); law = self.law
        def Jt(t):                       # upper estimate of int F(t gd(v)) from the two rules
            vals = []
            for tag in ('fine', 'coarse'):
                R, gd = self._gd_cache[tag]
                I = R.integrate(law.F(torch.tensor(t*gd)).numpy(), kth)
                vals.append(I['value'] + I['tail_sum'])
            return vals[0] + abs(vals[0] - vals[1])
        dfi, dco = gapinfo['fine']['diss'], gapinfo['coarse']['diss']
        J_up = dfi + abs(dfi - dco); J_lo = dfi - abs(dfi - dco)
        D1 = J_lo - gapinfo['G_up']                      # J*(1) >= D1
        p_lo, p_hi = law.p_range
        lo, hi = p_lo*max(D1, 0.0), p_hi*J_up
        hs = np.geomspace(1e-3, 0.5, 40) if hs is None else hs
        best_lo = best_hi = None
        for h in hs:
            L = (D1 - Jt(1 - h))/h; U = (Jt(1 + h) - D1)/h
            if L > lo: lo, best_lo = L, h
            if U < hi: hi, best_hi = U, h
        return dict(W_lo=lo, W_hi=hi, h_lo=best_lo, h_hi=best_hi, J_lo=D1, J_up=J_up, p_range=(p_lo, p_hi),
                    Q_in=self.dom.inflow_flux(), moving_walls=self.dom.has_moving_walls())

def strain_with_grad(S, P, chunk=3000):
    """eps11, eps12 of the network velocity and their spatial gradients at points P"""
    S._place(DT)
    out = []
    for i in range(0, len(P), chunk):
        X = S._points(P[i:i + chunk], requires_grad=True)
        f = S.fields(X, smooth=0.0, need_dual=False)
        a, b = f['e11'], f['e12']
        ga = torch.autograd.grad(a.sum(), X, retain_graph=True)[0]
        gb = torch.autograd.grad(b.sum(), X)[0]
        out.append(torch.stack([a, b, ga[:, 0], ga[:, 1], gb[:, 0], gb[:, 1]], 1).detach().cpu().numpy())
    return np.concatenate(out)
