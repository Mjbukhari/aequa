"""
Convex generalised-Newtonian laws for the Aequa certificate.

Every law is written through a scalar potential F of the shear rate gd = sqrt(2)|eps|:
    Phi(eps) = F(gd),   S = dPhi(eps) = F'(gd) 2 eps/gd,   Phi*(m) = F*(|m|/sqrt(2)).
F' is the flow curve tau(gd); it is nondecreasing for every law here, so Phi is convex and
the Fenchel-Young density  F(gd) + F*(t) - m:eps  is >= 0 pointwise (certificate note, Sec. 2).

What each law provides, and the result of the certificate note it serves
-----------------------------------------------------------------------
F, dF, d2F           potential, flow curve, tangent viscosity
Fstar                convex conjugate, evaluated as a guaranteed UPPER bound (Lemma 2.3), so that a
                     computed gap can only over-estimate the true gap
p, pconj             growth exponents of F and F* at infinity (Lemma 3.1: corner-exponent floors)
tau_sup              sup of the flow curve (finite only for saturating laws; then F* = +inf above it)
floor                nu_* = inf_g min(F''(g), F'(g)/g): unconditional Bregman floor (Lemma 2.4)
shear_thinning       True when the flow curve is concave (Assumption ST of the inertial notes);
                     then the sharp tube weight of Thm 3.4 is nu0 = F''(gmax)
nu0(gmax, gd_v)      weight of the energy norm actually used by the certificate
"""
import math
import numpy as np
import torch

DT = torch.float64
INF = float('inf')

# composite geometric Gauss-Legendre rule on (0, 1]: panels [2^-(k+1), 2^-k], k = 0..NPAN-1
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)
_NPAN = 60
_nodes, _weights = [], []
for _k in range(_NPAN):
    _a, _b = 2.0**-(_k + 1), 2.0**-_k
    _nodes.append(0.5*(_b - _a)*(_GL_X + 1) + _a); _weights.append(0.5*(_b - _a)*_GL_W)
_QN = torch.tensor(np.concatenate(_nodes), dtype=DT)
_QW = torch.tensor(np.concatenate(_weights), dtype=DT)
# the omitted first panel (0, 2^-60] contributes at most F'(g 2^-60) g 2^-60 (F' nondecreasing):
# relative size < 1e-18 for every law here.


class _FByQuadrature(torch.autograd.Function):
    """F(g) = int_0^g F'(s) ds by the composite geometric rule above (error ~ 1e-15 relative,
    the integrand is analytic on every panel); the backward pass uses the exact derivative F'(g)."""
    @staticmethod
    def forward(ctx, g, law):
        with torch.no_grad():
            gg = g.reshape(-1, 1)
            val = (law.dF(gg*_QN[None, :])*_QW[None, :]).sum(-1)*g.reshape(-1)
            val = val.reshape(g.shape)
        ctx.save_for_backward(g); ctx.law = law
        return val
    @staticmethod
    def backward(ctx, gout):
        g, = ctx.saved_tensors
        return gout*ctx.law.dF(g), None


class _ImplicitConjUpper(torch.autograd.Function):
    """Guaranteed upper bound for F*(t) = sup_g (t g - F(g)).

    A bracket g_lo <= g* <= g_hi with F'(g_lo) <= t <= F'(g_hi) is found by doubling and then
    bisection.  By convexity of F,
        F*(t) = t g* - F(g*) <= t g_lo - F(g_lo) + (t - F'(g_lo)) (g_hi - g_lo)      (Lemma 2.3)
    and the right side is what is returned; after the bisection its excess over F*(t) is below
    1e-20 relative.  t >= tau_sup gives +inf.  The derivative of F* is g* (taken as the midpoint)."""
    @staticmethod
    def forward(ctx, t, law):
        with torch.no_grad():
            t0 = t.clamp_min(0.0)
            out = torch.zeros_like(t0)
            base = law.dF(torch.zeros_like(t0))          # F'(0) = yield stress (0 for most laws)
            act = t0 > base                             # F*(t) = 0 when t <= F'(0)
            sat = act & (t0 >= law.tau_sup)
            act = act & ~sat
            hi = torch.ones_like(t0)
            for _ in range(2100):                       # doubling: F'(hi) >= t
                need = act & (law.dF(hi) < t0)
                if not bool(need.any()): break
                hi = torch.where(need, 2*hi, hi)
            lo = 0.5*hi
            for _ in range(2100):                       # halving: F'(lo) <= t
                need = act & (law.dF(lo) > t0) & (lo > 1e-300)
                if not bool(need.any()): break
                lo = torch.where(need, 0.5*lo, lo)
            lo = torch.where(act & (law.dF(lo) > t0), torch.zeros_like(lo), lo)
            for _ in range(110):
                mid = torch.where(lo > 0, torch.sqrt(lo*hi), 0.5*(lo + hi))
                up = law.dF(mid) > t0
                hi = torch.where(act & up, mid, hi); lo = torch.where(act & ~up, mid, lo)
            Flo = law.F(lo)
            val = t0*lo - Flo + (t0 - law.dF(lo)).clamp_min(0.0)*(hi - lo)
            out = torch.where(act, val, out)
            out = torch.where(sat, torch.full_like(out, INF), out)
            g = torch.where(act, 0.5*(lo + hi), torch.zeros_like(lo))
        ctx.save_for_backward(g)
        return out
    @staticmethod
    def backward(ctx, gout):
        g, = ctx.saved_tensors
        return gout*g, None


def _cq_lower():
    """c_q >= 1/24 for every q in [0, 1] (inertial notes, Lemma 3.8)."""
    return 1.0/24.0


class Law:
    name = 'law'
    yield_stress = 0.0
    p = 2.0                  # F(g) <= c (1 + g^p)
    tau_sup = INF
    floor = 0.0
    shear_thinning = True

    @property
    def pconj(self):
        return INF if self.p <= 1 else self.p/(self.p - 1.0)

    # corner-exponent floors (certificate note, Lemma 3.1)
    @property
    def lam_min(self):
        return max(1.0, 2.0 - 2.0/self.p)
    @property
    def lam_min_inertial(self):
        # with inertia the proofs also need grad v in L^q, q >= t* = p*/(p* - 2), p* = 2p/(2 - p)
        # (Lemma 3.2); for 1 < p < 4/3 this is the stronger floor lambda > 4/p - 2
        if self.p < 4.0/3.0 and self.p > 1.0:
            return max(self.lam_min, 4.0/self.p - 2.0)
        return self.lam_min
    @property
    def beta_min(self):
        if math.isfinite(self.tau_sup):
            return 2.0                      # saturating flow curve: the stress must stay bounded
        return 2.0 - 2.0/self.pconj

    @property
    def p_range(self):
        """(p_lo, p_hi) with p_lo <= g F'(g)/F(g) <= p_hi for all g > 0 (certificate note, Lemma 6.3);
        default: F convex with F(0) = 0 gives p_lo = 1; a concave flow curve gives p_hi = 2"""
        pr = getattr(self, '_pr', None)
        return pr if pr is not None else (1.0, 2.0 if self.shear_thinning else self.p)
    def F(self, g): raise NotImplementedError
    def dF(self, g): raise NotImplementedError
    def d2F(self, g):
        g = g.detach().clone().requires_grad_(True)
        return torch.autograd.grad(self.dF(g).sum(), g)[0]
    def Fstar(self, t):
        return _ImplicitConjUpper.apply(t, self)
    def tangent_min(self, g):
        """min(F'', F'/g): half the smallest eigenvalue of the Hessian of Phi at shear rate g (Lemma 3.3)"""
        g = g.clamp_min(1e-300)
        return torch.minimum(self.d2F(g), self.dF(g)/g)
    def nu0(self, gmax, gd_v=None):
        """weight nu0 with  D_Phi(eps(v), eps(u); S) >= nu0 |eps(v) - eps(u)|^2.
        shear-thinning laws: Thm 3.4 (sharp), valid where |eps(u) - eps(v)| <= tube, gmax = sqrt2(|eps v| + tube);
        other laws: an unconditional bound (floor, or Lemma 3.8 for power-law thickening)."""
        gmax = torch.as_tensor(gmax, dtype=DT)
        if self.shear_thinning:
            return torch.maximum(self.d2F(gmax.clamp_min(1e-300)), torch.full_like(gmax, self.floor))
        return torch.full_like(gmax, self.floor)
    def nu0_is_conditional(self):
        return self.shear_thinning and self.floor == 0.0
    def __repr__(self): return self.name


class Newtonian(Law):
    def __init__(self, mu):
        self.mu = mu; self.name = f'Newtonian(mu={mu})'; self.p = 2.0; self.floor = float(mu)
    p_range = (2.0, 2.0)
    def F(self, g): return 0.5*self.mu*g*g
    def dF(self, g): return self.mu*g
    def d2F(self, g): return self.mu + 0*g
    def Fstar(self, t): return t.clamp_min(0)**2/(2*self.mu)


class HerschelBulkley(Law):
    """tau = tau_y + K gd^n  (n = 1: Bingham; tau_y = 0: power law).  n <= 1: shear-thinning (ST);
    1 < n <= 2: shear-thickening, certified with the weight of Lemma 3.8 (r = n + 1 in (2, 3])."""
    def __init__(self, K, n, tau_y=0.0):
        if n <= 0: raise ValueError("Herschel-Bulkley needs n > 0")
        self.K, self.n, self.ty = K, n, tau_y; self.r = n + 1.0; self.rc = self.r/(self.r - 1.0)
        self.yield_stress = tau_y; self.p = self.r
        self.shear_thinning = n <= 1.0
        self.floor = float(K) if n == 1.0 else 0.0
        self.name = f'HerschelBulkley(K={K}, n={n}, tau_y={tau_y})'
    @property
    def p_range(self):                  # g tau'/tau = n K g^n/(tau_y + K g^n) in [0, n] (= n if tau_y = 0)
        return (self.r, self.r) if self.ty == 0 else (1.0, self.r)
    def F(self, g): return self.ty*g + self.K*g**self.r/self.r
    def dF(self, g): return self.ty + self.K*g**self.n
    def d2F(self, g): return self.K*self.n*g.clamp_min(1e-300)**(self.n - 1)
    def Fstar(self, t):
        return (t - self.ty).clamp_min(0.0)**self.rc*self.K**(1 - self.rc)/self.rc
    def nu0(self, gmax, gd_v=None):
        if self.n <= 1.0:
            return Law.nu0(self, gmax)
        if self.n > 2.0:
            raise ValueError("thickening certificate (Lemma 3.8) is proved for 1 < n <= 2 only")
        if gd_v is None:
            raise ValueError("the thickening weight needs the network shear rate gd_v")
        gd_v = torch.as_tensor(gd_v, dtype=DT)
        return 2.0*self.K*_cq_lower()*gd_v.clamp_min(0.0)**(self.n - 1.0)
    def nu0_is_conditional(self):
        return self.n < 1.0

def PowerLaw(K, n):
    L = HerschelBulkley(K, n, 0.0); L.name = f'PowerLaw(K={K}, n={n})'; return L
def Bingham(mu_p, tau_y):
    L = HerschelBulkley(mu_p, 1.0, tau_y); L.name = f'Bingham(mu_p={mu_p}, tau_y={tau_y})'; return L


class Carreau(Law):
    """eta = eta_inf + (eta0 - eta_inf) (1 + (lam gd)^2)^((n-1)/2),  n > 0, eta0 >= eta_inf >= 0"""
    def __init__(self, eta0, eta_inf, lam, n):
        if n <= 0 or eta0 < eta_inf or eta_inf < 0: raise ValueError("Carreau needs n > 0, eta0 >= eta_inf >= 0")
        self.e0, self.ei, self.lam, self.n = eta0, eta_inf, lam, n
        self.shear_thinning = n <= 1.0                         # Lemma 3.7
        self.p = 2.0 if (eta_inf > 0 or n == 1.0) else (n + 1.0)
        if n > 1.0: self.p = n + 1.0
        # floor: n <= 1: eta_inf (Lemma 3.7); n >= 1: eta0 (h'(x) >= (1+x^2)^mu >= 1, certificate note)
        self.floor = float(eta_inf) if n <= 1.0 else float(eta0)
        self.name = f'Carreau(eta0={eta0}, eta_inf={eta_inf}, lam={lam}, n={n})'
    @property
    def p_range(self):                  # g tau'/tau lies between n and 1
        return (min(2.0, self.n + 1), max(2.0, self.n + 1))
    def F(self, g):
        l2 = self.lam**2
        return 0.5*self.ei*g*g + (self.e0 - self.ei)/(l2*(self.n + 1))*((1 + l2*g*g)**((self.n + 1)/2) - 1)
    def dF(self, g):
        return (self.ei + (self.e0 - self.ei)*(1 + (self.lam*g)**2)**((self.n - 1)/2))*g
    def d2F(self, g):
        x2 = (self.lam*g)**2
        return self.ei + (self.e0 - self.ei)*(1 + x2)**((self.n - 3)/2)*(1 + self.n*x2)


class CarreauYasuda(Law):
    """eta = eta_inf + (eta0 - eta_inf) (1 + (lam gd)^a)^((n-1)/a),  a > 0, n > 0"""
    def __init__(self, eta0, eta_inf, lam, n, a):
        if n <= 0 or a <= 0 or eta0 < eta_inf or eta_inf < 0:
            raise ValueError("Carreau-Yasuda needs n > 0, a > 0, eta0 >= eta_inf >= 0")
        self.e0, self.ei, self.lam, self.n, self.a = eta0, eta_inf, lam, n, a
        self.shear_thinning = n <= 1.0                         # concavity: certificate note, Lemma 2.5
        self.p = 2.0 if (eta_inf > 0 or n == 1.0) else (n + 1.0)
        if n > 1.0: self.p = n + 1.0
        self.floor = float(eta_inf) if n <= 1.0 else float(eta0)
        self.name = f'CarreauYasuda(eta0={eta0}, eta_inf={eta_inf}, lam={lam}, n={n}, a={a})'
    def dF(self, g):
        return (self.ei + (self.e0 - self.ei)*(1 + (self.lam*g.clamp_min(0))**self.a)**((self.n - 1)/self.a))*g
    def d2F(self, g):
        x = (self.lam*g.clamp_min(0))**self.a; mu = (self.n - 1)/self.a
        return self.ei + (self.e0 - self.ei)*(1 + x)**(mu - 1)*(1 + self.n*x)
    @property
    def p_range(self):                  # g tau'/tau lies between n and 1
        return (min(2.0, self.n + 1), max(2.0, self.n + 1))
    def F(self, g): return _FByQuadrature.apply(g, self)


class Casson(Law):
    """sqrt(tau) = sqrt(tau_y) + sqrt(K gd)  (for tau > tau_y);  floor K"""
    def __init__(self, K, tau_y):
        self.K, self.ty = K, tau_y; self.yield_stress = tau_y
        self.p = 2.0; self.floor = float(K)
        self._pr = (1.0, 2.0) if tau_y > 0 else (2.0, 2.0)
        self.name = f'Casson(K={K}, tau_y={tau_y})'
    def F(self, g):
        return self.ty*g + (4.0/3.0)*math.sqrt(self.ty*self.K)*g**1.5 + 0.5*self.K*g*g
    def dF(self, g):
        return (math.sqrt(self.ty) + torch.sqrt(self.K*g.clamp_min(0)))**2
    def d2F(self, g):
        g = g.clamp_min(1e-300)
        return math.sqrt(self.ty*self.K)/torch.sqrt(g) + self.K
    def Fstar(self, t):
        s = t.clamp_min(self.ty); st = math.sqrt(self.ty)
        prim = lambda u: (u*u/2 - (4.0/3.0)*st*u**1.5 + self.ty*u)/self.K
        return prim(s) - prim(torch.as_tensor(self.ty, dtype=DT))


class Cross(Law):
    """eta = eta_inf + (eta0 - eta_inf)/(1 + (lam gd)^m),  0 < m <= 1 (concave flow curve).
    eta_inf = 0 and m = 1: the flow curve saturates at eta0/lam, so F* = +inf above it."""
    def __init__(self, eta0, eta_inf, lam, m):
        if not (0 < m <= 1): raise ValueError("Cross law is convex here only for 0 < m <= 1")
        if eta0 < eta_inf or eta_inf < 0: raise ValueError("Cross needs eta0 >= eta_inf >= 0")
        self.e0, self.ei, self.lam, self.m = eta0, eta_inf, lam, m
        self.floor = float(eta_inf)
        if eta_inf > 0: self.p = 2.0
        elif m < 1: self.p = 2.0 - m
        else: self.p = 1.0; self.tau_sup = eta0/lam
        self._pr = (2.0 - m, 2.0)          # g tau'/tau in [1 - m, 1]
        self.name = f'Cross(eta0={eta0}, eta_inf={eta_inf}, lam={lam}, m={m})'
    def dF(self, g):
        return (self.ei + (self.e0 - self.ei)/(1 + (self.lam*g.clamp_min(0))**self.m))*g
    def d2F(self, g):
        x = (self.lam*g.clamp_min(0))**self.m
        return self.ei + (self.e0 - self.ei)*(1 + (1 - self.m)*x)/(1 + x)**2
    def F(self, g): return _FByQuadrature.apply(g, self)


def Williamson(eta0, eta_inf, lam):
    """Williamson's law in its Cross-type form eta = eta_inf + (eta0 - eta_inf)/(1 + lam gd)
    (Cross with m = 1)."""
    L = Cross(eta0, eta_inf, lam, 1.0); L.name = f'Williamson(eta0={eta0}, eta_inf={eta_inf}, lam={lam})'
    return L


def check_convex(law, gmax=1e3, n=400):
    """numerical sanity check: F' nondecreasing on a grid, and the Fenchel-Young equality
    F(g) + F*(F'(g)) = g F'(g) at matched pairs (the computed F* is an upper bound, so the
    defect must be >= 0 and tiny)."""
    g = torch.logspace(-6, math.log10(gmax), n, dtype=DT)
    d = law.dF(g)
    mono = bool((d[1:] - d[:-1] >= -1e-12*d.abs().max()).all())
    fy = law.F(g) + law.Fstar(d) - d*g
    return mono, float(fy.abs().max()/max(1.0, float((d*g).abs().max())))
