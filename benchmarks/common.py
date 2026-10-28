"""Shared pieces of the benchmark suite: the Carreau test law, the exact Couette solution in an
annulus, error functionals against an exact solution, and result I/O.

Everything here is independent of the method being tested, so Aequa, Deep Ritz, the PINN and the
finite-element reference are all measured by exactly the same code.
"""
import json, math, os, platform, time
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- test law (numpy version)
# Carreau: eta = eta_inf + (eta0 - eta_inf) (1 + (lam gd)^2)^((n - 1)/2)
CARREAU = dict(eta0=1.0, eta_inf=0.05, lam=2.0, n=0.5)


def carreau_eta(gd, eta0=1.0, eta_inf=0.05, lam=2.0, n=0.5):
    return eta_inf + (eta0 - eta_inf)*(1.0 + (lam*gd)**2)**((n - 1.0)/2.0)


def carreau_tau(gd, **k):
    return carreau_eta(gd, **k)*gd


def carreau_F(gd, eta0=1.0, eta_inf=0.05, lam=2.0, n=0.5):
    l2 = lam**2
    return 0.5*eta_inf*gd**2 + (eta0 - eta_inf)/(l2*(n + 1))*((1 + l2*gd**2)**((n + 1)/2) - 1)


def carreau_gd_of_tau(tau, **k):
    """inverse flow curve (monotone), vectorised bisection"""
    tau = np.asarray(tau, float)
    lo = np.zeros_like(tau); hi = np.ones_like(tau)
    while np.any(carreau_tau(hi, **k) < tau):
        hi = np.where(carreau_tau(hi, **k) < tau, 2*hi, hi)
    for _ in range(200):
        mid = 0.5*(lo + hi)
        up = carreau_tau(mid, **k) > tau
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    return 0.5*(lo + hi)


# ---------------------------------------------------------------- exact Couette flow
class CouetteExact:
    """Inner cylinder r = Ri rotating with angular speed Om, outer cylinder r = Ro at rest.
    Torque balance gives tau_r_theta = C/r^2; the shear rate is gd = r |d omega/dr| = gd(C/r^2),
    and C is fixed by Om = int_Ri^Ro gd(C/r^2)/r dr.  Valid for any monotone law, with or without
    inertia (the flow is circular, so inertia only changes the pressure)."""
    def __init__(self, Ri=0.5, Ro=1.0, Om=2.0, law=CARREAU):
        self.Ri, self.Ro, self.Om, self.k = Ri, Ro, Om, dict(law)
        f = lambda C: quad(lambda r: carreau_gd_of_tau(np.array([C/r**2]), **self.k)[0]/r, Ri, Ro,
                           epsabs=1e-14, epsrel=1e-13, limit=200)[0] - Om
        Chi = 1.0
        while f(Chi) < 0: Chi *= 2
        self.C = brentq(f, 0.0, Chi, xtol=1e-15, rtol=1e-15)
        # omega(r) on a fine table (Gauss-Kronrod per point would be slow): cumulative integral
        rr = np.linspace(Ri, Ro, 20001)
        g = carreau_gd_of_tau(self.C/rr**2, **self.k)/rr
        from scipy.integrate import cumulative_simpson
        I = cumulative_simpson(g[::-1], x=-rr[::-1], initial=0.0)          # int_r^Ro, built from Ro inwards
        self._r = rr; self._om = I[::-1]                                   # omega(r) = int_r^Ro gd/s ds
        self.table_err = abs(self._om[0] - Om)/Om                         # table vs the quad() value of C
        self._om = self._om*(Om/self._om[0])

    def omega(self, r):
        return np.interp(r, self._r, self._om)

    def velocity(self, x, y):
        r = np.hypot(x, y); w = self.omega(r)
        return -w*y, w*x

    def strain(self, x, y):
        """(eps11, eps12) of the exact flow; eps_r_theta = r omega'/2 = -gd/2"""
        r = np.hypot(x, y); c, s = x/r, y/r
        gd = carreau_gd_of_tau(self.C/r**2, **self.k)
        ert = -0.5*gd
        return ert*(-2*s*c), ert*(c*c - s*s)

    def stress(self, x, y):
        """deviatoric stress S = 2 eta eps (S11, S12)"""
        a, b = self.strain(x, y)
        gd = 2*np.sqrt(a*a + b*b)
        eta = carreau_eta(gd, **self.k)
        return 2*eta*a, 2*eta*b

    def dissipation(self):
        """J* = int F(gd(u*)) over the annulus (1D integral in r)"""
        return quad(lambda r: 2*math.pi*r*carreau_F(carreau_gd_of_tau(np.array([self.C/r**2]), **self.k)[0], **self.k),
                    self.Ri, self.Ro, epsabs=1e-14, epsrel=1e-13, limit=200)[0]

    def power(self):
        """W* = torque x angular speed = 2 pi C Om (per unit length)"""
        return 2*math.pi*self.C*self.Om


def annulus_rule(Ri, Ro, nr=64, nt=256, ng=6):
    """tensor Gauss rule in polar coordinates on the annulus (spectrally accurate for smooth
    integrands; used only to MEASURE errors, identically for every method)"""
    g, w = np.polynomial.legendre.leggauss(ng)
    re = np.linspace(Ri, Ro, nr + 1); te = np.linspace(-math.pi, math.pi, nt + 1)
    R = (0.5*np.diff(re)[:, None]*(g + 1) + re[:-1, None]).ravel(); WR = (0.5*np.diff(re)[:, None]*w).ravel()
    T = (0.5*np.diff(te)[:, None]*(g + 1) + te[:-1, None]).ravel(); WT = (0.5*np.diff(te)[:, None]*w).ravel()
    RR, TT = np.meshgrid(R, T, indexing='ij')
    W = np.outer(WR, WT)*RR
    return np.c_[(RR*np.cos(TT)).ravel(), (RR*np.sin(TT)).ravel()], W.ravel()


def error_report(P, W, ex, u, v, e11, e12, nu_floor=CARREAU['eta_inf']):
    """errors of a velocity field (u, v, eps11, eps12 at the points P) against the exact solution"""
    ue, ve = ex.velocity(P[:, 0], P[:, 1]); ae, be = ex.strain(P[:, 0], P[:, 1])
    L2 = math.sqrt((((u - ue)**2 + (v - ve)**2)*W).sum()/(((ue**2 + ve**2))*W).sum())
    de = 2*((e11 - ae)**2 + (e12 - be)**2)                      # |eps(v - u)|^2 (Frobenius)
    H1 = math.sqrt((de*W).sum()/((2*(ae**2 + be**2))*W).sum())
    # primal Bregman distance E = int F(gd v) - F(gd u) - S(u):(eps v - eps u)
    gdv = 2*np.sqrt(e11**2 + e12**2); gdu = 2*np.sqrt(ae**2 + be**2)
    S11, S12 = ex.stress(P[:, 0], P[:, 1])
    E = float(((carreau_F(gdv, **ex.k) - carreau_F(gdu, **ex.k) - 2*(S11*(e11 - ae) + S12*(e12 - be)))*W).sum())
    return dict(rel_L2_velocity=L2, rel_energy_seminorm=H1, eps_err_sq=float((de*W).sum()),
                bregman_E=E, nu_floor_energy=float(nu_floor*(de*W).sum()))


def save(name, data):
    data = dict(data)
    data['_machine'] = dict(platform=platform.platform(), processor=platform.processor(),
                            cpus=os.cpu_count(), date=time.strftime('%Y-%m-%d %H:%M'))
    with open(os.path.join(RESULTS, name + '.json'), 'w') as f:
        json.dump(data, f, indent=1, default=float)


def load(name):
    with open(os.path.join(RESULTS, name + '.json')) as f:
        return json.load(f)
