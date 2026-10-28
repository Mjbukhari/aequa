#!/usr/bin/env python3
"""Herschel-Bulkley fluid (Carbopol 0.13 %, Hadj Arab et al.) through a planar 4:1 contraction.

Replaces contraction_training_patched.py and contraction_gap_quadrature.py of v0.2. Everything now
goes through the library, so every number carries the guarantees of the certificate note:

  * float64 throughout; exact law (no Papanastasiou, no regularisation in the certificate);
  * v exactly divergence free with exact no-slip and exact developed HB profiles at inlet and outlet;
    stress exactly in equilibrium (Lemma 3.1); corner exponents above the floors of Lemma 3.1
    (p = n + 1 = 1.41: lambda > 1, beta > 2 - 2/p' = 1.418);
  * the gap G by the deterministic rule with exact geometry, graded sectors at both re-entrant
    corners and a sector tail bound (Sec. 5), reported with its two-resolution difference;
  * the statement matches the regime: rho_hat > 0 here, so a bound on E + E* needs theta < 1
    (Thm 6.1); for 1 <= theta < 2 or theta >= 2 no bound is claimed. v0.2 printed
    'energy error <= G' at every flow rate, which is only proved for rho = 0.

The v0.2 script trained ONE network with Q as an input. A parametric network is certified only at
the Q values where its gap is evaluated, so here each flow rate gets its own certified network,
obtained by continuation in Q (warm start from the previous rate).

Scaling (downstream half-width h = 25 mm, depth 20 mm): velocity U = Q/(2 h e), strain U/h,
stress s0 = K (U/h)^n.  Scaled law: tau = B + gd^n with B = tau_y/s0, inertia rho_hat = rho U^2/s0.
Lengths in units of h: upstream half-width 4, downstream half-width 1.

Pressure drop between the taps of the experiment (x* = -28 and x* = 32) is obtained from the
computed drop between two sections inside the developed regions plus the exact developed
pressure gradients outside them. Pressure is NOT covered by the gap certificate (it needs an
inf-sup constant, or the concavity argument of the planned Delta-p theorem), so it is printed as
[uncertified].

Run:  python contraction_hb.py          (full: about 1 to 2 h per flow rate on a laptop CPU)
      QUICK=1 python contraction_hb.py  (pipeline check only, few iterations)
"""
import os
import json
import math
import numpy as np
import aequa as aq
from aequa.profiles import Developed

QUICK = bool(int(os.environ.get("QUICK", "0")))
OUT = os.path.abspath(os.environ.get("OUT", "out_contraction_hb"))
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ fluid, rig, flow rates
TAU_Y, K_C, N_C, RHO = 3.31, 3.72, 0.41, 1000.0          # Pa, Pa s^n, -, kg/m^3
h_SI, DEPTH = 0.025, 0.020                                 # m
Q_LIST = [1.24, 3.05, 4.68, 7.21, 8.50]                    # l/min
TAP_UP, TAP_DN = -28.0, 32.0                               # pressure taps, in units of h

H_UP, H_DN = 4.0, 1.0
L_UP, L_DN = (20.0, 12.0) if not QUICK else (12.0, 8.0)   # developed well before both ends


def U_dn(Q): return Q/60000.0/(2*h_SI*DEPTH)               # m/s
def s0(Q): return K_C*(U_dn(Q)/h_SI)**N_C                  # Pa
def B_of(Q): return TAU_Y/s0(Q)
def rho_hat(Q): return RHO*U_dn(Q)**2/s0(Q)


TRAIN = dict(iters=6000, lbfgs=300, width=48, depth=4, batch=2048) if not QUICK else \
        dict(iters=200, lbfgs=20, width=16, depth=2, batch=512)


def centreline_pressure(res, x):
    return res(np.asarray(x, float), np.zeros(len(x)))['p']


def main():
    dom = aq.contraction(L_UP, H_UP, L_DN, H_DN)
    prev, table = None, {}
    for Q in Q_LIST:
        B, rh = B_of(Q), rho_hat(Q)
        law = aq.HerschelBulkley(1.0, N_C, B)
        D_up = Developed(law, -H_UP, H_UP, H_DN/H_UP)          # same flux (2) in both channels
        D_dn = Developed(law, -H_DN, H_DN, 1.0)
        f_up = lambda x, y, D=D_up: D.speed(y)
        f_dn = lambda x, y, D=D_dn: D.speed(y)
        print(f"\n=== Q = {Q} l/min:  B = {B:.4f}, rho_hat = {rh:.4f}, plug half-widths "
              f"{D_up.plug:.3f} (up), {D_dn.plug:.3f} (down) ===", flush=True)
        res = aq.solve(dom, law, {'inlet': aq.Profile(f_up, (1, 0)), 'outlet': aq.Profile(f_dn, (1, 0)),
                                  'wall': aq.NoSlip()},
                       rho=rh, warm_start=prev, verbose=True, **TRAIN)
        prev = res

        # pressure drop between the taps [uncertified]: sections 3 h inside each end, developed
        # gradients G (scaled) outside them
        xa, xb = -L_UP + 3.0, L_DN - 3.0
        pa, pb = centreline_pressure(res, [xa, xb])
        dp = (pa - pb) + D_up.G*(xa - TAP_UP) + D_dn.G*(TAP_DN - xb)
        dp_fd = D_up.G*(0.0 - TAP_UP) + D_dn.G*(TAP_DN - 0.0)   # fully developed, no contraction loss

        g, th = res.gapinfo, res.theta
        U, S = U_dn(Q), s0(Q)
        Wscale = S*U*h_SI*DEPTH                                  # scaled dissipation -> W
        if th is None: regime, bound = "theta not computed", None
        elif th < 1: regime, bound = "gap regime (theta < 1)", res.G/(1 - th)
        elif th < 2: regime, bound = "hybrid regime (1 <= theta < 2): no bound reported", None
        else: regime, bound = "not certified (theta >= 2)", None
        J = g['fine']['diss']
        table[f"{Q:g}"] = dict(
            Q_lmin=Q, B=B, rho_hat=rh, U_dn=U, stress_scale_Pa=S,
            G=res.G, G_fine=g['fine']['G'], G_coarse=g['coarse']['G'], tails=g['fine']['tail_sum'],
            dissipation=J, dissipation_W=J*Wscale, rel_gap=res.G/J,
            corner_lambda=[float(x) for x in res.solver.exponents()[0]],
            theta=th, regime=regime, bound_E_plus_Estar=bound,
            bound_rel=(None if bound is None else math.sqrt(bound/J)),
            dp_taps_Pa_uncertified=float(dp*S), dp_developed_Pa=float(dp_fd*S),
            excess_dp_Pa_uncertified=float((dp - dp_fd)*S))
        json.dump(table, open(os.path.join(OUT, "contraction_hb_summary.json"), "w"), indent=2)

    print("\n" + "="*112)
    print(f"{'Q l/min':>8} {'B':>7} {'rho_hat':>8} {'G/J':>9} {'theta':>7}  {'E+E* <=':>10}  "
          f"{'dp taps Pa':>11} {'excess dp Pa':>13}  regime")
    for s in table.values():
        th = "-" if s['theta'] is None else f"{s['theta']:.3f}"
        bd = "-" if s['bound_E_plus_Estar'] is None else f"{s['bound_E_plus_Estar']:.3e}"
        print(f"{s['Q_lmin']:>8.2f} {s['B']:>7.3f} {s['rho_hat']:>8.4f} {100*s['rel_gap']:>8.3f}% {th:>7}  {bd:>10}  "
              f"{s['dp_taps_Pa_uncertified']:>11.2f} {s['excess_dp_Pa_uncertified']:>13.2f}  {s['regime']}")
    print("="*112)
    print("G/J, theta, E+E*: certificate quantities ([quadrature] / [sampled] tiers, see summary above).")
    print("dp: [uncertified] post-processing of the certified pair.")
    print("summary written to", os.path.join(OUT, "contraction_hb_summary.json"))


if __name__ == "__main__":
    main()
