"""T-shaped mixer with a rotating cylinder at the junction, power-law fluid (n = 0.6).

All three openings carry the developed power-law profile of their branch. The flux balance
(inlet flux = sum of outlet fluxes) is checked by the library; any residual is removed by a
uniform normal correction and reported as 'data mismatch' in the summary (Remark 4.4).
n = 0.6 gives p = 1.6 < 2, so the law has no positive viscosity floor: the norm statement is
the tube statement of Cor. 7.4 with s = s0 U/L.
"""
import aequa as aq

law = aq.PowerLaw(K=1.0, n=0.6)
dom = aq.t_junction(L_in=3.0, W_in=1.0, L_branch=4.0, W_branch=1.0,
                    obstacles=[aq.circle((0.0, 0.5), 0.2, 'rotor')])
inlet = aq.developed_profile(law, -0.5, 0.5, mean_speed=1.0, coord='x')     # flux 1 upward
branch = aq.developed_profile(law, 0.0, 1.0, mean_speed=0.5)                 # flux 1/2 each side

res = aq.solve(dom, law,
               {'inlet':   aq.Profile(inlet, (0, 1)),
                'outlet1': aq.Profile(branch, (1, 0)),
                'outlet2': aq.Profile(branch, (-1, 0)),
                'wall':    aq.NoSlip(),
                'rotor':   aq.Rotating((0.0, 0.5), omega=2.0)},
               rho=0.2, s0=0.1)
