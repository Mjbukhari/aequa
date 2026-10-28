"""Carreau fluid past a rotating cylinder in a channel: the one-call use of aequa.

Inlet and outlet carry the developed Carreau profile (computed by aequa.developed_profile to
~1e-15), so the imposed data are those of the flow the channel actually develops. The summary
labels each number as [exact], [quadrature] or [sampled] and states which theorem applies:
with rho = 1 the inertial theorem needs theta < 1 for a bound (gap regime).
"""
import aequa as aq

law = aq.Carreau(eta0=1.0, eta_inf=0.05, lam=2.0, n=0.5)
dom = aq.channel(8.0, 2.0, x0=-3.0, obstacles=[aq.circle((0.0, 0.0), 0.3, 'cylinder')])
flow = aq.developed_profile(law, -1.0, 1.0, mean_speed=1.0)

bcs = {'inlet': aq.Profile(flow, (1, 0)), 'outlet': aq.Profile(flow, (1, 0)),
       'wall': aq.NoSlip(), 'cylinder': aq.Rotating((0.0, 0.0), omega=3.0)}

# Stokes solve first (G = E + E* exactly), then continuation to rho = 1
stokes = aq.solve(dom, law, bcs, rho=0.0)
res = aq.solve(dom, law, bcs, rho=1.0, warm_start=stokes)
