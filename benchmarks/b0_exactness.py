"""Benchmark 0: properties that hold for EVERY network, checked on random (untrained) networks.

  * the identity G = E + E* (Stokes) against exact developed channel flows, 8 laws
  * the certified power enclosure contains the exact power, 7 laws
These are the test-suite checks (tests/test_identity.py, tests/test_power.py), with the numbers saved.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
import aequa as aq
from aequa.profiles import developed_profile
from test_identity import _channel_case
from common import save

laws = [aq.Newtonian(1.0), aq.PowerLaw(1.0, 0.5), aq.Bingham(1.0, 0.5), aq.HerschelBulkley(3.72, 0.41, 3.31),
        aq.Carreau(1.0, 0.05, 2.0, 0.5), aq.CarreauYasuda(1.0, 0.05, 2.0, 0.4, 1.5), aq.Casson(0.5, 0.3),
        aq.Cross(1.0, 0.05, 2.0, 0.7)]
ident = []
for law in laws:
    G, E, Es, sc, G_lib = _channel_case(law)
    ident.append(dict(law=law.name, G=G, E=E, Estar=Es, rel_defect=abs(G - E - Es)/sc))
    print(f"{law.name:55s} G={G:.6e}  E+E*={E + Es:.6e}  defect {abs(G - E - Es)/sc:.1e}", flush=True)

power = []
for law in laws[:5] + laws[6:]:
    dom = aq.channel(2.0, 2.0, x0=0.0)
    prof = developed_profile(law, -1.0, 1.0, 1.0); D = prof.developed
    dom.set_bc({'inlet': aq.Profile(prof, (1, 0)), 'outlet': aq.Profile(prof, (1, 0)), 'wall': aq.NoSlip()})
    S = aq.Solver(dom, law, rho=0.0, width=16, depth=2)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in S.parameters(): p.add_(0.05*torch.randn(p.shape, generator=g, dtype=torch.float64))
    gap = S.certify_gap(h=0.1); pb = S.power_bounds(gap)
    W = D.G*2.0*2.0
    power.append(dict(law=law.name, W_exact=W, W_lo=pb['W_lo'], W_hi=pb['W_hi'], G_over_J=gap['G_up']/gap['fine']['diss'],
                      contains=bool(pb['W_lo'] <= W <= pb['W_hi'])))
    print(f"{law.name:55s} W*={W:.6f} in [{pb['W_lo']:.6f}, {pb['W_hi']:.6f}]", flush=True)
save('b0_exactness', dict(identity=ident, power=power))
