"""Certified power W* = int S*:eps(u*) (rho = 0, certificate note Thm 6.4) against exact channel flows:
the enclosure must contain the exact value for every network (here random ones), for every law."""
import numpy as np
import torch
import aequa as aq
from aequa.profiles import developed_profile

DT = torch.float64


def test_power_contains_exact():
    laws = [aq.Newtonian(1.0), aq.PowerLaw(1.0, 0.5), aq.Bingham(1.0, 0.5), aq.HerschelBulkley(3.72, 0.41, 3.31),
            aq.Carreau(1.0, 0.05, 2.0, 0.5), aq.Casson(0.5, 0.3), aq.Cross(1.0, 0.05, 2.0, 0.7)]
    for law in laws:
        dom = aq.channel(2.0, 2.0, x0=0.0)
        prof = developed_profile(law, -1.0, 1.0, 1.0); D = prof.developed
        dom.set_bc({'inlet': aq.Profile(prof, (1, 0)), 'outlet': aq.Profile(prof, (1, 0)), 'wall': aq.NoSlip()})
        S = aq.Solver(dom, law, rho=0.0, width=16, depth=2)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for p in S.parameters(): p.add_(0.05*torch.randn(p.shape, generator=g, dtype=DT))
        gap = S.certify_gap(h=0.1)
        pb = S.power_bounds(gap)
        W = D.G*2.0*2.0                       # exact: pressure gradient x length x flux
        print(f"{law.name:50s} W* = {W:.6f} in [{pb['W_lo']:.6f}, {pb['W_hi']:.6f}]  (G/J = {gap['G_up']/gap['fine']['diss']:.2e})")
        assert pb['W_lo'] <= W <= pb['W_hi']
        assert abs(pb['Q_in'] - 2.0) < 1e-9 and not pb['moving_walls']


if __name__ == "__main__":
    test_power_contains_exact()
