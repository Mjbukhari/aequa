"""Standard benchmark geometries (tests/test_gallery.py imposes boundary data on each of them).
For every geometry: area, obstacles, re-entrant corners, and the Stokes corner exponent
lambda(omega) that the solver uses to initialise its corner terms (above the floor of Lemma 3.1)."""
import numpy as np
import aequa as aq
from aequa.solver import stokes_corner_exponent

geoms = {
    'cylinder in channel':  aq.channel(8, 2, x0=-2, obstacles=[aq.circle((0, 0), 0.25, 'cyl')]),
    'square / triangle':    aq.channel(8, 2, x0=-2, obstacles=[aq.square((0, 0), 0.4, 'sq'), aq.triangle((2.5, 0), 0.5, 'tri')]),
    'inclined ellipse':     aq.channel(6, 2, x0=-2, obstacles=[aq.ellipse((0, 0), 0.4, 0.15, 'ell', angle=0.4)]),
    'T-junction mixer':     aq.t_junction(3, 1, 4, 1, obstacles=[aq.circle((0, 0.5), 0.2, 'rotor')]),
    'cross-slot':           aq.cross_slot(3, 1),
    '4:1 contraction':      aq.contraction(10, 4, 8, 1),
    'sudden expansion':     aq.expansion(4, 1, 10, 4),
    'backward-facing step': aq.backward_step(2, 0.5, 6, 1),
    'lid-driven cavity':    aq.cavity(1, 1, obstacles=[aq.circle((0.5, 0.4), 0.15, 'cyl')]),
    'peristaltic channel':  aq.wavy_channel(8, lambda x: 1 + 0.3*np.sin(np.pi*x/2), lambda x: -1 - 0.3*np.sin(np.pi*x/2)),
    'eccentric annulus':    aq.annulus(1.0, 0.4, inner_center=(0.2, 0.0)),
}
if __name__ == "__main__":
    for k, d in geoms.items():
        rc = aq.reentrant_corners(d)                       # [(vertex, fluid angle omega)]
        lams = sorted({round(stokes_corner_exponent(om), 4) for _, om in rc})
        print(f"{k:22s} area {d.area():7.3f}  obstacles {len(d.hole_loops)}  re-entrant corners {len(rc)}"
              + (f"  Stokes lambda {lams}" if lams else ""))
