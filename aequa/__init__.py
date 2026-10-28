"""AEQUA: gap-certified neural solver for generalised-Newtonian flow (prototype)."""
from .geometry import (Domain, Segment, Circle, Curve, NoSlip, MovingWall, Rotating, Profile, Velocity,
                       polygon_loop, circle, square, rectangle, triangle, regular_polygon, ellipse,
                       closed_curve, graph, channel, contraction, expansion, backward_step, t_junction,
                       cross_slot, cavity, wavy_channel, annulus, poiseuille_profile, powerlaw_profile,
                       reentrant_corners)
from .rheology import (Newtonian, PowerLaw, HerschelBulkley, Bingham, Carreau, CarreauYasuda, Casson, Cross,
                       Williamson, check_convex)
from .profiles import developed_profile, Developed
from .solver import Solver
from .theta import theta_bound
from .api import solve, Result

__version__ = "0.3.1"
