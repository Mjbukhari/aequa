# aequa 0.3.1

Gap-certified neural solver for steady generalised-Newtonian flow in 2D.

```python
import aequa as aq

law  = aq.Carreau(eta0=1.0, eta_inf=0.05, lam=2.0, n=0.5)
dom  = aq.channel(8.0, 2.0, x0=-3.0, obstacles=[aq.circle((0, 0), 0.3, 'cylinder')])
flow = aq.developed_profile(law, -1, 1, mean_speed=1.0)      # developed Carreau profile
res  = aq.solve(dom, law,
                {'inlet': aq.Profile(flow, (1, 0)), 'outlet': aq.Profile(flow, (1, 0)),
                 'wall': aq.NoSlip(), 'cylinder': aq.Rotating((0, 0), omega=3.0)},
                rho=0.25)
print(res.summary())     # every number labelled [exact] / [quadrature] / [sampled]
u = res(x, y)            # velocity and pressure anywhere
```

## What the certificate says

The network returns a pair (v, m): v is exactly divergence free with the exact boundary data, and
m is exactly in equilibrium, for every value of the weights. The primal-dual gap

G = ∫ F(γ̇(v)) + F*(|m|/√2) − m : ε(v) dx ≥ 0

then controls the error:

| Regime | Statement printed |
|---|---|
| ρ = 0 (Stokes) | G = E + E* exactly; energy enclosure J(v) − G ≤ J* ≤ J(v); norm bounds below |
| ρ > 0, θ < 1 | E + E* ≤ G / (1 − θ) |
| 1 ≤ θ < 2 | existence and uniqueness near v hold, but no numerical bound is printed (needs the residual, not evaluated in 0.3) |
| θ ≥ 2 | not certified |

Norm bounds: ν* ∫|ε(u − v)|² ≤ bound when the law has a viscosity floor ν* > 0 (unconditional);
the Lemma 3.8 weight for shear-thickening Herschel-Bulkley (1 < n ≤ 2); for floorless
shear-thinning laws the sharp weight F''(√2((1+λ)|ε(v)| + s)) in the tube statement of Cor. 7.4,
with s = s0·U/L.

## Benchmarks

`benchmarks/BENCHMARKS.md` has the full results, the figures and the scripts that produce them. In short, on a
2-core CPU with small networks:

* **Exactness.** G = E + E* holds to 2e-10 for eight laws on random networks, and the certified power encloses the exact value.
* **Couette flow of a Carreau fluid (exact solution).** The certified energy-error bound is within a factor 1.1 (sharp tube
  weight) to 2.7 (unconditional floor) of the true error; the certified enclosures of the minimal dissipation and of the power
  contain the exact values. Compared on the same problem with Deep Ritz, a hard-constraint PINN, a standard PINN and
  Taylor-Hood FEM: FEM is the fastest in 2D, and some baselines reach similar or smaller errors, but none of them prints a
  guaranteed bound.
* **Flow past a cylinder (no exact solution).** The certified enclosures of the dissipation, power and pressure drop contain
  the converged FEM reference.

## Guarantee levels

Every number in `res.summary()` carries one of three labels.

* **[exact]** holds for every weight value, by construction: div v = 0, v = v_D, div of the Airy
  stress = 0, corner exponents above the floors λ > max(1, 2 − 2/p), β > 2 − 2/p' (finite
  dissipation and finite conjugate energy at re-entrant corners), the guaranteed upper bound used
  for implicit conjugates F*.
* **[quadrature]** G by a deterministic rule with exact polygon and circle geometry, positive
  weights, geometric grading into every corner and an analytic tail bound for the corner sectors.
  The value used is fine + |fine − coarse| + tails. This is a converged quadrature, not an
  interval enclosure.
* **[sampled]** θ (inertial problems only): the constants come from network values on a grid with a
  Lipschitz safety margin. Not interval-verified.

## What is new in 0.3.1

* Training sampler: the band of extra samples around an obstacle is now weighted by the exact area of
  its intersection with the domain. Before, a band that left the domain (an annulus, a cylinder close
  to a wall) was overweighted, which biased the Monte Carlo estimate of the training loss. The
  certificate was never affected (it uses the deterministic rule), but the training objective was.
* `benchmarks/`: reproducible comparison with Deep Ritz, PINNs and finite elements.

## What is new in 0.3

* Exponent floors at corners (Lemma 3.1); learned exponents can no longer leave the range where
  the energies are finite.
* With inertia, the stronger corner floor λ > 4/p − 2 for 1 < p < 4/3 (Lemma 3.2: the inertial
  terms of the proof need ∇v ∈ L^q with q ≥ p*/(p* − 2)).
* Vertex data check (Lemma 4.2): a velocity jump at a vertex (sharp lid) is rejected for p ≥ 2,
  allowed for p < 2; such vertices get the graded singular rule and tail bound. Inertia together
  with a data jump, and inertia for saturating laws (p = 1), are reported as not certified.
* No-slip vertex blending: the lift is exactly constant near rigid vertices (removes O(u/r²)
  cancellation in second derivatives).
* Deterministic quadrature (`aequa/quadrature.py`) replaces Monte Carlo in the certificate.
* Rheology rewritten: guaranteed upper bound for implicit conjugates, floors and growth exponents
  per law, sharp tube weight ν0; new laws Carreau-Yasuda and Williamson.
* Developed channel profiles for every law (`aq.developed_profile`, accurate to ~1e-15), including
  yield-stress plugs.
* Certified power and pressure drop for Stokes flow (note, Thm 6.4 and Cor. 6.5): sharp (width G/J) for
  Newtonian and power-law fluids, width of order sqrt(G/J) for laws with several exponents.
* Summary states the theorem that applies, per regime; no bound is printed where none is proved.
* θ: line-stress free ends placed outside the domain (required by the proof).
* Test suite: identity G = E + E* against exact solutions for 8 laws (defect ≤ 2e-10),
  admissibility, quadrature on 5 geometries and singular corner densities, gallery boundary data.

The full mapping from printed numbers to theorems, with proofs of the new lemmas, is in
`docs/certificate_note.pdf`.

## Rheology

Newtonian, power law, Bingham, Herschel-Bulkley (shear-thinning and 1 < n ≤ 2), Carreau,
Carreau-Yasuda, Casson, Cross, Williamson.

## Geometry

Outer loop and obstacles made of segments, circles, arcs, graphs y = h(x) and smooth closed
curves. Ready-made: channel, 4:1 contraction, expansion, backward step, T-junction, cross-slot,
lid-driven cavity, peristaltic channel, eccentric annulus. Obstacles: circle, square, rectangle,
triangle, regular polygon, ellipse, any closed curve, any tilt.

## Tests

```
pip install -e .[test]
pytest
```

## Not yet

* bound in the hybrid regime 1 ≤ θ < 2 (needs the residual |m − S(ε(v))|) and the θ ≥ 2 check
* interval-arithmetic verification of G and θ
* certified pressure drop with inertia (the Stokes case is certified, Thm 6.4)
* outflow (traction) and periodic boundaries in the one-call API
* moving boundaries, 3D, GPU batching

## License

AGPL-3.0, see LICENSE. For commercial licensing, contact M. Javed Bukhari (javedbukhari829@gmail.com).
