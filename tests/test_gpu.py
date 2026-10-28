"""Device and precision handling.

* On every machine: float32 ('mixed') training runs end to end on the CPU, the parameters come
  back in float64 and the certificate is computed in float64.
* On a machine with a CUDA GPU (skipped otherwise): the same network gives the same fields away
  from the walls (1e-8), the same integrals of the gap and shear rate and the same certificate G
  (1e-7) on the GPU as on the CPU, on geometries with obstacles, re-entrant corners and curved
  walls, for laws with closed-form and implicit conjugates; and a short aq.solve run on the GPU
  returns a finite certificate.
"""
import numpy as np
import pytest
import torch
import aequa as aq
from aequa.solver import resolve_device, resolve_precision

par = lambda a, b, U: (lambda x, y: 6*U*(y - a)*(b - y)/(b - a)**2)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")


def _cases():
    """(name, domain factory, bcs, law): one per kind of geometry the lift and the solver handle"""
    top = lambda x: 1.0 + 0.3*np.sin(2*np.pi*x/4); bot = lambda x: -top(x)
    return [
        ('cylinder, Carreau', lambda: aq.channel(4.0, 2.0, x0=-2.0, obstacles=[aq.circle((0, 0), 0.3, 'cyl')]),
         {'inlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
          'wall': aq.NoSlip(), 'cyl': aq.Rotating((0, 0), 1.0)}, aq.Carreau(1.0, 0.05, 2.0, 0.5)),
        ('contraction, Herschel-Bulkley', lambda: aq.contraction(4, 2, 4, 1),
         {'inlet': aq.Profile(par(-2, 2, 0.5), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
          'wall': aq.NoSlip()}, aq.HerschelBulkley(1.0, 0.5, 0.5)),
        ('ellipse, Carreau-Yasuda', lambda: aq.channel(4.0, 2.0, x0=-2.0, obstacles=[aq.ellipse((0, 0), 0.4, 0.15, 'ell', angle=0.4)]),
         {'inlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
          'wall': aq.NoSlip(), 'ell': aq.NoSlip()}, aq.CarreauYasuda(1.0, 0.05, 2.0, 0.4, 1.5)),
        ('wavy channel, Casson', lambda: aq.wavy_channel(4.0, top, bot, x0=0.0),
         {'inlet': aq.Profile(par(-1, 1, 1.0), (1, 0)), 'outlet': aq.Profile(par(-1, 1, 1.0), (1, 0)),
          'wall': aq.NoSlip()}, aq.Casson(0.5, 0.3)),
    ]


def _randomise(S, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in S.parameters():
            p.add_(0.3*torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.device, p.dtype))


def test_resolve():
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device('auto').type == ('cuda' if torch.cuda.is_available() else 'cpu')
    assert resolve_precision('auto', torch.device('cpu')) == torch.float64
    assert resolve_precision('mixed', torch.device('cpu')) == torch.float32
    with pytest.raises(ValueError):
        resolve_precision('half', torch.device('cpu'))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device('cuda')


@pytest.mark.parametrize('case', range(4))
def test_float32_forward_stays_float32(case):
    """every table used in the forward pass must follow the input's dtype: a float64 table left
    behind would silently promote the result to float64 here, and on a GPU it would sit on the
    wrong device. So this CPU test also guards the GPU path."""
    name, mk, bcs, law = _cases()[case]
    dom = mk(); dom.set_bc(bcs)
    S = aq.Solver(dom, law, width=16, depth=2, device='cpu', precision='mixed')
    _randomise(S)
    S._place(torch.float32)
    X, w = S.sample(256); X.requires_grad_(True)
    assert X.dtype == torch.float32 and w.dtype == torch.float32
    f = S.fields(X, smooth=S.smooth)
    for k in ('u', 'v', 'p', 'gdot', 'gap', 'm11', 'm12'):
        assert f[k].dtype == torch.float32, f"{name}: field {k} is {f[k].dtype}"
    loss = (w*f['gap']).sum(); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in S.parameters())
    S._place(torch.float64)
    assert np.all(np.isfinite(S.evaluate(S.quadrature(h=0.25)[0], keys=('gap',))['gap']))


def test_mixed_precision_training_on_cpu():
    """float32 training must run (geometry and rheology tables follow the input dtype), and
    everything after training must be float64"""
    name, mk, bcs, law = _cases()[0]
    dom = mk(); dom.set_bc(bcs)
    S = aq.Solver(dom, law, width=16, depth=2, device='cpu', precision='mixed')
    S.train(iters=30, n=256, lr=3e-3, verbose=0, lbfgs=3, lbfgs_points=512)
    assert len(S.history) > 0 and np.all(np.isfinite(S.history))
    assert all(p.dtype == torch.float64 for p in S.parameters())
    gap = S.certify_gap(h=0.25)
    assert np.isfinite(gap['G_up']) and gap['G_up'] >= 0


@cuda
@pytest.mark.parametrize('case', range(4))
def test_gpu_matches_cpu(case):
    name, mk, bcs, law = _cases()[case]
    d1, d2 = mk(), mk(); d1.set_bc(bcs); d2.set_bc(bcs)
    Sc = aq.Solver(d1, law, width=16, depth=2, device='cpu')
    _randomise(Sc)
    Sg = aq.Solver(d2, law, width=16, depth=2, device='cuda')
    Sg.load_state_dict(Sc.state_dict())
    P, W = Sc.quadrature(h=0.25)
    keys = ('u', 'v', 'p', 'gdot', 'gap')
    fc, fg = Sc.evaluate(P, keys=keys), Sg.evaluate(P, keys=keys)
    # Pointwise comparison away from the walls only. The rule puts points within ~1e-9 of the walls
    # (and into the corners); there the second derivatives of the lift are so ill-conditioned that
    # moving a point by 1e-13 changes gdot by O(1) on the CPU alone, so CPU and GPU rounding cannot
    # agree there. Those points carry weights ~1e-15, and the integrals below are compared on all points.
    dist = np.min([pc.dist(torch.tensor(P[:, 0]), torch.tensor(P[:, 1])).numpy() for pc in d1.pieces_all], 0)
    x0, x1, y0, y1 = d1.bbox
    away = dist > 1e-3*max(x1 - x0, y1 - y0)
    assert away.sum() > 0.1*len(P)
    for k in keys:
        scale = max(1e-12, float(np.abs(fc[k][away]).max()))
        err = float(np.abs(fc[k][away] - fg[k][away]).max())/scale
        assert err < 1e-8, f"{name}: {k} differs by {err:.2e} (relative, away from the walls)"
    for k in ('gap', 'gdot'):
        Ic, Ig = float((W*fc[k]).sum()), float((W*fg[k]).sum())
        assert abs(Ic - Ig) <= 1e-7*max(1e-12, abs(Ic)), f"{name}: integral of {k}: {Ic} on CPU, {Ig} on GPU"
    Gc, Gg = Sc.certify_gap(h=0.25)['G_up'], Sg.certify_gap(h=0.25)['G_up']
    assert abs(Gc - Gg) <= 1e-7*max(1e-12, abs(Gc)), f"{name}: G {Gc} on CPU, {Gg} on GPU"


@cuda
def test_solve_on_gpu():
    name, mk, bcs, law = _cases()[0]
    res = aq.solve(mk(), law, bcs, iters=50, lbfgs=5, width=16, depth=2, verbose=False, device='cuda')
    S = res.solver
    assert S.device.type == 'cuda' and S.train_dtype == torch.float32
    assert all(p.dtype == torch.float64 and p.device.type == 'cuda' for p in S.parameters())
    assert np.isfinite(res.G) and res.G >= 0
    out = res(np.array([0.5, 1.0]), np.array([0.2, -0.4]))
    assert np.all(np.isfinite(out['u']))
    assert 'certificate' in res.summary().lower() or 'AEQUA' in res.summary()
