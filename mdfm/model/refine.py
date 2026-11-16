"""Refinement of the amortized posterior (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5 and departure 7
of section 1.5, task B4.15; design section 4 and architecture note section 4, item 3).

refine(model, post, batch, steps) moves the posterior means (z, beta) and every nuisance tensor by `steps`
Adam steps (lr 0.05 [cfg]) on -sum_w log_joint (mfm.MFM.log_joint), with every model parameter frozen:
gradients are taken with torch.autograd.grad with respect to the refined variables only, so no parameter's
.grad is written and the optimizer of the model never sees a refinement step. Stopped gradients: nothing
is unrolled, the result is detached; the encoder's variances (logvar_z, logvar_b) are kept. Steps: 5 in
training (phase B, every N-th optimizer step) and 20 at evaluation (design section 4); points per stream
and step: 64 in training, 256 at evaluation [cfg] (Batch.subsample, fresh draws each step, weights rescaled
so each step's objective estimates the full one).

Adam's step is a length in the units of each variable, so the variables are stepped in natural units:
- rotations (the IMU mount residual and the camera rotation, 6D forms) in their tangent space: a rotation
  vector omega (rad), started at 0 and composed on the right of the starting rotation, R0 Exp(omega), 3
  coordinates for 3 degrees of freedom where the 6D form has 6;
- the IMU biases in units of their device prior's standard deviation (MFM.nuisance_scales: 0.02 g for the
  acc biases, 0.223 to 2.104 deg/s for the gyro biases of the 13 xsens13 sites), not in g and deg/s;
- every other variable in its own unit (latents, betas, log scales, bounded residuals).
Adam runs with betas BETAS = (0.5, 0.9) [cfg]: a run of 5 to 20 steps needs little momentum, and the
default (0.9, 0.999) overshoots (the head and trunk mounts of a walk turned away before coming back). The
choice was made on random draws of the 20 degree mount test of tests/test_refine.py.

With a body given (a KnotBody, for example the truth knots of a synthetic window), z and beta are neither
read nor moved: only the nuisances move, and the body's kinematics are computed once, for every kernel
point, since they do not depend on the refined variables (all points are used). This is the mode of the
tests that run before any training: a mount 20 degrees off its prior centre is refined back.

straight_through(post, refined) is the training form: z_st = z0 + sg(zK - z0) for every refined mean, so
the decoder and the kernels are scored at the refined values while the gradient reaches the encoder
through the amortized ones (the amortization loss of spec section 4 pulls mu_z toward sg(zK)).
"""
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_rot6d, rot6d_to_matrix
from ..kernels.nuisance import identity6d
from .mfm import Posterior

LR = 0.05             # [cfg]
BETAS = (0.5, 0.9)    # [cfg] Adam momentum of the refinement
STEPS_TRAIN = 5
STEPS_EVAL = 20
POINTS_TRAIN = 64     # [cfg] kernel points per stream-window and step
POINTS_EVAL = 256     # [cfg]
# 6D fields stepped in the tangent space: name -> True when the field is a residual on the identity
ROTATIONS = {"imu.mount_res6d": True, "camera.rot6d": False}


def _codec(name, value, scales):
    """(u0, to_value): the start of the refined variable and the map from it to the nuisance field."""
    if name in ROTATIONS:
        e6 = identity6d(value.dtype, value.device) if ROTATIONS[name] else torch.zeros(6, dtype=value.dtype,
                                                                                         device=value.device)
        R0 = rot6d_to_matrix(value.detach() + e6)
        return (torch.zeros(value.shape[:-1] + (3,), dtype=value.dtype, device=value.device),
                lambda u: matrix_to_rot6d(R0 @ axis_angle_to_matrix(u)) - e6)
    s = scales.get(name)
    if s is None:
        return value.detach().clone(), lambda u: u
    return value.detach() / s, lambda u: u * s


def refine(model, post, batch, steps, lr=LR, points_per_stream=None, body=None, generator=None, history=None,
           betas=BETAS):
    """Posterior with refined means (module docstring), detached. post: the amortized Posterior; batch: the
    mfm.Batch it came from; points_per_stream: None for every point, else the draw per stream-window and
    step (a torch.Generator on any device may be given); body: a KnotBody held fixed; history: a list that
    receives the objective -sum_w log_joint (float) before each step; betas: Adam's. steps = 0 returns post
    detached."""
    post = post.detach()
    if steps <= 0:
        return post
    scales = model.nuisance_scales(post.nuis, batch)
    codecs = {k: _codec(k, v, scales) for k, v in post.nuis.tensors().items()}
    variables = {k: u0.requires_grad_(True) for k, (u0, _) in codecs.items()}
    decode = lambda: {k: codecs[k][1](u) for k, u in variables.items()}  # noqa: E731
    z = beta = None
    if body is None:
        z = post.mu_z.clone().requires_grad_(True)
        beta = post.mu_b.clone().requires_grad_(True)
    params = ([z, beta] if body is None else []) + list(variables.values())
    opt = torch.optim.Adam(params, lr=lr, betas=betas)
    with torch.enable_grad():
        kin = None
        if body is not None:
            with torch.no_grad():
                kin = model.kinematics(body, batch)
        for _ in range(int(steps)):
            full = points_per_stream is None or body is not None
            b = batch if full else batch.subsample(points_per_stream, generator)
            loss = -model.log_joint(z, beta, post.nuis.with_tensors(decode()), b, body=body, kin=kin).sum()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for p, g in zip(params, grads):
                p.grad = torch.zeros_like(p) if g is None else g
            opt.step()
            if history is not None:
                history.append(float(loss.detach()))
        with torch.no_grad():
            nuis = post.nuis.with_tensors({k: v.detach() for k, v in decode().items()})
    return Posterior(post.mu_z if z is None else z.detach(), post.logvar_z,
                     post.mu_b if beta is None else beta.detach(), post.logvar_b, nuis, post.mask)


def straight_through(post, refined):
    """Posterior whose means are post's plus the detached refinement (refined - post): refined values
    forward, the amortized means' gradient backward; post's variances (with their gradient) kept."""
    st = lambda a, b: a + (b - a).detach()  # noqa: E731
    ref = refined.nuis.tensors()
    nuis = post.nuis.with_tensors({k: st(v, ref[k]) for k, v in post.nuis.tensors().items()})
    return Posterior(st(post.mu_z, refined.mu_z), post.logvar_z, st(post.mu_b, refined.mu_b), post.logvar_b, nuis,
                     post.mask)
