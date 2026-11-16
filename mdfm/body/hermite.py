"""Cubic Hermite interpolation on a uniform clock, differentiable in the query time.

The body lives on a 100 Hz internal clock (design section 1); sensors sample it at their own
timestamps. Tangents are Catmull-Rom central differences (one-sided at the ends), so the
interpolant is C1 and its first and second time derivatives exist almost everywhere; they
are taken by automatic differentiation through this function.

hermite() reads one sequence (T, ...). hermite_gather() reads a batch of windows (W, T, ...)
at flat queries (window w, time t in s from that window's start): the value at t reads knots
i - 1 to i + 2 of window w only (support() lists them), so knots outside every support are
never read. Both clamp queries outside the sampled span [0, L - 1] knots to its nearest end
(position()): the value there is the end knot's. A query on an end knot, or within 1e-3 knot
outside it (the rounding of t x rate at a window's first or last knot), keeps the time
derivatives of the end segment, in float32 and float64 alike; farther out the derivatives are
zero. The segment index is piecewise constant, only the fractional position is a
differentiable function of t. The batched path is split so that a caller can gather the knots
once and evaluate the time-dependent part under forward-mode transforms
(mdfm.body.kinematics): position, segment, support_of, basis_matrix, gather_knots, weights,
combine.
"""
import torch

# knots outside the span within which a clamped query keeps the end segment's derivatives: the
# rounding of t x rate at an end knot (float32 at 10 s and 100 Hz: about 1e-4 knot)
_TOL = 1e-3
# Hermite basis h00, h10, h01, h11 (columns) as polynomials in s (rows: powers 0 to 3)
_H = ((1.0, 0.0, 0.0, 0.0),
      (0.0, 1.0, 0.0, 0.0),
      (-3.0, -2.0, 3.0, -1.0),
      (2.0, 1.0, -2.0, 1.0))


def hermite(y, t, rate=100.0, t0=0.0):
    """Interpolate samples y (T, ...) taken at t0 + i / rate at query times t (N,) -> (N, ...).

    Queries outside the sampled span are clamped to it (position()). The fractional position is
    a differentiable function of t, the segment index is not (it is piecewise constant)."""
    T = y.shape[0]
    h = 1.0 / rate
    m = torch.cat([(y[1:2] - y[0:1]) / h, (y[2:] - y[:-2]) / (2 * h), (y[-1:] - y[-2:-1]) / h], 0)
    u = position(t - t0, rate, T)
    i = torch.floor(u.detach()).long().clamp(0, T - 2)
    s = u - i.to(u.dtype)
    shape = (-1,) + (1,) * (y.dim() - 1)
    s = s.reshape(shape)
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


# ---------------------------------------------------------------- the batched path

def check_queries(W, T, w, t, lengths=None):
    """Raise ValueError unless T >= 2, w (N,) is int64, t (N,) is floating, the ids of a CPU w lie
    in [0, W) and lengths is None or (W,) integers in [2, T]; return lengths as int64 on the
    device of w (or None). The ids of a GPU w are not read back (no device synchronization per
    call): an id outside [0, W) stops the gather with a device-side assertion."""
    if w.dim() != 1 or t.shape != w.shape:
        raise ValueError(f"w and t must be (N,) of one length, got {tuple(w.shape)} and {tuple(t.shape)}")
    if w.dtype != torch.int64:
        raise ValueError(f"window ids must be int64, got {w.dtype}")
    if not t.is_floating_point():
        raise ValueError(f"times must be floating point seconds, got {t.dtype}")
    if T < 2:
        raise ValueError(f"at least 2 knots per window, got {T}")
    if w.device.type == "cpu" and w.numel() and bool(((w < 0) | (w >= W)).any()):
        raise ValueError(f"window ids outside [0, {W})")
    if lengths is None:
        return None
    lengths = torch.as_tensor(lengths, device=w.device)
    if tuple(lengths.shape) != (W,) or lengths.is_floating_point() or lengths.dtype == torch.bool:
        raise ValueError(f"lengths must be ({W},) integers, got {tuple(lengths.shape)} {lengths.dtype}")
    if bool(((lengths < 2) | (lengths > T)).any()):
        raise ValueError(f"lengths must lie in [2, {T}]")
    return lengths.to(torch.int64)


def knots_per_query(T, w, lengths=None):
    """Valid knots of each query's window: the int T when lengths is None, else lengths[w]
    (N,) int64."""
    return T if lengths is None else lengths.to(device=w.device, dtype=torch.int64)[w]


def position(t, rate, L):
    """Knot position u = t * rate (N,) clamped to [0, L - 1]; L an int or (N,) int64. Its time
    derivative is rate inside the span, on its ends and within 1e-3 knot outside them (so a
    query on an end knot keeps the end segment's derivatives whichever way t x rate rounds),
    and zero farther out. The value outside the span is exactly the end."""
    u = t * rate
    ud = u.detach()
    if torch.is_tensor(L):
        c = torch.minimum(ud.clamp(min=0), (L - 1).to(u.dtype))
    else:
        c = ud.clamp(0, L - 1)
    # d is 0 inside the span and exact within the tolerance outside it (Sterbenz), so u - d there
    # equals the end knot exactly while carrying the tangent of u
    d = ud - c
    return torch.where(d.abs() <= _TOL, u - d, c)


def segment(u, L):
    """Segment i (N,) int64 in [0, L - 2] of the clamped positions u (not differentiable)."""
    i = torch.floor(u.detach()).long()
    if torch.is_tensor(L):
        return torch.minimum(i.clamp(min=0), L - 2)
    return i.clamp(0, L - 2)


def support_of(i, L):
    """Segment i (N,) -> (N, 4) knot ids i - 1, i, i + 1, i + 2 clamped to [0, L - 1]."""
    k = i[:, None] + torch.arange(-1, 3, device=i.device)
    last = (L - 1)[:, None] if torch.is_tensor(L) else torch.tensor(L - 1, device=i.device)
    return torch.minimum(k.clamp(min=0), last)


def support(t, T, rate=100.0):
    """Knot ids (N, 4) int64 that the value at times t (N,) s from the window start reads: i - 1,
    i, i + 1, i + 2 clamped to [0, T - 1], i the segment of t (queries outside the span clamped
    by position()). T: knots per window, an int or (N,) int64 per query."""
    L = T.to(device=t.device, dtype=torch.int64) if torch.is_tensor(T) else int(T)
    return support_of(segment(position(t.detach(), rate, L), L), L)


def basis_matrix(k, dtype):
    """B (N, 4, 4) with value = [1, s, s^2, s^3] @ B @ (the four support knots k (N, 4)):
    Catmull-Rom tangents times h, (y[k2] - y[k0]) / (k2 - k0) at knot i and (y[k3] - y[k1]) /
    (k3 - k1) at i + 1 (central inside, one-sided at the window's first and last knot)."""
    d0 = (k[:, 2] - k[:, 0]).to(dtype)
    d1 = (k[:, 3] - k[:, 1]).to(dtype)
    z, o = torch.zeros_like(d0), torch.ones_like(d0)
    # rows: y_i, h m_i, y_(i+1), h m_(i+1) as combinations of the four support knots
    A = torch.stack([torch.stack([z, o, z, z], -1),
                     torch.stack([-1 / d0, z, 1 / d0, z], -1),
                     torch.stack([z, z, o, z], -1),
                     torch.stack([z, -1 / d1, z, 1 / d1], -1)], -2)
    H = torch.tensor(_H, dtype=dtype, device=k.device)
    return (H[:, :, None] * A[:, None, :, :]).sum(-2)


def gather_knots(y, w, k):
    """Knots y (W, T, ...) of the supports k (N, 4) in windows w (N,) -> (N, 4, ...); the
    gradient flows back to y."""
    W, T = y.shape[:2]
    flat = (w[:, None] * T + k).reshape(-1)
    return y.reshape((W * T,) + y.shape[2:]).index_select(0, flat).reshape(k.shape + y.shape[2:])


def weights(u, i, B):
    """Weights (N, 4) of the four support knots at positions u (N,) of segments i. Products of
    tiny matrices here and below are broadcast sums, not batched GEMM, which runs far below
    bandwidth on 1 x 4 rows on the GPU."""
    s = u - i.to(u.dtype)
    s2 = s * s
    powers = torch.stack([torch.ones_like(s), s, s2, s2 * s], -1)
    return (powers[:, :, None] * B).sum(1)


def combine(c, K):
    """Weights c (N, 4) times support knots K (N, 4, ...) -> (N, ...)."""
    N = K.shape[0]
    return (c[:, :, None] * K.reshape(N, 4, -1)).sum(1).reshape((N,) + K.shape[2:])


def hermite_gather(y, w, t, rate=100.0, lengths=None):
    """Interpolate the knots y (W, T, ...) of W windows, each sampled at k / rate from its own
    start, at flat queries: window w (N,) int64 and time t (N,) s from that window's start ->
    (N, ...) in the dtype of y. Catmull-Rom tangents with one-sided ends and the clamp of
    hermite(), per window; the value at t reads knots i - 1 to i + 2 of window w only
    (support()). lengths (W,) int64 gives the valid knots of each window in a padded buffer
    (each at least 2; default T): the end tangent and the clamp then use the window's own last
    knot, so the result and its time derivatives equal those of hermite(y[w, :lengths[w]], t).
    ValueError on bad ids, shapes or lengths (check_queries)."""
    lengths = check_queries(y.shape[0], y.shape[1], w, t, lengths)
    L = knots_per_query(y.shape[1], w, lengths)
    u = position(t.to(y.dtype), rate, L)
    i = segment(u, L)
    k = support_of(i, L)
    return combine(weights(u, i, basis_matrix(k, y.dtype)), gather_knots(y, w, k))
