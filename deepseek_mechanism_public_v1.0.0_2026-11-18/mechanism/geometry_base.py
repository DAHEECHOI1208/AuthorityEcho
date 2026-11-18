import hashlib
import numpy as np
from .io import require

def vector(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 1 and len(x) > 1 and np.isfinite(x).all(), 'Invalid vector')
    return x

def unit(x):
    x = vector(x)
    n = np.linalg.norm(x)
    require(n > 1e-12, 'Degenerate direction/center')
    return x / n

def direction(mu):
    mu = np.asarray(mu)
    require(mu.ndim == 2 and mu.shape[0] == 3, 'Centers must be HR,HA,BA')
    return unit(unit(mu[1]) - unit(mu[0]))

def scores(h, mu):
    c = np.asarray([np.dot(unit(h), unit(m)) for m in mu], dtype=np.float64)
    return dict(cos_HR=float(c[0]), cos_HA=float(c[1]), cos_BA=float(c[2]), g=float(c[1] - c[0]), f=float(c[0] - c[2]), norm=float(np.linalg.norm(h)))

def restore(h, donor, u, alpha):
    h, donor, u = (vector(h), vector(donor), unit(u))
    require(h.shape == donor.shape == u.shape and np.isfinite(alpha) and (0 <= alpha <= 1), 'Shape or alpha outside preregistered [0,1]')
    delta = float(alpha) * np.dot(donor - h, u) * u
    return (h + delta, delta)

def bf16(x):
    """IEEE round-to-nearest-even after float64->float32, widened to float32.

    GPU hook verifies this CPU oracle against the actual torch BF16 conversion.
    Nonfinite/subnormal ambiguity is rejected by finite-value checks at use.
    """
    a = np.asarray(x, dtype=np.float32)
    require(np.isfinite(a).all(), 'Nonfinite FP32 writeback')
    b = a.view(np.uint32)
    out = (b + np.uint32(32767) + (b >> 16 & 1) & np.uint32(4294901760)).view(np.float32)
    require(np.isfinite(out).all(), 'Nonfinite BF16 writeback')
    return out

def random_direction(u, seed, identity, repeat):
    text = f'{seed}|{identity}|{repeat}'.encode()
    derived = int.from_bytes(hashlib.sha256(text).digest()[:16], 'little')
    rng = np.random.Generator(np.random.PCG64(derived))
    v = rng.standard_normal(len(u))
    u = unit(u)
    v = unit(v - np.dot(v, u) * u)
    return (v, str(derived))

def matched_random(h, u, norm, seed, identity, repeat, steps=48):
    """One preregistered direction. Match *post-BF16* norm by scalar bisection.

    The same direction is never resampled on poor matching; ties choose the smaller
    scalar. Quantization need not permit an exact norm. No behavioral outcomes enter.
    """
    h = vector(h)
    require(norm >= 0 and np.isfinite(norm), 'Invalid norm target')
    v, derived = random_direction(u, seed, identity, repeat)
    require(np.array_equal(bf16(h).astype(np.float64), h), 'Baseline must be actual BF16 values')
    if norm == 0:
        return (h.copy(), np.zeros_like(h), {'derived_seed': derived, 'scalar': 0.0, 'norm_target': 0.0, 'norm_error': 0.0, 'relative_norm_error': 0.0, 'direction_dot_u': float(v @ u)})

    def candidate(s):
        y = bf16(h + s * v).astype(np.float64)
        return (abs(np.linalg.norm(y - h) - norm), s, y)
    lo, hi = (0.0, max(norm * 2, 1e-12))
    for _ in range(32):
        if np.linalg.norm(candidate(hi)[2] - h) >= norm:
            break
        hi *= 2
    require(np.linalg.norm(candidate(hi)[2] - h) >= norm, 'Random norm bracket failed')
    best = min([candidate(lo), candidate(hi), candidate(norm)], key=lambda x: (x[0], x[1]))
    for _ in range(steps):
        mid = (lo + hi) / 2
        c = candidate(mid)
        best = min([best, c], key=lambda x: (x[0], x[1]))
        if np.linalg.norm(c[2] - h) < norm:
            lo = mid
        else:
            hi = mid
    err, s, y = best
    return (y, s * v, {'derived_seed': derived, 'scalar': float(s), 'norm_target': float(norm), 'norm_error': float(np.linalg.norm(y - h) - norm), 'relative_norm_error': float(err / norm), 'direction_dot_u': float(v @ u)})

def effect(h, donor, mu, alpha, mode, seed=0, identity='', repeat=0, random_norm_target=None):
    h, donor = (vector(h), vector(donor))
    u = direction(mu)
    proposed, restore_delta = restore(h, donor, u, alpha)
    restoration_actual = bf16(proposed).astype(np.float64)
    target_norm = float(np.linalg.norm(restoration_actual - h))
    extra = {}
    if mode in ['sham', 'baseline']:
        proposed, intended = (h.copy(), np.zeros_like(h))
        y = h.copy()
    elif mode == 'restore':
        intended = restore_delta
        y = restoration_actual
    elif mode == 'random':
        if random_norm_target is not None:
            target_norm = float(random_norm_target)
        y, intended, extra = matched_random(h, u, target_norm, seed, identity, repeat)
        proposed = h + intended
    else:
        raise ValueError('Unknown arm')
    actual = y - h
    before, after = (scores(h, mu), scores(y, mu))
    rec = dict(mode=mode, alpha=float(alpha), projection_h=float(h @ u), projection_donor=float(donor @ u), projection_after=float(y @ u), signed_restore_coefficient=float(alpha * ((donor - h) @ u)), intended_delta_norm=float(np.linalg.norm(intended)), actual_delta_norm=float(np.linalg.norm(actual)), rounding_error_norm=float(np.linalg.norm(actual - intended)), intended_orthogonal_delta_norm=float(np.linalg.norm(intended - intended @ u * u)), actual_orthogonal_delta_norm=float(np.linalg.norm(actual - actual @ u * u)), actual_delta_dot_u=float(actual @ u), actual_delta_cos_u=float(actual @ u / np.linalg.norm(actual)) if np.linalg.norm(actual) > 0 else None, actual_cosine_to_before=float(unit(y) @ unit(h)), before=before, after=after, delta_g=after['g'] - before['g'], delta_f=after['f'] - before['f'], restoration_actual_norm_target=target_norm, **extra)
    return (y.astype(np.float32), rec, {'before': h.astype(np.float32), 'donor': donor.astype(np.float32), 'intended_delta': intended, 'actual_delta': actual, 'after': y.astype(np.float32), 'u': u})
