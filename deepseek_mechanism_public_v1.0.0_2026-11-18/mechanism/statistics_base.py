from .io import require
import numpy as np

def paired(values, clusters, seed=20261102, repeats=2000, family=1):
    """values Nx2 has None/NaN for unobserved outcome; resample whole clusters."""
    a = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    require(a.ndim == 2 and a.shape[1] == 2 and (len(a) == len(clusters)), 'Malformed paired results')
    valid = np.isfinite(a).all(axis=1)
    b = a[valid]
    n = len(a)
    nv = len(b)
    result = {'eligible_n': n, 'paired_valid_n': nv, 'baseline_missing_n': int((~np.isfinite(a[:, 0])).sum()), 'arm_missing_n': int((~np.isfinite(a[:, 1])).sum()), 'paired_missing_n': n - nv, 'baseline_numerator': int(b[:, 0].sum()) if nv else 0, 'arm_numerator': int(b[:, 1].sum()) if nv else 0, 'denominator': nv, 'baseline_pct': float(b[:, 0].mean() * 100) if nv else None, 'arm_pct': float(b[:, 1].mean() * 100) if nv else None, 'difference_pp': float((b[:, 1] - b[:, 0]).mean() * 100) if nv else None, 'shared_component_n': len(set(clusters.tolist())), 'ci_level': 1 - 0.05 / family, 'ci_pp': None, 'bootstrap_valid': 0, 'fixed_centers': True}
    if n:
        alo = np.where(np.isfinite(a), a, 0)
        ahi = np.where(np.isfinite(a), a, 1)
        result['all_eligible_missingness_bounds_pp'] = [float((alo[:, 1] - ahi[:, 0]).mean() * 100), float((ahi[:, 1] - alo[:, 0]).mean() * 100)]
    ids = sorted(set(clusters.tolist()))
    occupied = len(set(clusters[valid].tolist()))
    result['paired_valid_shared_component_n'] = occupied
    result['ci_unavailable_reason'] = 'fewer_than_two_valid_components' if occupied < 2 else None
    if nv and occupied >= 2:
        sums = np.array([(a[(clusters == c) & valid, 1] - a[(clusters == c) & valid, 0]).sum() for c in ids])
        counts = np.array([np.sum((clusters == c) & valid) for c in ids])
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=repeats)
        den = draws @ counts
        good = den > 0
        est = draws[good] @ sums / den[good] * 100
        result['bootstrap_valid'] = int(good.sum())
        if len(est) >= 0.95 * repeats:
            result['ci_pp'] = np.quantile(est, [0.025 / family, 1 - 0.025 / family]).tolist()
        else:
            result['ci_unavailable_reason'] = 'bootstrap_valid_fraction_below_minimum'
    result['few_clusters_warning'] = len(ids) < 10
    return result
