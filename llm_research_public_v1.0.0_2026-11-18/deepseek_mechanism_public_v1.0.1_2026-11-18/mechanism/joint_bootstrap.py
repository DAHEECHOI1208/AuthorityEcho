from collections import Counter
from .validation_base import MAIN
def joint_bootstrap(fv, np, rows, X, frame, p, progress=None):
    """Joint component draws preserve fit/evaluate overlap; refit HR/BA each draw.

    Use sufficient component sums and float64 projections; weighted AUROC counts
    identical scores as half ties. No independent pseudo fit/evaluate split.
    """
    k = len(frame)
    nboot = p['bootstrap_repeats']
    draws = np.random.default_rng(p['bootstrap_seed']).multinomial(k, np.ones(k) / k, size=nboot)
    B, N = fv.basis_for(rows, X, frame)
    Y = X / np.linalg.norm(X, axis=1)[:, None]
    bases = [B[g].reshape(4 * k, -1) for g in range(2)]
    projections = [Y @ b.T for b in bases]
    grams = [b @ b.T for b in bases]
    cluster_index = np.array([frame.index(r['shared_component_id']) for r in rows])
    names = list(fv.COMPARISONS)
    parts = {}
    for name, (positive, negative) in fv.COMPARISONS.items():
        parts[name] = []
        for condition in MAIN:
            ix = np.array([i for i, r in enumerate(rows) if r['condition_id'] == condition and r['group'] in [positive, negative]], dtype=int)
            parts[name].append((ix, np.array([rows[i]['group'] == positive for i in ix], dtype=bool)))
    values = np.full((len(names), nboot), np.nan, dtype=np.float64)
    fit_valid = np.zeros(nboot, dtype=bool)
    missing_fit = Counter()
    missing_eval = Counter()
    for start in range(0, nboot, 64):
        weights = draws[start:start + 64]
        counts = np.einsum('gck,bk->bgc', N, weights)
        valid = np.all(counts > 0, axis=(1, 2))
        for counts_one in counts[~valid]:
            for gi, ci in np.argwhere(counts_one == 0):
                missing_fit[['HR', 'BA'][gi] + '_' + MAIN[ci]] += 1
        ids = np.flatnonzero(valid)
        if len(ids):
            co = (weights[ids, None, None, :] / counts[ids, :, :, None] / 4).reshape(len(ids), 2, 4 * k)
            norm2 = np.stack([np.einsum('bi,ij,bj->b', co[:, g], grams[g], co[:, g]) for g in range(2)], axis=1)
            good = np.isfinite(norm2).all(axis=1) & (norm2 > 0).all(axis=1)
            ids, co, norms = (ids[good], co[good], np.sqrt(norm2[good]))
            score_matrix = projections[0] @ co[:, 0].T / norms[:, 0] - projections[1] @ co[:, 1].T / norms[:, 1]
            for col, offset in enumerate(ids):
                b = start + int(offset)
                fit_valid[b] = True
                row_weights = weights[offset, cluster_index]
                s = score_matrix[:, col]
                for ni, name in enumerate(names):
                    aucs = [fv.weighted_auc(s[ix], labels, row_weights[ix]) for ix, labels in parts[name]]
                    if any((v is None for v in aucs)):
                        missing_eval[name] += 1
                    else:
                        values[ni, b] = float(np.mean(aucs))
        if progress and (start == 0 or (start + 64) % 512 == 0 or start + 64 >= nboot):
            progress(min(start + 64, nboot), nboot)
    result = {}
    for ni, name in enumerate(names):
        vv = values[ni, np.isfinite(values[ni])]
        fraction = len(vv) / nboot
        raw_ci = np.percentile(vv, [2.5, 97.5]).tolist() if len(vv) else None
        ci = raw_ci if fraction >= p['minimum_valid_bootstrap_fraction'] else None
        result[name] = {'ci95': ci, 'percentile95_valid_replicates_aux': raw_ci, 'width': ci[1] - ci[0] if ci else None, 'valid_replicates': len(vv), 'total_replicates': nboot, 'valid_fraction': fraction, 'unstable_ci': fraction < p['minimum_valid_bootstrap_fraction'], 'minimum_valid_fraction': p['minimum_valid_bootstrap_fraction'], 'fit_invalid_replicates': int((~fit_valid).sum()), 'fit_missing_by_group_condition': dict(missing_fit), 'missing_evaluation_class_replicates': missing_eval[name], 'CI_scope': 'apparent_same_sample_candidatewise_not_selection_adjusted'}
    return (result, values, fit_valid)
