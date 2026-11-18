import collections
import numpy as np
MAIN = ['C2','C3-E','C3-P','C4']
COMPARISONS = {'H_R':('HR','BR'),'H_A':('HA','BA'),'R_H':('HR','HA'),'R_B':('BR','BA')}
from .io import ROOT,read
DEFAULT_PROTOCOL = read(ROOT/'configs/f_protocol.json')
import numpy as np

def weighted_auc(scores, positive, weights):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive, dtype=bool)
    w = np.asarray(weights, dtype=float)
    keep = w > 0
    s = s[keep]
    y = y[keep]
    w = w[keep]
    p = w[y].sum()
    n = w[~y].sum()
    if p == 0 or n == 0:
        return None
    if not np.isfinite(s).all():
        raise ValueError('Invalid measurement: nonfinite score')
    order = np.argsort(s, kind='stable')
    s = s[order]
    y = y[order]
    w = w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    pw = np.add.reduceat(w * y, starts)
    nw = np.add.reduceat(w * ~y, starts)
    return float(np.sum(pw * (np.cumsum(nw) - nw + 0.5 * nw)) / (p * n))

def metrics(rs, scores, weights=None, conditions=MAIN):
    w = np.ones(len(rs)) if weights is None else np.asarray(weights)
    out = {}
    for name, (pos, neg) in COMPARISONS.items():
        cond = {}
        for c in conditions:
            ids = [i for i, r in enumerate(rs) if r['condition_id'] == c and r['group'] in [pos, neg]]
            cond[c] = weighted_auc(np.asarray(scores)[ids], [rs[i]['group'] == pos for i in ids], w[ids])
        ids = [i for i, r in enumerate(rs) if r['condition_id'] in conditions and r['group'] in [pos, neg]]
        available = [v for v in cond.values() if v is not None]
        out[name] = {'conditions': cond, 'macro': float(np.mean(available)) if len(available) == len(conditions) else None, 'status': 'estimable' if len(available) == len(conditions) else 'not_estimable', 'partial_condition_mean_aux': float(np.mean(available)) if available else None, 'pooled_aux': weighted_auc(np.asarray(scores)[ids], [rs[i]['group'] == pos for i in ids], w[ids])}
    return out

def basis_for(fit, X, clusters):
    """Sums of parent means for HR/BA x condition x shared component."""
    k = len(clusters)
    index = {c: i for i, c in enumerate(clusters)}
    B = np.zeros((2, 4, k, X.shape[1]), dtype=np.float64)
    N = np.zeros((2, 4, k), dtype=float)
    buckets = collections.defaultdict(list)
    for i, r in enumerate(fit):
        if r['group'] in ['HR', 'BA'] and r['condition_id'] in MAIN:
            buckets[r['parent_question_id'], r['group'], r['condition_id'], r['shared_component_id']].append(i)
    for (parent, g, c, component), ids in buckets.items():
        gi = ['HR', 'BA'].index(g)
        ci = MAIN.index(c)
        ki = index[component]
        B[gi, ci, ki] += X[ids].mean(axis=0)
        N[gi, ci, ki] += 1
    return (B, N)

def coefficients(N, w):
    counts = (N * w[None, None, :]).sum(axis=2)
    if np.any(counts == 0):
        return None
    return w[None, None, :] / counts[:, :, None] / 4

def centers(fit, X, clusters=None, weights=None):
    clusters = clusters or sorted({r['shared_component_id'] for r in fit})
    B, N = basis_for(fit, X, clusters)
    c = coefficients(N, np.ones(len(clusters)) if weights is None else weights)
    if c is None:
        return None
    mu = (B * c[:, :, :, None]).sum(axis=(1, 2))
    norm = np.linalg.norm(mu, axis=1)
    if np.any(norm == 0) or not np.isfinite(mu).all():
        return None
    return mu

def cosine_scores(X, mu):
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms == 0) or not np.isfinite(X).all():
        raise ValueError('Invalid activation vectors')
    return X / norms[:, None] @ (mu[0] / np.linalg.norm(mu[0]) - mu[1] / np.linalg.norm(mu[1]))

def bootstrap(fit, X, ev, Y, nboot, seed, fit_clusters, eval_clusters, minimum_valid_fraction=None):
    """Resample whole components in each role independently; refit all four centers.
Projection via sufficient sums is algebraically equivalent to materializing draws.
"""
    minimum_valid_fraction = DEFAULT_PROTOCOL['minimum_valid_bootstrap_fraction'] if minimum_valid_fraction is None else minimum_valid_fraction
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError('Invalid minimum bootstrap fraction')
    B, N = basis_for(fit, X, fit_clusters)
    k = len(fit_clusters)
    q = len(eval_clusters)
    rng = np.random.default_rng(seed)
    fw = rng.multinomial(k, np.ones(k) / k, size=nboot)
    ew = rng.multinomial(q, np.ones(q) / q, size=nboot)
    idx = np.array([eval_clusters.index(r['shared_component_id']) for r in ev])
    Ynorm = Y / np.linalg.norm(Y, axis=1)[:, None]
    projections = [Ynorm @ B[g].reshape(4 * k, -1).T for g in range(2)]
    grams = [B[g].reshape(4 * k, -1) @ B[g].reshape(4 * k, -1).T for g in range(2)]
    comparisons = {}
    for name, (pos, neg) in COMPARISONS.items():
        comparisons[name] = []
        for cond in MAIN:
            ids = np.array([i for i, r in enumerate(ev) if r['condition_id'] == cond and r['group'] in [pos, neg]], dtype=int)
            comparisons[name].append((ids, np.array([ev[i]['group'] == pos for i in ids], dtype=bool)))
    samples = {n: [] for n in COMPARISONS}
    fit_invalid = 0
    eval_missing = collections.Counter()
    fit_missing_by_group_condition = collections.Counter()
    for b in range(nboot):
        co = coefficients(N, fw[b])
        if co is None:
            fit_invalid += 1
            for gi, ci in np.argwhere((N * fw[b][None, None, :]).sum(axis=2) == 0):
                fit_missing_by_group_condition[['HR', 'BA'][gi] + '_' + MAIN[ci]] += 1
            continue
        co = co.reshape(2, 4 * k)
        norms = [np.sqrt(max(0, float(co[g] @ grams[g] @ co[g]))) for g in range(2)]
        if min(norms) == 0:
            fit_invalid += 1
            continue
        s = projections[0] @ co[0] / norms[0] - projections[1] @ co[1] / norms[1]
        weights = ew[b, idx]
        for name, parts in comparisons.items():
            vals = [weighted_auc(s[ids], labels, weights[ids]) for ids, labels in parts]
            if any((v is None for v in vals)):
                eval_missing[name] += 1
            else:
                samples[name].append(float(np.mean(vals)))
    out = {}
    for name, vs in samples.items():
        ci = np.percentile(vs, [2.5, 97.5]).tolist() if vs else None
        out[name] = {'ci95': ci, 'width': ci[1] - ci[0] if ci else None, 'valid_replicates': len(vs), 'valid_fraction': len(vs) / nboot, 'unstable_ci': len(vs) / nboot < minimum_valid_fraction, 'minimum_valid_fraction': minimum_valid_fraction, 'missing_evaluation_class_replicates': eval_missing[name], 'missing_evaluation_class_fraction': eval_missing[name] / nboot, 'fit_invalid_replicates': fit_invalid, 'fit_invalid_fraction': fit_invalid / nboot, 'fit_missing_by_group_condition': dict(fit_missing_by_group_condition), 'total_replicates': nboot}
    return out

def ci_issues(m, protocol):
    issues = []
    ci = m.get('ci95')
    macro = m.get('macro')
    if macro is None or not np.isfinite(macro):
        issues.append('missing_or_invalid_macro')
    if ci is None or len(ci) != 2 or (not np.isfinite(ci).all()) or (ci[0] > ci[1]):
        issues.append('missing_or_invalid_CI')
    fraction = m.get('valid_fraction')
    total = m.get('total_replicates')
    valid = m.get('valid_replicates')
    if fraction is None or total is None or valid is None or (total <= 0) or (not 0 <= valid <= total) or (not np.isfinite(fraction)) or (not np.isclose(fraction, valid / total, rtol=0, atol=1e-12)):
        issues.append('missing_or_invalid_bootstrap_evidence')
    elif fraction < protocol['minimum_valid_bootstrap_fraction']:
        issues.append('bootstrap_valid_fraction_below_minimum')
    if m.get('unstable_ci') is not False:
        issues.append('unstable_CI')
    return issues

def choose(candidates, kind, tolerance=None, protocol=None):
    protocol = protocol or DEFAULT_PROTOCOL
    tolerance = protocol['selection_tie_tolerance'] if tolerance is None else tolerance
    names = ['H_R', 'H_A'] if kind == 'H' else ['R_H', 'R_B']
    ranked = []
    excluded = []
    for c in candidates:
        stats = c.get('comparisons', {})
        issues = {n: ci_issues(stats.get(n, {}), protocol) for n in names}
        if any(issues.values()):
            excluded.append({'layer': c['layer'], 'position': c['position'], 'reason': 'required_selection_evidence_ineligible', 'comparison_reasons': issues})
            continue
        ranked.append((min((stats[n]['ci95'][0] for n in names)), c))
    if not ranked:
        return {'status': 'inconclusive', 'candidate': None, 'reason': 'no_eligible_select_candidate', 'excluded': excluded}
    best = max((v for v, c in ranked))
    ties = [(v, c) for v, c in ranked if best - v <= tolerance + 1e-12]
    v, c = min(ties, key=lambda item: (0 if item[1]['position'] == 't_C' else 1, item[1]['layer']))
    return {'status': 'selected', 'candidate': {'layer': c['layer'], 'position': c['position']}, 'objective': v, 'max_objective': best, 'tie_tolerance': tolerance, 'excluded': excluded, 'minimum_valid_bootstrap_fraction': protocol['minimum_valid_bootstrap_fraction'], 'selection_uses_only': 'calibration_fit_and_readout_select'}

ROLES=['calibration_fit','readout_select','readout_check']
GROUPS=['HR','HA','BR','BA']
def support(rs, protocol=None):
    protocol = protocol or DEFAULT_PROTOCOL
    out = []
    for role in ROLES:
        rr = [r for r in rs if r['dev_role'] == role]
        for c in ['ALL'] + MAIN:
            for group in ['HR', 'HA', 'BR', 'BA']:
                subset = [r for r in rr if r['group'] == group and (c == 'ALL' or r['condition_id'] == c)]
                n = len({r['parent_question_id'] for r in subset})
                k = len({r['shared_component_id'] for r in subset})
                target = (protocol['fit_min_parents_each_center'] if role == 'calibration_fit' and group in ['HR', 'BA'] else protocol['select_check_min_parents_each_group']) if c == 'ALL' else protocol['per_condition_min_parents_each_class']
                out.append({'dev_role': role, 'condition_id': c, 'group': group, 'rows': len(subset), 'parents': n, 'components': k, 'parent_target': target, 'low_support': n < target, 'component_warning': k < (protocol['component_warning_group'] if c == 'ALL' else protocol['component_warning_condition']), 'component_count_not_a_sufficiency_test': True})
    return out
def evaluate_candidate(fit, X, ev, Y, protocol, fit_clusters, eval_clusters):
    mu = centers(fit, X, fit_clusters)
    if mu is None:
        return ({'status': 'not_estimable', 'reason': 'missing_fit_HR_or_BA_in_required_condition'}, None)
    s = cosine_scores(Y, mu)
    point = metrics(ev, s)
    boot = bootstrap(fit, X, ev, Y, protocol['bootstrap_repeats'], protocol['bootstrap_seed'], fit_clusters, eval_clusters, protocol['minimum_valid_bootstrap_fraction'])
    return ({'status': 'evaluated', 'comparisons': {k: {**point[k], **boot[k]} for k in COMPARISONS}}, mu)
def verdict_details(kind, result, rs, protocol):
    names = ['H_R', 'H_A'] if kind == 'H' else ['R_H', 'R_B']
    sp = support([r for r in rs if r['condition_id'] in MAIN], protocol)
    reasons = []
    precision = []
    performance = []
    reversals = []
    needed = [x for x in sp if x['condition_id'] == 'ALL' and (x['dev_role'] != 'calibration_fit' or x['group'] in ['HR', 'BA'])]
    overall_low = [x for x in needed if x['low_support']]
    condition_low = [x for x in sp if x['condition_id'] != 'ALL' and x['low_support'] and (x['dev_role'] != 'calibration_fit' or x['group'] in ['HR', 'BA'])]
    if overall_low:
        reasons.append('overall_parent_support_below_target')
    if condition_low:
        reasons.append('per_condition_parent_support_below_target')
    if result.get('status') != 'evaluated':
        precision.append({'reason': 'candidate_not_evaluated'})
    else:
        for name in names:
            m = result.get('comparisons', {}).get(name, {})
            precision.extend(({'comparison': name, 'reason': why} for why in ci_issues(m, protocol)))
            ci = m.get('ci95')
            macro = m.get('macro')
            if ci is not None and len(ci) == 2 and np.isfinite(ci).all():
                if ci[1] - ci[0] > protocol['ci_width_max'] + 1e-12:
                    precision.append({'comparison': name, 'reason': 'CI_width_exceeds_maximum', 'observed': ci[1] - ci[0]})
                if ci[0] < protocol['ci_lower_min']:
                    performance.append({'comparison': name, 'reason': 'CI_lower_below_target'})
            if macro is not None and np.isfinite(macro) and (macro < protocol['auroc_min']):
                performance.append({'comparison': name, 'reason': 'AUROC_below_target'})
            for c, value in m.get('conditions', {}).items():
                classcounts = [x for x in sp if x['dev_role'] == 'readout_check' and x['condition_id'] == c and (x['group'] in COMPARISONS[name])]
                if value is not None and value < 0.5 and (len(classcounts) == 2) and all((x['parents'] >= protocol['per_condition_min_parents_each_class'] for x in classcounts)):
                    reversals.append({'comparison': name, 'condition': c, 'AUROC': value})
    if precision or overall_low:
        status = 'inconclusive'
    elif reversals:
        status = 'condition_dependent_adoption_withheld'
    elif condition_low:
        status = 'low_support'
    elif performance:
        status = 'not_supported'
    else:
        status = 'exploratory_numerical_support_only'
    return {'status': status, 'reason_codes': reasons + sorted({x['reason'] for x in precision + performance}) + (['well_supported_condition_reversal'] if reversals else []), 'precision_reasons': precision, 'performance_reasons': performance, 'condition_reversals': reversals, 'overall_low_support': overall_low, 'per_condition_low_support': condition_low, 'component_warnings': [x for x in sp if x['component_warning']], 'low_support': bool(overall_low or condition_low), 'general_adoption_withheld': status != 'exploratory_numerical_support_only', 'numerical_criteria_met': not (precision or performance or reversals), 'causal_or_independent_HR_validation': False, 'priority': 'precision_or_overall_support > supported_reversal > condition_support_warning > performance > exploratory_only'}
