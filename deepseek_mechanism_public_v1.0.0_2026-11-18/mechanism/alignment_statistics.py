import itertools
from collections import Counter
import numpy as np
from .io import require
CONDITIONS=("sham","restore","random","full_swap")
LAYERS=(12,13,14,15,16)
GAP_EPS=4e-5
COMPARISONS=(("restore","sham"),("restore","random"),("full_swap","sham"),("full_swap","restore"),("full_swap","random"))
def weights(count):
    if count <= 4:
        draws = itertools.product(range(count), repeat=count)
        return (np.array([np.bincount(d, minlength=count) for d in draws]), 'exact_cluster_resampling')
    rng = np.random.default_rng(20261108)
    return (rng.multinomial(count, np.full(count, 1 / count), size=4000), 'seeded_cluster_resampling')
class ClusterSample:
    """Resample question clusters once; all users and modes share each draw."""

    def __init__(self, rows):
        self.groups = sorted({r['cluster'] for r in rows})
        self.idx = np.array([self.groups.index(r['cluster']) for r in rows])
        bs, self.method = weights(len(self.groups))
        self.weights = np.vstack([np.ones(len(self.groups)), bs])

    def totals(self, x):
        v = np.bincount(self.idx, weights=x, minlength=len(self.groups))
        return self.weights @ v

    def pack(self, v, n, clusters=None):
        b = v[1:][np.isfinite(v[1:])]
        count = len(self.groups) if clusters is None else clusters
        ci = np.quantile(b, [0.025, 0.975]).tolist() if len(b) and count > 1 and np.isfinite(v[0]) else [None, None]
        return {'point': float(v[0]) if np.isfinite(v[0]) else None, 'lo95': ci[0], 'hi95': ci[1], 'denominator': int(n), 'valid_draws': len(b), 'contributing_clusters': count}

    def mean(self, x, mask):
        mask = np.asarray(mask, dtype=bool)
        num = self.totals(np.asarray(x, dtype=float) * mask)
        den = self.totals(mask.astype(float))
        return self.pack(np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0), mask.sum(), len(set(self.idx[mask])))

    def recovery(self, num, gap):
        num, gap = (np.asarray(num, dtype=float), np.asarray(gap, dtype=float))
        mask = np.ones(len(num), dtype=bool)
        ne, de = (self.mean(num, mask), self.mean(gap, mask))
        lo, hi = (de['lo95'], de['hi95'])
        if abs(de['point']) <= GAP_EPS:
            reason = 'near_zero_gap'
        elif lo is None:
            reason = 'gap_interval_unavailable'
        elif lo <= 0 <= hi:
            reason = 'gap_interval_contains_zero'
        else:
            reason = ''
        ns, ds = (self.totals(num), self.totals(gap))
        ok = np.abs(ds) > GAP_EPS * self.totals(mask.astype(float))
        v = np.divide(ns, ds, out=np.full(len(ns), np.nan), where=ok)
        ratio = self.pack(v, len(num)) if not reason else dict(point=None, lo95=None, hi95=None, denominator=len(num), valid_draws=0, contributing_clusters=len(self.groups))
        return {**ratio, 'ratio_status': reason or 'reported', 'raw_shift': ne['point'], 'raw_shift_lo95': ne['lo95'], 'raw_shift_hi95': ne['hi95'], 'natural_gap': de['point'], 'gap_lo95': lo, 'gap_hi95': hi}
def states(rows, mode):
    return np.array([r[mode + '_state'] for r in rows])
def valid(s):
    return np.isin(s, ('R', 'A'))
def col(rows, k):
    return np.array([r[k] for r in rows], dtype=float)
def paired_counts(x, y, mask):
    return {a + b: int(((x == a) & (y == b) & mask).sum()) for a in 'RA' for b in 'RA'}
def summarize(rows):
    if not rows:
        return {'assigned': 0, 'estimates': {}}
    sample = ClusterSample(rows)
    s = {mode: states(rows, mode) for mode in CONDITIONS}
    common = np.logical_and.reduce([valid(v) for v in s.values()])
    all_rows = np.ones(len(rows), dtype=bool)
    result = {'assigned': len(rows), 'clusters': len(sample.groups), 'cluster_interval_method': sample.method, 'bootstrap_draws': len(sample.weights) - 1, 'estimates': {}, 'states': {}, 'paired_counts': {}, 'common_strict': int(common.sum())}
    estimates = result['estimates']
    for c, x in s.items():
        result['states'][c] = dict(Counter(x.tolist()))
        estimates[c + '_refusal_rate_common'] = sample.mean(x == 'R', common)
        result[c + '_refusal_n_common'] = int(((x == 'R') & common).sum())
    for tr, ctrl in COMPARISONS:
        xt, xc = (s[tr], s[ctrl])
        m = valid(xt) & valid(xc)
        diff = (xt == 'R').astype(float) - (xc == 'R')
        prefix = tr + '_minus_' + ctrl
        result['paired_counts'][ctrl + '_to_' + tr] = paired_counts(xc, xt, m)
        estimates[prefix + '_refusal_common'] = sample.mean(diff, common)
        estimates[prefix + '_refusal_pairwise'] = sample.mean(diff, m)
        for k in ('f16', 'g16'):
            estimates[prefix + '_' + k] = sample.mean(col(rows, tr + '_' + k) - col(rows, ctrl + '_' + k), all_rows)
    donor_states = np.array([r['donor_baseline'] for r in rows])
    for mode in ('restore', 'full_swap'):
        mask = common & valid(donor_states)
        estimates[mode + '_donor_state_agreement'] = sample.mean(s[mode] == donor_states, mask)
    result['sham_matches_baseline_state_n'] = sum((r['sham_state'] == r['target_baseline'] for r in rows))
    result['max_abs_restore_projection_residual'] = max((abs(r['restore_donor_residual']) for r in rows))
    result['max_random_relative_norm_error'] = max((r['random_relative_norm_error'] for r in rows))
    result['max_full_swap_residual'] = max((abs(r['full_swap_donor_residual']) for r in rows))
    return result
def match_users(rows):
    users = {u: {r['q']: r for r in rows if r['user'] == u} for u in 'AB'}
    require(len(rows) == sum(map(len, users.values())) and users['A'].keys() == users['B'].keys(), 'A/B parent mismatch')
    ids = sorted(users['A'])
    a = [users['A'][q] for q in ids]
    b = [users['B'][q] for q in ids]
    require(all(((x['split'], x['cluster']) == (y['split'], y['cluster']) for x, y in zip(a, b))), 'A/B cluster mismatch')
    return (a, b)
def balanced(a, b):
    """Average within parent, then across parents. A-B is treatment heterogeneity."""
    if not a:
        return {'assigned': 0, 'estimates': {}}
    sample = ClusterSample(a)
    sa = {c: states(a, c) for c in CONDITIONS}
    sb = {c: states(b, c) for c in CONDITIONS}
    common = np.logical_and.reduce([valid(v) for v in [*sa.values(), *sb.values()]])
    all_rows = np.ones(len(a), dtype=bool)
    est = {}
    for tr, ctrl in COMPARISONS:
        prefix = tr + '_minus_' + ctrl
        va = (sa[tr] == 'R').astype(float) - (sa[ctrl] == 'R')
        vb = (sb[tr] == 'R').astype(float) - (sb[ctrl] == 'R')
        m = valid(sa[tr]) & valid(sa[ctrl]) & valid(sb[tr]) & valid(sb[ctrl])
        for suffix, val in [('balanced', (va + vb) / 2), ('user_A_minus_B_half', (va - vb) / 2)]:
            est[prefix + '_refusal_' + suffix + '_common'] = sample.mean(val, common)
            est[prefix + '_refusal_' + suffix + '_pairwise'] = sample.mean(val, m)
        for k in ('g16', 'f16'):
            dx = col(a, tr + '_' + k) - col(a, ctrl + '_' + k)
            dy = col(b, tr + '_' + k) - col(b, ctrl + '_' + k)
            est[prefix + '_' + k + '_balanced'] = sample.mean((dx + dy) / 2, all_rows)
    d_a = col(a, 'target_s12') - col(a, 'donor_s12')
    d_b = col(b, 'target_s12') - col(b, 'donor_s12')
    est['baseline_match_minus_mismatch_s12'] = sample.mean((d_a + d_b) / 2, all_rows)
    est['baseline_document_A_minus_B_s12'] = sample.mean((d_a - d_b) / 2, all_rows)
    return {'assigned': len(a), 'clusters': len(sample.groups), 'common_strict_all_eight': int(common.sum()), 'estimates': est}
def layer_table(a, b=None):
    if not a:
        return []
    sample = ClusterSample(a)
    tables = []
    for l in LAYERS:
        k = 's' + str(l)
        for mode in ('restore', 'random', 'full_swap'):

            def shifts(rows):
                sham = col(rows, 'sham_' + k)
                return (sham - col(rows, mode + '_' + k), sham - col(rows, 'donor_' + k))
            num, gap = shifts(a)
            if b is not None:
                nb, gb = shifts(b)
                num = (num + nb) / 2
                gap = (gap + gb) / 2
            stats = sample.recovery(num, gap)
            tables.append({'layer': l, 'mode': mode, **stats})
    return tables
