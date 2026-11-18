import warnings
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
SEED = 20261102
NBOOT = 2000
MODELS = ['A', 'B', 'C', 'M', 'MG', 'N', 'NG', 'F', 'FG']
COMPS = [('C', 'B'), ('MG', 'M'), ('NG', 'N'), ('FG', 'F')]
X_NAMES = ['H_question_chars', 'H_tokens0', 'H_added_tokens', 'H_document_chars4']
B_NAMES = ['B_question_chars', 'B_tokens0', 'B_added_tokens']
EXTRA = {'A': [], 'B': ['H_g0'], 'C': ['H_g0', 'H_dg'], 'M': ['H_g0', 'H_dm'], 'MG': ['H_g0', 'H_dm', 'H_dg'], 'N': ['H_g0', 'H_dm', 'B_g0', 'B_dg', 'B_dm'] + B_NAMES, 'NG': ['H_g0', 'H_dm', 'B_g0', 'B_dg', 'B_dm'] + B_NAMES + ['H_dg'], 'F': ['H_g0', 'H_f0', 'H_df'], 'FG': ['H_g0', 'H_f0', 'H_df', 'H_dg']}

def metric(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])), 'brier': float(brier_score_loss(y, p))}

def aucmany(y, p, W):
    order = np.argsort(p, kind='stable')
    ss = p[order]
    yy = y[order]
    ww = W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
    pos = np.add.reduceat(ww * yy[None, :], starts, axis=1)
    neg = np.add.reduceat(ww * (1 - yy)[None, :], starts, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (pos * (np.cumsum(neg, axis=1) - neg + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))

def metricmany(y, p, W):
    pp = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    den = W.sum(1)
    return {'auc': aucmany(y, p, W), 'log_loss': W @ -(y * np.log(pp) + (1 - y) * np.log1p(-pp)) / den, 'brier': W @ (y - p) ** 2 / den}

def bootweights(g):
    levels = sorted(set(g))
    ix = np.array([levels.index(t) for t in g])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(levels), size=(NBOOT, len(levels)))
    counts = np.stack([np.bincount(v, minlength=len(levels)) for v in draws])
    return (counts[:, ix].astype(float), counts, levels)

def interval(v):
    vv = np.asarray(v)
    vv = vv[np.isfinite(vv)]
    if len(vv) < 0.95 * NBOOT:
        return (np.nan, np.nan, len(vv))
    a, b = np.quantile(vv, [0.025, 0.975])
    return (float(a), float(b), len(vv))

def simultaneous(point, boots):
    ok = np.all(np.isfinite(boots), axis=1)
    v = boots[ok]
    se = np.std(v, axis=0, ddof=1)
    use = se > 1e-14
    if ok.sum() < 0.95 * NBOOT:
        return (np.full_like(point, np.nan), np.full_like(point, np.nan), np.nan)
    if not use.any():
        return (point.copy(), point.copy(), 0.0)
    crit = float(np.quantile(np.max(np.abs((v[:, use] - point[use]) / se[use]), axis=1), 0.95))
    return (point - crit * se, point + crit * se, crit)

def fitone(rows, features, y, tr, te, layer, model, scheme, fold):
    names = X_NAMES + EXTRA[model]
    xx = np.column_stack([features[k] for k in names])
    assert np.isfinite(xx).all()
    scaler = StandardScaler().fit(xx[tr])
    levels = sorted({rows[i]['depth1'] for i in tr})
    cats = np.array([[float(r['depth1'] == t) for t in levels] for r in rows])
    z = np.concatenate([scaler.transform(xx), cats], axis=1)
    assert len(np.unique(y[tr])) == 2
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000, tol=1e-09, random_state=SEED)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        clf.fit(z[tr], y[tr])
    if any(('converg' in str(w.message).lower() for w in ws)):
        raise RuntimeError(str(ws))
    pred = clf.predict_proba(z[te])[:, 1]
    mm = metric(y[te], pred)
    fr = {'scheme': scheme, 'layer': layer, 'model': model, 'fold': fold, 'train_n': len(tr), 'eval_n': len(te), 'eval_RA': int(y[te].sum()), 'unknown_topic_n': sum((rows[i]['depth1'] not in levels for i in te)), **mm}
    coeff = {'scheme': scheme, 'layer': layer, 'model': model, 'fold': fold, 'train_ids': [rows[i]['parent_id'] for i in tr], 'test_ids': [rows[i]['parent_id'] for i in te], 'names': names + ['depth1=' + k for k in levels], 'coef': clf.coef_[0], 'intercept': float(clf.intercept_[0]), 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'iterations': int(clf.n_iter_[0])}
    return (pred, fr, coeff)
