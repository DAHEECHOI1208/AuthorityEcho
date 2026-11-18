import math
import numpy as np
def calibrate_threshold(negative_scores, budget):
    x = np.sort(np.asarray(negative_scores, dtype=float))
    if len(x) == 0 or not 0 <= budget < 1 or (not np.all(np.isfinite(x))):
        raise ValueError('Invalid threshold calibration inputs')
    allowed = math.floor(len(x) * budget + 1e-12)
    tau = float(x[len(x) - allowed - 1])
    if np.count_nonzero(x > tau) > allowed:
        raise AssertionError('Threshold FPR rule violated')
    return tau

def confusion(y, score, tau, weights=None):
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    alert = score > tau
    tp, fp, tn, fn = [float(w[mask].sum()) for mask in ((y == 1) & alert, (y == 0) & alert, (y == 0) & ~alert, (y == 1) & ~alert)]

    def div(a, b):
        return a / b if b else None
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'TPR': div(tp, tp + fn), 'FPR': div(fp, fp + tn), 'precision': div(tp, tp + fp), 'FNR': div(fn, tp + fn), 'alert_fraction': div(tp + fp, w.sum())}
