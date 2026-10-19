# metrics.py
# challenge metrics: mA, instance F1, score = harmonic mean of both
# score is computed per dataset and then averaged

import numpy as np


def compute_metrics(probs, y, mask, thr):
    pred = (probs >= thr[None, :]).astype(np.float32)
    m = mask > 0

    # mA = mean of (tpr + tnr) / 2
    tp = (pred * y * m).sum(0)
    pos = (y * m).sum(0)
    tn = ((1 - pred) * (1 - y) * m).sum(0)
    neg = ((1 - y) * m).sum(0)
    tpr = np.where(pos > 0, tp / np.maximum(pos, 1), np.nan)
    tnr = np.where(neg > 0, tn / np.maximum(neg, 1), np.nan)
    mA = np.nanmean(np.nanmean(np.stack([tpr, tnr]), 0))

    # instance f1, per image then mean
    P = pred * m
    G = y * m
    inter = (P * G).sum(1)
    p_sum = P.sum(1)
    g_sum = G.sum(1)
    prec = np.where(p_sum > 0, inter / np.maximum(p_sum, 1), 0)
    rec = np.where(g_sum > 0, inter / np.maximum(g_sum, 1), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0).mean()

    score = 2 * mA * f1 / max(mA + f1, 1e-9)
    return {"mA": float(mA * 100), "F1": float(f1 * 100), "score": float(score * 100)}


def dataset_wise_metrics(probs, y, mask, thr, groups):
    # same as the challenge, score each dataset then take mean
    results = []
    for g in np.unique(groups):
        idx = groups == g
        results.append(compute_metrics(probs[idx], y[idx], mask[idx], thr))
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def find_best_thresholds(probs, y, mask, groups=None, rounds=2):
    # change one threshold at a time, keep it if score goes up
    n_attr = probs.shape[1]
    thr = np.full(n_attr, 0.5, dtype=np.float32)
    if groups is None:
        groups = np.zeros(len(probs), dtype=int)
    grid = np.linspace(0.1, 0.9, 17)

    best = dataset_wise_metrics(probs, y, mask, thr, groups)["score"]
    for _ in range(rounds):
        for a in range(n_attr):
            best_t = thr[a]
            for t in grid:
                thr[a] = t
                s = dataset_wise_metrics(probs, y, mask, thr, groups)["score"]
                if s > best + 1e-6:
                    best = s
                    best_t = t
            thr[a] = best_t
    return thr
