# Adapted from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def dprint(s):
    from datetime import datetime as dt
    import sys
    print(f'\t\t\t[{dt.now()}] {s}')
    sys.stdout.flush()


def geometric_median_mb(X, eps=1e-5):
    return geometric_median(X, eps, dist_fn='mahalanobis')

def geometric_median_eu(X, eps=1e-5):
    return geometric_median(X, eps, dist_fn='euclidean')

def _is_zero(x):
    return np.isclose(x, 0)


def geometric_median(X, eps, dist_fn, max_iters=1000):
    y = np.mean(X, 0)

    for counter in range(max_iters):
        counter += 1
        D = cdist(X, [y], metric=dist_fn, VI=np.linalg.pinv(np.cov(np.vstack([X, y]), rowvar=False)))
        nonzeros = np.logical_not(_is_zero(D))
        nonzeros = nonzeros[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs # can this be zero!?
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if _is_zero(r) else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        error = euclidean(y, y1) / np.linalg.norm(y)
        y = y1

        if error < eps:
            break
    return y
