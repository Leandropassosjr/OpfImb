import sys
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import special
from scipy.spatial.distance import cdist

from oversampling.core.gp import GaussianParams
from oversampling.core import gmean


def get(name: str) -> callable:
    if name == 'sampling':
        return sampling
    if name == 'interpolation':
        return interpolation
    if name == 'geometric_euclidean' or name == 'median_centroid_eu':
        return geometric_euclidean
    if name == 'geometric_mb' or name == 'median_centroid_mb':
        return geometric_mb
    if name == 'normal_mb' or name == 'mean_centroid_mb':
        return normal_mb
#     if name == 'probabilistic_geometric':
#         return probabilistic_geometric
    raise ValueError('Invalid generation function name.')


def sampling(gp: GaussianParams,
             n_new_samples: int,
             cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples by sampling from a Gaussian Distribution with
    parameters gp.

    # Arguments
        gp: Gaussian parameters.
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].`
    """
    return np.random.multivariate_normal(gp.mean, gp.std, n_new_samples)


def interpolation(gp: GaussianParams,
                  n_new_samples: int,
                  cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples through `z' = a * p + (1 - a) * z`,
    where `z` is sampled from a Gaussian Distribution with parameters
    `gp`, `a` is a random uniform number in the interval `[0,1)` and
    `p` is the nearest sample to `z`.

    # Arguments
        gp: Gaussian parameters.
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].`
    """

    # z' = a*p + (1 - a)*z
    z = np.random.multivariate_normal(gp.mean, gp.std, n_new_samples)
    knn = NearestNeighbors(n_neighbors=1).fit(cluster_x)
    _, nearest_idx = knn.kneighbors(z)
    
    # nearest_idx[:, 0] makes a (n, 1) ndarray into a (n,) ndarray,
    # thus making p having shape (n_samples, n_features) insted
    # of (n_samples, 1, n_features)
    p = cluster_x[nearest_idx[:, 0]]
    a = np.random.uniform(size=(p.shape[0], 1))
    return a * p + (1 - a) * z


def _mahalanobis(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Compute the Mahalanobis distance between x and mu. This method
    should not be used if the amount of features is larger than the amount
    of samples in each cluster, as pinv may yield unstable results."""
    inv_cov = np.linalg.pinv(np.cov(x, rowvar=False))
    d = cdist(x, mu[None, :], metric='mahalanobis', VI=inv_cov)

    # Ensure no-nans, see
    # https://stackoverflow.com/questions/29717269/scipy-nan-when-calculating-mahalanobis-distance
    return np.nan_to_num(d, nan=np.nan_to_num(d, nan=1000).min())


def _geometric_generation(n_new_samples: int,
                          _cluster_x: np.ndarray,
                          centroid_fn: callable,
                          distance_fn: callable):
    cluster_x = _cluster_x.copy()

    new_samples = []
    for i in range(n_new_samples):
        n_samples = cluster_x.shape[0]
        center_point = centroid_fn(cluster_x)
        distances = 1 / (1 + distance_fn(cluster_x, center_point))

        max_radius_idx = np.random.choice(np.arange(0, n_samples))
        max_radius = distances[max_radius_idx]
        border_sample = cluster_x[max_radius_idx]

        # z_prime = [1, n_features]
        a = np.random.uniform(high=max_radius, size=1).item()
        z_prime = (a * border_sample + (1 - a) * center_point).reshape(1, -1)

        new_samples.append(z_prime)
        cluster_x = np.vstack([cluster_x, z_prime])
    return np.concatenate(new_samples)


def probabilistic_geometric(_: GaussianParams,
                            n_new_samples: int,
                            _cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples by iterativelly computing the cluster geometric
    mean, finding a maximum interpolation radius with Mahalanobis distance
    and performing a linear interpolation with scalar `a`. Closer radii are more
    likely to be sampled and `a` in uniformly samples from the interval
    `[0.01, 1 - max_radius)`. Finally, the interpolation is performad as
    `z' = a * p + (1 - a) * z`, where `p` is the geometric median,

    # Arguments:
        gp: Gaussian parameters (not used).
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].
    """
    cluster_x = _cluster_x.copy()

    new_samples = []
    for i in range(n_new_samples):
        n_samples = cluster_x.shape[0]   # [n_samples, n_features]

        center_point = gmean.geometric_median_mb(cluster_x)
        distances = 1 / (1 + _mahalanobis(cluster_x, center_point))

        # softmax(-x) = softmin(x) -> lowest values get the highest probas
        # max subtraction avoids overflow for midly large numbers
        probs = special.softmax(-(distances - np.max(distances)))
        max_radius_idx = np.random.choice(np.arange(0, n_samples), p=probs)
        border_sample = cluster_x[max_radius_idx]

        threshold = 1e-2
        high = 1 - distances[max_radius_idx]
        low = threshold if threshold < high else 0
        a = np.random.uniform(low, high)

        z_prime = ((a * border_sample) + (1 - a) * center_point).reshape(1, -1)
        new_samples.append(z_prime)
        cluster_x = np.vstack([cluster_x, z_prime])
    return np.concatenate(new_samples)


def geometric_euclidean(_: GaussianParams,
                        n_new_samples: int,
                        _cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples by iterativelly computing the cluster geometric
    median, finding a maximum interpolation radius with euclidean distance
    and performing a linear interpolation between the geometric median and
    the sample that lies in this radius, i.e.: `z' = a * p + (1 - a) * z`,
    where `p` is the geometric median, `z` is the distant sample and `a`
    is a random number. Uses euclidean distance.

    # Arguments
        gp: Gaussian parameters.
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].
    """
    def euclidean(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x - mu, axis=-1)

    return _geometric_generation(n_new_samples,
                                 _cluster_x,
                                 gmean.geometric_median_eu,
                                 euclidean)


def geometric_mb(_: GaussianParams,
                 n_new_samples: int,
                 _cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples by iterativelly computing the cluster geometric
    median, finding a maximum interpolation radius with euclidean distance
    and performing a linear interpolation between the geometric median and
    the sample that lies in this radius, i.e.: `z' = a * p + (1 - a) * z`,
    where `p` is the geometric median, `z` is the distant sample and `a`
    is a random number. Use mahalanobis distance.

    # Arguments
        gp: Gaussian parameters.
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].
    """
    return _geometric_generation(n_new_samples,
                                 _cluster_x,
                                 gmean.geometric_median_mb,
                                 _mahalanobis)


def normal_mb(_: GaussianParams,
              n_new_samples: int,
              _cluster_x: np.ndarray) -> np.ndarray:
    """Generate new samples by iterativelly computing the cluster mean
    (not median) and finding a maximum interpolation radius with mahalanobis
    distance and performing a linear interpolation between the geometric
    median and the sample that lies in this radius, i.e.:
    `z' = a * p + (1 - a) * z`, where `p` is the geometric median, `z` is
    the distant sample and `a` is a random number. Use mahalanobis distance.

    # Arguments
        gp: Gaussian parameters.
        n_new_samples: Number of samples to generate.
        cluster_x: All cluster feature vectors.

    # Return
        A ndarray with shape `[n_new_samples, n_features].
    """

    return _geometric_generation(n_new_samples,
                                 _cluster_x,
                                 centroid_fn=lambda x: np.mean(x, axis=0),
                                 distance_fn=_mahalanobis)
