from typing import NamedTuple, List, Tuple, Dict, Optional

import numpy as np


EPSILON = 1e-10

def get(name: str) -> callable:
    if name == 'mean_gaussian':
        return mean_gaussian
    if name == 'weighted_gaussian':
        return weighted_gaussian
    if name == 'prototype_gaussian':
        return prototype_gaussian
    if name == 'inv_dist_gaussian':
        return inv_dist_gaussian
    raise ValueError('Invalid est function')


def mean_gaussian(X: np.ndarray, sample_ids: List[int], subgraph):
    """Compute standard mean and covariance matrix for a normal distr.
    
    # Arguments
        X: The feature matrix with shape `[n_samples, n_features]`.
        sample_ids: List of index of elements from the current cluster.
        subgraph: Unsuperfised opf clf.subgraph.
    """
    cluster_X = X[sample_ids]

    cluster_mean = cluster_X.mean(axis=0)
    cluster_cova = np.cov(cluster_X, rowvar=False)
    return cluster_mean, cluster_cova


def weighted_gaussian(X: np.ndarray, sample_ids: List[int], subgraph):
    """Compute standard mean and covariance matrix for a normal distr.
    Each feature vector is weighted by its OPF density rho.
    
    # Arguments
        X: The feature matrix with shape `[n_samples, n_features]`.
        sample_ids: List of index of elements from the current cluster.
        subgraph: Unsuperfised opf clf.subgraph.
    """
    cluster_X = X[sample_ids]

    rho = np.asarray([subgraph.nodes[idx].density for idx in sample_ids])
    rho = rho / rho.sum()
    cluster_mean = (rho[:, None] * cluster_X).sum(axis=0)
    cluster_cova = np.cov(cluster_X, rowvar=False, aweights=rho)
    return cluster_mean, cluster_cova


def prototype_gaussian(X: np.ndarray, sample_ids: List[int], subgraph):
    """Compute standard mean and covariance matrix for a normal distr.
    The mean vector is the cluster prototype feature vector.

    NOTICE: It might perform better with large K values, causing a single
    cluster to emerge.
    
    # Arguments
        X: The feature matrix with shape `[n_samples, n_features]`.
        sample_ids: List of index of elements from the current cluster.
        subgraph: Unsuperfised opf clf.subgraph.
    """
    cluster_X = X[sample_ids]

    root_idx = subgraph.nodes[sample_ids[0]].root
    cluster_mean = X[root_idx, :]

    cluster_cova = np.cov(cluster_X, rowvar=False)
    return cluster_mean, cluster_cova


def inv_dist_gaussian(X: np.ndarray, sample_ids: List[int], subgraph):
    """Compute standard mean and covariance matrix for a normal distr.
    Each feature vector is weighted by 1/(dist to cluster centroid).
    
    NOTICE: OPF uses log_euclidean_distance, this function uses the euclidean
    distance only, fix that.
    
    # Arguments
        X: The feature matrix with shape `[n_samples, n_features]`.
        sample_ids: List of index of elements from the current cluster.
        subgraph: Unsuperfised opf clf.subgraph.
    """
    cluster_X = X[sample_ids]
    cluster_mean = cluster_X.mean(axis=0)
    
    distances = 1 / (np.linalg.norm(cluster_X - cluster_mean, axis=-1) + EPSILON)
    distances = distances / distances.sum()

    cluster_mean = (distances[:, None] * cluster_X).sum(axis=0)
    cluster_cova = np.cov(cluster_X, rowvar=False)
    return cluster_mean, cluster_cova
