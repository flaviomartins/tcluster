import numpy as np

from tcluster.cluster import k_means


N = 1000
dim = 10
n_clusters = 10
init_size = 100
tol = .001
max_iter = 10
metric = "cosine"
X = np.random.exponential(size=(N, dim))


def test_kmeans_random():
    cluster_centers_, labels_, inertia_ = k_means(X, n_clusters, init='random', n_init=1, max_iter=max_iter,
                                                  tol=tol, metric=metric, verbose=True)


def test_kmeans_k_init():
    cluster_centers_, labels_, inertia_ = k_means(X, n_clusters, init='k-means++', n_init=1, max_iter=max_iter,
                                                  tol=tol, metric=metric, verbose=True)


def test_kmeans_jsd():
    cluster_centers_, labels_, inertia_ = k_means(X, n_clusters, init='random', n_init=1, max_iter=max_iter,
                                                  tol=tol, metric="jsd", verbose=True)


def test_kmeans_nkl():
    cluster_centers_, labels_, inertia_ = k_means(X, n_clusters, init='random', n_init=1, max_iter=max_iter,
                                                  tol=tol, metric="nkl", verbose=True)