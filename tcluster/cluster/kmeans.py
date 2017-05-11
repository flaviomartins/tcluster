#!/usr/bin/env python
# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np

# http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, pairwise_distances

from tcluster.metrics.jsd import jensen_shannon_divergence
from tcluster.metrics.nkl import nkl_metric

__date__ = "2011-11-17 Nov denis"


# X sparse, any cdist metric: real app ?
# centres get dense rapidly, metrics in high dim hit distance whiteout
# vs unsupervised / semi-supervised svm


def kmeans(X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, a=.1, verbose=1):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError("kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape))
    if verbose:
        print("kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric))
    allx = np.arange(N)
    prevdist = 0
    for jiter in range(1, maxiter + 1):
        if metric in ['euclidean', 'euc']:
            D = euclidean_distances(X, centres)
        elif metric in ['cosine', 'cos']:
            D = cosine_distances(X, centres)
        elif metric in ['jsd', 'jensen-shannon']:
            D = pairwise_distances_sparse(X, centres, metric=jensen_shannon_divergence)
        elif metric in ['nkl', 'negative-kullback-leibler']:
            centres_mean = centres.mean(axis=0)
            D = pairwise_distances_sparse(X, centres, p_B=centres_mean, a=a, metric=nkl_metric)
        else:
            D = pairwise_distances_sparse(X, centres, metric=metric, p=p)  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx, xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print("kmeans: av |X - nearest centre| = %.4g" % avdist)
        if (1 - delta) * prevdist <= avdist <= prevdist \
                or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where(xtoc == jc)[0]
            if len(c) > 0:
                centres[jc] = X[c].mean(axis=0)
    if verbose:
        print("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc))
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[xtoc == j]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile(dist, (50, 90))
        print("kmeans: cluster 50 % radius", r50.astype(int))
        print("kmeans: cluster 90 % radius", r90.astype(int))
        # scale L1 / dim, L2 / sqrt(dim) ?
    return np.array(centres), xtoc, distances


def kmeanssample(X, k, nsample=0, **kwargs):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
    # merge w kmeans ? mttiw
    # v large N: sample N^1/2, N^1/2 of that
    # seed like sklearn ?
    N, dim = X.shape
    if nsample is None or nsample == 0:
        nsample = max(2 * np.sqrt(N), 10 * k)
    Xsample = randomsample(X, int(nsample))
    pass1centres = randomsample(X, int(k))
    samplecentres = kmeans(Xsample, pass1centres, **kwargs)[0]
    return kmeans(X, samplecentres, **kwargs)


def pairwise_distances_sparse(X, Y, **kwargs):
    """ -> |X| x |Y| distances array, any pairwise_distances metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2 * issparse(X) + issparse(Y)
    if sxy == 0:
        return pairwise_distances(X, Y, n_jobs=-1, **kwargs)
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = pairwise_distances(x.todense(), Y, **kwargs)[0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:, k] = pairwise_distances(X, y.todense(), **kwargs)[0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j, k] = pairwise_distances(x.todense(), y.todense(), **kwargs)[0]
    return d


def randomsample(X, n):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample(range(X.shape[0]), int(n))
    return X[sampleix]


def nearestcentres(X, centres, metric="euclidean", p=2, a=.1, precomputed_centres_mean=None):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    if metric in ['cosine', 'cos']:
        D = cosine_distances(X, centres)
    elif metric in ['jsd', 'jensen-shannon']:
        D = pairwise_distances_sparse(X, centres, metric=jensen_shannon_divergence)
    elif metric in ['nkl', 'negative-kullback-leibler']:
        if precomputed_centres_mean is None:
            centres_mean = centres.mean(axis=0)
        else:
            centres_mean = precomputed_centres_mean
        D = pairwise_distances_sparse(X, centres, p_B=centres_mean, a=a, metric=nkl_metric)
    else:
        D = pairwise_distances_sparse(X, centres, metric=metric, p=p)  # |X| x |centres|
    return D.argmin(axis=1)


def Lqmetric(x, y=None, q=.5):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q).mean() if y is not None \
        else (np.abs(x) ** q).mean()


class KMeans(object):
    """ km = Kmeans( X, k= or centres=, ... )
        in: n_clusters for kmeans
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """

    def __init__(self, n_clusters=8, init='random', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto', metric='euclidean', p=2, a=.1):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.a = a

    def fit(self, X, y=None):
        randomcentres = randomsample(X, self.n_clusters)
        self.cluster_centers_, self.labels_, self.distances = kmeans(X, centres=randomcentres, delta=self.tol,
                                                                     maxiter=self.max_iter, metric=self.metric,
                                                                     p=self.p, a=self.a, verbose=self.verbose)
        return self

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)


class SampleKMeans(KMeans):
    """ km = SampleKmeans( X, k= or centres=, ... )
        in: n_clusters= [init_size=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """

    def __init__(self, n_clusters=8, init='random', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=1e-4, max_no_improvement=10,
                 init_size=None, n_init=3, metric='euclidean', p=2, a=.1, reassignment_ratio=0.01):

        super(SampleKMeans, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            verbose=verbose, random_state=random_state, tol=tol, n_init=n_init, metric=metric, p=p, a=a)

        self.init_size = init_size

    def fit(self, X, y=None):
        self.cluster_centers_, self.labels_, self.distances = kmeanssample(X, self.n_clusters, self.init_size,
                                                                           delta=self.tol, maxiter=self.max_iter,
                                                                           metric=self.metric, p=self.p, a=self.a,
                                                                           verbose=self.verbose)
        return self

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)
