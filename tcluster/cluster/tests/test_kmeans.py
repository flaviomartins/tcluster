import numpy as np

from tcluster.cluster.kmeans import kmeanssample, randomsample, kmeans


N = 1000
dim = 10
ncluster = 10
kmsample = 100
kmdelta = .001
kmiter = 10
metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
X = np.random.exponential(size=(N, dim))


def test_kmeans_random():
    randomcentres = randomsample(X, ncluster)
    centres, xtoc, dist = kmeans(X, randomcentres,
                                 delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2)


def test_kmeans_sample():
    centres, xtoc, dist = kmeanssample(X, ncluster, nsample=kmsample,
                                       delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2)


def test_kmeans_jsd():
    centres, xtoc, dist = kmeanssample(X, ncluster, nsample=kmsample,
                                       delta=kmdelta, maxiter=kmiter, metric="jsd", verbose=2)


def test_kmeans_nkl():
    centres, xtoc, dist = kmeanssample(X, ncluster, nsample=kmsample,
                                       delta=kmdelta, maxiter=kmiter, metric="nkl", verbose=2)