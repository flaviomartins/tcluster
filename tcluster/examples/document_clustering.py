"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the tcluster can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from nltk import sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from tcluster.cluster import KMeans, SampleKMeans, MiniBatchKMeans
from tcluster.metrics import nkl_transform, purity_score

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--minibatch",
              action="store_true", dest="minibatch", default=False,
              help="Use k-means minibatch algorithm.")
op.add_option("--sample",
              action="store_true", dest="sample", default=False,
              help="Use k-means sample algorithm.")
op.add_option("--batch-size", type=float, default=.01,
              help="Batch size for k-means minibatch algorithm.")
op.add_option("--init-size", type=float, default=.03,
              help="Number of samples for k-means algorithm training phase.")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--max-iter", type=int, default=50,
              help="Maximum number of iterations")
op.add_option("--n-init", type=int, default=1,
              help="Number of time the k-means algorithm will be run with different centroid seeds.")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--no-remove",
              action="store_false", dest="remove_extra", default=True,
              help="Use this to keep 'headers', 'footers', 'quotes'")
op.add_option("--metric",
              dest="metric", type="str", default="euclidean",
              help="Specify the distance metric to use for KMeans.")
op.add_option("--a", type=float, default=.7,
              help="JM smoothing (NKL parameter).")
op.add_option("--norm",
              dest="norm", type="str", default="l2",
              help="Use this norm to normalize document vectors.")
op.add_option("--short",
              action="store_true", dest="short", default=False,
              help="Use short sentences (short text simulation).")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

remove = ()
if opts.remove_extra:
    remove = ('headers', 'footers', 'quotes')

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             remove=remove,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

if opts.short:
    print("Partitioning 20 newsgroups dataset into sentences:")
    data = []
    labels = []
    for text, label in zip(dataset.data, dataset.target):
        for line in sent_tokenize(text):
            data.append(line)
            labels.append(label)

    dataset.data = data
    labels = np.array(labels)

    print("%d sentences" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer(norm=opts.norm))
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm=opts.norm,
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf, norm=opts.norm)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


if opts.batch_size < 1:
    batch_size = int(opts.batch_size * X.shape[0])
else:
    batch_size = int(max(opts.batch_size, 10 * true_k))

if opts.init_size < 1:
    init_size = int(opts.init_size * X.shape[0])
else:
    init_size = int(max(opts.init_size, 10 * true_k))


init = 'k-means++' if opts.metric in ['cosine', 'euclidean', 'l2'] else 'random'


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init=init, max_iter=opts.max_iter, n_init=opts.n_init,
                         max_no_improvement=opts.max_iter / 10, compute_labels=True,
                         metric=opts.metric, metric_kwargs={'a': opts.a},
                         init_size=init_size, batch_size=batch_size, verbose=opts.verbose)
elif opts.sample:
    km = SampleKMeans(n_clusters=true_k, init=init, max_iter=opts.max_iter, n_init=opts.n_init,
                      metric=opts.metric, metric_kwargs={'a': opts.a},
                      init_size=init_size, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init=init, max_iter=opts.max_iter, n_init=opts.n_init,
                max_no_improvement=opts.max_iter / 10,
                metric=opts.metric, metric_kwargs={'a': opts.a},
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
print("Number of iterations %d: cluster sizes:" % km.n_iter_, np.bincount(km.labels_))
print("Purity: %0.3f" % purity_score(labels, km.labels_))
homogeneity, completeness, v_measure_score = metrics.homogeneity_completeness_v_measure(labels, km.labels_)
print("NMI: %0.3f" % v_measure_score)
print("ARI: %0.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print("AMI: %0.3f" % metrics.adjusted_mutual_info_score(labels, km.labels_))
print("Homogeneity: %0.3f" % homogeneity)
print("Completeness: %0.3f" % completeness)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    cluster_centers_ = km.cluster_centers_
    if opts.metric in ['nkl', 'negative-kullback-leibler']:
        cluster_centers_ = nkl_transform(cluster_centers_, a=opts.a)

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()