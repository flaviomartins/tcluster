"""K-means clustering"""

# Authors: Flavio Martins <flaviomartins@acm.org>
# License: BSD 3 clause
# Some code shamelessly lifted from sklearn

import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.six import string_types

from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances, \
    pairwise_distances_argmin_min
from sklearn.utils import check_random_state, check_array, as_float_array
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.sparsefuncs import mean_variance_axis

from tcluster.metrics.jsd import jensen_shannon_divergence
from tcluster.metrics.nkl import nkl_metric


###############################################################################
# Initialization heuristic


def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters: integer
        The number of seeds to choose

    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state: numpy.RandomState
        The generator used to initialize the centers.

    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


###############################################################################
# K-means batch estimation by EM (expectation maximization)

def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def k_means(X, n_clusters, init='random', precompute_distances='auto',
            n_init=10, max_iter=300, verbose=False,
            tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
            algorithm="full", return_n_iter=False,
            metric='euclidean', metric_kwargs=None):
    """K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'random', 'sample', or ndarray, or a callable}, optional
        Method for initialization, default to 'random':

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    algorithm : "full", default="full"
        K-means algorithm to use. The classical EM-style algorithm is "full".

    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

    tol : float, optional
        The relative increment in the results before declaring convergence.

    verbose : boolean, optional
        Verbosity mode.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    metric : default="euclidean", optional
        Valid values for metric are:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix inputs.
    
        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
          See the documentation for scipy.spatial.distance for details on these
          metrics. These metrics do not support sparse matrix inputs.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter: int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.

    """
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if metric == 'euclidean':
        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X) or hasattr(init, '__array__'):
            X_mean = X.mean(axis=0)
        if not sp.issparse(X):
            # The copy was already done above
            X -= X_mean

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if metric == 'euclidean':
            init -= X_mean
        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    if metric == 'euclidean':
        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = None

    best_labels, best_inertia, best_centers = None, None, None
    if n_clusters == 1:
        algorithm = "full"
    if algorithm == "auto":
        algorithm = "full"
    if algorithm == "full":
        kmeans_single = _kmeans_single_lloyd
    else:
        raise ValueError("Algorithm must be 'auto' or 'full', got"
                         " %s" % str(algorithm))
    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
                precompute_distances=precompute_distances, tol=tol,
                x_squared_norms=x_squared_norms, random_state=random_state,
                metric=metric, metric_kwargs=metric_kwargs)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_single)(X, n_clusters, max_iter=max_iter, init=init,
                                   verbose=verbose, tol=tol,
                                   precompute_distances=precompute_distances,
                                   x_squared_norms=x_squared_norms,
                                   # Change seed to ensure variety
                                   random_state=seed,
                                   metric=metric, metric_kwargs=metric_kwargs)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if metric == 'euclidean':
        if not sp.issparse(X):
            if not copy_x:
                X += X_mean
            best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


def _kmeans_single_lloyd(X, n_clusters, max_iter=300, init='k-means++',
                         verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4,
                         precompute_distances=True,
                         metric='euclidean', metric_kwargs=None):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters: int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init: {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol: float, optional
        The relative increment in the results before declaring convergence.

    verbose: boolean, optional
        Verbosity mode

    x_squared_norms: array
        Precomputed x_squared_norms.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    -------
    centroid: float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label: integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia: float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances, metric=metric, metric_kwargs=metric_kwargs)

        # computation of the means is also called the M-step of EM
        centers = _centers(X, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances, metric=metric, metric_kwargs=metric_kwargs)

    return best_labels, best_inertia, best_centers, i + 1


def _labels_inertia(X, x_squared_norms, centers,
                    precompute_distances=True, distances=None,
                    metric='euclidean', metric_kwargs=None):
    """Compute labels and inertia using a full distance matrix.

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.

    x_squared_norms : numpy array, shape (n_samples,)
        Precomputed squared norms of X.

    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of distances of samples to their closest cluster center.

    """
    n_samples = X.shape[0]

    # Breakup nearest neighbor distance computation into batches to prevent
    # memory blowup in the case of a large number of samples and clusters.
    # TODO: Once PR #7383 is merged use check_inputs=False in metric_kwargs.
    if metric == 'euclidean':
        euclidean_kwargs = {'squared': True}
        if metric_kwargs is not None:
            euclidean_kwargs.update(metric_kwargs)
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric='euclidean', metric_kwargs=euclidean_kwargs)
    elif metric in ['jsd', 'jensen-shannon']:
        D = pairwise_distances_sparse(
            X=X, Y=centers, metric=jensen_shannon_divergence)
        labels = D.argmin(axis=1)
        mindist = D[np.arange(n_samples), labels]
    elif metric in ['nkl', 'negative-kullback-leibler']:
        centers_mean = centers.mean(axis=0)
        nkl_kwargs = {'p_B': centers_mean}
        if metric_kwargs is not None:
            nkl_kwargs.update(metric_kwargs)
        D = pairwise_distances_sparse(
            X=X, Y=centers, metric=nkl_metric, metric_kwargs=nkl_kwargs)
        labels = D.argmin(axis=1)
        mindist = D[np.arange(n_samples), labels]
    else:
        if metric == 'cosine':
            metric_kwargs = None
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric=metric, metric_kwargs=metric_kwargs)
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    inertia = mindist.sum()
    return labels, inertia


def _centers(X, labels, n_clusters, distances):
    """
    M step of the K-means EM algorithm
    
        Computation of cluster centers / means.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
    
        labels : array of integers, shape (n_samples)
            Current label assignment
    
        n_clusters : int
            Number of desired clusters
    
        distances : array-like, shape (n_samples)
            Distance to closest cluster for each sample.
    
        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            The resulting centers
    """
    centers = np.empty((n_clusters, X.shape[1]), np.float64)
    for i in range(n_clusters):
        k = np.where(labels == i)[0]
        if len(k) > 0:
            centers[i] = X[k].mean(axis=0)
    return centers


def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None):
    """Compute the initial centroids

    Parameters
    ----------

    X: array, shape (n_samples, n_features)

    k: int
        number of centroids

    init: {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization

    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    x_squared_norms:  array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers: array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, string_types) and init == 'k-means++':
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif isinstance(init, string_types) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        centers = np.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(X, k, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    _validate_center_shape(X, k, centers)
    return centers


class KMeans(object):
    """K-Means clustering

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])

    See also
    --------

    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    ------
    The k-means problem is solved using Lloyd's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    """

    def __init__(self, n_clusters=8, init='random', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto', metric='euclidean', metric_kwargs=None):

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
        self.metric_kwargs = metric_kwargs

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    def fit(self, X, y=None):
        """Compute k-means clustering.
    
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True,
                metric=self.metric, metric_kwargs=self.metric_kwargs)
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
                 init_size=None, n_init=3, metric='euclidean', metric_kwargs=None, reassignment_ratio=0.01):

        super(SampleKMeans, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            verbose=verbose, random_state=random_state, tol=tol, n_init=n_init,
            metric=metric, metric_kwargs=metric_kwargs)

        self.init_size = init_size

    def fit(self, X, y=None):
        super(SampleKMeans, self).fit(X, y)
        return self

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)


def pairwise_distances_sparse(X, Y, metric, metric_kwargs=None):
    if metric_kwargs is None:
        metric_kwargs = {}

    """ -> |X| x |Y| distances array, any pairwise_distances metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2 * sp.issparse(X) + sp.issparse(Y)
    if sxy == 0:
        return pairwise_distances(X, Y, metric=metric, **metric_kwargs)
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = pairwise_distances(x.todense(), Y, metric=metric, **metric_kwargs)[0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:, k] = pairwise_distances(X, y.todense(), metric=metric, **metric_kwargs)[0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j, k] = pairwise_distances(x.todense(), y.todense(), metric=metric, **metric_kwargs)[0]
    return d


def nearestcentres(X, centers, metric="euclidean", p=2, a=.1, precomputed_centres_mean=None):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    if metric in ['jsd', 'jensen-shannon']:
        D = pairwise_distances_sparse(X, centers, metric=jensen_shannon_divergence)
    elif metric in ['nkl', 'negative-kullback-leibler']:
        if precomputed_centres_mean is None:
            centers_mean = centers.mean(axis=0)
        else:
            centers_mean = precomputed_centres_mean
        D = pairwise_distances_sparse(X, centers, p_B=centers_mean, a=a, metric=nkl_metric)
    else:
        D = pairwise_distances_sparse(X, centers, metric=metric, p=p)  # |X| x |centres|
    return D.argmin(axis=1)
