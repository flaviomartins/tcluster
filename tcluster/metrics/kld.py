from __future__ import division
from builtins import range

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import _copy_array_if_base_present
from scipy.special import rel_entr


def kld_cdist_sparse(X, Y, p_B, **kwargs):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2 * issparse(X) + issparse(Y)
    if sxy == 0:
        return kld_cdist(X, Y, p_B, **kwargs)
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = kld_cdist(x.todense(), Y, p_B, **kwargs)[0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:, k] = kld_cdist(X, y.todense(), p_B, **kwargs)[0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j, k] = kld_cdist(x.todense(), y.todense(), p_B, **kwargs)[0]
    return d


def kld_cdist(XA, XB, p_B=None, a=0.1):
    # You can also call this as:
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.

    # Store input arguments to check whether we can modify later.
    input_XA, input_XB = XA, XB

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(XA)
    XB = _copy_array_if_base_present(XB)

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    dm = np.zeros((mA, mB), dtype=np.double)
    for i in range(0, mA):
        for j in range(0, mB):
            dm[i, j] = kld_metric(XA[i, :], XB[j, :], p_B, a)

    return dm


def kld_metric(X, Y, p_B, a=0.1):
    """Compute Kulkarni's Negative Kullback-Liebler metric
    Parameters
    ----------
    X : array-like
        possibly unnormalized distribution.
    Y : array-like
        possibly unnormalized distribution. Must be of same shape as ``X``.
    p_B : array-like
        possibly unnormalized distribution. Must be of same shape as ``X``.
        e.g., background model, 
    a : float
        tune the weight of the background model p_B
    Returns
    -------
    j : float
    See Also
    --------
    entropy : function
        Computes entropy and K-L divergence
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    a_p_B = a * p_B
    p_D = (1 - a) * X + a_p_B
    p_C = Y
    m = (p_C + p_D)
    m /= 2.
    return 0.5 * np.sum(rel_entr(p_C, m) + rel_entr(p_D, m), axis=1)
