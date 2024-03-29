from __future__ import division

import numpy as np
from scipy.special import xlogy, rel_entr

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)


def np_nkl_distance(X, Y, p_B=None, a=0.1, base=None):
    """Compute Negative Kullback-Liebler metric
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

    with np.errstate(divide='ignore', invalid='ignore'):
        agg = xlogy(p_C, a_p_B/p_D) + xlogy(p_D, a_p_B/p_C)
        agg[~np.isfinite(agg)] = 0

    nkl = np.sum(agg, axis=1)
    if base is not None:
        nkl /= np.log(base)
    return nkl


def nkl_transform(X, a=0.1):
    return rel_entr(X, a * X.mean(axis=0))
