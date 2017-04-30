from __future__ import division

import numpy as np
from scipy.special import rel_entr


def jensen_shannon_divergence(X, Y):
    """Compute Jensen-Shannon Divergence
    Parameters
    ----------
    X : array-like
        possibly unnormalized distribution.
    Y : array-like
        possibly unnormalized distribution. Must be of same shape as ``X``.
    Returns
    -------
    j : float
    See Also
    --------
    entropy : function
        Computes entropy and K-L divergence
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    m = .5 * (X + Y)
    return 0.5 * np.sum(rel_entr(X, m) + rel_entr(Y, m), axis=1)


def jensen_shannon_distance(X, Y):
    """Compute Jensen-Shannon Distance
    Parameters
    ----------
    X : array-like
        possibly unnormalized distribution.
    Y : array-like
        possibly unnormalized distribution. Must be of same shape as ``X``.
    Returns
    -------
    j : float
    See Also
    --------
    jensen_shannon_divergence : function
        Computes Jensen-Shannon Divergence 
    """
    return np.sqrt(jensen_shannon_divergence(X, Y))
