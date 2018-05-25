from __future__ import division

import numpy as np
from scipy.special import rel_entr

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from .js_div import js_div


def jensenshannon(p, q, base=None):
    """
    Compute the Jensen-Shannon distance (metric) between
    two 1-D probability arrays. This is the square root
    of the Jensen-Shannon divergence.

    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,

    .. math::

       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}

    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.

    This routine will normalize `p` and `q` if they don't sum to 1.0.

    Parameters
    ----------
    p : (N,) array_like
        left probability vector
    q : (N,) array_like
        right probability vector
    base : double, optional
        the base of the logarithm used to compute the output
        if not given, then the routine uses the default base of
        scipy.stats.entropy.

    Returns
    -------
    js : double
        The Jensen-Shannon distance between `p` and `q`

    .. versionadded:: 1.0.2

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    1.0
    >>> distance.jensenshannon([1.0, 0.0], [0.5, 0.5])
    0.46450140402245893
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    0.0

    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)


def jensenshannon_divergence(X, Y, base=None):
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
    js = js_div(X, Y)
    if base is not None:
        js /= np.log(base)
    return js / 2.0


def jensenshannon_distance(X, Y, base=None):
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
    js = js_div(X, Y)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)


def np_jensenshannon_divergence(X, Y, base=None):
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
    js = np.sum(rel_entr(X, m) + rel_entr(Y, m), axis=1)
    if base is not None:
        js /= np.log(base)
    return .5 * js


def np_jensenshannon_distance(X, Y, base=None):
    return np.sqrt(np_jensenshannon_divergence(X, Y, base))