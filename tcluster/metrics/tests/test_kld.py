from __future__ import division

import numpy as np
from numpy.testing import assert_, assert_array_almost_equal
from scipy.stats import entropy

from tcluster.metrics.kld import kld_metric, kld_cdist


def test_kld_basic():
    for _ in range(8):
        a = np.random.random((1, 16))
        a = a / a.sum()
        b = np.random.random((1, 16))
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(kld_metric(a, b, p_B=c, a=0.5) > 0.)
        assert_(kld_metric(a, b, p_B=c, a=0.5) >
                kld_metric(a, c, p_B=c, a=0.5))


def test_kld_known_result():
    a = np.array([[1, 0, 0, 0]]).astype(np.float)
    b = np.array([[0, 1, 0, 0]]).astype(np.float)
    c = a + b
    c = c / c.sum()
    assert_(kld_metric(a, b, p_B=c, a=0.5) < np.log(2))


def test_kld_stats_entropy():
    a = np.array([1, 0, 0, 0], float)
    a = a / a.sum()
    b = np.array([1, 1, 1, 1], float)
    b = b / b.sum()
    m = a + b
    expected = (entropy(a, m) + entropy(b, m)) / 2

    calculated = kld_metric(a, b, p_B=m, a=0.5)
    assert_(calculated < expected)


def test_kld_cdist():
    a = np.random.random((1, 12))
    b = np.random.random((10, 12))
    c = b.sum()
    c = c / c.sum()
    direct = kld_metric(a, b, p_B=c, a=0.5)
    indirect = kld_cdist(a, b, p_B=c, a=0.5)[0]
    assert_array_almost_equal(direct, indirect)
