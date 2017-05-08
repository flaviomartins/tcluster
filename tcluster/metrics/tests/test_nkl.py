from __future__ import division

import numpy as np
from numpy.testing import assert_
from scipy.stats import entropy

from tcluster.metrics.nkl import nkl_metric


def test_kld_basic():
    for _ in range(8):
        a = np.random.random((1, 16))
        a = a / a.sum()
        b = np.random.random((1, 16))
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(nkl_metric(a, b, p_B=c, a=0.5) > 0.)
        assert_(nkl_metric(a, b, p_B=c, a=0.5) <
                nkl_metric(a, c, p_B=c, a=0.5))


def test_kld_known_result():
    a = np.array([[1, 0, 0, 0]]).astype(np.float)
    b = np.array([[0, 1, 0, 0]]).astype(np.float)
    c = a + b
    c = c / c.sum()
    assert_(0. < nkl_metric(a, b, p_B=c, a=0.5) < np.log(2))


def test_kld_stats_entropy():
    a = np.array([1, 0, 0, 0], float)
    a = a / a.sum()
    b = np.array([1, 1, 1, 1], float)
    b = b / b.sum()
    c = a + b
    c = c / c.sum()
    expected = (entropy(a, c) + entropy(b, c)) / 2

    calculated = nkl_metric(a, b, p_B=c, a=0.5)
    assert_(0. < calculated < expected)
