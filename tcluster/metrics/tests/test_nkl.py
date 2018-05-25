from __future__ import division

import numpy as np
from numpy.testing import assert_, assert_almost_equal

from tcluster.metrics.nkl import np_nkl_distance
from tcluster.metrics.nkl_dist import nkl_distance


def test_nkl_basic():
    for _ in range(8):
        a = np.random.random(16)
        a = a / a.sum()
        b = np.random.random(16)
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(nkl_distance(a, b, c, 0.5) < 0.)
        assert_(nkl_distance(a, b, c, 0.5) >
                nkl_distance(a, c, c, 0.5))


def test_nkl_known_result():
    a = np.array([1, 0, 0, 0]).astype(np.float)
    b = np.array([0, 1, 0, 0]).astype(np.float)
    c = a + b
    c = c / c.sum()
    assert_(-np.log(2) < nkl_distance(a, b, c, 0.5) < 0.)


def test_nkl_special_xlogy():
    a = np.array([1, 0, 0, 0], float)
    a = a / a.sum()
    b = np.array([1, 1, 1, 1], float)
    b = b / b.sum()
    c = a + b
    c = c / c.sum()

    expected = np_nkl_distance(a, b, p_B=c, a=0.5)
    calculated = nkl_distance(a, b, c, 0.5)
    assert_almost_equal(calculated, expected)
