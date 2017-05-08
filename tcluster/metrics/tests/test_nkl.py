from __future__ import division

import numpy as np
from numpy.testing import assert_

from tcluster.metrics.nkl import nkl_metric


def test_nkl_basic():
    for _ in range(8):
        a = np.random.random((1, 16))
        a = a / a.sum()
        b = np.random.random((1, 16))
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(nkl_metric(a, b, p_B=c, a=0.5) < 0.)
        assert_(nkl_metric(a, b, p_B=c, a=0.5) >
                nkl_metric(a, c, p_B=c, a=0.5))


def test_nkl_known_result():
    a = np.array([[1, 0, 0, 0]]).astype(np.float)
    b = np.array([[0, 1, 0, 0]]).astype(np.float)
    c = a + b
    c = c / c.sum()
    assert_(-np.log(2) < nkl_metric(a, b, p_B=c, a=0.5) < 0.)
