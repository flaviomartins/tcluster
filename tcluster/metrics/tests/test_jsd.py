from __future__ import division

import numpy as np
from numpy.testing import assert_, assert_almost_equal
from scipy.stats import entropy

from tcluster.metrics.jsd import jensen_shannon_divergence


def test_jsd_basic():
    for _ in range(8):
        a = np.random.random(16)
        a = a / a.sum()
        b = np.random.random(16)
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(jensen_shannon_divergence(a, a) < 1e-4)
        assert_(jensen_shannon_divergence(a, b) > 0.)
        assert_(jensen_shannon_divergence(a, b) >
                jensen_shannon_divergence(a, c))


def test_jsd_known_result():
    a = np.array([1, 0, 0, 0]).astype(np.float)
    b = np.array([0, 1, 0, 0]).astype(np.float)
    assert_almost_equal(jensen_shannon_divergence(a, b), np.log(2))


def test_jsd_stats_entropy():
    a = np.array([1, 0, 0, 0], float)
    a = a / a.sum()
    b = np.array([1, 1, 1, 1], float)
    b = b / b.sum()
    m = a + b
    expected = (entropy(a, m) + entropy(b, m)) / 2

    calculated = jensen_shannon_divergence(a, b)
    assert_almost_equal(calculated, expected)
