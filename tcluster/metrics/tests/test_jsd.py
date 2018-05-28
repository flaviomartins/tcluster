from __future__ import division

import os
import numpy as np
from numpy.testing import assert_, assert_almost_equal
from scipy.spatial.distance import pdist
from scipy.spatial.tests.test_distance import _assert_within_tol
from scipy.stats import entropy

from tcluster.metrics.jsd import jensenshannon, np_jensenshannon_distance, np_jensenshannon_divergence
from tcluster.metrics.js_div import jensenshannon_distance, jensenshannon_divergence


_filenames = [
    "iris.txt",
    "pdist-boolean-inp.txt",
    "pdist-double-inp.txt",
    "pdist-jensenshannon-ml-iris.txt",
    "pdist-jensenshannon-ml.txt",
]


# A hashmap of expected output arrays for the tests. These arrays
# come from a list of text files, which are read prior to testing.
# Each test loads inputs and outputs from this dictionary.
eo = {}


def load_testing_files():
    for fn in _filenames:
        name = fn.replace(".txt", "").replace("-ml", "")
        fqfn = os.path.join(os.path.dirname(__file__), 'data', fn)
        fp = open(fqfn)
        eo[name] = np.loadtxt(fp)
        fp.close()


load_testing_files()


def test_jsd_basic():
    for _ in range(8):
        a = np.random.random(16)
        a = a / a.sum()
        b = np.random.random(16)
        b = b / b.sum()
        c = a + b
        c = c / c.sum()

        assert_(jensenshannon_divergence(a, a) < 1e-4)
        assert_(jensenshannon_divergence(a, b) > 0.)
        assert_(jensenshannon_divergence(a, b) >
                jensenshannon_divergence(a, c))


def test_jsd_known_result():
    a = np.array([1, 0, 0, 0], float)
    b = np.array([0, 1, 0, 0], float)
    assert_almost_equal(jensenshannon_divergence(a, b), np.log(2))


def test_jsd_known_result1():
    assert_almost_equal(jensenshannon([1.0, 0.0], [0.5, 0.5]), 0.46450140402245893)
    assert_almost_equal(jensenshannon_distance(np.asarray([1.0, 0.0], float), np.array([0.5, 0.5], float)), 0.46450140402245893)
    assert_almost_equal(np_jensenshannon_distance(np.array([1.0, 0.0], float), np.array([0.5, 0.5], float)), 0.46450140402245893)


def test_jsd_known_result2():
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.8325546111576977)
    assert_almost_equal(jensenshannon_distance(np.asarray([1.0, 0.0, 0.0], float), np.array([0.0, 1.0, 0.0], float)), 0.8325546111576977)
    assert_almost_equal(np_jensenshannon_distance(np.array([1.0, 0.0, 0.0], float), np.array([0.0, 1.0, 0.0], float)), 0.8325546111576977)


def test_jsd_stats_entropy():
    a = np.array([1, 0, 0, 0], float)
    a = a / a.sum()
    b = np.array([1, 1, 1, 1], float)
    b = b / b.sum()
    m = a + b
    expected = (entropy(a, m) + entropy(b, m)) / 2

    calculated = jensenshannon_divergence(a, b)
    assert_almost_equal(calculated, expected)


# def test_pdist_jensenshannon_random():
#     eps = 1e-08
#     X = eo['pdist-double-inp']
#     Y_right = eo['pdist-jensenshannon']
#     Y_test1 = pdist(X, jensenshannon)
#     _assert_within_tol(Y_test1, Y_right, eps)
#
#
# def test_pdist_jensenshannon_random_float32():
#     eps = 1e-07
#     X = np.float32(eo['pdist-double-inp'])
#     Y_right = eo['pdist-jensenshannon']
#     Y_test1 = pdist(X, jensenshannon)
#     _assert_within_tol(Y_test1, Y_right, eps)
#
#
# def test_pdist_jensenshannon_random_nonC():
#     eps = 1e-08
#     X = eo['pdist-double-inp']
#     Y_right = eo['pdist-jensenshannon']
#     Y_test2 = pdist(X, jensenshannon)
#     _assert_within_tol(Y_test2, Y_right, eps)
#
#
# def test_pdist_jensenshannon_iris():
#     eps = 1e-12
#     X = eo['iris']
#     Y_right = eo['pdist-jensenshannon-iris']
#     Y_test1 = pdist(X, jensenshannon_distance)
#     _assert_within_tol(Y_test1, Y_right, eps)
#
#
# def test_pdist_jensenshannon_iris_float32():
#     eps = 1e-04
#     X = np.float32(eo['iris'])
#     Y_right = eo['pdist-jensenshannon-iris']
#     Y_test1 = pdist(X, jensenshannon)
#     _assert_within_tol(Y_test1, Y_right, eps)
#
#
# def test_pdist_jensenshannon_iris_nonC():
#     eps = 5e-13
#     X = eo['iris']
#     Y_right = eo['pdist-jensenshannon-iris']
#     Y_test2 = pdist(X, jensenshannon)
#     _assert_within_tol(Y_test2, Y_right, eps)
