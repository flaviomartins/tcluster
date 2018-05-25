# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log, sqrt


ctypedef np.float64_t DOUBLE


cdef double clip(double x, double low) nogil:
    if x < low:
        return low
    return x


@cython.cdivision(True)
cdef inline DOUBLE kl(DOUBLE x, DOUBLE y) nogil:
    return x * log(x / y)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DOUBLE jensenshannon_divergence(DOUBLE[::1] v1, DOUBLE[::1] v2) nogil:
    cdef Py_ssize_t d, dim
    cdef DOUBLE xd, yd, md, agg
    dim = v1.shape[0]
    agg = 0.
    for d in prange(dim, nogil=True):
        xd, yd = v1[d], v2[d]
        md = (xd + yd) / 2.0
        if xd > 0.:
            agg += kl(xd, md)
        if yd > 0.:
            agg += kl(yd, md)
    return clip(agg / 2.0, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DOUBLE jensenshannon_distance(DOUBLE[::1] v1, DOUBLE[::1] v2) nogil:
    return sqrt(jensenshannon_divergence(v1, v2))
