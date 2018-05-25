# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log


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
cpdef DOUBLE js_div(DOUBLE[::1] v1, DOUBLE[::1] v2) nogil:
    cdef Py_ssize_t d, dim
    cdef DOUBLE xd, yd, md, agg
    dim = v1.shape[0]
    agg = 0.
    for d in prange(dim, nogil=True):
        xd, yd = v1[d], v2[d]
        if xd == yd:
            continue
        md = .5 * (xd + yd)
        if not (md > 0. or md < 0.):
            continue
        if xd > 0.:
            agg += kl(xd, md)
        if yd > 0.:
            agg += kl(yd, md)
    return clip(agg, 0)
