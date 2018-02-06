# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef double clip(double x, double low) nogil:
    if x < low:
        return low
    return x


@cython.cdivision(True)
cdef inline DTYPE_t kl(DTYPE_t x, DTYPE_t y) nogil:
    return x * log(x / y)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef DTYPE_t js_div(DTYPE_t[::1] v1, DTYPE_t[::1] v2) nogil:
    cdef Py_ssize_t d, dim
    cdef DTYPE_t xd, yd, md, agg
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
