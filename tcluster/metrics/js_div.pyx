# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log

@cython.cdivision(True)
cdef inline double kl(double x, double y) nogil:
    return x * log(x / y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def js_div(np.ndarray[np.double_t, ndim=1] v1, np.ndarray[np.double_t, ndim=1] v2):
    cdef int d, dim
    cdef double xd, yd, md, agg
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
    return agg
