# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log

@cython.cdivision(True)
cdef inline double kl(double x, double y, double z) nogil:
    return x * log(y / z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def nkl_dist(np.ndarray[np.double_t, ndim=1] v1, np.ndarray[np.double_t, ndim=1] v2,
            np.ndarray[np.double_t, ndim=1] b, double a):
    cdef int d, dim
    cdef double xd, yd, bd, a_bd, pd, pc, agg
    dim = v1.shape[0]
    agg = 0.
    for d in prange(dim, nogil=True):
        xd, yd, bd = v1[d], v2[d],  b[d]
        a_bd = a * bd
        pd = (1. - a) * xd + a_bd
        pc = yd
        if pd == pc:
            continue
        if not (a_bd > 0. or a_bd < 0.):
            continue
        if pd > 0. and pc > 0.:
            agg += kl(pd, a_bd, pc)
            agg += kl(pc, a_bd, pd)
    return agg
