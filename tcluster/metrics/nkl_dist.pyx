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


@cython.cdivision(True)
cdef inline DTYPE_t kl(DTYPE_t x, DTYPE_t y, DTYPE_t z) nogil:
    return x * log(y / z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef DTYPE_t nkl_dist(DTYPE_t[::1] v1, DTYPE_t[::1] v2, DTYPE_t[::1] b, DTYPE_t a) nogil:
    cdef Py_ssize_t d, dim
    cdef DTYPE_t xd, yd, bd, a_bd, pd, pc, agg
    dim = v1.shape[0]
    agg = 0.
    for d in prange(dim, nogil=True):
        xd, yd, bd = v1[d], v2[d], b[d]
        if not (bd > 0. or bd < 0.):
            continue
        a_bd = a * bd
        pd = (1. - a) * xd + a_bd
        pc = yd
        if pd > 0. and pc > 0.:
            agg += kl(pd, a_bd, pc)
            agg += kl(pc, a_bd, pd)
    return agg
