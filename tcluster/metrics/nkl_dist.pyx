# -*- coding: utf-8 -*-
# author: Flavio Martins
# creation date: 1/5/2017
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.math cimport log


ctypedef np.float64_t DOUBLE


@cython.cdivision(True)
cdef inline DOUBLE kl(DOUBLE x, DOUBLE y, DOUBLE z) nogil:
    return x * log(y / z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef DOUBLE nkl_distance(DOUBLE[::1] v1, DOUBLE[::1] v2, DOUBLE[::1] p_B, DOUBLE a) nogil:
    cdef Py_ssize_t d, dim
    cdef DOUBLE xd, yd, bd, a_bd, pd, pc, agg
    dim = v1.shape[0]
    agg = 0.
    for d in prange(dim, nogil=True):
        xd, yd, bd = v1[d], v2[d], p_B[d]
        if not (bd > 0. or bd < 0.):
            continue
        a_bd = a * bd
        pd = (1. - a) * xd + a_bd
        pc = yd
        if pd > 0. and pc > 0.:
            agg += kl(pd, a_bd, pc)
            agg += kl(pc, a_bd, pd)
    return agg
