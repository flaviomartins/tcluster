"""
The :mod:`tcluster.cluster` module gathers clustering algorithms
algorithms.
"""

from .k_means_ import k_means, KMeans, MiniBatchKMeans, SampleKMeans

__all__ = ['KMeans',
           'MiniBatchKMeans',
           'SampleKMeans',
           'k_means']
