"""
The :mod:`tcluster.cluster` module gathers clustering algorithms
algorithms.
"""

from .kmeans import k_means, KMeans, SampleKMeans

__all__ = ['KMeans',
           'SampleKMeans',
           'k_means']
