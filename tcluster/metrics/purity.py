import numpy as np


def purity_score(labels_true, labels_pred):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param labels_true: the ground truth classes
    :type labels_true: numpy.array

    :param labels_pred: the cluster assignments array
    :type labels_pred: numpy.array
    
    :returns: the purity score
    :rtype: float
    """

    A = np.c_[(labels_pred, labels_true)]

    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]
