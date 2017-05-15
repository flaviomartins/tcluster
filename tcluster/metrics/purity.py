import numpy as np


def purity_score(labels_true, labels_pred):
    """Purity between two clusterings
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    Returns
    -------
    purity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    """

    A = np.c_[(labels_pred, labels_true)]

    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]
