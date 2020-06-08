import numpy as np

from scipy.special import gammaln, factorial
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS


def kernels(X, Y, metric, **kwargs):
    metric = str(metric)
    if metric == 'rbf':
        gamma = kwargs.pop('gamma')
        return pairwise_kernels(X=X, Y=Y, metric=metric, gamma=gamma)
    elif metric == 'cosine':
        return pairwise_kernels(X=X, Y=Y, metric=metric)
    elif metric == 'categorical':
        gamma = kwargs.pop('gamma')
        return np.exp(-gamma * cdist(XA=X, XB=Y, metric='hamming'))


def euler_beta(a, axis=1):
    """
    Represents Euler beta function: B(a(i)) = Gamma(a(i,1))*...*Gamma(a_n)/Gamma(a(i,1)+...+a(i,n))

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.
    axis: int
        Determines along which axis the Euler beta function is computed.

    Returns
    -------
    result: array-like, shape (m)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=axis)-gammaln(np.sum(a, axis=axis)))


def multinomial_coefficient(a, axis=1):
    """
    Computes Multinomial coefficient: Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.
    axis: int
        Determines along which axis the Euler beta function is computed.

    Returns
    -------
    result: array-like, shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m))
    """
    return factorial(np.sum(a, axis=axis))/np.prod(factorial(a), axis=axis)


def gen_l_vec_list(m_approx, n_classes):
    """
    Creates all possible class labeling vectors for given number of hypothetically acquired labels and given number of
    classes.

    Parameters
    ----------
    m_approx: int
        Number of hypothetically acquired labels..
    n_classes: int,
        Number of classes

    Returns
    -------
    label_vec_list: array-like, shape = [n_label_vectors, n_classes]
        All possible label vectors for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[label_vec_res - (m_approx - sum(labelVec)) <= 1.e-10]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list

def np_ix(A, v1, v2):
    return A[v1][:, v2]
