import numpy as np
import itertools

from sklearn.utils import check_array

from src.base.query_strategy import QueryStrategy
from src.utils.mathematical_functions import gen_l_vec_list, euler_beta, multinomial_coefficient


class PAL(QueryStrategy):
    """PAL

    This class implements multi-class probabilistic active learning (McPAL) [1] strategy.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples and class labels.
    n_classes: int
        Number of classes.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    m_max: int, optional (default=2)
        Maximal number of hypothetically acquired labels.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    n_classes_: int
        Number of classes.
    alpha_c_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    m_max_: int, optional (default=2)
        Maximal number of hypothetically acquired labels.
    S_: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    densities_: array-like, shape (n_samples)
        Density for all labeled and unlabeled samples.
        Default density is one for each sample.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra Spiliopoulou.
        Multi-Class Probabilistic Active Learning,
        vol. 285 of Frontiers in Artificial Intelligence and Applications, pages 586-594. IOS Press, 2016
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.n_classes_ = kwargs.pop('n_classes', None)
        if not isinstance(self.n_classes_, int) or self.n_classes_ < 2:
            raise TypeError(
                "'n_classes' must be an integer and at least 2"
            )

        self.m_max_ = kwargs.pop('m_max', 2)
        if not isinstance(self.n_classes_, int) or self.n_classes_ < 0:
            raise TypeError(
                "'m_max' must be positive integer"
            )

        self.S_ = check_array(kwargs.pop('S', np.eye(len(self.data_set_))))
        if np.size(self.S_, axis=0) != np.size(self.S_, axis=1) or np.size(self.S_, axis=0) != len(self.data_set_):
            raise ValueError(
                "'S' must be a matrix with the shape (n_samples, n_samples)"
            )

        self.densities_ = np.sum(self.S_, axis=1)

        self.alpha_c_ = kwargs.pop('alpha_c', 1)

    def compute_scores(self, unlabeled_indices):
        """Compute score for each unlabeled sample. Score is to be maximized.

        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)

        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled sample.
        """
        # compute frequency estimates for evaluation set (K_x) and candidate set (K_c)
        labeled_indices = self.data_set_.get_labeled_indices()
        y_labeled = np.array(self.data_set_.y_[labeled_indices].reshape(-1), dtype=int)
        Z = np.eye(self.n_classes_)[y_labeled]
        K_x = self.S_[:, labeled_indices] @ Z
        K_c = K_x[unlabeled_indices]

        # calculate loss reduction for each unlabeled sample
        gains = pal_gain(K_c=K_c, alpha_c=self.alpha_c_)
        gains *= self.densities_[unlabeled_indices]

        return gains


def pal_gain(K_c, m_max=2, alpha_c=1):
    """
    Calculates the expected performance gains given a maximal number of hypothetically acquired labels,
    the observed kernel frequency estimates and prior for the Dirichlet distribution.

    Parameters
    ----------
    K_c: array-like, shape (n_samples, n_classes)
        Observed kernel frequency estimates of the candidate samples.
    m_max: int, optional (default=2)
        Maximal number of hypothetically acquired labels.
    alpha_c : int | array-like, shape (n_classes), optional (default=1)
        Prior probabilities for the Dirichlet distribution.
        Default is 1 for all classes.

    Returns
    -------
    gains: array-like, shape (n_samples)
        Expected performance gains for given parameters.
    """
    n_classes = len(K_c[0])
    n_samples = len(K_c)

    # uniform risk matrix
    R = 1 - np.eye(n_classes)

    # generate labeling vectors for all possible m values
    l_vec_list = np.vstack([gen_l_vec_list(m, n_classes) for m in range(m_max + 1)])
    m_list = np.sum(l_vec_list, axis=1)
    n_l_vecs = len(l_vec_list)

    # compute optimal decision for all combination of k- and l-vectors
    k_l_vec_list = np.swapaxes(np.tile(K_c, (n_l_vecs, 1, 1)), 0, 1) + l_vec_list
    y_hats = np.argmin(k_l_vec_list @ R, axis=2)

    # add prior_classes to k-vectors
    alpha_c = alpha_c * np.ones(n_classes)
    K_c = np.asarray(K_c) + alpha_c

    # all combination of k-, l-, and prediction indicator vectors
    combs = [K_c, l_vec_list, np.eye(n_classes)]
    combs = np.asarray([list(elem) for elem in list(itertools.product(*combs))])

    # three factors of the closed form solution
    factor_1 = 1 / euler_beta(K_c)
    factor_2 = multinomial_coefficient(l_vec_list)
    factor_3 = euler_beta(np.sum(combs, axis=1)).reshape(n_samples, n_l_vecs, n_classes)

    # expected risk for each m
    m_sums = np.asarray(
        [factor_1[k_idx] * np.bincount(m_list, factor_2 * [R[:, y_hats[k_idx, l_idx]] @ factor_3[k_idx, l_idx]
                                                           for l_idx in range(n_l_vecs)]) for k_idx in
         range(n_samples)])

    # compute performance gains as risk reductions
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize performance gains by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)