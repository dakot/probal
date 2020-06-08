import copy
import numpy as np

from src.base.query_strategy import QueryStrategy
from src.query_strategies.mdsp import MDSP

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


class ALCE(QueryStrategy):
    """ALCE
    Active Learning with Cost Embedding [1] is a multi-class cost-sensitive query strategy assuming that
    each class has at least one sample in the labeled pool.

    Parameters
    ----------
    C: array-like, shape=(n_classes, n_classes)
        The ith row, jth column represents the cost of the ground truth being
        ith class and prediction as jth class.
    mds_params: dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    nn_params: dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    embed_dim : int, optional (default: None)
        Number of dimensions of the embedded space. If it is None, embed_dim = n_classes is defined.
    base_regressor: sklearn regressor

    Attributes:
    -----------
    C_: array-like, shape=(n_classes, n_classes)
        The ith row, jth column represents the cost of the ground truth being
        ith class and prediction as jth class.
    mds_params_: dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    nn_params_: dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    embed_dim : int, optional (default: None)
        Number of dimensions of the embedded space. If it is None, embed_dim = n_classes is defined.
    base_regressor_: sklearn regressor
    regressors_: array-like, shape (embed_dim)
        Copy of base regressor for each embedded dimension.
    n_classes_: int
        Number of classes.
    class_embed_: array-like, shape (n_classes, n_samples)
        Stores the position of the dataset in the embedding space.

    References:
    -----------
    [1] Kuan-Hao, and Hsuan-Tien Lin. "A Novel Uncertainty Sampling Algorithm
        for Cost-sensitive Multiclass Active Learning", In Proceedings of the
        IEEE International Conference on Data Mining (ICDM), 2016
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.base_regressor_ = kwargs.pop('base_regressor', None)

        self.C_ = kwargs.pop('C', None)
        if self.C_ is not None:
            self.C_ = check_array(self.C_)
            if np.size(self.C_, axis=0) != np.size(self.C_, axis=1):
                raise ValueError(
                    "C must be a square matrix"
                )

        self.n_classes_ = len(self.C_)
        self.embed_dim_ = kwargs.pop('embed_dim', None)
        if self.embed_dim_ is None:
            self.embed_dim_ = self.n_classes_
        self.regressors_ = [copy.deepcopy(self.base_regressor_) for _ in range(self.embed_dim_)]

        self.mds_params_ = {
            'metric': False,
            'n_components': self.embed_dim_,
            'n_uq': self.n_classes_,
            'max_iter': 300,
            'eps': 1e-6,
            'dissimilarity': "precomputed",
            'n_init': 8,
            'n_jobs': 1,
            'random_state': copy.deepcopy(self.random_state_)
        }
        mds_params = kwargs.pop('mds_params', {})
        self.mds_params_.update(mds_params)

        self.nn_params_ = kwargs.pop('nn_params', {})
        self.nn_ = NearestNeighbors(n_neighbors=1, **self.nn_params_)

        dissimilarity = np.zeros((2 * self.n_classes_, 2 * self.n_classes_))
        dissimilarity[:self.n_classes_, self.n_classes_:] = self.C_
        dissimilarity[self.n_classes_:, :self.n_classes_] = self.C_.T
        mds_ = MDSP(**self.mds_params_)
        embedding = mds_.fit(dissimilarity).embedding_

        self.class_embed_ = embedding[:self.n_classes_, :]
        self.nn_.fit(embedding[self.n_classes_:, :])

    def compute_scores(self, unlabeled_indices):
        """
        Compute score for each unlabeled sample. Score is to be maximized.

        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)

        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled sample.
        """
        X_unlabeled = self.data_set_.X_[unlabeled_indices]
        labeled_indices = self.data_set_.get_labeled_indices()
        X_labeled = self.data_set_.X_[labeled_indices]
        y_labeled = self.data_set_.y_[labeled_indices]

        return cost_sensitive_uncertainty(X_unlabeled=X_unlabeled, X_labeled=X_labeled, y_labeled=y_labeled,
                                          nn=self.nn_, regressors=self.regressors_, class_embed=self.class_embed_,
                                          embed_dim=self.embed_dim_)


def cost_sensitive_uncertainty(X_unlabeled, X_labeled, y_labeled, nn, regressors, class_embed, embed_dim):
    """
    Computes the uncertainty scores according to active learning with cost embedding (ALCE).

    Parameters
    ----------
    X_labeled: array-like, shape (n_labeled_samples, n_features)
        Labeled samples.
    y_labeled: array-like, shape (n_labeled_samples)
        Class labels of labeled samples.
    X_unlabeled: array-like, shape (n_unlabeled_samples)
        Unlabeled samples.
    nn: sklearn.neighbors import NearestNeighbors
        k Nearest Neighbors classifier from sklearn.
    regressors: array-like, shape (embed_dim)
        List of base regressors, one for each embedded dimension.
    class_embed: array-like, shape (n_classes, n_samples)
        Stores the position of the dataset in the embedding space.
    embed_dim : int, optional (default: None)
        Number of dimensions of the embedded space. If it is None, embed_dim = n_classes is defined.
    """
    if len(np.unique(y_labeled)) != len(class_embed):
        return -np.zeros(len(X_unlabeled))

    pred_embed = np.zeros((len(X_unlabeled), embed_dim))
    y_labeled = np.array(y_labeled, np.int)
    for i in range(embed_dim):
        regressors[i].fit(X_labeled, class_embed[y_labeled, i])
        pred_embed[:, i] = regressors[i].predict(X_unlabeled)

    dist, _ = nn.kneighbors(pred_embed)
    return dist[:, 0]