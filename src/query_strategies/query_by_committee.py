import numpy as np
import warnings

from src.base.query_strategy import QueryStrategy

from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_random_state


class QBC(QueryStrategy):
    """QBC

    The Query-By-Committee (QBC) algorithm minimizes the version space, which is the set of hypotheses that are
    consistent with the current labeled training data.
    This class implement the query-by-bagging method, which uses the bagging in sklearn to
    construct the committee. So your model should be a sklearn model.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.
    model: model used for committee construction
        Model implementing the methods 'fit' and and 'predict_proba'.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    model_: model used for committee construction
        Model implementing the methods 'fit' and and 'predict_proba'.

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.
    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and bagging.
        In Proceedings of the International Conference on Machine Learning (ICML),
        pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)
        self.model_ = kwargs.get('model', None)
        if self.model_ is not None and (
                getattr(self.model_, 'fit', None) is None or getattr(self.model_, 'predict_proba', None) is None):
            raise TypeError(
                "'model' must implement the methods 'fit' and 'predict_proba'"
            )

    def compute_scores(self, unlabeled_indices):
        """
        Compute score for each unlabeled_indices sample. Score is to be maximized.

        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)

        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled_indices sample.
        """
        labeled_indices = self.data_set_.get_labeled_indices()
        X_labeled = self.data_set_.X_[labeled_indices]
        y_labeled = np.array(self.data_set_.y_[labeled_indices], dtype=int)
        X_unlabeled = self.data_set_.X_[unlabeled_indices]

        n_detected_classes = len(np.unique(y_labeled))
        if n_detected_classes <= 1:
            scores = np.zeros((len(unlabeled_indices)))
        else:
            scores = calc_avg_KL_divergence(model=self.model_, X_labeled=X_labeled, y_labeled=y_labeled,
                                            X_unlabeled=X_unlabeled, random_state=self.random_state_)

        return scores


def calc_avg_KL_divergence(model, X_labeled, y_labeled, X_unlabeled, random_state):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the level of disagreement in QBC.

    Parameters
    ----------
    model: sklearn classifier with predict_proba method
        Model whose expected error reduction is measured.
    X_labeled: array-like, shape (n_labeled_samples, n_features)
        Labeled samples.
    y_labeled: array-like, shape (n_labeled_samples)
        Class labels of labeled samples.
    X_unlabeled: array-like, shape (n_unlabeled_samples)
        Unlabeled samples.

    Returns
    -------
    scores: np.ndarray, shape (n_unlabeled_samples)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning for
        text classification. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """
    random_state = check_random_state(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_features = X_labeled.shape[1]
        max_features = random_state.choice(np.arange(np.ceil(n_features / 2), n_features))
        max_features = int(max_features)
        bagging = BaggingClassifier(base_estimator=model, n_estimators=25,
                                    max_features=max_features, random_state=random_state).fit(X=X_labeled, y=y_labeled)
        est_arr = bagging.estimators_
        est_features = bagging.estimators_features_
        P = [est_arr[e_idx].predict_proba(X_unlabeled[:, est_features[e_idx]]) for e_idx in range(len(est_arr))]
        P = np.array(P)
        P_com = np.mean(P, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.nansum(np.nansum(P * np.log(P / P_com), axis=2), axis=0)
    return scores
