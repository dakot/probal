import copy
import numpy as np

from src.base.query_strategy import QueryStrategy
from src.classifier.parzen_window_classifier import PWC
from src.utils.mathematical_functions import np_ix

from sklearn.utils import check_array


class EER(QueryStrategy):
    """EER

    This class implements the expected error reduction algorithm with different loss functions:
     - log loss (log-loss) [1]
     - and 0/1-loss (zero-one-loss) [1].

    Parameters
    ----------
    n_classes: int
        Number of classes.
    model: model to be trained
        Model implementing the methods 'fit' and and 'predict_proba'.
    S: array-like, shape (n_samples, n_samples), optional (default=None)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j. This matrix can be provided to speed up the expected error
        computation for the Parzen window classifier.
    method: {'log-loss', 'zero-one-loss'}, optional (default='log_loss')
        Variant of expected error reduction to be used.
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    n_classes_: int
        Number of classes.
    model_: model to be trained
        Model implementing the methods 'fit' and 'predict_proba'.
    S_: array-like, shape (n_samples, n_samples), optional (default=None)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j. This matrix can be provided to speed up the expected error
        computation for the Parzen window classifier.
    method_: {'log-loss', 'zero-one-loss'}
        Variant of expected error reduction to be used.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    """

    LOG_LOSS = 'log-loss'
    ZERO_ONE_LOSS = 'zero-one-loss'

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.n_classes_ = kwargs.pop('n_classes', None)
        if not isinstance(self.n_classes_, int) or self.n_classes_ < 2:
            raise TypeError(
                "n_classes must be an integer and at least 2"
            )

        self.model_ = kwargs.get('model', None)
        if self.model_ is not None and (
                getattr(self.model_, 'fit', None) is None or getattr(self.model_, 'predict_proba', None) is None):
            raise TypeError(
                "'model' must implement the methods 'fit' and 'predict_proba'"
            )

        self.S_ = kwargs.pop('S', None)
        self.S_ = self.S_ if self.S_ is None else check_array(self.S_)
        if self.S_ is not None and (
                np.size(self.S_, axis=0) != np.size(self.S_, axis=1) or np.size(self.S_, axis=0) != len(
            self.data_set_)):
            raise ValueError(
                "S must be a squared matrix where the number of rows is equal to the number of samples"
            )

        self.eps_ = float(kwargs.pop('eps', 0.001))

        self.method_ = kwargs.get('method', EER.LOG_LOSS)
        if self.method_ not in [EER.LOG_LOSS, EER.ZERO_ONE_LOSS]:
            raise ValueError(
                "supported methods are [{}, {}], the given one " "is: {}".format(EER.LOG_LOSS,
                                                                                 EER.ZERO_ONE_LOSS,
                                                                                 self.method_)
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

        if self.S_ is None:
            scores = -expected_error_reduction(model=self.model_, X_labeled=X_labeled, y_labeled=y_labeled,
                                               X_unlabeled=X_unlabeled, method=self.method_, eps=self.eps_)
        else:
            scores = -expected_error_reduction_pwc(n_classes=self.n_classes_, S=self.S_, method=self.method_, eps=self.eps_,
                                                   unlabeled_indices=unlabeled_indices, labeled_indices=labeled_indices,
                                                   y_labeled=np.array(self.data_set_.y_[labeled_indices], dtype=int))

        return scores


def expected_error_reduction(model, X_labeled, y_labeled, X_unlabeled, method='log-loss', eps=0.001):
    """
    Computes the expected error. The log-loss and zero-one-loss functions are supported.

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
    method: {'log-loss', 'zero-one-loss'}, optional (default='log-loss')
        Variant of expected error reduction to be used.

    Returns
    -------
    errors: np.ndarray, shape (n_unlabeled_samples)
        The negative expected errors.
    """
    model = copy.deepcopy(model)
    model.fit(X_labeled, y_labeled)
    if isinstance(model, PWC) and eps > 0:
        K = model.predict_proba(X_unlabeled, normalize=False) + eps
        P = K / np.sum(K, axis=1, keepdims=True)
    else:
        P = model.predict_proba(X_unlabeled)
    n_classes = P.shape[1]
    errors = np.zeros(len(X_unlabeled))
    errors_per_class = np.zeros(n_classes)
    for i, x in enumerate(X_unlabeled):
        for yi in range(n_classes):
            model.fit(np.vstack((X_labeled, [x])), np.append(y_labeled, [[yi]]))
            if isinstance(model, PWC) and eps > 0:
                K = model.predict_proba(X_unlabeled, normalize=False) + eps
                P_new = K / np.sum(K, axis=1, keepdims=True)
            else:
                P_new = model.predict_proba(X_unlabeled)
            if method == 'log-loss':
                with np.errstate(divide='ignore', invalid='ignore'):
                    err = -np.nansum(P_new * np.log(P_new))
            elif method == 'zero-one-loss':
                err = np.sum(1 - np.max(P_new, axis=1))
            else:
                raise ValueError(
                    "supported methods are ['log-loss', 'zero-one-loss'], the given one is: {}".format(method)
                )
            errors_per_class[yi] = P[i, yi] * err
        errors[i] = errors_per_class.sum()
    return errors


def expected_error_reduction_pwc(n_classes, S, unlabeled_indices, labeled_indices, y_labeled, method='log-loss', eps=0.001):
    """
    Fast computation of the expected error for the Parzen window classifier.
    The log-loss and zero-one-loss functions are supported.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
    y_labeled: array-like, shape (n_labeled_samples)
        Class labels of all labeled samples.
    unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Indices of unlabeled samples.
    labeled_indices: array-like, shape (n_labeled_samples)
        Indices of labeled samples.
    method: {'log-loss', 'zero-one-loss'}, optional (default='log-loss')
        Variant of expected error reduction to be used.

    Returns
    -------
    errors: np.ndarray, shape (n_unlabeled_samples)
        The negative expected errors.
    """
    Z = np.eye(n_classes)[y_labeled]
    K = (np_ix(S, unlabeled_indices, labeled_indices) @ Z) + eps
    P = K / np.sum(K, axis=1, keepdims=True)

    errors = np.zeros(len(unlabeled_indices))
    errors_per_class = np.zeros(n_classes)

    for i, u_idx in enumerate(unlabeled_indices):
        for yi in range(n_classes):
            K_new = np.array(K)
            K_new[:, yi] += S[unlabeled_indices, u_idx]
            P_new = K_new / np.sum(K_new, axis=1, keepdims=True)
            if method == 'log-loss':
                with np.errstate(divide='ignore', invalid='ignore'):
                    err = -np.nansum(P_new * np.log(P_new))
            elif method == 'zero-one-loss':
                err = np.sum(1 - np.max(P_new, axis=1))
            else:
                raise ValueError(
                    "supported methods are ['log-loss', 'zero-one-loss'], the given one is: {}".format(method)
                )
            errors_per_class[yi] = P[i, yi] * err
        errors[i] = errors_per_class.sum()
    return errors
