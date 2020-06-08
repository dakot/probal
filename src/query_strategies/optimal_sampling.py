import copy
import numpy as np

from src.base.query_strategy import QueryStrategy
from src.utils.evaluation_functions import misclf_rate

from sklearn.utils import check_X_y, check_array


class OS(QueryStrategy):
    """OS

    This class implements the optimal sampling (OS) algorithm. It assumes that the class labels of all samples are
    known in advance. In each iteration, it selects the sample leading to the minimal error on all available samples.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    y: array-like, shape (n_samples)
        True class labels of all samples in 'data_set'.
    model: model to be trained
        Model implementing the methods 'fit' and 'predict'.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    y_: array-like, shape (n_samples)
        True class labels of all samples in 'data_set_'.
    model: model to be trained
        Model implementing the methods 'fit' and 'predict'.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.model_ = kwargs.get('model', None)
        if self.model_ is not None and (
                getattr(self.model_, 'fit', None) is None or getattr(self.model_, 'predict', None) is None):
            raise TypeError(
                "'model' must implement the methods 'fit' and 'predict'"
            )

        self.S_ = kwargs.pop('S', None)
        self.S_ = self.S_ if self.S_ is None else check_array(self.S_)
        if self.S_ is not None and (
                np.size(self.S_, axis=0) != np.size(self.S_, axis=1) or np.size(self.S_, axis=0) != len(
            self.data_set_)):
            raise ValueError(
                "S must be a squared matrix where the number of rows is equal to the number of samples"
            )

        self.y_ = kwargs.pop('y', None)
        check_X_y(X=self.data_set_.X_, y=self.y_)

    def compute_scores(self, unlabeled_indices):
        """Compute score for each unlabeled_indices sample. Score is to be maximized.

        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)

        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled_indices sample.
        """
        labeled_indices = self.data_set_.get_labeled_indices()
        if self.S_ is None:
            scores = -compute_errors(model=self.model_, X=self.data_set_.X_, y=self.y_,
                                     unlabeled_indices=unlabeled_indices,
                                     labeled_indices=labeled_indices)
        else:
            scores = -compute_errors_pwc(S=self.S_, y=self.y_, unlabeled_indices=unlabeled_indices,
                                         labeled_indices=labeled_indices)

        return scores


def compute_errors(model, X, y, unlabeled_indices, labeled_indices):
    """
    Computes the actual error when adding an unlabeled sample to the set of labeled samples.

    Parameters
    ----------
    model: sklearn classifier with 'fit' and 'predict' method
        Model whose actual error is measured.
    X: array-like, shape (n_samples, n_features)
        All available samples
    y: array-like, shape (n_samples)
        Class labels of all available samples.
    unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Indices of unlabeled samples.
    labeled_indices: array-like, shape (n_labeled_samples)
        Indices of labeled samples.

    Returns
    -------
    errors: np.ndarray, shape (n_unlabeled_samples)
        The actual error for each unlabeled sample.
    """
    model = copy.deepcopy(model)
    errors = np.zeros(len(unlabeled_indices))
    n_labeled_samples = len(labeled_indices)
    labeled_indices_new = np.zeros(n_labeled_samples + 1, dtype=int)
    labeled_indices_new[:n_labeled_samples] = labeled_indices
    for i in range(len(unlabeled_indices)):
        labeled_indices_new[-1] = unlabeled_indices[i]
        model.fit(X[labeled_indices_new], y[labeled_indices_new])
        errors[i] = misclf_rate(y_true=y, y_pred=model.predict(X))
    return errors


def compute_errors_pwc(S, y, unlabeled_indices, labeled_indices):
    """
    Fast computation of the actual error for the Parzen window classifier when adding an unlabeled sample to the set of
    labeled samples.

    Parameters
    ----------
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
    y: array-like, shape (n_samples)
        Class labels of all available samples.
    unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Indices of unlabeled samples.
    labeled_indices: array-like, shape (n_labeled_samples)
        Indices of labeled samples.

    Returns
    -------
    errors: np.ndarray, shape (n_unlabeled_samples)
        The actual error for each unlabeled sample.
    """
    n_classes = len(np.unique(y))
    errors = np.zeros(len(unlabeled_indices))
    n_labeled_samples = len(labeled_indices)
    labeled_indices_new = np.zeros(n_labeled_samples + 1, dtype=int)
    labeled_indices_new[:n_labeled_samples] = labeled_indices
    for i in range(len(unlabeled_indices)):
        labeled_indices_new[-1] = unlabeled_indices[i]
        Z = np.eye(n_classes)[y[labeled_indices_new]]
        K = S[:, labeled_indices_new] @ Z
        y_pred = np.argmax(K, axis=1)
        errors[i] = misclf_rate(y_true=y, y_pred=y_pred)
    return errors
