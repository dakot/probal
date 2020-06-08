import numpy as np

from src.base.query_strategy import QueryStrategy


class US(QueryStrategy):
    """US

    This class implements different variants of the uncertainty sampling (US) algorithm [1]:
     - least confident (lc),
     - smallest margin (sm),
     - and entropy based uncertainty (entropy).

    Parameters
    ----------
    model: model to be trained
        Model implementing the methods 'fit' and 'predict_proba'.
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample whose maximal posterior probability is minimal.
        Smallest margin (sm) queries the sample whose posterior probability gap between
        the most and the second most probable class label is minimal.
        Entropy queries the sample whose posterior's have the maximal entropy.
    data_set: base.DataSet
        Data set containing samples and class labels.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    model_: model to be trained
        Model implementing the methods 'fit' and 'predict_proba'.
    method_: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample whose maximal posterior probability is minimal.
        Smallest margin (sm) queries the sample whose posterior probability gap between
        the most and the second most probable class label is minimal.
        Entropy queries the sample whose posterior's have the maximal entropy
    data_set_: base.DataSet
        Data set containing samples and class labels.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    """

    LC = 'lc'
    SM = 'sm'
    ENTROPY = 'entropy'

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.model_ = kwargs.get('model', None)
        if self.model_ is None:
            raise TypeError(
                "missing required keyword-only argument 'model'"
            )
        if getattr(self.model_, 'fit', None) is None or getattr(self.model_, 'predict_proba', None) is None:
            raise TypeError(
                "'model' must implement the methods 'fit' and 'predict_proba'"
            )

        self.method_ = kwargs.get('method', US.LC)
        if self.method_ not in [US.LC, US.SM, US.ENTROPY]:
            raise ValueError(
                "supported methods are [{}, {}, {}], the given one is: {}".format(US.LC,
                                                                                  US.SM,
                                                                                  US.ENTROPY,
                                                                                  self.method_)
            )

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
        labeled_indices = self.data_set_.get_labeled_indices()
        self.model_.fit(self.data_set_.X_[labeled_indices], self.data_set_.y_[labeled_indices])
        P = self.model_.predict_proba(self.data_set_.X_[unlabeled_indices])

        return uncertainty_scores(P=P, method=self.method_)


def uncertainty_scores(P, method='lc'):
    """
    Computes uncertainty scores. Three methods are available: least confident (lc), smallest margin (sm), and entropy
    based uncertainty.

    Parameters
    ----------
    P: array-like, shape (n_samples, n_classes)
        Class membership probabilities for each sample.
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample whose maximal posterior probability is minimal.
        Smallest margin (sm) queries the sample whose posterior probability gap between
        the most and the second most probable class label is minimal.
        Entropy queries the sample whose posterior's have the maximal entropy.
    """
    if method == 'lc':
        return 1 - np.max(P, axis=1)
    elif method == 'sm':
        P = -(np.partition(-P, 1, axis=1)[:, :2])
        return 1 - np.abs(P[:, 0] - P[:, 1])
    elif method == 'entropy':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nansum(-P * np.log(P), axis=1)
    else:
        raise ValueError(
            "supported methods are ['lc', 'sm', 'entropy'], the given one is: {}".format(method)
        )
