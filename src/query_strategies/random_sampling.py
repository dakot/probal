import numpy as np

from src.base.query_strategy import QueryStrategy


class RS(QueryStrategy):
    """RS

    This class implements the random sampling (RS) algorithm.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

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
        return np.ones(len(unlabeled_indices))

