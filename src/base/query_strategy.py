import numpy as np

from abc import ABC, abstractmethod
from src.base.data_set import DataSet
from sklearn.utils import check_random_state


class QueryStrategy(ABC):
    """QueryStrategy
    Given a pool of labeled and unlabeled samples, a query strategy selects which unlabeled sample is to be queried
    next.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples and class labels.
    random_state: numeric | np.random.RandomState
        Random state for sample selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples and class labels.
    random_state_: numeric | np.random.RandomState
        Random state for sample selection.
    """

    def __init__(self, data_set, **kwargs):
        self.data_set_ = data_set
        if not isinstance(self.data_set_, DataSet):
            raise TypeError(
                "'data_set' must be instance of the class 'base.DataSet'"
            )
        self.random_state_ = check_random_state(kwargs.pop('random_state', None))

    @abstractmethod
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
        pass

    def make_query(self):
        """
        Returns the indices of the samples and the indices of the annotators who shall label the selected samples.

        Returns
        -------
        sample_indices: array-like, shape (n_selected_samples)
            The indices of the samples to be labeled.
        """
        # determine indices of unlabeled samples
        unlabeled_indices = self.data_set_.get_unlabeled_indices()

        # compute score for each sample
        scores = self.compute_scores(unlabeled_indices)

        # determine sample with maximal score
        sample_ids = np.where(scores == np.nanmax(scores))[0]
        random_id = self.random_state_.randint(low=0, high=len(sample_ids))
        sample_indices = unlabeled_indices[sample_ids[random_id]]

        return [sample_indices]
