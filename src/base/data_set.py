import numpy as np

from sklearn.utils import check_array, check_consistent_length


class DataSet(object):
    """DataSet

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y: array-like, shape (n_samples)
        Class labels of the given samples X.

    Attributes
    ----------
    X_: numpy.ndarray, shape (n_samples, n_features)
        Samples of the whole data set.
    y_: numpy.ndarray, shape (n_samples)
        Class labels of the given samples X.
    """

    def __init__(self, X, y=None):
        # set parameters
        self.X_ = check_array(X, copy=True)
        self.y_ = np.full(len(self.X_), np.nan) if y is None else check_array(y, force_all_finite=False, copy=True)
        check_consistent_length(self.X_, self.y_)

    def __len__(self):
        """
        Number of all samples in this object.

        Returns
        -------
        n_samples: int
        """
        return len(self.X_)

    def len_labeled(self):
        """
        Number of labeled samples in this object.

        Returns
        -------
        n_samples : int | array-like
            Total number of labeled samples or number of labeled samples for each annotator.
        """
        return len(self.get_labeled_indices())

    def len_unlabeled(self):
        """
        Number of unlabeled samples in this object.

        Returns
        -------
        n_samples : int | array-like
            Total number of unlabeled samples or number of unlabeled samples for each annotator.
        """
        return self.__len__() - self.len_labeled()

    def get_labeled_indices(self):
        """
        Returns indices of all labeled samples.

        Returns
        -------
        labeled_indices: array-like, shape (n_labeled_samples)
            Indices of labeled samples.
        """
        return np.where(~np.isnan(self.y_))[0]

    def get_unlabeled_indices(self):
        """
        Returns indices of all unlabeled samples.

        Returns
        -------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)
            Indices of unlabeled samples.
        """
        return np.where(np.isnan(self.y_))[0]

    def update_entries(self, sample_indices, y):
        """
        Updates labels and confidence scores for given samples X.

        Parameters
        ----------
        sample_indices: array-like, shape (n_samples)
            Indices of samples whose labels are updated.
        y: array-like, shape (n_samples, n_annotators)
            Class labels of the unlabeled samples X.
        """
        y = check_array(y, ensure_2d=False, force_all_finite=False)
        y = y if y.ndim == 2 else y.reshape((-1, 1))

        not_nan = ~np.isnan(y)
        ids = np.zeros_like(self.y_, dtype=bool)
        ids[sample_indices] = not_nan
        self.y_[ids] = y[not_nan]







