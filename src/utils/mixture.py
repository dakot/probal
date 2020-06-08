import numpy as np

from sklearn.utils import check_random_state


class Mixture():
    def __init__(self, priors, base_dists, classes=None):
        """
        Represents a Mixture of distributions.

        Parameters
        ----------
        priors: array-like, shape=[n_distributions]
            Prior probabilities for the given distributions.
        base_dists: array-like, shape=[n_distributions]
            Underlying distributions.
        classes: array-like, shape=[n_distributions]
            Class label of each distribution.
        """
        self.priors = priors
        self.base_dists = base_dists
        if classes is None:
            classes = [None] * len(priors)
        self.classes = classes
        self.n_dists = len(self.priors)
        self.n_classes = len(np.unique(self.classes))

    def rvs(self, size, random_state=None):
        """Random variates of given type.

        Parameters
        ----------
        size: array-like, shape=[n_samples, n_features]
            Sizes of the resulting data set.
        random_state: None|int|RandomState
            Random state to reproduce results.

        Returns
        -------
        X: array-like, shape=[n_samples, n_features]
            Dataset with samples as feature vectors.
        Y: array-like, shape=[n_samples]
            Class label of each sample.
        """
        random_state = check_random_state(random_state)
        n_inst_per_base_dists = random_state.multinomial(size[0], self.priors)
        X = list()
        Y = list()
        for i, n_inst_per_base_dist in enumerate(n_inst_per_base_dists):
            X.append(self.base_dists[i].rvs([n_inst_per_base_dist, *size[1:]]))
            Y.append(np.ones((n_inst_per_base_dist, 1)) * self.classes[i])
        resort = random_state.permutation(size[0])
        X = np.vstack(X)[resort]
        Y = np.array(np.vstack(Y)[resort].ravel(), int)
        return X, Y

    def pdf(self, x, c=None):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x: array-like, shape=[n_samples, n_features]
            Sample to evaluate pdf.
        c: array-like, shape=[n_samples]
            Class labels.

        Returns
        -------
        densities: array-like, shape=[n_samples]
            Density of a sample, if it belongs to class c.
        """
        if c is None:
            c = list(np.unique(self.classes))
        if type(c) is not list:
            c = [c]
        c_idx = np.where([self.classes[i] in c for i in range(self.n_dists)])[0]
        return np.sum([self.priors[i] * self.base_dists[i].pdf(x)
                       for i in c_idx], axis=0)