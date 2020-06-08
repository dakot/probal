import numpy as np

from sklearn.metrics import accuracy_score


def misclf_rate(y_true, y_pred):
    """
    Computes the misclassification rate given the true and predicted labels.

    Parameters
    ----------
    y_true: array-like, shape (n_labels)
        Ground truth (correct) labels.
    y_pred: array-like, shape (n_labels)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    misclf_rate: float in [0, 1]
        Fraction of wrongly classified samples.
    """
    return 1 - accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)


def eval_perfs(clf, X_train, y_train, X_test, y_test, perf_funcs, perf_results=None):
    """
    Evaluates performances of a classifier on train and test data given a list of performance functions.
    Stores evaluated performances into a given dictionary.

    Parameters
    ----------
    clf: model
        Model to be evaluated. Must implement predict method.
    X_train: array-like, shape (n_training_samples, n_features)
        Training samples.
    y_train: array-like, shape (n_training_samples)
        Class labels of training samples.
    X_train: array-like, shape (n_test_samples, n_features)
        Test samples.
    y_train: array-like, shape (n_test_samples)
        Class labels of test samples.
    perf_funcs: dict-like
        Dictionary of performance functions to be used where 'y_true' and 'y_pred' are expected as parameters.
        An example entry is given by perf_funcs['key'] = [perf_func, kwargs], where 'kwargs' are keyword-only arguments
        passed to the 'predict' method of 'clf'.
    perf_results: dict-like, optional (default={})
        Dictionary of performances.

    Returns
    -------
    perf_results: dict-like
        Dictionary of updated performances.
    """
    # check parameters
    if not callable((getattr(clf, 'predict', None))):
        raise TypeError("'clf' must be an instance with the method 'predict'")
    perf_results = {} if perf_results is None else perf_results
    if not isinstance(perf_results, dict):
        raise TypeError("'perf_results' must be a dictionary")
    if not isinstance(perf_funcs, dict):
        raise TypeError("'perf_funcs' must be a dictionary")

    # create storage for performance measurements
    if len(perf_results) == 0:
        for key in perf_funcs:
            perf_results['train-' + key] = []
            perf_results['test-' + key] = []

    # compute performances
    for perf_name, perf_func in perf_funcs.items():
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        perf_results['train-' + perf_name].append(perf_func(y_pred=y_train_pred, y_true=y_train))
        perf_results['test-' + perf_name].append(perf_func(y_pred=y_test_pred, y_true=y_test))

    return perf_results


def compute_statistics(dic):
    """
    Calculation of means and standard deviations of the lists stored in a given dictionary.

    Parameters
    ----------
    dic: dictionary, shape = {key_1: [[...]], key_n: [[...]]}
        Dictionary with lists of lists as values.

    Returns
    -------
    statistics: dictionary, shape = {key-1-mean: [...], key-1-std: [...], ..., key-n-mean: [...], key-n-std: [...]}
        Dictionary with lists of std and mean values.
    """
    if not isinstance(dic, dict):
        raise TypeError("'dic' must be a dictionary")

    statistics = {}

    # Iteration over all keys.
    for key in dic:
        # Calculation of means and std values.
        arr = np.array(dic[key])
        if np.size(arr, axis=0) > 1:
            statistics[key + '-mean'] = np.mean(arr, axis=0)
            statistics[key + '-std'] = np.std(arr, ddof=1, axis=0)
        else:
            statistics[key + '-mean'] = dic[key][0]
            statistics[key + '-std'] = len(dic[key][0]) * 0

    return statistics
