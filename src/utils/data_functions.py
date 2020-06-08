import numpy as np
import os.path
import pandas as pd

from itertools import compress

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y


def load_data(data_set_name):
    """
    Loads data of given data set name.

    Parameters
    ----------
    data_set_name: str
        Name of the data set.

    Returns
    -------
    X: array-like, shape = [n_samples, n_features]
        Samples as feature vectors.
    y: array-like, shape = [n_samples]
        Class labels of samples.
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    try:
        # look locally for data
        relative_path = '../../data/' + data_set_name + '.csv'
        data_set = pd.read_csv(os.path.join(abs_path, relative_path))
        columns = list(data_set.columns.values)
        features = list(compress(columns, [c.startswith('x_') for c in columns]))

        # Getting data.
        X = np.array(data_set[features], dtype=np.float64)

        # Getting assumed true labels.
        y = data_set['y']
    except FileNotFoundError:
        relative_path = '../../data_set_ids.csv'
        data_set = pd.read_csv(os.path.join(abs_path, relative_path))
        idx = data_set[data_set['name'] == data_set_name].index.values.astype(int)[0]
        data_set = fetch_openml(data_id=data_set.at[idx, 'id'])
        X = data_set.data
        y = data_set.target

    le = LabelEncoder()
    le.fit(y)
    y = le.fit_transform(y)
    X, y = check_X_y(X=X, y=y)

    return X, y
