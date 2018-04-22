
"""
Simple ECOC Classifier
edit by Tycho Zhong
"""

import numpy as np
import warnings
from Decoding.Decoder import get_decoder
import copy


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]


def _fit_ternary(estimator, X, y):
    """Fit a single ternary estimator. not offical editing.
        delete item from X and y when y = 0
        edit by elfen.
    """
    X, y = X[y != 0], y[y != 0]
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        warnings.warn('only one class')
    else:
        estimator = copy.deepcopy(estimator)
        estimator.fit(X, y)
    return estimator


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]
    return score


def _sigmoid_normalize(X):
    return 1 / (1 + np.exp(-X))


def _min_max_normalize(X):
    """Min max normalization
    warning: 0 value turns not 0 in most cases.
    """
    res = []
    for x in X:
        x_min, x_max = min(x), max(x)
        x_range = x_max - x_min
        res.append([float(i-x_min)/x_range for i in x])
    return np.array(res)


class SimpleECOCClassifier:
    """ A simple ECOC classifier
    Parameters:
        estimator: object
            unfitted base classifier object.
        code_matrix: 2-d array
            code matrix (Classes×Dichotomies).
        decoder: str
            indicates the type of decoder, get a decoder object immediately when initialization.
            For more details, check Decoding.Decoder.get_decoder.
        soft: bool, default True.
            Whether to use soft distance to decode.

    Attributes:
        estimator_type: str, {'decision_function','predict_proba'}
            which type the estimator belongs to.
            'decision_function' - predict value range (-∞,+∞)
            'predict_proba' - predict value range [0,1]
        classes_: set
            the set of labels.
        estimators_: 1-d array
            trained classifers.

    Methods:
        fit(X, y): Fit the model according to the given training data.
        predict(X): Predict class labels for samples in X.
        fit_predict(X, y, test_X): fit(X, y) then predict(X_test).

    Descriptions:
        fit(X, y): Fit the model according to the given training data.
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                y: array-like, shape = [n_samples]
                    Target vector relative to X
            Returns:
                self: object
                    Returns self.

        predict(X): Predict class labels for samples in X.
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Samples.
            Returns:
                C: array, shape = [n_samples]
                    Predicted class label per sample.

        fit_predict(X, y, test_X): fit(X, y) then predict(X_test).
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                y: array-like, shape = [n_samples]
                    Target vector relative to
                X_test: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Samples.
            Returns:
                C: array, shape = [n_samples]
                    Predicted class label per sample.
            Notes: This is a combination of two methods fit & predict, with X, y for fit and X_test for predict.
                Run fit first and then run predict
    """
    def __init__(self, estimator, code_matrix, decoder='AED', soft=True):
        self.estimator = estimator  # classifier
        self.code_matrix = code_matrix  # code matrix
        self.decoder = get_decoder(decoder)  # decoder
        self.soft = soft  # if using soft distance.

    def fit(self, X, y):
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        else:
            self.estimator_type = 'predict_proba'

        self.classes_ = np.unique(y)
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)

        self.estimators_ = [_fit_ternary(self.estimator, X, Y[:, i]) for i in
                            range(Y.shape[1])]
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        Y = np.array([_predict_binary(self.estimators_[i], X) for i in range(len(self.estimators_))]).T

        # Y_min, Y_max = Y.min(), Y.max()
        # print('%s: (%f , %f)' % (self.estimator_type, Y_min, Y_max))

        if self.estimator_type == 'decision_function':
            Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]

        Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]

        pred = self.decoder.decode(Y, self.code_matrix).argmin(axis=1)
        return self.classes_[pred]

    def fit_predict(self, X, y, test_X):
        self.fit(X, y)
        return self.predict(test_X)

