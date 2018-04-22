"""
Feature Selection.

Code by Tycho Zhong, Dec 6, 2017.
"""

import numpy as np
from DataComplexity import datacomplexity


class FeatureSelector(object):
    """ Select k best features.
    Parameters:
        k: int or 'all'
            Number of top features to select. The “all” option bypasses selection, for use in a parameter search.

    Attributes:
        feature_length: int
            Number of features.
        scores_: array-like, shape=(n_features,)
            Scores of features.
        ranking_indices: array-like, shape=(n_features,)
            Ranking feature indices by scores.
        support: array-like, shape=(k,)
            Indices of k best (high scores) features, ranking by indices.
        mask: array-like of bool, shape=(n_features,)
            Indicates whether features are selected. An array full of bool, total k Trues.

    Methods:
        fit(X, y): Run score function on (X, y) and get the appropriate features.
        transform(X): Reduce X to the selected features.
        fit_transform(X, y): Fit to data, then transform it.
        get_support(indices=False): Get a mask, or integer index, of the features selected
        get_ranking_indices(): Get ranked feature indices by scores.

    Descriptions:
        fit(X, y): Run score function on (X, y) and get the appropriate features.
            Parameters:
                X : array-like, shape = [n_samples, n_features]
                    The training input samples.
                y : array-like, shape = [n_samples]
                    The target values (class labels in classification, real numbers in regression).
            Returns:
                self : object
                    Returns self.

        transform(X): Reduce X to the selected features.
            Parameters:
                X : array of shape [n_samples, n_features]
                    The input samples.
            Returns:
                X_r : array of shape [n_samples, n_selected_features]
                    The input samples with only the selected features.

        fit_transform(X, y): Fit to data, then transform it.
                Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
            Parameters:
                X : numpy array of shape [n_samples, n_features]
                    Training set.
                y : numpy array of shape [n_samples]
                    Target values.
            Returns:
                X_new : numpy array of shape [n_samples, n_features_new]
                    Transformed array.

        get_support(indices=False): Get a mask, or integer index, of the features selected.
            Parameters:
                indices : bool (default False)
                    If True, the return value will be an array of integers, rather than a boolean mask.
            Returns:
                support : array
                    An index that selects the retained features from a feature vector.
                    If indices is False, this is a boolean array of shape [# input features],
                    in which an element is True iff its corresponding feature is selected for
                    retention. If indices is True, this is an integer array of shape
                    [# output features] whose values are indices into the input feature vector.

        get_ranking_indices(): Get ranked feature indices by scores.
            Returns:
                ranking_indices: array-like
                    Feature indices ranked by feature scores in a descending order.
                    Top elements of the array got high scores.
    """
    def __init__(self, k):
        if k == 'all':
            self.k = k
        else:
            self.k = int(k)

    def _check_params(self, X, y):

        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features; got %r."
                             "Use k='all' to return all features."
                             % self.k)

    def fit(self, X, y):
        self._check_params(X, y)
        self.feature_length = X.shape[1]
        self.scores_ = np.array([self._value(f, y) for f in X.T])
        self.ranking_indices = self.scores_.argsort()[::-1]
        if self.k == 'all':
            self.support = np.sort(self.ranking_indices)
        else:
            self.support = np.sort(self.ranking_indices[:self.k])
        return self

    def transform(self, X):
        return X[:, self.get_support()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        if not hasattr(self, 'support'):
            raise KeyError('please fit() first')
        return self.support if indices else self._get_support_mask()

    def get_ranking_indices(self):
        if not hasattr(self, 'ranking_indices'):
            raise KeyError('please fit() first')
        return self.ranking_indices

    def _get_support_mask(self):
        if not hasattr(self, 'mask'):
            bs = []
            j = 0
            for i in range(self.feature_length):
                if j < len(self.support) and i == self.support[j]:
                    bs.append(True)
                    j += 1
                else:
                    bs.append(False)
            self.mask = np.array(bs)
        return self.mask

    def _value(self, f, y):
        """Calculate importance of feature f."""
        raise NotImplementedError('Unimplemented Class.')


class BSSWSS(FeatureSelector):
    """
    BSSWSS, a features select algorithm from:
    AC Lorena, IG Costa, N Spolaôr, MCPD Souto. Analysis of complexity indices for classification problems: Cancer gene expression data [J]. Neurocomputing, 2012, 75(1): 33-42
    """
    def _value(self, f, labels):
        names = sorted(set(labels))
        wss, bss = np.array([]), np.array([])
        for name in names:
            f_k = f[labels == name]
            f_m = f_k.mean()
            d_m = (f_m - f.mean()) ** 2
            d_z = (f_k - f_m) ** 2
            bss = np.append(bss, d_m)
            wss = np.append(wss, d_z)
        z, m = bss.sum(), wss.sum()
        bsswss = z / m if m > 0 else 0
        return bsswss


class VarianceSelector(FeatureSelector):
    """
    Select Features with high variances.
    """
    def _value(self, f, y):
        return f.var()


class DataComplexitySelector(FeatureSelector):
    """
    Use data complexities as selectors.

    Attributes:
        dc: object
            The Data Complexity object.

    Methods:
        _set_dc(): Call the Data complexity object.
        _multi(X, y): dc calculate data complexities only when data contains just two class label.
            This methods calculate all two-class combinations, and return the average value.

    Descriptions:
        _set_dc(): Call the Data complexity object.

        _multi(X, y): dc calculate data complexities only when data contains just two class label.
            This methods calculate all two-class combinations, and return the average value.
            Parameters:
                X : numpy array of shape [n_samples, n_features]
                    Training set.
                y : numpy array of shape [n_samples]
                    Target values.

            Returns:
                v : Average value of data complexities.
    """
    def _set_dc(self):
        """Select a data complexity."""
        raise NotImplementedError('Unimplemented Class.')

    def _value(self, f, y):
        raise NotImplementedError('Unimplemented Class.')

    def _multi(self, f, y):
        y_names = np.unique(y)
        y_cnt = len(y_names)
        v_mat = np.zeros((y_cnt, y_cnt))
        self._set_dc()
        for i in range(y_cnt):
            for j in range(i + 1, y_cnt):
                v_mat[i, j] = v_mat[j, i] = self.dc.fi_score_value(f[y == y_names[i]], f[y == y_names[j]])
        return v_mat.sum() / (2 * y_cnt * (y_cnt - 1))


class F1Selector(DataComplexitySelector):
    def _set_dc(self):
        self.dc = datacomplexity.DCF1()

    def _value(self, f, y):
        return self._multi(f, y)


class F2Selector(DataComplexitySelector):
    def _set_dc(self):
        self.dc = datacomplexity.DCF2()

    def _value(self, f, y):
        return 1 - self._multi(f, y)


class F3Selector(DataComplexitySelector):
    def _set_dc(self):
        self.dc = datacomplexity.DCF3()

    def _value(self, f, y):
        return self._multi(f, y)
