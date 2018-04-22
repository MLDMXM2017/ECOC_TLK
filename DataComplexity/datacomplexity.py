"""
Data Complexities are measures of separability of a two-class data.
Code by Tycho Zhong, Dec 6, 2017.
"""


import numpy as np


def get_data_complexity(dc_code):
    """ Get a data complexity object by dc_code.
    Parameters:
        dc_code: str
            indicate the type of data complexity to be returned.
            Recognizable de_code list below:
            'F1' - Maximum Fisher’s discriminant ratio (noted as F1)
            'F2' - Volume of overlap region (noted as F2)
            'F3' - Maximal (individual) feature efficiency (noted as F3)
            'N1' - Fraction of points on class boundary (noted as N1)
            'N2' - Ratio of average intra/inter class nearest neighbor distance (noted as N2)
            'N3' - Error rate of 1 nearest neighbor classifier (noted as N3)

    Returns:
        o: object
        data complexity object.
    """
    if dc_code == 'F1':
        return DCF1()
    elif dc_code == 'F2':
        return DCF2()
    elif dc_code == 'F3':
        return DCF3()
    elif dc_code == 'N1':
        return DCN1()
    elif dc_code == 'N2':
        return DCN2()
    elif dc_code == 'N3':
        return DCN3()
    else:
        raise ValueError('data complexity: Unknown value for dc_code: %s' % dc_code)


class DataComplexity(object):
    """The data complexity measure complexities of binary-class-data.
    Methods:
        score(X, y): Return the complexity score of X after checking X & y.
            SMALL SCORES MEAN LOW DATA COMPLEXITIES.
            If y contains more than 2 classes, for anyone class, seen this class as positive and the other as negative,
            calculate data complexities. And then return mean value of those data complexities.
        _score(X, y): Return the complexity score of X, should be implemented by subclasses.
        _check_Xy( X, y): Check if X and y meet demands.

    Description:
        score(X, y): Return the complexity score of X after checking X & y.
        Parameters:
            X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
            y: array-like, shape = [n_samples]
                    Target vector relative to X
        Notes:
            The public method is socre(X, y), of which the returning SMALL SCORES MEAN LOW DATA COMPLEXITIES.
    """
    def score(self, X, y):
        """ Return the complexity score of X, SMALL SCORES MEAN LOW DATA COMPLEXITIES.
        If y contains more than 2 classes, for anyone class, seen this class as positive and the other as negative,
        calculate data complexities. And then return mean value of those data complexities.
        X: data
        y: label of data, y should contains only {-1, 1} or {0, 1}
        """
        X, y = self._check_Xy(X, y)
        y_ = np.unique(y)

        if y_.shape[0] == 2:
            neg_ind = y == y_[0]
            pos_ind = y == y_[1]
            y[neg_ind] = -1
            y[pos_ind] = 1
            return self._score(X, y)

        elif y_.shape[0] > 2:
            sum_scores = []
            for y_i in y_.flat:
                y_cpy = y.copy()
                neg_ind = y_cpy != y_i
                pos_ind = y_cpy == y_i
                y_cpy[neg_ind] = -1
                y_cpy[pos_ind] = 1
                sum_scores.append(self._score(X, y_cpy))
            return np.array(sum_scores).mean()

        else:
            raise ValueError('y contains only 1 class.')
    
    def _score(self, X, y):
        """To be implemented by subclasses."""
        raise NotImplementedError('Unimplemented function.')
    
    def _check_Xy(self, X, y):
        """Check if X and y meet demands."""
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(y) is not np.ndarray:
            y = np.array(y)
        if X.dtype not in [np.float64, np.int, float, int]:
            X = X.astype(np.float64)
        if X.ndim != 2:
            raise ValueError('X should be 2-d data.')
        if y.ndim != 1:
            raise ValueError('y should be 1-d array.')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y should be the same length.')
        return X, y


class DCF1(DataComplexity):
    """ Maximum Fisher’s discriminant ratio (noted as F1)
    It's a measure of data overlap.
    The range is [0, +∞)
    """
    def _score(self, X, y):
        v = np.array([self.fi_score_value(X[y == 1, i], X[y == -1, i]) for i in range(X.shape[1])])
        return 1 / v.max()

    def fi_score_value(self, c1, c2):
        return (c1.mean() - c2.mean()) ** 2 / (c1.var() + c2.var())
    

class DCF2(DataComplexity):
    """Volume of overlap region (noted as F2)
    This measure is defined as the overlap of the tails of the two
    class-conditional distributions.
    The range is [0, 1]
    """
    def _score(self, X, y):
        v = np.array([self.fi_score_value(X[y == 1, i], X[y == -1, i]) for i in range(X.shape[1])])
        return v.mean()

    def fi_score_value(self, c1, c2):
        a_min, a_max = np.array([c1.min(), c2.min()]), np.array([c1.max(), c2.max()])
        min_max, max_min, max_max, min_min = a_min.max(), a_max.min(), a_max.max(), a_min.min()
        if min_max <= max_min:
            return (max_min - min_max) / (max_max - min_min)
        else:  # no overlap
            return 0


class DCF3(DataComplexity):
    """ Maximal (individual) feature efficiency (noted as F3)
    This is a measure of efficiency of individual features that describe
    how much each feature contributes to the separation ofthe two classes.
    The range is [0, 1]
    """
    def _score(self, X, y):
        v = np.array([self.fi_score_value(X[y == 1, i], X[y == -1, i]) for i in range(X.shape[1])])
        return 1 - v.mean()

    def fi_score_value(self, c1, c2):
        a_min, a_max = np.array([c1.min(), c2.min()]), np.array([c1.max(), c2.max()])
        min_max, max_min, max_max, min_min = a_min.max(), a_max.min(), a_max.max(), a_min.min()
        if min_max <= max_min:
            n1, n2 = len(c1), len(c2)
            m1, m2 = sum(c1 > max_min) + sum(c1 < min_max), sum(c2 > max_min) + sum(c2 < min_max)
            return (m1 + m2) / (n1 + n2)
        else:  # no overlap
            return 1


class DCN1(DataComplexity):
    """ Fraction of points on class boundary (noted as N1)
    N1 is calculated by means of constructing a class-blind minimum
    spanning tree over the entire data set, counting the number of
    points incident to an edge which goes across the two classes.
    The range is [0, 1]
    """
    def append_uniq(self, a, i):
        if i not in a:
            a = np.append(a, i)
        return a

    def _score(self, X, y):
        edges = MinimumSpanningTree(X).getEdges()

        bp = np.array([]) # boundary point
        for e in edges:
            if y[e[0]] != y[e[1]]:
                bp = self.append_uniq(bp, e[0])
                bp = self.append_uniq(bp, e[1])
        return 1 - len(bp) / len(y)


class DCN2(DataComplexity):
    """ Ratio of average intra/inter class nearest neighbor distance (noted as N2)
    The range is [0, +∞)
    """
    def _score(self, X, y):
        mst = MinimumSpanningTree(X)
        # edges = mst.getEdges()
        distance = mst.getDistanceMetic()

        # different classes and same class
        dc_count, sc_count, dc_distance, sc_distance = 0, 0, 0.0, 0.0
        # nearest = np.zeros((len(y), 2), float)  # [if_boundary_point, nearest_distance]
        for i in range(0, len(y)):
            nearest_j, nearest_d = -1, np.inf
            for j in range(0, len(y)):
                if nearest_d > distance[i, j]:
                    nearest_j, nearest_d = j, distance[i, j]
            if y[i] != y[nearest_j]:# different classes
                # nearest[i, 0] = 1
                dc_count += 1
                dc_distance += nearest_d
            else:# same class
                # nearest[i, 0] = 0
                sc_count += 1
                sc_distance += nearest_d
            # nearest[i, 1] = nearest_d

        if dc_count == 0:
            dc_count = 1
        if sc_count == 0:
            sc_count = 1
        if dc_distance == 0:
            dm = distance.copy()
            dm[dm == np.inf] = 0
            dc_distance = dm.mean()

        return 1 / ((sc_distance / sc_count) / (dc_distance / dc_count))


class DCN3(DataComplexity):
    """ Error rate of 1 nearest neighbor classifier (noted as N3)
    The range is [0, +∞)
    """
    def _score(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=1)
        wrong = 0  # count wrong classifications
        for i in range(0, len(y)): # leave_one_out method cross validation
            X_train, X_test = np.append(X[:i, :], X[i+1:, :], axis=0), [X[i]] # 2-d array
            y_train, y_test = np.append(y[:i], y[i+1:]), y[i]
            neigh.fit(X_train, y_train)

            if neigh.predict(X_test) != y_test:
                wrong += 1
        error_rate = wrong / len(y)
        return 1 - error_rate


class MinimumSpanningTree(object):
    """Minimum Spanning Tree
    coded for DCN1 & DCN2.
    """
    def __init__(self, X):
        self.X = X
        self.metric = None
        self.edges = None

    def euclidean_distance(self, X):
        n_sample = len(X)
        d = np.zeros((n_sample, n_sample), float)
        for i in range(0, n_sample):
            d[i, i] = np.inf
            for j in range(i + 1, n_sample):
                d[j, i] = d[i, j] = np.linalg.norm(X[i] - X[j])  # euclidean distances of two sample
        return d

    def append_uniq(self, a, i):
        if i not in a:
            a = np.append(a, i)
        return a

    def is_connect(self, e_sel, v_sel, edge):
        # if the graph is connected return true, else false
        vectors, new_v, old_v = np.array([]), np.array([]), np.array([])
        if (edge[0] in v_sel) & (edge[1] in v_sel):
            old_v = np.append(old_v, edge[0])
            while len(old_v) > 0:
                for v in old_v:
                    for e in e_sel:
                        if (v == e[0]):
                            if (e[1] not in vectors) & (e[1] not in old_v):
                                new_v = np.append(new_v, e[1])
                        elif (v == e[1]):
                            if (e[0] not in vectors) & (e[0] not in old_v):
                                new_v = np.append(new_v, e[0])
                vectors = np.append(vectors, old_v)
                old_v, new_v = new_v, np.array([])
            if edge[1] in vectors:
                return True
            else:
                return False
        else:
            return False

    def kruskal_tree(self, d):
        e_sel, v_sel = np.array([[-1, -1]]), np.array([])
        d_min = 0
        while (d_min != np.inf) & ((len(v_sel) < len(d)) | (len(e_sel) < len(d))):
            d_min, e_min = np.inf, np.array([])
            for i in range(0, len(d)):
                for j in range(i + 1, len(d)):
                    if d[i, j] < d_min:
                        d_min = d[i, j]
                        e_min = np.array([[i, j]])
                    elif d[i, j] == d_min:
                        np.append(e_min, [[i, j]])
            for e in e_min:
                if not self.is_connect(e_sel, v_sel, e):
                    v_sel = self.append_uniq(v_sel, e[0])
                    v_sel = self.append_uniq(v_sel, e[1])
                    e_sel = np.append(e_sel, [e], axis=0)
                d[e[0], e[1]] = np.inf
        return np.delete(e_sel, 0, axis=0)

    def getDistanceMetic(self):
        if self.metric is None:
            self.metric = self.euclidean_distance(self.X)
        return self.metric

    def getEdges(self):
        if self.edges is None:
            self.edges = self.kruskal_tree(self.getDistanceMetic().copy())
        return self.edges
