"""Normalize data
There are some normalizer classes that are described below.
They server only 2-D data and they deal data in three ways(overall, rows and columns)

1. Min-max normalizer
2. Z-score normalizer
3. Log normalizer
4. Atan normalizer
5. Sigmoid normalizer
"""

import abc
import numpy as np


def check_data(data):
    """check data
    1. if data is ndarray.
    2. if data is 2-D data.
    3. if data type is float64.
    """
    if type(data) is not np.ndarray:
        data = np.array(data)
    if len(data.shape) != 2:
        raise ValueError('The data must be 2-D.')
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    return data


class Normalizer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def normalise(self, data, axis=None):
        """
            axis=None: overall normalization.
            axis=0: row normalization.
            axis=1: column normalization.
            """
        pass


class MinMaxNormalizer(Normalizer):
    """Min-max normalizer
    x' = (x-min)/(max-min)
    value range: [0, 1]
    """

    @staticmethod
    def nor(d, mini, maxi):
        return (d - mini) / (maxi - mini)

    @staticmethod
    def check_min_max(mini, maxi):
        if mini == maxi:
            raise ValueError('Min value equals max value while doing min-max normalization.')

    def normalise(self, data, axis=None):
        data = check_data(data)

        if axis is None:
            mini, maxi = data.min(), data.max()
            self.check_min_max(mini, maxi)
            data = self.nor(data, mini, maxi)

        elif axis == 0:
            for i in range(data.shape[0]):
                mini, maxi = data[i].min(), data[i].max()
                self.check_min_max(mini, maxi)
                data[i] = self.nor(data[i], mini, maxi)

        elif axis == 1:
            for j in range(data.shape[1]):
                mini, maxi = data[:, j].min(), data[:, j].max()
                self.check_min_max(mini, maxi)
                data[:, j] = self.nor(data[:, j], mini, maxi)

        return data


class ZscoreNormalizer(Normalizer):
    """ Z-score Normalizer
    x' = (x-mu)/delta, where mu is the mean value and delta standard deviation.
    value range: (-∞,+∞)
    """

    @staticmethod
    def nor(d, mu, delta):
        return (d - mu) / delta

    def normalise(self, data, axis=None):
        data = check_data(data)

        if axis is None:
            mu, delta = data.mean(), data.std()
            data = self.nor(data, mu, delta)

        elif axis == 0:
            for i in range(data.shape[0]):
                mu, delta = data[i].mean(), data[i].std()
                data[i] = self.nor(data[i], mu, delta)

        elif axis == 1:
            for j in range(data.shape[1]):
                mu, delta = data[:, j].mean(), data[:, j].std()
                data[:, j] = self.nor(data[:, j], mu, delta)

        return data


class LogNormalizer(Normalizer):
    """ Log Normalizer
    x' = log10(x)/log10(max)
    value range: [0, 1]
    """
    @staticmethod
    def nor(d, maxi):
        return np.log10(d) / np.log10(maxi)

    @staticmethod
    def check_log(data):
        mini = data.min()
        if mini < 1:
            raise ValueError('Values must be bigger than 1.')

    def normalise(self, data, axis=None):
        data = check_data(data)
        self.check_log(data)

        if axis is None:
            maxi = data.max()
            data = self.nor(data, maxi)
        elif axis == 0:
            for i in range(data.shape[0]):
                maxi = data[i].max()
                data[i] = self.nor(data[i], maxi)
        elif axis == 1:
            for j in range(data.shape[1]):
                maxi = data[:, j].max()
                data[:, j] = self.nor(data[:, j], maxi)

        return data


class AtanNormalizer(object):
    """ Atan Normalizer
    x' = atan(x)*2/pi
    value range: (-1, 1)
    """
    @staticmethod
    def nor(d):
        return np.arctan(d)*2/np.pi

    def normalise(self, data, axis=None):
        data = check_data(data)
        return self.nor(data)


class SigmoidNormalizer(object):
    """ Sigmoid Normalizer
    x' = 1/(1+e^(-x))
    value range: (0, 1)
    """
    @staticmethod
    def nor(d):
        return 1 / (1 + np.e ** (-d))

    def normalise(self, data, axis=None):
        data = check_data(data)
        return self.nor(data)
