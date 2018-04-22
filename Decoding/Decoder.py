"""
ECOC Decoder
The code is based on numpy (http://www.numpy.org/), and almost all data objects are formed as numpy.ndarray.

The implementation of decoders come from:
Sergio Escalera SERGIO, Oriol Pujol ORIOL, Petia Radeva. Error-Correcting Ouput Codes Library. Journal of Machine Learning Research 11 (2010) 661-664.

Code by Tycho Zhong, Dec 6, 2017.
"""

import numpy as np


def get_decoder(dec):
    """ Get a decoder object.
    Parameters:
        dec: str
            Indicates a kind of decoder. Cognitive dec list below.
            'HD' - Hamming Decoder.
            'IHD' - Inverse Hamming Decoder.
            'LD' - Laplacian Decoder.
            'ED' - Euclidean Decoder.
            'AED' - Attenuated Euclidean Decoder.
            'RED' - Ratio Euclidean Decoder.
            'EuD' - Euler Decoder.
            'LLB' - Linear Loss Based Decoder.
            'ELB' - Exponential Loss Based Decoder.
            'LLW' - Linear Loss Weighted Decoder.
            'ELW' - Exponential Loss Weighted Decoder.
            'PD' - Probabilistic Decoder (Coming soon).

    Returns:
        o: object
            A decoder object.
    """
    if dec == 'HD':
        return HammingDecoder()
    elif dec == 'IHD':
        return InverseHammingDecoder()
    elif dec == 'LD':
        return LaplacianDecoder()
    elif dec == 'ED':
        return EuclideanDecoder()
    elif dec == 'AED':
        return AttenuatedEuclideanDecoder()
    elif dec == 'RED':
        return RatioEuclideanDecoder()
    elif dec == 'EuD':
        return EulerDecoder()
    elif dec == 'LLB':
        return LinearLossBasedDecoder()
    elif dec == 'ELB':
        return ExponentialLossBasedDecoder()
    elif dec == 'LLW':
        return LinearLossWeightedDecoder()
    elif dec == 'ELW':
        return ExponentialLossWeightedDecoder()
    elif dec == 'PD':
        raise NotImplementedError('The Probabilistic Decoder is unimplemented.')
        # return ProbabilisticDecoder()
    else:
        raise KeyError('Unknown code %s.' % dec)


class Decoder(object):
    """ ECOC Decoder
    Methods:
        decode(Y, M): decode Y (predict matrix), by M (code matrix), into distance matrix.

    Description:
        decode(Y, M): decode Y (predict matrix), by M (code matrix), into distance matrix.
            Parameters:
                Y: 2-d array, shape = [n_samples, n_dichotomies]
                M: 2-d array, shape = [n_classes, n_dichotomies]
            Returns:
                D: 2-d array, shape = [n_samples, n_classes]
    """
    def _check_param(self, Y, M):
        """Check Y and M, check the column number of Y and M."""
        Y, M = self._check_y(Y), self._check_matrix(M)
        if Y.shape[1] != M.shape[1]:
            raise ValueError('Different column numbers of Y and M')
        return Y, M

    def _check_matrix(self, M):
        """Check matrix object type, dimension, data type, and weather it is tenary code."""
        if type(M) is not np.ndarray:
            M = np.array(M)
        if M.ndim != 2:
            raise ValueError('Matrix must be 2-d ndarray.')
        if M.dtype not in [np.float64, np.int, float, int]:
            # raise TypeError('Matrix dtype is not float or int.')
            M = M.astype(np.int)
        m = np.unique(M)
        if len(m) == 2 and -1 in m and 1 in m:
            self.tenary = False
        elif len(m) == 3 and -1 in m and 0 in m and 1 in m:
            self.tenary = True
        else:
            raise ValueError('Matrix contains codes not in [-1,0,1].')
        return M

    def _check_y(self, Y):
        """Check matrix object type, dimension, data type, and value range."""
        if type(Y) is not np.ndarray:
            Y = np.array(Y)
        if Y.ndim != 2:
            raise ValueError('Y must be 2-d ndarray.')
        if Y.dtype not in [np.float64, np.int, float, int]:
            # raise TypeError('Y dtype is not float or int.')
            Y = Y.astype(np.float64)
        if Y.max() > 1 and Y.min() < -1:
            raise ValueError('Values in Y out of range[-1,1].')
        return Y

    def decode(self, Y, M):
        """
        Y: The predict (Samples×Dichotomies)
        M: The Matrix (Classes×Dichotomies)
        Column numbers of Y must equals column numbers of M.
        Return the distance matrix represent distance between samples and classes,
        where rows represent the samples and columns represent the classes
        """
        pass

    @staticmethod
    def _distance(y, m):
        """The distance between 1-D array y and 1-D array m."""
        pass


class OrdinaryDecoder(Decoder):

    def decode(self, Y, M):
        Y, M = self._check_param(Y, M)
        return np.array([[self._distance(y, m) for m in M] for y in Y])

    @staticmethod
    def _distance(y, m):
        raise NotImplementedError('Unimplemented class.')


class HardDecoder(Decoder):

    def decode(self, Y, M):
        Y, M = self._check_param(Y, M)
        self._check_hard(Y)
        return np.array([[self._distance(y, m) for m in M] for y in Y])

    @staticmethod
    def _distance(y, m):
        raise NotImplementedError('Unimplemented class.')

    def _check_hard(self, Y):
        """check if Y contains only -1, 0 or 1"""
        y_ = np.unique(Y)
        if len(y_) > 3:
            raise ValueError('Y contains values not in [-1,0,1]')
        for i in y_.flat:
            if i not in [-1, 0, 1]:
                raise ValueError('Y contains values not in [-1,0,1]')


class WeightedDecoder(Decoder):

    def decode(self, Y, M, W):
        """W represents the weight vector."""
        Y, M = self._check_param(Y, M)
        W = self._check_weight(W, M)
        return np.array([[self._distance(y, m, W) for m in M] for y in Y])

    @staticmethod
    def _distance(y, m, W):
        """Calculate distances with weights."""
        raise NotImplementedError('Unimplemented class.')

    def _check_weight(self, W, M):
        if type(W) is not np.ndarray:
            W = np.array(W)
        if W.shape[1] != M.shape[1]:
            raise ValueError('Length of W must be the same with column number of Matrix.')


class HammingDecoder(HardDecoder):
    """Hamming Decoder (HD)
    Hamming decoder must check if Y contains only -1, 0 or 1.
    """
    @staticmethod
    def _distance(y, m):
        return sum(abs(y-m))/2


class InverseHammingDecoder(HammingDecoder):
    """Inverse Hamming Decoder (IHD)."""

    def decode(self, Y, M):
        Y, M = self._check_param(Y, M)
        self._check_hard(Y)

        n = M.shape[1]
        delta = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                delta[i, j] = delta[j, i] = self._distance(M[i, :], M[j, :])
        L = np.array([[self._distance(y, m) for m in M] for y in Y])
        return np.dot(np.linalg.inv(delta), L.T) * -1


class LaplacianDecoder(HardDecoder):
    """Laplacian Decoder (LD)."""
    @staticmethod
    def _distance(y, m):
        c, e, k = sum(y == m), sum(y != m) - sum(y == 0), 2
        return (c + e + k) / (c + 1)


class EulerDecoder(OrdinaryDecoder):
    """Scikit-learn default decoder."""
    def decode(self, Y, M):
        Y, M = self._check_param(Y, M)
        return -1 * np.dot(Y, M.T) / np.einsum('ij,ij->i', M, M)


class EuclideanDecoder(OrdinaryDecoder):
    """Euclidean Decoder (ED)."""
    @staticmethod
    def _distance(y, m):
        return np.sqrt(sum((y-m)**2))


class AttenuatedEuclideanDecoder(OrdinaryDecoder):
    """Attenuated Euclidean Decoder (AED).
    Note that this decoder is originally designed for tenary code matrix.
    """
    @staticmethod
    def _distance(y, m):
        return np.sqrt(sum(((y-m)**2)*abs(m)))


class RatioEuclideanDecoder(OrdinaryDecoder):
    """Ratio Euclidean Decoder
    Attenuated Euclidean Distance with ratio.
    """
    @staticmethod
    def _distance(y, m):
        mm = abs(m)
        return np.sqrt(sum((y-m)**2 * mm) / sum(mm))


class LinearLossBasedDecoder(OrdinaryDecoder):
    """Linear Loss-Based Decoder (LLB).
    """
    @staticmethod
    def _distance(y, m):
        return sum(-1 * (y * m))


class LinearLossWeightedDecoder(WeightedDecoder):
    """Linear Loss-based Weighted Decoder (LLW).
    LLB with a weight vector.
    """
    @staticmethod
    def _distance(y, m, W):
        return sum(-1 * W * (y * m))


class ExponentialLossBasedDecoder(OrdinaryDecoder):
    """Exponential Loss-Based Decoder (ELB)."""
    @staticmethod
    def _distance(y, m):
        return sum(np.exp(-1 * (y * m)))


class ExponentialLossWeightedDecoder(WeightedDecoder):
    """Exponential Loss-Based  Weighted Decoder (ELW).
    ELB with a weight vector.
    """
    @staticmethod
    def _distance(y, m, W):
        return sum(W * np.exp(-1 * (y * m)))


class ProbabilisticDecoder(OrdinaryDecoder):
    """Probabilistic-based Decoder (PD)."""
    @staticmethod
    def _distance(y, m):
        raise NotImplementedError('Undo class.')
