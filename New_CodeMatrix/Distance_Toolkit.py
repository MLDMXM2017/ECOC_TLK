import numpy as np


# 计算汉明距离
def hamming_distance(x, y, weights=None):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sum(((1 - np.sign(x * y))/2) * weights)
    return distance
# 计算欧式距离
def euclidean_distance(x, y, weights=None):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sqrt(np.sum(np.power(x - y, 2)*weights))
    return distance
# 计算闵可夫斯基距离
def minkowski_distance(x, y, weights=None):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    assert len(x) == len(y)
    n = len(x)
    distance = np.power(np.sum(np.power(x - y, n)*weights), n)
    return distance

def y_euclidean_distance(x, y,  weights=None):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sqrt(np.sum(np.abs(x)*np.power(x - y, 2)*weights))
    return distance