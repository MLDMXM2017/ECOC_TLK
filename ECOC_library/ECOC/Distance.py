# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/20
# file: Complexity_tool.py
# description:  this model define some method to calculate distance


"""
1.hmming_distance
2.euclidean_distance
"""

import numpy as np
import copy


def hamming_distance(x, y, weights=None):
    """
    calculate hamming distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: hamming distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sum(((1 - np.sign(x * y)) / 2) * weights)
    return distance


def euclidean_distance(x, y, weights=None):
    """
    calulate euclidean distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: euclidean distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    temp_x = copy.deepcopy(x)
    temp_y = copy.deepcopy(y)
    distance = np.sqrt(np.sum(np.power(temp_x - temp_y, 2) * weights))
    return distance


def y_euclidean_distance(x, y, weights=None):
    """
    a different euclidean distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sqrt(np.sum(np.abs(y) * np.power(x - y, 2) * weights))
    return distance
