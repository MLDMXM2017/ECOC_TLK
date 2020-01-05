# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/20 12:15
# file: Get_complexity.py
# description: this module defines the F1 value

import numpy as np
import copy
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier

from ECOC_library.DC.Complexity_tool import *
import logging


def get_complexity_F1(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the F1 index
    F1 is the sum of the (mean1-mean2)^2/var1^2+Var2^2 on each dimension

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """

    FK = []
    K = len(group1_data[1])
    for i in range(K):
        temp1_data = [x[i] for x in group1_data]
        temp2_data = [x[i] for x in group2_data]

        mean_v = np.power(np.mean(temp1_data) - np.mean(temp2_data), 2)
        var_v = np.power(np.var(temp1_data), 2) + np.power(np.var(temp2_data), 2)

        if var_v == 0:
            logging.debug('ERROR-F1: the denominator of the ' + str(i) + ' dimension of F1 is zero!')

        else:
            FK.append(mean_v / var_v)

    F1 = np.max(FK)

    return F1


def get_complexity_F2(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the F2 index
    F2 is the sum of the overlapping region/total region on each dimension

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    K = len(group1_data[1])
    F2 = 0
    for i in range(K):
        temp1_data = [x[i] for x in group1_data]
        temp2_data = [x[i] for x in group2_data]
        M = min(max(temp1_data), max(temp2_data)) - max(min(temp1_data), min(temp2_data))
        D = max(max(temp1_data), max(temp2_data)) - max(min(temp1_data), min(temp2_data))
        if (D == 0):
            logging.debug('ERROR-F2: the denominator of the ' + str(i) + ' dimension of F2 is zero!')
        else:
            F2 = F2 + M / D

    return F2


def get_complexity_F3(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the F3 index
    F3 is the sum of the overlapping region/total region on each dimension

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    K = len(group1_data[1])
    data_size = len(group1_data + group2_data)
    if (data_size == 0):
        logging.error('ERROR-F3: data size is empty')
        return 0

    F3 = 0
    for i in range(K):
        temp1_data = [x[i] for x in group1_data]
        temp2_data = [x[i] for x in group2_data]
        mean1_data = np.mean(temp1_data)
        mean2_data = np.mean(temp2_data)
        if (mean1_data > mean2_data):
            small = temp2_data
            large = temp1_data
        else:
            small = temp1_data
            large = temp2_data

        min_data = min(large)
        lager_min = [x for x in small if x > min_data]
        max_data = max(small)
        smaller_max = [x for x in large if x < max_data]
        F3 = F3 + (len(lager_min) + len(smaller_max)) / data_size

    return F3


def get_complexity_N2(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the N2 index
    N2 is the sum of the nearest inner-neighbor/ nearest intra-neighbor of each sample

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    K = len(group1_data[1])
    N2 = 0
    tinner1_dis = [get_point_group_min_dis(x, group1_data) for x in group1_data]
    tintra1_dis = [get_point_group_min_dis(x, group2_data) for x in group1_data]

    tinner2_dis = [get_point_group_min_dis(x, group2_data) for x in group2_data]
    tintra2_dis = [get_point_group_min_dis(x, group1_data) for x in group2_data]

    inner_dis = sum(tinner1_dis + tinner2_dis)
    intra_dis = sum(tintra1_dis + tintra2_dis)

    if (intra_dis == 0):
        logging.error('ERROR-N2: the denominator of N2 is zero!')
        return 0
    else:
        N2 = inner_dis / intra_dis
        return N2


def get_complexity_N3(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the N3 index
    N3 is the error rate of  KNN classifier for testing whole samples
    leave one out method is used

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    data = list(group1_data) + list(group2_data)
    label = list(group1_label) + list(group2_label)

    if (len(data) == 0):
        logging.error('ERROR-N3: the data size is empty')
        return 0

    error = 0
    for inx, x in enumerate(data):
        temp_data = copy.deepcopy(data)
        temp_label = copy.deepcopy(label)
        del temp_data[inx]
        del temp_label[inx]
        y_pred = neighbors.KNeighborsClassifier().fit(temp_data, temp_label).predict([x])
        if (y_pred != label[inx]):
            error = error + 1

    N3 = error / len(data)

    return N3


def get_complexity_N4(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the N4 index
    N4 is the error rate of  KNN classifier for samples Created by linear interpolation
    train data is the group data and train label is [1,-1]
    test data is the interpolation data and test label is [1,-1]


    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    if (len(group1_data) == 0 or len(group2_data) == 0):
        logging.error('ERROR-N4: the data size is empty')
        return 0

    train_data = group1_data + group2_data
    train_label = [1] * len(group1_data) + [-1] * len(group2_data)

    if (len(train_data) == 0):
        logging.error('ERROR-N4: the data size is empty')
        return 0

    interpolation1_data = create_interpolation_data(group1_data, group1_label)
    interpolation2_data = create_interpolation_data(group2_data, group2_label)

    test_data = interpolation1_data + interpolation2_data
    test_label = [1] * len(interpolation1_data) + [-1] * len(interpolation2_data)

    if (len(test_data) == 0):
        logging.error('ERROR-N4: the interpolation data size is empty')
        return 0

    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(train_data, train_label)
    y_pred = classifier.predict(test_data)

    N3 = sum([1 for inx, y in enumerate(y_pred) if y != test_label[inx]]) / len(test_label)

    return N3


def get_complexity_L3(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the L3 index
    L3 is the error rate of  linear classifier for samples Created by linear interpolation

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    if (len(group1_data) == 0 or len(group2_data) == 0):
        logging.error('ERROR-L3: the data size is empty')
        return 0

    train_data = group1_data + group2_data
    train_label = [1] * len(group1_data) + [-1] * len(group2_data)

    if (len(train_data) == 0):
        logging.error('ERROR-L3: the data size is empty')
        return 0

    interpolation1_data = create_interpolation_data(group1_data)
    interpolation2_data = create_interpolation_data(group2_data)

    test_data = interpolation1_data + interpolation2_data
    test_label = [1] * len(interpolation1_data) + [-1] * len(interpolation2_data)

    if (len(test_data) == 0):
        logging.error('ERROR-L3: the interpolation data size is empty')
        return 0

    y_pred = SGDClassifier.fit(group1_data + group2_data, group1_label + group2_label).fit_predict(test_data)

    L3 = sum([1 for inx, y in enumerate(y_pred) if y != test_label[inx]]) / len(test_label)

    return L3


def get_complexity_Cluster(group1_data, group1_label, group2_data, group2_label):
    """
    this fun is calculate the Cluster index
    Cluster is the error rate of self-created Cluster method
    by adjusting label with max mis-allocation to reduce Cluster

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """
    if (len(group1_data) == 0 or len(group2_data) == 0):
        logging.error('ERROR-Cluster: the data size is empty')
        return 0

    while (True):
        cluster_error, cluster_label = get_cluster_error(group1_data, group1_label, group2_data, group2_label)
        adjusted_ulabel1, adjusted_ulabel2 = adjust_cluster(group1_data, group1_label, group2_data, group2_label,
                                                            cluster_label)

        if (list(np.unique(group1_label)) == adjusted_ulabel1 or list(np.unique(group2_label)) == adjusted_ulabel2):
            break

        # type judge
        if (type(group1_data) != list):
            group1_data = list(group1_data)

        if (type(group2_data) != list):
            group2_data = list(group2_data)

        if (type(group1_label) != list):
            group1_label = list(group1_label)

        if (type(group2_label) != list):
            group2_label = list(group2_label)

        adjusted_group1_data, adjusted_group1_label = get_data_subset(group1_data + group2_data,
                                                                      group1_label + group2_label, adjusted_ulabel1)
        adjusted_group2_data, adjusted_group2_label = get_data_subset(group1_data + group2_data,
                                                                      group1_label + group2_label, adjusted_ulabel2)

        adjusted_cluster_error, adjusted_cluster_label = get_cluster_error(adjusted_group1_data, adjusted_group1_label,
                                                                           adjusted_group2_data, adjusted_group2_label)
        if (adjusted_cluster_error < cluster_error):
            cluster_error = adjusted_cluster_error
            cluster_label = adjusted_cluster_label
            group1_data = adjusted_group1_data
            group1_label = adjusted_group1_label
            group2_data = adjusted_group2_data
            group2_label = adjusted_group2_label
        else:
            break

    return cluster_error


def get_complexity_D2(data, label, k=5):
    """
    this complxity only need one group data and label
    :param data:  group data
    :param label: group label
    :return:
    """
    if (len(data) == 0):
        logging.error('ERROR-D2: the data size is empty')
        return 0

    K = len(data[0])
    sum = 0
    for i, each in enumerate(data):
        neighbors = get_Kneighbors(data, each, k=3)
        f = 1
        for k in range(K):  # each dimension
            k_column = [row[k] for row in neighbors]
            if (np.max(k_column) - np.min(k_column)) != 0:
                f = f * (np.max(k_column) - np.min(k_column))
                # logging.info("k:%-10d f:%f" %(k,f))
        sum = sum + f
        # logging.info("i:%-10d sum:%f" %(i,sum))
    D2 = sum / len(data)

    return D2
