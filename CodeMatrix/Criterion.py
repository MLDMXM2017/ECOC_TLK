"""
This model define some common criterion for Agg_ECOC , D_ECOC and CL_ECOC
"""

from CodeMatrix import Distance
import numpy as np


def mean_center_distance_score(class_data, class_label):
    """
    calculate mean distance between different centers
    :param class_data:
    :param class_label:
    :return:
    """
    distance_sum = 0
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_center)):
        for j in range(i+1, len(class_center)):
            distance_sum = distance_sum + Distance.euclidean_distance(class_center[i], class_center[j])
    class_num = len(np.unique(class_label))
    total_num = np.power(class_num, 2) - class_num
    if total_num == 0:
        total_num = 1
    score = 2 * distance_sum / total_num
    return score


def max_center_distance_score(class_data, class_label):
    """
    calculate max distance between different centers
    :param class_data:
    :param class_label:
    :return:
    """
    max_distance = -np.inf
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_center)):
        for j in range(i+1, len(class_center)):
            distance = Distance.euclidean_distance(class_center[i], class_center[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def min_center_distance_score(class_data, class_label):
    """
    calculate min distance between different centers
    :param class_data:
    :param class_label:
    :return:
    """
    min_distance = np.inf
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_label)):
        for j in range(i+1, len(class_center)):
            distance = Distance.euclidean_distance(class_center[i], class_center[j])
            if distance < min_distance:
                min_distance = distance
    return min_distance


def mean_distance_score(class_data, class_label):
    """
    calculate mean distance between different samples
    :param class_data:
    :param class_label:
    :return:
    """
    center = np.mean(class_data, axis=0)
    distance_sum = 0
    for i in range(len(class_label)):
        distance_sum = distance_sum + Distance.euclidean_distance(center, class_data[i])
    return distance_sum / len(class_label)


def max_distance_score(class_data, class_label):
    """
    calculate max distance between different samples
    :param class_data:
    :param class_label:
    :return:
    """
    max_distance = -np.inf
    center = np.mean(class_data, axis=0)
    for i in range(len(class_label)):
        distance = Distance.euclidean_distance(center, class_data[i])
        if distance > max_distance:
            max_distance = distance
    return max_distance


def min_distance_score(class_data, class_label):
    """
    calculate min distance between different samples
    :param class_data:
    :param class_label:
    :return:
    """
    min_distance = np.inf
    center = np.mean(class_data, axis=0)
    for i in range(len(class_label)):
        distance = Distance.euclidean_distance(center, class_data[i])
        if distance < min_distance:
            min_distance = distance
    return min_distance


def divide_score(class_1_data, class_1_label, class_2_data, class_2_label, *, K=None, score=mean_center_distance_score):
    """
    use the above methods to evaluate the score of a certain partition
    :param class_1_data:
    :param class_1_label:
    :param class_2_data:
    :param class_2_label:
    :param K:
    :param score:
    :return:
    """
    if K is None:
        K = len(np.unique(class_1_label)) + len(np.unique(class_2_label))
    if 1 < len(np.unique(class_1_label)) < K-1:
        class_1_s = score(class_1_data, class_1_label)
    else:
        class_1_s = 0
    if 1 < len(np.unique(class_2_label)) < K-1:
        class_2_s = score(class_2_data, class_2_label)
    else:
        class_2_s = 0
    class_1_center = np.average(class_1_data, axis=0)
    class_2_center = np.average(class_2_data, axis=0)
    class_1_2_s = Distance.euclidean_distance(class_1_center, class_2_center)
    if 1 < len(np.unique(class_1_label)) < K-1:
        confidence_score = class_1_2_s / (class_1_s + class_2_s)
    else:
        confidence_score = 0
    return confidence_score


def agg_score(class_1_data, class_1_label, class_2_data, class_2_label, score=mean_distance_score):
    """
    use the above methods to evaluate the score of a certain agglomeration
    :param class_1_data:
    :param class_1_label:
    :param class_2_data:
    :param class_2_label:
    :param score:
    :return:
    """
    class_1_distance = score(class_1_data, class_1_label)
    class_2_distance = score(class_2_data, class_2_label)
    class_1_center = np.mean(class_1_data, axis=0)
    class_2_center = np.mean(class_2_data, axis=0)
    distance_between_two_class = Distance.euclidean_distance(class_1_center, class_2_center)
    return 2 * distance_between_two_class / (class_1_distance + class_2_distance)