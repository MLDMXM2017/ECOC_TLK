"""
this model define a sequential floating forward searching(SFFS) method
"""

from CodeMatrix.Matrix_tool import _get_data_subset
import numpy as np
from CodeMatrix import Criterion


def sffs(data, labels, judge_score=Criterion.divide_score, **param):
    """
    Sequential floating forward searching(SFFS) method
    :param data: data
    :param labels: label
    :param judge_score: a callable object to evaluate the score for partition
    :param param: params for judge_score
    :return: partition labels
    """
    target_label = []
    other_label = []
    best_target_label = []
    best_other_label = []
    unique_label = np.unique(labels)
    score_list = {}
    target_label_list = []
    target_label_list_pre_len = 0
    K = 0
    pre_K = None
    while True:
        best_score = -np.inf
        update_flag = 0
        for label in unique_label:
            if label in target_label:
                continue
            target_label.append(label)
            if target_label in target_label_list:
                continue
            other_label = list(unique_label)
            for i in target_label:
                other_label.remove(i)
            target_data, target_labels = _get_data_subset(data, labels, target_label)
            other_data, other_labels = _get_data_subset(data, labels, other_label)
            score = judge_score(target_data, target_labels, other_data, other_labels, **param)
            if score > best_score:
                best_score = score
                best_target_label = list(target_label)
                best_other_label = list(other_label)
                update_flag = 1
            target_label.pop()
        if update_flag:
            score_list[K + 1] = best_score
            K = K + 1
            target_label = list(best_target_label)
            target_label_list.append(list(target_label))
            other_label = list(best_other_label)
        else:
            break
        while True:
            best_score = -np.inf
            if len(target_label) < 2:
                break
            for label in target_label:
                target_label_temp = list(target_label)
                target_label_temp.remove(label)
                if target_label_temp in target_label_list:
                    continue
                other_label_temp = list(other_label)
                other_label_temp.append(label)
                target_data, target_labels = _get_data_subset(data, labels, target_label_temp)
                other_data, other_labels = _get_data_subset(data, labels, other_label_temp)
                score = judge_score(target_data, target_labels, other_data, other_labels)
                if score > best_score:
                    best_score = score
                    best_target_label = list(target_label_temp)
                    best_other_label = list(other_label_temp)
            if K-1 > 0 and best_score > score_list[K]:
                score_list[K+1] = best_score
                target_label = list(best_target_label)
                target_label_list.append(list(target_label))
                other_label = list(best_other_label)
                K = K + 1
            else:
                break
        if K-1 > 0 and score_list[K] - score_list[pre_K] < 0.00001:
            break
        if len(target_label) >= len(unique_label)-1:
            break
        if target_label_list_pre_len == len(target_label_list):
            break
        target_label_list_pre_len = len(target_label)
        pre_K = K
    try:
        return target_label_list[pre_K-1], [label for label in unique_label if label not in target_label_list[pre_K-1]]
    except TypeError:
        return target_label, other_label