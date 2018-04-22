"""
产生编码矩阵的工具
"""

import numpy as np
from itertools import combinations
from scipy.special import comb
from CodeMatrix.Matrix_tool import _exist_same_col, _exist_same_row, _exist_two_class, _get_data_subset, _estimate_weight
from CodeMatrix.SFFS import sffs
from CodeMatrix import Criterion
from CodeMatrix.Distance import euclidean_distance
import copy


def _matrix(X, y):
    """ This is an example function.

    Description:
        _matrix(X, y):
            Parameters:
                X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                y: array-like, shape = [n_samples]
                    Target vector relative to X
            Returns:
                M: 2-d array, shape = [n_classes, n_dichotomies]
                    The coding matrix.
    """
    pass


def ova(X, y):
    """
        ONE-VERSUS-ONE ECOC
    """
    index = {l: i for i, l in enumerate(np.unique(y))}
    matrix = np.eye(len(index)) * 2 - 1
    return matrix, index


def ovo(X, y):
    """
        ONE-VERSUS-ONE ECOC
    """

    index = {l: i for i, l in enumerate(np.unique(y))}
    groups = combinations(range(len(index)), 2)
    matrix_row = len(index)
    matrix_col = np.int(comb(len(index), 2))
    col_count = 0
    matrix = np.zeros((matrix_row, matrix_col))
    for group in groups:
        class_1_index = group[0]
        class_2_index = group[1]
        matrix[class_1_index, col_count] = 1
        matrix[class_2_index, col_count] = -1
        col_count += 1
    return matrix, index


def dense_rand(X, y):
    """
        Dense random ECOC
    """

    while True:
        index = {l: i for i, l in enumerate(np.unique(y))}
        matrix_row = len(index)
        if matrix_row > 3:
            matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
        else:
            matrix_col = matrix_row
        matrix = np.random.random((matrix_row, matrix_col))
        class_1_index = matrix > 0.5
        class_2_index = matrix < 0.5
        matrix[class_1_index] = 1
        matrix[class_2_index] = -1
        if (not _exist_same_col(matrix)) and (not _exist_same_row(matrix)) and _exist_two_class(matrix):
            return matrix, index


def sparse_rand(X, y):
    """
        Sparse random ECOC
    """

    while True:
        index = {l: i for i, l in enumerate(np.unique(y))}
        matrix_row = len(index)
        if matrix_row > 3:
            matrix_col = np.int(np.floor(15 * np.log10(matrix_row)))
        else:
            matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
        matrix = np.random.random((matrix_row, matrix_col))
        class_0_index = np.logical_and(0.25 <= matrix, matrix < 0.75)
        class_1_index = matrix >= 0.75
        class_2_index = matrix < 0.25
        matrix[class_0_index] = 0
        matrix[class_1_index] = 1
        matrix[class_2_index] = -1
        if (not _exist_same_col(matrix)) and (not _exist_same_row(matrix)) and _exist_two_class(matrix):
            return matrix, index


def decoc(X, y):
    """
        Discriminant ECOC
    """

    index = {l: i for i, l in enumerate(np.unique(y))}
    matrix = None
    labels_to_divide = [np.unique(y)]
    while len(labels_to_divide) > 0:
        label_set = labels_to_divide.pop(0)
        datas, labels = _get_data_subset(X, y, label_set)
        class_1_variety_result, class_2_variety_result = sffs(datas, labels)
        new_col = np.zeros((len(index), 1))
        for i in class_1_variety_result:
            new_col[index[i]] = 1
        for i in class_2_variety_result:
            new_col[index[i]] = -1
        if matrix is None:
            matrix = copy.copy(new_col)
        else:
            matrix = np.hstack((matrix, new_col))
        if len(class_1_variety_result) > 1:
            labels_to_divide.append(class_1_variety_result)
        if len(class_2_variety_result) > 1:
            labels_to_divide.append(class_2_variety_result)
    return matrix, index


def agg_ecoc(X, y):
    """
        Agglomerative ECOC
    """

    index = {l: i for i, l in enumerate(np.unique(y))}
    matrix = None
    labels_to_agg = np.unique(y)
    labels_to_agg_list = [[x] for x in labels_to_agg]
    label_dict = {labels_to_agg[value]: value for value in range(labels_to_agg.shape[0])}
    num_of_length = len(labels_to_agg_list)
    class_1_variety = []
    class_2_variety = []
    while len(labels_to_agg_list) > 1:
        score_result = np.inf
        for i in range(0, len(labels_to_agg_list) - 1):
            for j in range(i + 1, len(labels_to_agg_list)):
                class_1_data, class_1_label = _get_data_subset(X, y, labels_to_agg_list[i])
                class_2_data, class_2_label = _get_data_subset(X, y, labels_to_agg_list[j])
                score = Criterion.agg_score(class_1_data, class_1_label, class_2_data, class_2_label,
                                            score=Criterion.max_distance_score)
                if score < score_result:
                    score_result = score
                    class_1_variety = labels_to_agg_list[i]
                    class_2_variety = labels_to_agg_list[j]
        new_col = np.zeros((num_of_length, 1))
        for i in class_1_variety:
            new_col[label_dict[i]] = 1
        for i in class_2_variety:
            new_col[label_dict[i]] = -1
        if matrix is None:
            matrix = new_col
        else:
            matrix = np.hstack((matrix, new_col))
        new_class = class_1_variety + class_2_variety
        labels_to_agg_list.remove(class_1_variety)
        labels_to_agg_list.remove(class_2_variety)
        labels_to_agg_list.insert(0, new_class)
    return matrix, index


def cl_ecoc(X, y):
    """
        Centroid loss ECOC, which use regressors as base estimators
    """

    index = {l: i for i, l in enumerate(np.unique(y))}
    matrix = None
    labels_to_divide = [np.unique(y)]
    while len(labels_to_divide) > 0:
        label_set = labels_to_divide.pop(0)
        datas, labels = _get_data_subset(X, y, label_set)
        class_1_variety_result, class_2_variety_result = sffs(datas, labels, score=Criterion.max_center_distance_score)
        class_1_data_result, class_1_label_result = _get_data_subset(X, y, class_1_variety_result)
        class_2_data_result, class_2_label_result = _get_data_subset(X, y, class_2_variety_result)
        class_1_center_result = np.average(class_1_data_result, axis=0)
        class_2_center_result = np.average(class_2_data_result, axis=0)
        belong_to_class_1 = [
            euclidean_distance(x, class_1_center_result) <= euclidean_distance(x, class_2_center_result)
            for x in class_1_data_result]
        belong_to_class_2 = [
            euclidean_distance(x, class_2_center_result) <= euclidean_distance(x, class_1_center_result)
            for x in class_2_data_result]
        class_1_true_num = {k: 0 for k in class_1_variety_result}
        class_2_true_num = {k: 0 for k in class_2_variety_result}
        for y in class_1_label_result[belong_to_class_1]:
            class_1_true_num[y] += 1
        for y in class_2_label_result[belong_to_class_2]:
            class_2_true_num[y] += 1
        class_1_label_count = {k: list(class_1_label_result).count(k) for k in class_1_variety_result}
        class_2_label_count = {k: list(class_2_label_result).count(k) for k in class_2_variety_result}
        class_1_ratio = {k: class_1_true_num[k] / class_1_label_count[k] for k in class_1_variety_result}
        class_2_ratio = {k: -class_2_true_num[k] / class_2_label_count[k] for k in class_2_variety_result}
        new_col = np.zeros((len(index), 1))
        for i in class_1_ratio:
            new_col[index[i]] = class_1_ratio[i]
        for i in class_2_ratio:
            new_col[index[i]] = class_2_ratio[i]
        if matrix is None:
            matrix = copy.copy(new_col)
        else:
            matrix = np.hstack((matrix, new_col))
        if len(class_1_variety_result) > 1:
            labels_to_divide.append(class_1_variety_result)
        if len(class_2_variety_result) > 1:
            labels_to_divide.append(class_2_variety_result)
    return matrix, index


def ecoc_one(train_X, train_y, valid_X, valid_y, estimator):
    """ ECOC-ONE:Optimal node embedded ECOC
    Description:
        ecoc_one(train_X, train_y, valid_X, valid_y, estimator):
            Parameters:
                train_X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples and n_features is the number of features.
                train_y: array-like, shape = [n_samples]
                    Target vector relative to train_X
                valid_X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                    Validate vector, where n_samples in the number of samples and n_features is the number of features.
                valid_y: array-like, shape = [n_samples]
                    Target vector relative to valid_X
                estimator: classifier object.
                    Classifier used in the validating process.
            Returns:
                M: 2-d array, shape = [n_classes, n_dichotomies]
                    The coding matrix.
    """

    index = {l: i for i, l in enumerate(np.unique(train_y))}
    matrix = None
    predictors = []
    predictor_weights = []
    labels_to_divide = [np.unique(train_y)]
    while len(labels_to_divide) > 0:
        label_set = labels_to_divide.pop(0)
        label_count = len(label_set)
        groups = combinations(range(label_count), np.int(np.ceil(label_count / 2)))
        score_result = 0
        est_result = None
        for group in groups:
            class_1_variety = np.array([label_set[i] for i in group])
            class_2_variety = np.array([l for l in label_set if l not in class_1_variety])
            class_1_data, class_1_label = _get_data_subset(train_X, train_y, class_1_variety)
            class_2_data, class_2_label = _get_data_subset(train_X, train_y, class_2_variety)
            class_1_cla = np.ones(len(class_1_data))
            class_2_cla = -np.ones(len(class_2_data))
            train_d = np.vstack((class_1_data, class_2_data))
            train_c = np.hstack((class_1_cla, class_2_cla))
            est = estimator.fit(train_d, train_c)
            class_1_data, class_1_label = _get_data_subset(valid_X, valid_y, class_1_variety)
            class_2_data, class_2_label = _get_data_subset(valid_X, valid_y, class_2_variety)
            class_1_cla = np.ones(len(class_1_data))
            class_2_cla = -np.ones(len(class_2_data))
            validation_d = np.array([])
            validation_c = np.array([])
            try:
                validation_d = np.vstack((class_1_data, class_2_data))
                validation_c = np.hstack((class_1_cla, class_2_cla))
            except Exception:
                if len(class_1_data) > 0:
                    validation_d = class_1_data
                    validation_c = class_1_cla
                elif len(class_2_data) > 0:
                    validation_d = class_2_data
                    validation_c = class_2_cla
            if validation_d.shape[0] > 0 and validation_d.shape[1] > 0:
                score = est.score(validation_d, validation_c)
            else:
                score = 0.8
            if score >= score_result:
                score_result = score
                est_result = est
                class_1_variety_result = class_1_variety
                class_2_variety_result = class_2_variety
        new_col = np.zeros((len(index), 1))
        for i in class_1_variety_result:
            new_col[index[i]] = 1
        for i in class_2_variety_result:
            new_col[index[i]] = -1
        if matrix is None:
            matrix = copy.copy(new_col)
        else:
            matrix = np.hstack((matrix, new_col))
        predictors.append(est_result)
        predictor_weights.append(_estimate_weight(1 - score_result))
        if len(class_1_variety_result) > 1:
            labels_to_divide.append(class_1_variety_result)
        if len(class_2_variety_result) > 1:
            labels_to_divide.append(class_2_variety_result)
    return matrix, index, predictors, predictor_weights