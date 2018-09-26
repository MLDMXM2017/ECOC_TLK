import numpy as np
import copy
import Distance_Toolkit

#根据每一列从数据集中抽取数据
def get_data_from_col(data, label, col, index, hardcode=False, reallabel=False):
    data_result = None
    label_result = None
    real_label_result = None
    for i in range(len(col)):
        if col[i] !=0:
            data_temp = np.array([data[k] for k in range(len(label)) if label[k] == index[i]])
            real_label_temp = [label[k] for k in range(len(label)) if label[k] == index[i]]
            if hardcode:
                label_temp = np.ones(data_temp.shape[0]) * np.sign(col[i])
            else:
                label_temp = np.ones(data_temp.shape[0]) * col[i]
            if label_temp.shape[0] > 0:
                if data_result is None:
                    data_result = copy.deepcopy(data_temp)
                    label_result = copy.deepcopy(label_temp)
                    real_label_result = copy.deepcopy(real_label_temp)
                else:
                    data_result = np.vstack((data_result, data_temp))
                    label_result = np.hstack((label_result, label_temp))
                    real_label_result = np.hstack((real_label_result, real_label_temp))
    if reallabel:
        return data_result, label_result, real_label_result
    else:
        return data_result, label_result
# 根据标签来提取数据
def get_data_by_label(data, label, target_labels):
    # assert set(target_labels).issubset(set(label))
    data_subset = np.array([data[i] for i in range(len(label)) if label[i] in target_labels])
    label_subset = [label[i] for i in range(len(label)) if label[i] in target_labels]
    return data_subset, label_subset
#判断矩阵中每一列中是否都存在正类和负类
def exist_two_class(matrix):
    for i in range(matrix.shape[1]):
        if not exist_two_class_for_col(matrix[:, i]):
            return False
    return True
#判断一个类中是否有两个类
def exist_two_class_for_col(column):
    column = np.sign(column)
    column_unique = np.unique(column)
    if (1 not in column_unique) or (-1 not in column_unique):
        return False
    return True
# 对于样本中只有只有一个样本的类进行复制
def duplicate_single_class(data, label, number=2):
    data = data.tolist()
    label = list(label)
    unique_label = np.unique(label)
    for lab in unique_label:
        if label.count(lab) < number:
            index = label.index(lab)
            for i in range(number-1):
                data.append(data[index])
                label.append(label[index])
    return np.array(data), np.array(label)

def coverage_rate(data_i, label_i, data_j, label_j, distance_measure=Distance_Toolkit.euclidean_distance):
    center_i = np.mean(data_i, axis=0)
    center_j = np.mean(data_j, axis=0)
    samples_i = {label:list(label_i).count(label) for label in np.unique(label_i)}
    samples_j = {label:list(label_j).count(label) for label in np.unique(label_j)}
    count_i = {label:0 for label in label_i}
    count_j = {label:0 for label in label_j}
    for index in range(data_i.shape[0]):
        distance_i = distance_measure(center_i, data_i[index])
        distance_j = distance_measure(center_j, data_i[index])
        if distance_i < distance_j:
            count_i[label_i[index]] += 1
    for index in range(data_j.shape[0]):
        distance_i = distance_measure(center_i, data_j[index])
        distance_j = distance_measure(center_j, data_j[index])
        if distance_j < distance_i:
            count_j[label_j[index]] -= 1
    coverage_rate_i = {label:count_i[label]/samples_i[label] for label in samples_i}
    coverage_rate_j = {label:count_j[label]/samples_j[label] for label in samples_j}
    coverage_rate_i.update(coverage_rate_j)
    return coverage_rate_i

def distance_coverage_rate(data_i, label_i, data_j, label_j, distance_measure=Distance_Toolkit.euclidean_distance):
    center_i = np.mean(data_i, axis=0)
    center_j = np.mean(data_j, axis=0)
    coverage_rate_i = {}
    coverage_rate_j = {}
    for label in np.unique(label_i):
        data_subset, label_subset = get_data_by_label(data_i, label_i, [label])
        distance_mean = np.mean([distance_measure(center_i, data) for data in data_subset])
        if distance_mean == 0:
            distance_mean = 0.0001
        coverage_rate_i[label] = 2 / (1 + np.exp(-distance_mean)) - 1
    for label in np.unique(label_j):
        data_subset, label_subset = get_data_by_label(data_j, label_j, [label])
        distance_mean = np.mean([distance_measure(center_j, data) for data in data_subset])
        if distance_mean == 0:
            distance_mean = 0.0001
        coverage_rate_j[label] = 2 / (1 + np.exp(-distance_mean)) - 1
    coverage_rate_i.update(coverage_rate_j)
    return coverage_rate_i

def upper_triangular_matrix(matrix):
    upper_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            upper_matrix[i, j] = matrix[i, j] + matrix[j, i]
    return upper_matrix

def get_node_label(matrix, index):
    node_label = []
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        positive_mask = column == 1
        node_label.append(sorted(list(np.array(index)[positive_mask])))
        negative_mask = column == -1
        node_label.append(sorted(list(np.array(index)[negative_mask])))
    return node_label

def column_in_matrix(matrix, column):
    if matrix is None:
        return  False
    else:
        column = np.sign(np.array(column).reshape((1, -1)))
        matrix = np.sign(matrix.T)
        for i in range(len(matrix)):
            if np.all(column == matrix[i, :]) or np.all(column == -matrix[i, :]):
                return True
    return False

def min_distance_between_row(matrix):
    distance = np.inf
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[0]):
            distance_temp = Distance_Toolkit.euclidean_distance(matrix[i,:], matrix[j,:])
            if distance_temp < distance:
                col_1 = matrix[i,:]
                col_2 = matrix[j,:]
                distance = distance_temp
    return distance




