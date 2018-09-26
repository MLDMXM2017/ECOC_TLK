import Distance_Toolkit
import numpy as np
import Matrix_Toolkit

def mean_center_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    distance_sum = 0
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_center)):
        for j in range(i+1, len(class_center)):
            distance_sum = distance_sum + distance_measure(class_center[i], class_center[j])
    class_num = len(np.unique(class_label))
    total_num = np.power(class_num, 2) - class_num
    if total_num == 0:
        total_num = 1
    score = 2 * distance_sum / total_num
    return score


def max_center_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    max_distance = -np.inf
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_center)):
        for j in range(i+1, len(class_center)):
            distance = distance_measure(class_center[i], class_center[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def min_center_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    min_distance = np.inf
    class_center = []
    for label in np.unique(class_label):
        class_center.append(np.average([class_data[i] for i in range(len(class_label)) if class_label[i] == label], axis=0))
    for i in range(len(class_label)):
        for j in range(i+1, len(class_center)):
            distance = distance_measure(class_center[i], class_center[j])
            if distance < min_distance:
                min_distance = distance
    return min_distance


def mean_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    center = np.mean(class_data, axis=0)
    distance_sum = 0
    for i in range(len(class_label)):
        distance_sum = distance_sum + distance_measure(center, class_data[i])
    return distance_sum / len(class_label)


def max_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    max_distance = -np.inf
    center = np.mean(class_data, axis=0)
    for i in range(len(class_label)):
        distance = distance_measure(center, class_data[i])
        if distance > max_distance:
            max_distance = distance
    return max_distance


def min_distance_score(class_data, class_label, distance_measure=Distance_Toolkit.euclidean_distance):
    min_distance = np.inf
    center = np.mean(class_data, axis=0)
    for i in range(len(class_label)):
        distance = distance_measure(center, class_data[i])
        if distance < min_distance:
            min_distance = distance
    return min_distance


def divide_score(class_1_data, class_1_label, class_2_data, class_2_label,
                 distance_measure=Distance_Toolkit.euclidean_distance,
                 K=None, score=mean_center_distance_score):
    class_1_data_ratio = len(class_1_label)/len(class_2_label)
    class_2_data_ratio = len(class_2_label)/len(class_1_label)
    data_ratio = max(class_1_data_ratio, class_2_data_ratio)
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
    class_1_2_s = distance_measure(class_1_center, class_2_center)
    if 1 < len(np.unique(class_1_label)) < K-1:
        confidence_score = class_1_2_s / (class_1_s + class_2_s + data_ratio)
    else:
        confidence_score = 0
    if str(confidence_score) == 'nan':
        confidence_score = np.inf
    return confidence_score

def divide_score_coverage(class_1_data, class_1_label, class_2_data, class_2_label,
                 distance_measure=Distance_Toolkit.euclidean_distance):
    rate = Matrix_Toolkit.coverage_rate(class_1_data, class_1_label, class_2_data, class_2_label, distance_measure)
    rate_array = np.array(list(rate.values()))
    mask = rate_array > 0
    positive_rate = rate_array[mask]
    negative_rate = rate_array[np.logical_not(mask)]
    positive_rate_mean = np.mean(positive_rate)
    negative_rate_mean = np.mean(np.abs(negative_rate))
    positive_unique_label = len(np.unique(class_1_label))
    negative_unique_label = len(np.unique(class_2_label))
    unique_label_ratio = min(positive_unique_label/negative_unique_label, negative_unique_label/positive_unique_label)
    # print('positive rate:', positive_rate_mean)
    # print('negative rate:', negative_rate_mean)
    # print('label ratio:', unique_label_ratio)
    # print('score:', positive_rate_mean*negative_rate_mean*unique_label_ratio)
    return positive_rate_mean*negative_rate_mean*unique_label_ratio



def agg_score(class_1_data, class_1_label, class_2_data, class_2_label,
              distance_measure=Distance_Toolkit.euclidean_distance,
              score=mean_distance_score):
    class_1_distance = score(class_1_data, class_1_label)
    class_2_distance = score(class_2_data, class_2_label)
    class_1_center = np.mean(class_1_data, axis=0)
    class_2_center = np.mean(class_2_data, axis=0)
    distance_between_two_class = distance_measure(class_1_center, class_2_center)
    return 2 * distance_between_two_class / (class_1_distance + class_2_distance)

