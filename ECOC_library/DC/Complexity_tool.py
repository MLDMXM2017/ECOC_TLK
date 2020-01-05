# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/20 12:15
# file: Complexity_tool.py
# description:  this module defines some fun to help calculate compelxity index

import numpy as np
from scipy import interpolate
import random
import math
import copy
from scipy.interpolate import griddata
import operator

def get_data_subset(data, label, target_label):
    """
    to get data with certain labels
    :param data: data set
    :param label: label corresponding to data
    :param target_label: the label which we want to get certain data
    :return:
    """

    data_subset = np.array([data[i] for i in range(len(label)) if label[i] in target_label])
    label_subset = np.array([label[i] for i in range(len(label)) if label[i] in target_label])
    return data_subset, label_subset


def get_point_point_dis(p1,p2):
    """

    :param p1: first point
    :param p2: second point
    :return: the Euclidean distance between p1 and p2
    """
    K = len(p1) #dimension
    dis = 0
    for inx in range(K):
        dis = dis + np.power(p1[inx] - p2[inx],2)

    dis = math.sqrt(dis)

    return dis

def get_point_group_min_dis(p,group_data):
    """

    :param p: point
    :param group_data: group data
    :return: the minimum dis between point and group
    """
    K = len(group_data) #samples numbers
    min_dis = float("inf")
    min_inx = 0
    for inx in range(K):
        dis = get_point_point_dis(p,group_data[inx])
        if(dis < min_dis and dis != 0):
            min_dis = dis
            min_inx = inx

    return min_dis

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

def create_interpolation_data(group_data,group_label):

    """
    this function use a half samples of each label to create new samples by their interpolation values

    :param group_data: the whole group data
    :param group_label:  the corresponding labels to data
    :return:
    """
    ulabel = np.unique(group_label) #labels vector
    interpolated_data = []
    for label in ulabel:
        label_data = [group_data[inx] for inx in range(len(group_label)) if group_label[inx] == label]
        if(len(label_data) > 10): # if samples number more than 10, random select a half samples
            random_sequence = [inx for inx in range(len(label_data))]
            random.shuffle(random_sequence)
            selected_data = [label_data[inx] for inx in random_sequence[0: len(random_sequence) / 2 ]]
        else:
            selected_data = label_data

        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        # grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        interpolated_data = interpolated_data + ovo_interpolation(selected_data)

    return interpolated_data

def ovo_interpolation(data):
    """
    this fun create interpolation values between data by OVO way

    :param data: data that needs to produce interpolated values
    :return:
    """
    res = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            res.append(interpolate.interp1d(data[i],data[j],kind="slinear"))
    return res


def get_cluster_error(group1_data,group1_label,group2_data,group2_label):

    """
    This function clusters the two groups of data and then calculate the accuracy of the clustering.

    :param group1_data:
    :param group1_label:
    :param group2_data:
    :param group2_label:
    :return:
    """

    K = len(group1_data[1])

    centroid1 = []
    centroid2 = []
    for inx in range(K):
        centroid1.append(np.mean([x[inx] for x in group1_data]))
        centroid2.append(np.mean([x[inx] for x in group2_data]))

    cluster_label = []
    for x in list(group1_data) + list(group2_data):
        if get_point_point_dis(x, centroid1) > get_point_point_dis(x, centroid2):
            cluster_label.append(-1)
        else:
            cluster_label.append(1)

    true_label = [1] * len(group1_data) + [-1] * len(group2_data)

    error = float(sum([1 for inx in range(len(true_label)) if true_label[inx] != cluster_label[inx]])) / float(len(true_label))

    return error,cluster_label


def adjust_cluster(group1_data,group1_label,group2_data,group2_label,cluster_label):
    ulabel1 = list(np.unique(group1_label))
    ulabel2 = list(np.unique(group2_label))

    c1_to_adjust = find_max_misallocate_class(group1_data,group1_label,cluster_label[0:len(group1_data)],1)
    c2_to_adjust = find_max_misallocate_class(group2_data,group2_label,cluster_label[len(group1_data):],-1)

    ulabel1[ulabel1.index(c1_to_adjust)] = c2_to_adjust
    ulabel2[ulabel2.index(c2_to_adjust)] = c1_to_adjust

    return ulabel1,ulabel2


def find_max_misallocate_class(group_data,group_label,cluster_label,group_option):
    """

    :param group_data:
    :param group_label:
    :param cluster_label:
    :param group_option:
    :return: class label
    """
    ulabel = {x: 0 for x in np.unique(group_label)}

    for inx in range(len(cluster_label)):
        if (cluster_label[inx] != group_option):
            ulabel[group_label[inx]] = ulabel[group_label[inx]] + 1

    class_to_adjust = min(ulabel.items(), key=lambda x: x[1])[0]

    return class_to_adjust

def get_Kneighbors(trainset,sample,k=3):
    """
    this method get neighbors by calculating the distance of all train samples to traget(sample) and sorting
    :param trainset:
    :param sample:
    :param k:
    :return:
    """
    distances = []
    for i,each in enumerate(trainset):
        if operator.eq(list(sample),list(each)) != 0:
            dis = get_point_point_dis(sample,each)
            distances.append((i,dis))

    distances.sort(key=operator.itemgetter(1))#根据距离排序

    neighbors = []
    for i,key in enumerate(distances):
        if i >= k:
            break
        neighbors.append(trainset[key[0]])

    return neighbors




