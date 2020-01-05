# -*- coding: utf-8 -*-
# author: fengkaijie
# time: 2018/1/29 18:33
# file: Matrix_tool.py
# description: this method define some tool kit to checkout matrix
# or get data subset from data set


import copy
import numpy as np
import logging
import operator
import math
import random

import Ternary_Operation
from Distance import euclidean_distance
from ECOC_library.Common.Evaluation_tool import Evaluation
from ECOC_library.DC import Get_Complexity

def get_data_from_col(data, label, col, index):
    """
    to get data subset form a col, where the value is not zero
    :param data: data set
    :param label: label corresponding to data
    :param col: the col we want to get data subset
    :param index: the index for matrix
    :return: data subset and corresponding labels
    """
    data_result = None
    cla_result = None
    for i in range(len(col)):
        if col[i] != 0:
            d = np.array([data[k] for k in range(len(label)) if label[k] == get_key(index, i)])
            c = np.ones(len(d)) * col[i]
            if d.shape[0] > 0 and d.shape[1] > 0:
                if data_result is None:
                    data_result = copy.copy(d)
                    cla_result = copy.copy(c)
                else:
                    data_result = np.vstack((data_result, d))
                    cla_result = np.hstack((cla_result, c))
    return data_result, cla_result


def closet_vector(vector, matrix, distance=euclidean_distance, weights=None):
    """
    find the closet coding vector in matrix
    :param vector: a predicted vector
    :param matrix: coding matrix
    :param distance: a callable object to calculate distance
    :param weights: the weights for each feature
    :return: the index corresponding to closet coding vector
    """
    d = np.inf
    index = None
    for i in range(matrix.shape[0]):
        if distance(vector, matrix[i], weights) < d:
            d = distance(vector, matrix[i], weights)
            index = i
    return index


def get_key(dictionary, value):
    for i in dictionary:
        if dictionary[i] == value:
            return i


def exist_same_row(matrix):
    """
    to checkout whether there are same rows in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    row_count = matrix.shape[0]
    for i in range(row_count):
        for j in range(i+1, row_count):
            if np.all([matrix[i] == matrix[j]]) or np.all([matrix[i] == -matrix[j]]):
                # print('matrix[i]:', matrix[i])
                # print('matrix[j]:', matrix[j])
                return True
    return False


def exist_same_col(matrix):
    """
    to checkout whether there are same cols in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    col_count = matrix.shape[1]
    for i in range(col_count):
        for j in range(i+1, col_count):
            if np.all([matrix[:, i] == matrix[:, j]]) or np.all([matrix[:, i] == -matrix[:, j]]):
                # print('matrix[:,i]:', matrix[:, i])
                # print('i:', i)
                # print('matrix[:,j]:', matrix[:, j])
                # print('j:', j)
                return True
    return False


def exist_two_class(matrix):
    """
    to ensure all cols in coding matrix have 1 and -1
    :param matrix: coding matrix
    :return: true or false
    """
    col_count = matrix.shape[1]
    for i in range(col_count):
        col_unique = np.unique(matrix[:, i])
        if (1 not in col_unique) or (-1 not in col_unique):
            # print('dont have two classes:', matrix[:, i])
            return False
    return True


def get_data_subset(data, label, target_label):
    """
    to get data with certain labels
    :param data: data set
    :param label: label corresponding to data
    :param target_label: the label which we want to get certain data
    :return:
    """
    print(target_label)
    data_subset = np.array([data[i] for i in range(len(label)) if label[i] in target_label])
    label_subset = np.array([label[i] for i in range(len(label)) if label[i] in target_label])
    return data_subset, label_subset


def get_subset_feature_from_matrix(matrix, index):
    """
    I forget what it uses to.
    :param matrix:
    :param index:
    :return:
    """
    res = []
    for i in range(matrix.shape[1]):
        class_1 = []
        class_2 = []
        for j in range(matrix.shape[0]):
            if matrix[j, i] > 0:
                class_1.append(get_key(index, j))
            elif matrix[j, i] < 0:
                class_2.append(get_key(index, j))
        res.append(class_1)
        res.append(class_2)
    return res


def create_confusion_matrix(y_true, y_pred, index):
    """
    create a confusion matrix
    :param y_true: true label
    :param y_pred: predicted label
    :param index: matrix index
    :return: confusion matrix
    """
    res = np.zeros((len(index), len(index)))
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            try:
                res[index[y_true[i]], index[y_pred[i]]] += 1
            except KeyError:
                pass
    return res + res.T


def have_same_col(col, matrix):
    """
    to checkout wheather the col in coding matrix
    :param col:certain col to checkout
    :param matrix:coding matrix
    :return:true or false
    """
    col = col.reshape((1, -1))
    for i in range(matrix.shape[1]):
        if np.all([col == matrix[:, i]]) or np.all([col == -matrix[:, i]]):
            return True
    return False


def create_col_from_partition(class_1_variety, class_2_variety, index):
    """
    create a col based on a certain partition
    :param class_1_variety: a part of partition as positive group
    :param class_2_variety: another part of partition as negative group
    :param index: index of coding matrix
    :return: a col
    """
    col = np.zeros((len(index), 1))
    for i in class_1_variety:
        col[index[i]] = 1
    for i in class_2_variety:
        col[index[i]] = -1
    return col


def estimate_weight(error):
    """
    to estimate weights for base estimators based on the error rates
    :param error: error rates
    :return:
    """
    if error < 0.0000001:
        error = 0.0000001
    if error == 1:
        error = 0.9999999
    return 0.5*np.log((1-error)/error)

def select_column(M,data,label,label_length):
    """
    this function is unknown
    :param M:
    :return:
    """

    # logging.info("before select_column:\r\n" + str(M))

    #calculate the column number of matrix
    index = {l: i for i, l in enumerate(np.unique(label))}
    cplx = {}
    for i,each in enumerate(np.unique(label)):
        cls_data,cls_label = get_data_subset(data,label,[each])
        cplx[each] = Get_Complexity.get_complexity_D2(cls_data,cls_label,k=5)
    bottom_stand = np.mean(list(cplx.values()))
    cplx_stand = np.mean(list(cplx.values())) * 2

    logging.info("class cplx:\r\n" + str(cplx))
    logging.info("var:%-10f mean:%-10f" %(np.var(list(cplx.values())), np.mean(list(cplx.values()))))
    logging.info("botttom th:%-10f top th:%-10f" %(bottom_stand, cplx_stand))

    cls_num = {}
    for i, each in enumerate(np.unique(label)):
        if cplx[each] > bottom_stand:
            cls_num[each] = 1
        if cplx[each] > cplx_stand:
            cls_num[each] = 2
    column_num = label_length + np.sum(list(cls_num.values()))

    logging.info("column num:%d" %np.sum(list(cls_num.values())))
    #column length is right, return M
    if len(M[0]) == column_num:
        return M

    #1.column length is less, polish M
    if len(M[0]) < column_num:
        cplx = sorted(cplx.items(),key=operator.itemgetter(1)) #small to big

        for i in range(int(column_num) -  len(M[0])):#add column_num-M column
            if column_num == len(M[0]):
                break
            if i < len(cplx):
                inx1 = index[cplx[i][0]]
                inx2 = index[cplx[-(i+1)][0]]
                if inx1 == inx2:
                    inx2 = index[cplx[-1][0]] #odd,mid+end
                new_col = np.zeros((len(index), 1))
                new_col[inx1] = 1
                new_col[inx2] = -1
                M = np.hstack((M, new_col))
        return M

    # 2.minus extra column
    else:
        # find a column with all class
        used_column = []
        M_backup = copy.deepcopy(M)
        for i in range(len(M[0])):
            i_column = [row[i] for row in M]
            if np.all(i_column) == True:
                GPM = i_column
                column_to_divide = [i_column]
                M_backup = np.delete(M_backup, i, axis=1)
                break

        while column_to_divide:
            column = column_to_divide.pop(0)

            pos_cls = [i for i,each in enumerate(column) if each == 1]
            neg_cls = [i for i,each in enumerate(column) if each == -1]

            del_inx = []
            for i in range(len(M_backup[0])):
                i_column = [row[i] for row in M_backup]
                if check_sub_tree(i_column,pos_cls) or check_sub_tree(i_column,neg_cls):
                    column_to_divide.append(copy.deepcopy(i_column))
                    GPM = np.vstack((GPM,i_column))
                    del_inx.append(i)
                    if len(del_inx) == 2:
                        break

            for inx, each in enumerate(del_inx):
                M_backup = np.delete(M_backup, each - inx, axis=1)

        #find a column with many 0 or without 0
        #calculate the num of zero
        zero_num = {}
        for i in range(len(M[0])):
            i_column = [row[i] for row in M]
            zero_num[i] = i_column.count(0) #dict{column_inx,cplx_value}

        zero_num = sorted(zero_num.items(), key=operator.itemgetter(1),reverse=True)  #big to small

        for each in zero_num:
            if column_num == len(GPM[0]):
                break
            key = each[0]
            i_column = [row[key] for row in M]
            if is_repeat(np.transpose(GPM),i_column) == False and len(GPM) < column_num:
                if is_comtain_cplx_cls(i_column,cls_num.keys(),label):#优先选择0多的并且包含复杂类的列
                    GPM = np.vstack((GPM, i_column))

        GPM = np.transpose(GPM)
        return GPM

def is_comtain_cplx_cls(column,cplx_cls,label):
    index = {l:i for i, l in enumerate(np.unique(label))}
    cplx_cls_index = []
    for each in cplx_cls:
        cplx_cls_index.append(index[each])
    for i in cplx_cls_index:
        if column[i] == 0:
            return False
    return True

def is_repeat(M,column):
    for i in range(len(M[0])):
        i_column = [row[i] for row in M]
        if operator.eq(i_column,column):
            return True
    return False

def check_sub_tree(column,cls):
    for i, each in enumerate(column):
        if i in cls and each == 0:
            return False
        elif i not in cls and each != 0:# not contain the cls
            return False

    return True

def left_right_create_parent(left,right,option,data,label):
    if option == '+':
        ternary_fun_name = 'ternary_add'
    elif option == '-':
        ternary_fun_name = 'ternary_subtraction'
    elif option == '*':
        ternary_fun_name = 'ternary_multiplication'
    elif option == '/':
        ternary_fun_name = 'ternary_divide'
    elif option == 'and':
        ternary_fun_name = 'ternary_and'
    elif option == 'or':
        ternary_fun_name = 'ternary_or'
    elif option == 'info':
        ternary_fun_name = 'ternary_info'
    elif option == 'DC':
        ternary_fun_name = 'ternary_DC'
    else:
        ValueError('ERROR:wrong ternary option!')

    ternary_fun = getattr(Ternary_Operation, ternary_fun_name)
    parent_node = ternary_fun(left, right, data=data, label=label)
    return parent_node


def get_2column(M):
    left_node = [[row[0]] for row in M]
    right_node = [[row[1]] for row in M]
    M = np.delete(M,[0,1],axis=1)
    return left_node, right_node, M

def insert_2column(M,left_node,right_node):
    if M is None:
        M = copy.deepcopy(np.hstack((left_node, right_node)))
    else:
        M = np.hstack((M, left_node, right_node))
    return M

def remove_unfit(M):
    delinx = []
    for i in range(len(M[0])):
        column = [row[i] for row in M]
        if 1 not in column or -1 not in column:
            delinx.append(i)
    for inx,each in enumerate(delinx):
        M = np.delete(M,each-inx,axis=1)

    return M

def remove_reverse(M):
    # logging.info("before remove_reverse:\r\n" + str(M))
    delete_row_index = []
    for i in range(len(M[0])):
        for j in range(i+1,len(M)):
            if operator.eq(list(M[i]),list(-M[j])):
                delete_row_index.append(j)

    for inx, each in enumerate(delete_row_index):
        M = np.delete(M, each - inx)

    delete_column_index = []
    for i in range(len(M[0])):
        for j in range(i+1,len(M[0])):
            i_column = [row[i] for row in M]
            j_column = [row[j] for row in M]

            if operator.eq(i_column,list(-np.array(j_column))):
                delete_column_index.append(j)
    delete_column_index = np.unique(delete_column_index)
    for inx, each in enumerate(delete_column_index):
        M = np.delete(M, each - inx, axis=1)

    # logging.info("after remove_reverse:\r\n" + str(M))
    return M


def remove_duplicate_row(M):#non sense
    for i in range(len(M[0])):
        i_row = M[i]
        for j in range(i+1, len(M)):
            j_row = M[j]
            if i_row == j_row:
                M = np.delete(M,j)
    return M

def remove_duplicate_column(M):
    del_inx = []
    for i in range(len(M[0])):
        i_column = [row[i] for row in M]
        for j in range(i+1, len(M[0])):
            j_column = [row[j] for row in M]
            if i_column == j_column:
                del_inx.append(j)
    del_inx = np.unique(del_inx)
    for inx, each in enumerate(del_inx):
        M = np.delete(M, each - inx, axis=1)
    return M

def change_subtree(target,source):

    #produce random changed col
    tar_col = random(0,len(target[0]))
    source_col = random(0,len(source[0]))

    #define changed class
    tar_class = []
    for i,row in enumerate(target):
        if row[tar_col] != 0:
            tar_class.append(i)

    source_class = []
    for i,row in enumerate(source):
        if row[source_col] != 0:
            tar_class.append(i)

    # save left column
    target_save_col = []
    for i in range(len(target[0])):
        col = [row[i] for row in target]
        for j in tar_class:
            if col[j] != 0: #left class
                target_save_col.append(i)

    source_save_col = []
    for i in range(len(source[0])):
        col = [row[i] for row in source]
        for j in source_class:
            if col[j] != 0: # changed class
                source_save_col.append(i)

    new_M = None
    #merge save column
    for i in range(target[0]):
        if i in target_save_col:
            col = [row[i] for row in target]
            if new_M is None:
                new_M = copy.copy(np.hstack((col)))
            else:
                new_M = np.hstack((new_M, col))

    for i in range(source[0]):
        if i in source_save_col:
            col = [row[i] for row in source]
            new_M = np.hstack((new_M, col))

    return target

def split_traindata(data,label):
    # train_data, train_label, val_data, val_label
    length = len(data)
    data_1 = data[:round(length/3 * 2)]
    label_1 = label[:round(length / 3 * 2)]

    data_2 = data[round(length / 3 * 2):]
    label_2 = label[:round(length / 3 * 2):]

    return data_1,label_1,data_2,label_2

def res_matrix(m,index,train_data,train_label,test_data,test_label,estimator,distance_measure):

    predictors = []
    for j in range(m.shape[1]):
        dat, cla = get_data_from_col(train_data, train_label, m[:, j], index)
        estimator = estimator().fit(dat, cla)
        predictors.append(estimator)

    predicted_label = []
    if len(predictors) == 0:
        logging.error('The Model has not been fitted!')

    if len(test_data.shape) == 1:
        test_data = np.reshape(test_data, [1, -1])

    for i in test_data:

        predicted_vector = []
        for i in predictors:
            predicted_vector.append(i.predict(np.array([test_data]))[0])

        value = closet_vector(predicted_vector, m, distance_measure)
        predicted_label.append(get_key(index, value))

    accuracy = Evaluation(test_label, predicted_label).evaluation(simple=True)

    return accuracy

def change_unfit_DC(M,data,label,dc_option):
    index = {l: i for i, l in enumerate(np.unique(label))}  # name:index
    for i in range(len(M[0])):
        column = list(M[:,i])
        if (1 not in column or -1 not in column) and column.count(0)  != len(column):#non all 0
            if 1 not in column:#no positive class
                reg = -1
            elif -1 not in column:#no negative
                reg = 1

            class_to_change = []
            for j in range(len(column)):
                if column[j] == reg:
                    class_to_change.append(j)

            cplx = {}
            for j, each in enumerate(np.unique(label)):
                if j in class_to_change:
                    cls_data, cls_label = get_data_subset(data, label, each)
                    cplx[j] = Get_Complexity.get_complexity_D2(cls_data, cls_label, k=5)

            cplx = sorted(cplx.items(), key=operator.itemgetter(1), reverse=True)
            class_to_change_index = cplx[0][0]  # big to small
            if M.shape[1] == 1:
                M[class_to_change_index] = -reg
            else:
                M[class_to_change_index,i]  = -reg
    return M

