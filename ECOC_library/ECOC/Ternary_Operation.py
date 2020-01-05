# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/3/12 19:47
# file: Ternary_Operation.py
# description: this model defines the ternary operation for self_adaptive ECOC

import logging
import numpy as np
import operator
import math
import copy

from ECOC_library.DC import Get_Complexity as GC
from ECOC_library.ECOC import Greedy_Search as GS


# jiafa
def ternary_add(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if (a == -1 and b == 1) or (a == 1 and b == -1) or (a == 0 and b == 0):
            res = 0
        elif (a == 0 and b == 1) or (a == 1 and b == 0) or (a == -1 and b == -1):
            res = 1
        elif (a == 0 and b == -1) or (a == -1 and b == 0) or (a == 1 and b == 1):
            res = -1
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))

        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))

    return parent


# jianfa
def ternary_subtraction(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if (a == 0 and b == 0) or (a == 1 and b == 1) or (a == -1 and b == -1):
            res = 0
        elif (a == 0 and b == -1) or (a == 1 and b == 0) or (a == -1 and b == 1):
            res = 1
        elif (a == 0 and b == 1) or (a == 1 and b == -1) or (a == -1 and b == 0):
            res = -1
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))

        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))

    return parent


# changfa
def ternary_multiplication(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if a == 0 or b == 0:
            res = 0
        elif (a == 1 and b == 1) or (a == -1 and b == -1):
            res = 1
        elif (a == 1 and b == -1) or (a == -1 and b == 1):
            res = -1
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))

        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))
    return parent


# chufa
def ternary_divide(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if b == 0:
            logging.error('ERROR: The denominator of Ternary Divide is zero!')
            res = 0
        if a == 0:
            res = 0
        elif (a == 1 and b == -1) or (a == -1 and b == 1):
            res = -1
        elif (a == 1 and b == 1) or (a == -1 and b == -1):
            res = 1
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))

        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))
    return parent


# yu
def ternary_and(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if (a == 1 and b == 1):
            res = 1
        elif a == -1 or b == -1:
            res = -1
        elif a == 0 or b == 0:
            res = 0
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))
    return parent


# huo
def ternary_or(left, right, **param):
    parent = None
    for i in range(len(left)):
        a = left[i][0]
        b = right[i][0]
        if a == 1 or b == 1:
            res = 1
        elif a == 0 or b == 0:
            res = 0
        elif a == -1 and b == -1:
            res = -1
        else:
            logging.error('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
            ValueError('ERROR: left %d, right %d, left and right node is wrong!' % (a, b))
        if parent is None:
            parent = copy.deepcopy(res)
        else:
            parent = np.row_stack((parent, res))
    return parent


def cal_info(vector):
    label = np.unique(vector)
    p = 0
    for i, each in enumerate(label):
        pi = list(vector).count(each) / float(len(vector))
        if pi != 1.0:
            p = p + pi * math.log(2, pi)
    return p


def ternary_info(left, right, **param):
    operation_name = {'Ad': ternary_add, 'Sub': ternary_subtraction, 'Mu': ternary_multiplication \
        , 'D': ternary_divide, 'A': ternary_and, 'O': ternary_or}
    ternary_res = {}
    for i, each in operation_name.items():
        ternary_res[i] = each(left, right)

    Info = {}
    for i, each in enumerate(ternary_res):
        Info[each] = cal_info(ternary_res[each])

    min_inx = sorted(Info.items(), key=operator.itemgetter(1))[0][0]
    return ternary_res[min_inx]


def ternary_DC(left, right, **param):
    operation_name = {'Ad': ternary_add, 'Sub': ternary_subtraction, 'Mu': ternary_multiplication \
        , 'D': ternary_divide, 'A': ternary_and, 'O': ternary_or}
    ternary_res = {}
    for i, each in operation_name.items():
        ternary_res[i] = each(left, right)

    all_classes = np.unique(param['label'])
    cplx = {}
    group1 = []
    group2 = []
    for each in ternary_res:
        class_label = np.unique(ternary_res[each])
        if 1 in class_label and -1 in class_label:
            for j in range(len(ternary_res[each])):
                if ternary_res[each][j] == 1:
                    group1.append(all_classes[j])
                elif ternary_res[each][j] == -1:
                    group2.append(all_classes[j])
            cplx[each] = GS.get_DC_value(param['data'], param['label'], group1, group2, dc_option='F1')

    min_info_inx = sorted(cplx.items(), key=operator.itemgetter(1))[0][0]
    return ternary_res[min_info_inx]
