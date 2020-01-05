# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/20 12:15
# file: Greedy_Search.py
# description: this model define how to generate ECOC by DC greedy strategy

import numpy as np
import random
import logging
import operator

from ECOC_library.DC import Get_Complexity as GC
from ECOC_library.ECOC import Matrix_tool as MT

def greedy_search(data, labels, dc_option = 'F1'):
    """

    :param data:
    :param labels:
    :param dc_option:
    :return: divided_partitons
    """
    group1 = []
    group2 = []

    classes = np.unique(labels)
    rand_classes = classes.tolist()
    random.shuffle(rand_classes)
    # print('random shuffle:',rand_classes)

    group1 = rand_classes[0:int(len(rand_classes) / 2)]
    group2 = rand_classes[int(len(rand_classes) / 2):]
    DC = get_DC_value(data,labels,group1,group2,dc_option)

    logging.info('=*=*=*=*=* Greedy Search  *=*=*=*=*=*=*')
    iteration = 0
    while(True):
        logging.info('iteration:%-8d %-2s %f' %(iteration,dc_option,DC))
        iteration = iteration + 1
        c1,c1_inx,c2,c2_inx = _get_most_cplx_class(data,labels,group1,group2,dc_option)
        post_group1,post_group2 = _swap_class(group1,group2,c1,c1_inx,c2,c2_inx)
        post_DC = get_DC_value(data,labels,post_group1,post_group2,dc_option)

        # judge whethe DC is lower than pre_DC or not
        flag = 0
        if operator.eq(dc_option,'F1') or operator.eq(dc_option,'F3'):
            if post_DC > DC:
                flag = 1
        else:
            if post_DC < DC:
                flag = 1
        if  flag == 1:
            DC = post_DC
            group1 = post_group1
            group2 = post_group2
        else:
            break

    return group1,group2


def get_DC_value(data,labels,group1,group2,dc_option):
    """
    
    :param data: the whole data
    :param labels: the whole labels
    :param group1: group1 classes 
    :param group2: group2 classes
    :param dc_option: select which dc is used
    :return: DC value
    """
    group1_data, group1_label = MT.get_data_subset(data, labels, group1)
    group2_data, group2_label = MT.get_data_subset(data, labels, group2)

    try:
        funname = 'get_complexity_' + dc_option
        fun = getattr(GC,funname)
        DC = fun(list(group1_data),list(group1_label),list(group2_data),list(group2_label))
    except:
        # logging.error('DC option is wrong')
        # raise NameError('DC option is wrong')
        return 0
    return DC

def _get_most_cplx_class(data,labels,group1,group2,dc_option):
   """
   
   :param data: the whole data
   :param labels: the whole label
   :param group1: group1 classes
   :param group2: group2 classes
   :param dc_option: select which dc is used
   :return: complex c1,complex c2
   """
   group = group1 + group2
   cplxs = np.zeros((len(group),len(group)))  
   for inx1 in range(len(group)):
       for inx2 in range(inx1+1,len(group)):
           cplxs[inx1][inx2] = get_DC_value(data,labels,[group[inx1]],[group[inx2]],dc_option)
           cplxs[inx2][inx1] = cplxs[inx1][inx2]

   cplxs_sum = list(cplxs.sum(axis=1)) #the sum of each row == each class to other class complexity

   if dc_option == 'F1' or dc_option == 'F3':
       c1_inx = cplxs_sum.index(min(cplxs_sum[0:len(group1)]))  # group1 most complex class
       c2_inx = cplxs_sum.index(min(cplxs_sum[len(group1):])) - len(group1)  # group2 most complex class

   else:
       c1_inx = cplxs_sum.index(max(cplxs_sum[0:len(group1)]))  # group1 most complex class
       c2_inx = cplxs_sum.index(max(cplxs_sum[len(group1):])) - len(group1)  # group2 most complex class

   c1 = group1[c1_inx]
   c2 = group2[c2_inx]
   return c1,c1_inx,c2,c2_inx

def _swap_class(group1,group2,c1,c1_inx,c2,c2_inx):
    """

    :param data: the whole data
    :param labels: the whole labels
    :param group1: group1 classes
    :param group2: group2 classes
    :return: adj_group1 adj_group2
    """

    if group1[c1_inx] !=c1 or group2[c2_inx]!=c2:
        logging.error("class idnex is not match the class")
        return

    group1[c1_inx] = c2
    group2[c2_inx] = c1

    return group1,group2


