# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/3/20 18:53
# file: Transition_tool.py
# description: this module offers some methods for tranlating discrete vaiables into continue vaiables

import numpy as np


def round_list(data):
    tdata = []
    for i in range(data.shape[0]):
        d = []
        for j in range(data.shape[1]):
            num = round(float(data[i][j]), 3)
            d.append(num)
        tdata.append(np.array(d))

    return tdata


def turn_label_2num(data):
    classes = {each: i for i, each in enumerate(np.unique(data))}
    new_data = []
    for each in data:
        new_data.append(classes[each])
    return new_data
