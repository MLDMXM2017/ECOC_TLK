# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/5/1 8:00
# file: LineChart.py
# description:

import numpy as np
import re
import matplotlib.pyplot as plt

def read_data(path,data,ecoc):
    file_object = open(path, 'rU')
    try:
        for line in file_object:
            if (data in line) & (ecoc in line):
                res = re.findall(r"(?<=\[).*?(?=\])",line)[0].replace(',','').split(' ')
                data = []
                for each in res:
                    data.append(float(each))
                return data
    finally:
        file_object.close()
    return 0

if __name__ == '__main__':
    SAT_ECOC_path = 'E:/paper/paper_writing/ACML/data/SAT_ECOC/classifier_numbers/SAT_ECOC/cls_acc_RandForReg.txt'
    other_ECOC_path = 'E:/paper/paper_writing/ACML/data/SAT_ECOC/classifier_numbers/Other_ECOC/cls_acc_RandForReg.txt'
    res_fig = 'E:/paper/paper_writing/ACML/data/SAT_ECOC/classifier_numbers'

    microarray_dataname = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2'\
                ,'Lung1','SRBCT']

    ecoc_name = ['Self_Adaption_ECOC F1 F1 DC', 'Self_Adaption_ECOC F2 F2 DC', 'Self_Adaption_ECOC N3 N3 DC' \
        , 'Self_Adaption_ECOC F1 F2 DC', 'Self_Adaption_ECOC F1 N3 DC', 'Self_Adaption_ECOC F2 N3 DC' \
        , 'Self_Adaption_ECOC F1 F2 N3 DC']

    other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'DC_ECOC F1' \
        , 'DC_ECOC F2', 'DC_ECOC N3']


    labels = ['b','h', 'i','j','k','l','m','n','o']
    legends = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2', 'Lung', 'SRBCT']
    color = ['red', 'blue', 'fuchsia', 'cyan', 'black', 'yellow', 'green', 'brown', 'forest green']
    markers = ['o', 'D', 's', '*', '^', '<', '>', 'H', 'v']
    index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

    fig, ax = plt.subplots()
    for inx,data in enumerate(microarray_dataname):

        list_data = []

        for ecoc in ecoc_name:#other ECOC
            classifier_arr = read_data(SAT_ECOC_path, data, ecoc)
            list_data.append(len(classifier_arr)) #classifer numbers

        for ecoc in other_ECOC:#other ECOC
            classifier_arr = read_data(other_ECOC_path, data, ecoc)
            list_data.append(len(classifier_arr)) # classifier numbers

        ax.plot(list_data, '-', ms=10, lw=2, alpha=0.7, color=color[inx],marker=markers[inx],markersize=5)  # will be used to label x-ticks
    plt.legend(legends, loc='left top', ncol=2, fontsize=10)

    ax.set_title('Classifier Numbers', va='bottom', fontproperties="SimHei", fontsize=13)
    ax.grid()
    plt.yticks(np.arange(2, 85, 4))
    plt.xticks(np.arange(15), (
    '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)'))
    # position bottom right
    plt.ylabel('classifier_numbers')
    plt.xlabel('ecoc_name')
    plt.savefig(res_fig + '/' + data + 'RandForReg.png')
    plt.show()




