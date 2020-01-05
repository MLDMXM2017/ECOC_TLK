# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/30 8:39
# file: BoxChart.py
# description:

import pandas as pd
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

    microarray_dataname = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2'\
                ,'Lung1','SRBCT']

    ecoc_name = ['Self_Adaption_ECOC F1 F1 DC', 'Self_Adaption_ECOC F2 F2 DC', 'Self_Adaption_ECOC N3 N3 DC' \
        , 'Self_Adaption_ECOC F1 F2 DC', 'Self_Adaption_ECOC F1 N3 DC', 'Self_Adaption_ECOC F2 N3 DC' \
        , 'Self_Adaption_ECOC F1 F2 N3 DC']

    other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'DC_ECOC F1' \
        , 'DC_ECOC F2', 'DC_ECOC N3']

    res_fig = 'E:/paper/paper_writing/ACML/data/SAT_ECOC/classifier_acc-F2-F2'
    labels = ['a','b','c','d','e','f','g','h', 'i','j','k','l','m','n','o']
    # labels = ['F2-F2', 'OVA', 'OVO', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'DECOC', 'ECOC-MDC-F1' \
    #     , 'ECOC-MDC-F2', 'ECOC-MDC-N3']

    for data in microarray_dataname:

        list_data = []

        for ecoc in ecoc_name:  # other ECOC
            classifier_arr = read_data(SAT_ECOC_path, data, ecoc)
            list_data.append(classifier_arr)

        for ecoc in other_ECOC:#other ECOC
            classifier_arr = read_data(other_ECOC_path, data, ecoc)
            list_data.append(classifier_arr)

        plt.ylabel('classifier_accuracy')
        plt.xlabel('ecoc_name')
        plt.title('classifier_accuracy')
        bplot = plt.boxplot(list_data,vert=True, patch_artist=True,labels=labels)  # will be used to label x-ticks
        plt.savefig(res_fig + '/' + data + 'RandForReg.png')
        print(res_fig + '/' + data + 'RandForReg.png')
        plt.show()




