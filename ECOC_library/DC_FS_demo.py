# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/29 18:23
# file: DC_FS_demo.py
# description: this module shows how use DC_feature_selection method

import numpy as np

from ECOC_library.Common.Evaluation_tool import Evaluation
from ECOC_library.ECOC.Classifier import OVO_ECOC
from ECOC_library.FS.DC_Feature_selection import DC_FS, select_data_by_feature_index
from ECOC_library.Common.Read_Write_tool import read_Microarray_Dataset


train_path = r'./Microarray_data/treated_data/Breast_train.csv'
test_path = r'./Microarray_data/treated_data/Breast_test.csv'
train_data, train_label = read_Microarray_Dataset(train_path)
test_data, test_label = read_Microarray_Dataset(test_path)


#new ECOC model
selected_data, selected_feature_inx = DC_FS(train_data,train_label)
test_data = np.array(select_data_by_feature_index(test_data, selected_feature_inx))
E = OVO_ECOC()
E.fit(selected_data, train_label)

predicted_label = E.predict(test_data)
res = Evaluation(test_label,predicted_label).evaluation(accuracy=True, Fscore=True)

print('selected feature num: %d'%len(selected_feature_inx))
print(res)

# selected feature num: 9211
# [0.20000000000000001, 0.20000000000000001]
