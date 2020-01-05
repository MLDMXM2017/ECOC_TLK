# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/30 16:08
# file: ECOC_demo.py
# description: this module comoputs the original data classification res

from ECOC_library.Common.Evaluation_tool import Evaluation
from ECOC_library.ECOC.Classifier import OVO_ECOC
from ECOC_library.Common.Read_Write_tool import read_Microarray_Dataset

train_path = r'./Microarray_data/treated_data/Breast_train.csv'
test_path = r'./Microarray_data/treated_data/Breast_test.csv'
train_data, train_label = read_Microarray_Dataset(train_path)
test_data, test_label = read_Microarray_Dataset(test_path)

#origin model
E = OVO_ECOC()
E.fit(train_data, train_label)

predicted_label = E.predict(test_data)
res = Evaluation(test_label,predicted_label).evaluation(accuracy=True, Fscore=True)

print('feature num: %d'%len(train_data[1]))
print(res)

# origin data res :
# feature num: 9216
# [0.20000000000000001, 0.20000000000000001]