# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 18-11-5
# file: DC_ECOC_demo
# description:

"""
There is an example of using OVO_ECOC class to validate on dermatology data
"""
from ECOC_library.Common.Read_Write_tool import read_UCI_Dataset
from ECOC_library.ECOC.Classifier import DC_ECOC

filepath = r'./UCI_data/treated_data/dermatology.csv'
data, label = read_UCI_Dataset(filepath)

# new ECOC model
E = DC_ECOC()
E.fit(data, label)

true_label = label[0]
predicted_label = E.predict(data[0])

# calculate the 3K cross validation accuracy
accuracy = E.validate(data, label)

# print accuracy
print("accuracy:", accuracy)
