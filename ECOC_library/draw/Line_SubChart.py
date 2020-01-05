# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/2 8:23
# file: Line_SubChart.py
# description:

import numpy as np

from ECOC_library.Common.Read_Write_tool import read_ECOC_res
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 数据
fold_path = r'E:/paper/paper_writing/ACML/data/SAT_ECOC/various_combination'
other_ecoc_path = r'E:/paper/paper_writing/ACML/data/Other_ECOC'
standards = ['accuracy', 'Fscore', 'precision', 'sensitivity',  'specifity']
ecoc_name = ['SAT F1 F1 DC', 'SAT F2 F2 DC', 'SAT N3 N3 DC','SAT F1 F2 DC','SAT F1 N3 DC','SAT F2 N3 DC','SAT F1 F2 N3 DC']
other_ecoc_name = ['OVA_ECOC','OVO_ECOC','Dense_random_ECOC','Sparse_random_ECOC','D_ECOC','DC_ECOC F1'\
              ,'DC_ECOC F2','DC_ECOC N3']
data_name = ['Breast','Cancers','DLBCL','Leukemia1','SRBCT','Avg']
color = ['red', 'blue', 'fuchsia', 'cyan', 'black', 'yellow', 'green']
markers = ['o','D','s','*','^','<']
fig, ax = plt.subplots()

#main axis
for i in range(len(data_name)):
    data = []
    for j in range(len(ecoc_name)):
        file_path = fold_path + '/accuracy_RandForReg.xls'
        data.append(read_ECOC_res(file_path, ecoc_name[j], data_name[i]))
    for j in range(len(other_ecoc_name)):
        file_path = other_ecoc_path + '/accuracy_RandForReg_backup.xls'
        data.append(read_ECOC_res(file_path, other_ecoc_name[j], data_name[i]))
    ax.plot(data, '-', ms=10, lw=2, alpha=0.7, mfc='black',color='black',marker=markers[i],markersize=8)
plt.legend(data_name,loc='upper center',ncol=3,fontsize=10)

plt.ylim(0.85,1.04)
ax.grid()
plt.xticks(np.arange(15), ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)','(j)','(k)','(l)','(m)','(n)','(o)') )
# position bottom right
ax.set_title('Accuracy', va='bottom', fontproperties="SimHei",fontsize=13)
plt.ylabel('accuracy_values')
plt.xlabel('ecoc_name')

#inset another axis
data1 = []
for k in range(len(standards)):
    file_path = fold_path + '/' + standards[k] + '_RandForReg.xls'
    data1.append(read_ECOC_res(file_path, ecoc_name[2], 'Avg'))

data2 = []
for k in range(len(standards)):
    file_path = fold_path + '/' + standards[k] + '_RandForReg.xls'
    data2.append(read_ECOC_res(file_path, ecoc_name[5], 'Avg'))

data = [data1, data2]
index = np.arange(len(data[0]))
a = plt.axes([.22, .25, .2, .2], axisbg='w')#only y axis
colors = ['w','black']
for i in range(len(data)):
    plt.bar(index + i * .25 + .1, data[i], width=.25, color=colors[i])
plt.ylim(0.85,1.02)

plt.title('Avg',fontsize=11)
# plt.xlabel('standards')
# plt.ylabel('values')
plt.xticks(np.arange(5),standards,rotation=45,fontsize=10)
plt.legend(['(c)','(f)'],fontsize=10,loc='center upper')

save_path = fold_path + '/line_sub.png'
print('save path:' + save_path)

plt.savefig(save_path)
plt.show()
