# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/30 9:33
# file: Line_LineChart.py
# description:

import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from ECOC_library.Common.Read_Write_tool import read_ECOC_res
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 数据
fold_path = r'E:/paper/paper_writing/ACML/data/SAT_ECOC/different_FS/SAT_ECOC'
other_ecoc_path = r'E:/paper/paper_writing/ACML/data/SAT_ECOC/different_FS/Other_ECOC'
standards = ['accuracy', 'Fscore', 'precision', 'sensitivity',  'specifity']
ecoc_name = ['Self_Adaption_ECOC F1 F1 DC', 'Self_Adaption_ECOC F2 F2 DC', 'Self_Adaption_ECOC N3 N3 DC'\
            ,'Self_Adaption_ECOC F1 F2 DC','Self_Adaption_ECOC F1 N3 DC','Self_Adaption_ECOC F2 N3 DC'\
            ,'Self_Adaption_ECOC F1 F2 N3 DC']
other_ecoc_name = ['OVA_ECOC','OVO_ECOC','Dense_random_ECOC','Sparse_random_ECOC','D_ECOC','DC_ECOC F1'\
              ,'DC_ECOC F2','DC_ECOC N3']
data_name = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2','Lung1','SRBCT']
legends = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2','Lung','SRBCT']
color = ['red', 'blue', 'fuchsia', 'cyan', 'black', 'yellow', 'green','brown','forest green']
markers = ['o','D','s','*','^','<','>','H','v']
index= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j','k','l','m','n','o']
fig, ax = plt.subplots()

#main axis
FS_name = 'linear_svc'

for i in range(len(data_name)):
    data = []
    for j in range(len(ecoc_name)):
        file_path = fold_path + '/accuracy_'+ FS_name + '.xls'
        data.append(read_ECOC_res(file_path, ecoc_name[j], data_name[i]))
    for j in range(len(other_ecoc_name)):
        file_path = other_ecoc_path + '/accuracy_'+ FS_name +'_backup.xls'
        data.append(read_ECOC_res(file_path, other_ecoc_name[j], data_name[i]))
    ax.plot(data, '-', ms=10, lw=2, alpha=0.7, mfc=color[i],color=color[i],marker=markers[i],markersize=7)
plt.legend(legends,loc='upper center',ncol=3,fontsize=9)
plt.ylim(0.75,1.075)
ax.grid()
plt.yticks(np.arange(0.750,1.075,0.05))
plt.xticks(np.arange(15), index, fontsize=10)
# position bottom right
ax.set_title('Accuracy', va='bottom', fontproperties="SimHei",fontsize=13)
plt.ylabel('accuracy_values')
plt.xlabel('ecoc_name')

#inset avg axis
data = []
for j in range(len(ecoc_name)):
    file_path = fold_path + '/accuracy_' + FS_name +'.xls'
    data.append(read_ECOC_res(file_path, ecoc_name[j], 'Avg'))
for j in range(len(other_ecoc_name)):
    file_path = other_ecoc_path + '/accuracy_' + FS_name + '_backup.xls'
    data.append(read_ECOC_res(file_path, other_ecoc_name[j],'Avg'))

a = plt.axes([.275, .148, .25, .20], axisbg='w')#only y axis
inx = np.arange(len(data))
for i in range(len(data)):
    plt.bar(range(len(data)), data, 0.8,color='blue')

plt.yticks(np.arange(0.875,1.010,0.025),fontsize=8)

plt.ylim(0.875,0.975)


plt.title('Average',fontsize=10)
plt.xticks(np.arange(15), index,fontsize=8)

save_path = fold_path + '/' + FS_name + '_line_line.png'
print('save path:' + save_path)

plt.savefig(save_path)
plt.show()
