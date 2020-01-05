# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/2 8:14
# file: RadarChart.py
# description:


import numpy as np
import matplotlib.pyplot as plt
from ECOC_library.Common.Read_Write_tool import read_ECOC_res

def draw_angle(ax,data,color,marker):
    angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    ax.plot(angles, data, color='black', marker= marker,linewidth=2,markersize=10,alpha=0.5)
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    return ax

if __name__ == '__main__':

    # 标签
    labels = np.array(['    accuracy', 'precision', 'specifity    ', 'sensitivity     '\
                        , 'Fscore'])
    # 数据个数
    dataLenth = 5

    # 数据
    fold_path = r'E:/paper/paper_writing/ACML/data/SAT_ECOC/various_DC_select_column'
    standards = ['accuracy','Fscore','precision','sensitivity','specifity']
    ecoc_name = ['SAT F1 F2 DC=F1','SAT F1 F2 DC=F2','SAT F1 F2 DC=N3']
    legends = ['F1','F2','N3']
    data_name = ['Breast', 'Cancers', 'DLBCL', 'Leukemia1', 'SRBCT', 'Avg']
    titles = ['Breast', 'Cancers', 'DLBCL', 'Leukemia', 'SRBCT', 'Avg']

    color = ['red','blue','fuchsia','cyan','black','yellow','green']
    linestyle=[':','-.','--','-']
    markers = ['o', '^','*']
    id=['(a)','(b)','(c)','(d)','(e)','(f)']


    for k in range(len(data_name)):
        if k != 3:
            continue
        fig = plt.figure(figsize=(4.3,3.5))
        ax = fig.add_subplot(111, polar=True)
        for j in range(len(ecoc_name)):
            data = []
            for i in range(len(standards)):
                file_path = fold_path + '/' + standards[i] + '_RandForReg.xls'
                data.append(read_ECOC_res(file_path,ecoc_name[j],data_name[k]))
            ax = draw_angle(ax,data,color=color[j],marker=markers[j])

        ax.legend(legends,shadow=True,loc='center',fontsize=9)
        ax.set_title(id[k] + titles[k], va='bottom', fontproperties="SimHei",fontsize=12)
        ax.grid(True)
        save_path = fold_path + '/' + data_name[k] + '.png'
        print('save path:' + save_path)
        plt.savefig(save_path,dpi=300)
        plt.show()




