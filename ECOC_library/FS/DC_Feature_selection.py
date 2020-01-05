# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/29 12:15
# file: DC_feature_selection.py
# description: this model defines a novel feature selection method with DC measures


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFpr
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import logging

from ECOC_library.Common.Transition_tool import turn_label_2num


from ECOC_library.DC.Get_Complexity import *
from ECOC_library.ECOC.Greedy_Search import greedy_search
from ECOC_library.ECOC.Matrix_tool import get_data_subset


def DC_FS(data, label):
    feature_num = len(data[1])
    feature_votes = init_vote_vector(data, label, feature_num, 'tree', 5, 'vertical')
    th = get_threhold(data, label, feature_num, feature_votes)

    selected_feature_inx = [i for i in range(feature_num) if feature_votes[i] >= th]
    new_data = []
    for i in range(len(data)):
        new_data.append([data[i][e] for e in selected_feature_inx])

    return new_data, selected_feature_inx


def init_vote_vector(data, label, feature_num, fs_option = 'RFLV', round = 5, partition_type = 'vertical'):

    parts_num = 3
    feature_vote = [0] * feature_num
    for i in range(round):
        data_parts, label_parts, feature_parts_inx = split_data(data, label, feature_num,parts_num, partition_type) #return a list containing various parts
        for j in range(len(data_parts)):
            part_data = data_parts[j]
            part_label = label_parts[j]
            part_feature_inx = feature_parts_inx[j]

            selected_data, selected_fs_inx = feature_method_selection(part_data,part_label,fs_option)
            part_selected_feature_inx = [part_feature_inx[k] for k in selected_fs_inx]

            for k in part_feature_inx:
                if(k not in part_selected_feature_inx):
                    feature_vote[k] = feature_vote[k] + 1

    return feature_vote


def feature_method_selection(data, label, fsname):
    """
    select features by option 'fsname'
    :param data:
    :param label:
    :param fsname:
    :return: new_data, selected data
    :return: selected_features_inx, the index of selected feature, starts with 0
    """
    if fsname == 'variance_threshold': #变化不大就舍弃，离散值
        model = VarianceThreshold() #th=1
        return model.fit_transform(data)

    elif fsname == 'select_kbest':
        model = SelectKBest(chi2, k=10) #特征值必须非负，chi2是分类

    elif fsname == 'rfe':#递归消除,耗时很长
        svc = SVC(kernel='linear', C=1)
        model = RFE(estimator=svc, n_features_to_select=10, step=1)

    elif fsname == 'rfecv': #交叉验证执行执行REF,label必须是数值
        svc = SVC(kernel="linear")
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(label, 1),
                      scoring='accuracy')

    elif fsname == 'RandLasso':#打乱重新选择，cannot perform reduce with flexible type
        model = RandomizedLogisticRegression()


    elif fsname == 'linear_svc':
        model = LinearSVC() #没有importance

    elif fsname == 'tree':
        model = ExtraTreesClassifier()

    elif fsname == 'fclassif':
        model = SelectFpr() #默认是f_classif，值越大，特征越有用

    elif fsname == 'pearsonr': #label必须是数值
        label = turn_label_2num(label)#结果是两个sample的相关性
        res = pearsonr(data,label)

    elif fsname == 'RandForReg': #label必须是数值
        label = turn_label_2num(label)
        model = RandomForestRegressor()

    else:
        logging.error('ERROR: feature selection option is wrong')

    model.fit(data, label)
    new_data = model.transform(data)  # selected importanted data

    return new_data

def split_data(data, label, feature_num, parts_num, partition_type):
    """

    :param data:
    :param label:
    :param feature_num: the number of the features
    :param parts_num: the number of paritions
    :param partition_type: horizontal or vertical
    :return: data_parts, list of various data list
    :return:label_parts, list of various label list
    :return:feature_parts_inx, list of various feature index in previous features vector

    """

    if partition_type == 'horizontal':
        each_data_num = round(len(data) / parts_num)
        data_parts = [data[i:i+parts_num] for i in range(0, len(data), parts_num)]
        label_parts = label * parts_num
        feature_parts_inx = range(feature_num) * parts_num

    elif partition_type == 'vertical':
        each_feature_num = int(math.ceil((feature_num / parts_num)))
        rand_inx = range(feature_num)
        random.shuffle(rand_inx)

        data_parts = []
        feature_parts_inx = []
        for i in range(0, parts_num):
            data_temp = [data[:,rand_inx[j + i * each_feature_num]] for j in range(each_feature_num) if (j + i * each_feature_num) < feature_num]
            data_temp = list(np.array(data_temp).T)
            data_parts.append(data_temp)

            feature_temp = [rand_inx[j + i * each_feature_num] for j in range(each_feature_num) if (j + i * each_feature_num) < feature_num]
            feature_parts_inx.append(feature_temp)

        label_parts = [list(label) for i in range(parts_num)]

    else:
        logging.error('ERROR: partition_type is wrong')

    return data_parts, label_parts, feature_parts_inx

def get_threhold(data, label, feature_num, feature_votes):
    """

    :param: data containing whole features
    :param: corresponding label
    :param: number of features
    :param: weights of features
    :return: th,threhold spliting features
    """
    e = []
    alfa = 0.5
    min_vote = int(round(np.min(feature_votes) + np.var(feature_votes)/2 ))
    max_vote = int(round(np.max(feature_votes) - np.var(feature_votes)/2 ))
    step = int(math.floor((max_vote - min_vote) / 3))
    if step == 0:
        step = 1

    for v in range(min_vote, max_vote, step):
        feat_feature_index = [i for i in range(len(feature_votes)) if feature_votes[i] < v ]
        feat_feature_num = len(feat_feature_index)
        feat_percentage = float(feat_feature_num) / float(len(feature_votes)) * 100

        selected_feature_index = [i for i in range(feature_num) if i not in feat_feature_index]
        DC = get_DC(data, label, selected_feature_index, 'F1')
        logging.info('feat_percentage is %, DC value is %f' %(feat_percentage,DC))
        e.append(alfa * DC + (1 - alfa) * feat_percentage)

    th = min(e)

    return th



def get_DC(data, label, selected_feature_inx, dc_option):
    new_data = []
    for i in range(len(data)):
        new_data.append([data[i][e] for e in selected_feature_inx])

    group1, group2 = greedy_search(new_data, label, dc_option)
    group1_data, group1_label = get_data_subset(data, label, group1)
    group2_data, group2_label = get_data_subset(data, label, group2)

    if dc_option == 'F1':
        DC = get_complexity_F1(group1_data,group1_label,group2_data,group2_label)
    elif dc_option == 'F2':
        DC = get_complexity_F2(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'F3':
        DC = get_complexity_F3(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'N2':
        DC = get_complexity_N2(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'N3':
        DC = get_complexity_N3(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'N4':
        DC = get_complexity_N4(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'L3':
        DC = get_complexity_L3(group1_data, group1_label, group2_data, group2_label)
    elif dc_option == 'Cluster':
        DC = get_complexity_Cluster(group1_data, group1_label, group2_data, group2_label)
    else:
        logging.error('ERROR: dc option is wrong!')

    return DC


def select_data_by_feature_index(data, feature_index):
    new_data = []
    for each in data:
        new_data.append([each[i] for i in feature_index])

    return new_data


def FS_selection(train_data,train_label,test_data,test_label,fsname,**param):
    data = list(train_data) + list(test_data)
    label = list(train_label) + list(test_label)

    data = feature_method_selection(data,label,fsname)

    if 'num' in param:
        new_data = None
        for row in data:
            if new_data is None:
                new_data = row[0:param['num']]
            else:
                new_data = np.row_stack((new_data,row[0:param['num']]))
        data = new_data

    train_data = data[0:len(train_data)]
    test_data = data[len(train_data):]

    return train_data,train_label,test_data,test_label



