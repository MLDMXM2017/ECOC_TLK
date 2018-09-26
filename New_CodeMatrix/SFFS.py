import numpy as np
import Criterion
import Matrix_Toolkit
import copy


def sffs_divide(data, label, judge_score=Criterion.divide_score_coverage, **param):
    unique_label = np.unique(label)
    unique_label_i = []
    unique_label_j = unique_label[:]
    unique_label_i_res = []
    unique_label_j_res = []
    while(list(unique_label_i) != list(unique_label_i_res)
          or list(unique_label_j) != list(unique_label_j_res)):
        unique_label_i_res = copy.deepcopy(unique_label_i)
        unique_label_j_res = copy.deepcopy(unique_label_j)
        unique_label_i, unique_label_j = invovle_process(data, label, unique_label_i, unique_label_j, judge_score)
        unique_label_i, unique_label_j = exclude_process(data, label, unique_label_i, unique_label_j, judge_score)
    return unique_label_i, unique_label_j

def sffs_agg(data, label, judge_score=Criterion.divide_score, **param):
    pass

def invovle_process(data, label, unique_label_i, unique_label_j, judge_score=Criterion.divide_score):
    unique_label_i = list(unique_label_i)
    unique_label_j = list(unique_label_j)
    if len(unique_label_j)==1:
        return unique_label_i, unique_label_j
    data_i, label_i = Matrix_Toolkit.get_data_by_label(data, label, unique_label_i)
    data_j, label_j = Matrix_Toolkit.get_data_by_label(data, label, unique_label_j)
    if unique_label_i == []:
        pre_score = -np.inf
        pre_unique_label_i = None
        pre_unique_label_j = None
    else:
        pre_score = judge_score(data_i, label_i, data_j, label_j)
        pre_unique_label_i = copy.deepcopy(unique_label_i)
        pre_unique_label_j = copy.deepcopy(unique_label_j)
    while(True):
        # print('new_loop...')
        # print(unique_label_j)
        highest_score = -np.inf
        unique_label_i_temp = None
        unique_label_j_temp = None
        for lab in unique_label_j:
            label_i_temp = list(unique_label_i)
            label_i_temp.insert(0, lab)
            label_j_temp = list(unique_label_j)
            label_j_temp.remove(lab)
            data_i_temp, label_i_temp = Matrix_Toolkit.get_data_by_label(data, label, label_i_temp)
            data_j_temp, label_j_temp = Matrix_Toolkit.get_data_by_label(data, label, label_j_temp)
            score = judge_score(data_i_temp, label_i_temp, data_j_temp, label_j_temp)
            if score > highest_score:
                highest_score = score
                unique_label_i_temp = np.unique(label_i_temp)
                unique_label_j_temp = np.unique(label_j_temp)
        # print(highest_score)
        unique_label_i = copy.deepcopy(unique_label_i_temp)
        unique_label_j = copy.deepcopy(unique_label_j_temp)
        if highest_score < pre_score:
            # print('pre score:', pre_score)
            # print('return pre:',pre_unique_label_i)
            return pre_unique_label_i, pre_unique_label_j
        if len(unique_label_j) == 1:
            # print('highest score:', highest_score)
            # print('return:',unique_label_i)
            return unique_label_i, unique_label_j

        pre_score = highest_score
        pre_unique_label_i = copy.deepcopy(unique_label_i)
        pre_unique_label_j = copy.deepcopy(unique_label_j)

def exclude_process(data, label, unique_label_i, unique_label_j, judge_score=Criterion.divide_score):
    unique_label_j, unique_label_i = invovle_process(data, label, unique_label_j, unique_label_i, judge_score)
    return unique_label_i, unique_label_j





