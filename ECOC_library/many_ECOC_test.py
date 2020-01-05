# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/2/28 10:05
# file: many_ECOC_test.py
# description:

import time

from ECOC_library.Common.Evaluation_tool import Evaluation
from ECOC_library.FS.DC_Feature_selection import *
from ECOC_library.Common import Read_Write_tool


def ECOC_Process(train_data, train_label, test_data, test_label, ECOC_name, **param):
    E = None
    if ECOC_name.find('DC_ECOC') >= 0:
        dc_option = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'Cluster']
        for i, each in enumerate(dc_option):
            if each in ECOC_name:
                E = eval('DC_ECOC()')
                E.fit(train_data, train_label, dc_option=each)
                break

    elif ECOC_name.find('Self_Adaption_ECOC') >= 0:
        ternary_option = []
        Ternary = ['+', '-', '*', '/', 'and', 'or', 'info', 'DC']
        for i, each in enumerate(Ternary):
            if each in ECOC_name:
                param['ternary_option'] = each
                break

        if 'base_M' in param:
            E = eval('Self_Adaption_ECOC()')
            E.fit(train_data, train_label, **param)

        else:
            dc_option = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'Cluster']
            dc_option = []
            for each in enumerate(dc_option):
                if each in ECOC_name:
                    dc_option.append([each])
            param['dc_option'] = dc_option

            E = eval('Self_Adaption_ECOC()')
            E.fit(train_data, train_label, **param)

    else:
        E = eval(ECOC_name + '(base_estimator=RandomForestClassifier)')

        E.fit(train_data, train_label)

    logging.info(ECOC_name + ' Matrix:\n' + str(E.matrix))
    predicted_label = E.predict(test_data)

    evaluation_option = ['simple_acc', 'accuracy', 'sensitivity', 'specifity', 'precision', 'Fscore']
    Eva = Evaluation(test_label, predicted_label)
    res = Eva.evaluation(option=evaluation_option)
    res['classifier_num'] = len(E.matrix[0])
    res['cls_acc'] = Eva.evaluate_classifier_accuracy(E.matrix, E.predicted_vector, test_label)
    res['diversity'] = Eva.evaluate_diversity(E.predicted_vector)
    return res


def get_base_M(path, ecoc, dataname):
    dc_option = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'Cluster']

    M = None
    ecoc_parts = ecoc.split(' ')
    for each in ecoc_parts:
        if each in dc_option:
            filepath = path + str(each) + '_' + str(dataname) + '.xls'
            if M == None:
                M = [Read_Write_tool.read_matirx(filepath)]
            else:
                M.append(Read_Write_tool.read_matirx(filepath))

    return M


if __name__ == '__main__':
    # current_time log_level function_name user_print_info
    LOG_FORMAT = "%(message)s"

    # set log filepath, log level and info format
    logging.basicConfig(filename='RandForest_OtherECOC_log.txt', level=logging.DEBUG, format=LOG_FORMAT)

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'fclassif', 'RandForReg', 'linearsvc_tree']

    microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2' \
        , 'Lung1', 'SRBCT']

    UCI_dataname = ['car', 'cleveland', 'dermatology', 'led7digit' \
        , 'led24digit', 'letter', 'nursery', 'penbased', 'satimage', 'segment' \
        , 'shuttle', 'vehicle', 'vowel', 'yeast', 'zoo']

    ecoc_name = ['Self_Adaption_ECOC F1 F2 +', 'Self_Adaption_ECOC F1 F2 -' \
        , 'Self_Adaption_ECOC F1 F2 *', 'Self_Adaption_ECOC F1 F2 /' \
        , 'Self_Adaption_ECOC F1 F2 and', 'Self_Adaption_ECOC F1 F2 or', 'Self_Adaption_ECOC F1 F2 info']

    ECOC_conbination = ['Self_Adaption_ECOC F1 F1 DC', 'Self_Adaption_ECOC F2 F2 DC', 'Self_Adaption_ECOC F3 F3 DC' \
        , 'Self_Adaption_ECOC N2 N2 DC', 'Self_Adaption_ECOC N3 N3 DC', 'Self_Adaption_ECOC Cluster Cluster DC' \
        , 'Self_Adaption_ECOC F1 F2 DC', 'Self_Adaption_ECOC F1 F3 DC', 'Self_Adaption_ECOC F1 N2 DC' \
        , 'Self_Adaption_ECOC F1 N3 DC', 'Self_Adaption_ECOC F1 Cluster DC', 'Self_Adaption_ECOC F2 F3 DC' \
        , 'Self_Adaption_ECOC F2 N2 DC', 'Self_Adaption_ECOC F2 N3 DC', 'Self_Adaption_ECOC F2 Cluster DC' \
        , 'Self_Adaption_ECOC F3 N2 DC', 'Self_Adaption_ECOC F3 N3 DC', 'Self_Adaption_ECOC F3 Cluster DC' \
        , 'Self_Adaption_ECOC N2 N3 DC', 'Self_Adaption_ECOC N2 Cluster DC', 'Self_Adaption_ECOC N3 Cluster DC' \
        , 'Self_Adaption_ECOC F1 F2 F3 DC', 'Self_Adaption_ECOC F1 F2 N2 DC', 'Self_Adaption_ECOC F1 F2 N3 DC' \
        , 'Self_Adaption_ECOC F1 F2 Cluster DC', 'Self_Adaption_ECOC F2 F3 N2 DC', 'Self_Adaption_ECOC F2 F3 N3 DC' \
        , 'Self_Adaption_ECOC F2 F3 Cluster DC', 'Self_Adaption_ECOC F3 N2 N3 DC', 'Self_Adaption_ECOC F3 N2 Cluster DC' \
        , 'Self_Adaption_ECOC F3 N3 Cluster DC']

    ECOC_coding_numbers = ['Self_Adaption_ECOC F1 F1', 'Self_Adaption_ECOC F1 F1 F1' \
        , 'Self_Adaption_ECOC F1 F1 F1 F1', 'Self_Adaption_ECOC F1 F1 F1 F1 F1']

    other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC' \
        , 'D_ECOC', 'DC_ECOC F1', 'DC_ECOC F2', 'DC_ECOC F3' \
        , 'DC_ECOC N2', 'DC_ECOC N3', 'DC_ECOC Cluster']

    data_folder_path = 'E:/workspace1/ECOCDemo/Microarray_data/FS_data/'
    matrix_folder_path = 'E:/workspace1/ECOCDemo/Microarray_res/DC_matrix/'
    res_folder_path = 'E:/workspace1/ECOCDemo/Microarray_res/RandForest/other_ECOC_backup/'

    selected_dataname = microarray_dataname
    selected_ecoc_name = other_ECOC
    selected_fs_name = fs_name

    for k in range(len(selected_fs_name)):

        if selected_fs_name[k] != 'RandForReg':
            continue

        # save evaluation varibles
        data_acc = []
        data_simacc = []
        data_precision = []
        data_specifity = []
        data_sensitivity = []
        data_cls_acc = []
        data_Fscore = []

        for i in range(len(selected_dataname)):

            train_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_train.csv'
            test_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_test.csv'
            train_data, train_label = Read_Write_tool.read_Microarray_Dataset(train_path)
            test_data, test_label = Read_Write_tool.read_Microarray_Dataset(test_path)

            acc = []
            simacc = []
            precision = []
            specifity = []
            sensitivity = []
            cls_acc = []
            Fscore = []

            for j in range(len(selected_ecoc_name)):

                print('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                print('FS: ' + selected_fs_name[k])
                print('Dataset： ' + selected_dataname[i])
                print('ECOC: ' + selected_ecoc_name[j])

                logging.info('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                logging.info('FS: ' + selected_fs_name[k])
                logging.info('Dataset： ' + selected_dataname[i])
                logging.info('ECOC: ' + selected_ecoc_name[j])

                param = {}
                if selected_ecoc_name[j].find('Self_Adaption_ECOC') >= 0:
                    final_folder_matrix_path = matrix_folder_path + selected_fs_name[k] + '/'
                    param['base_M'] = get_base_M(final_folder_matrix_path, selected_ecoc_name[j], selected_dataname[i])

                res = ECOC_Process(train_data, train_label, test_data, test_label, selected_ecoc_name[j], **param)
                if 'simple_acc' in res:
                    simacc.append(res['simple_acc'])
                if 'accuracy' in res:
                    acc.append(res['accuracy'])
                if 'sensitivity' in res:
                    sensitivity.append(res['sensitivity'])
                if 'specifity' in res:
                    specifity.append(res['specifity'])
                if 'precision' in res:
                    precision.append(res['precision'])
                if 'Fscore' in res:
                    Fscore.append(res['Fscore'])

                if 'diversity' in res:
                    txtname = res_folder_path + 'diversity_' + selected_fs_name[k] + '.txt'
                    content = 'Data: ' + selected_dataname[i] + '\t' + ' ECOC: ' + selected_ecoc_name[j] \
                              + '\t ' + str(res['diversity'])
                    Read_Write_tool.write_txt(txtname, content)

                if 'cls_acc' in res:
                    txtname = res_folder_path + 'cls_acc_' + selected_fs_name[k] + '.txt'
                    content = 'Data: ' + selected_dataname[i] + '\t' + ' ECOC: ' + selected_ecoc_name[j] \
                              + '\t ' + str(res['cls_acc'])
                    Read_Write_tool.write_txt(txtname, content)

                if 'classifier_num' in res:
                    txtname = res_folder_path + 'cls_num_' + selected_fs_name[k] + '.txt'
                    content = 'Data: ' + selected_dataname[i] + '\t' + ' ECOC: ' + selected_ecoc_name[j] \
                              + '\t ' + str(res['classifier_num'])
                    Read_Write_tool.write_txt(txtname, content)

            data_simacc.append(simacc)
            data_acc.append(acc)
            data_precision.append(precision)
            data_specifity.append(specifity)
            data_Fscore.append(Fscore)
            data_sensitivity.append(sensitivity)

        row_name = copy.deepcopy(selected_dataname)
        row_name.append('Avg')
        if np.all(data_simacc):
            save_filepath = res_folder_path + 'simple_acc_' + selected_fs_name[k] + '.xls'
            data_simacc.append(np.mean(data_simacc, axis=0))
            Read_Write_tool.write_file(save_filepath, data_simacc, selected_ecoc_name, row_name)

        if np.all(data_acc):
            save_filepath = res_folder_path + 'accuracy_' + selected_fs_name[k] + '.xls'
            data_acc.append(np.mean(data_acc, axis=0))
            Read_Write_tool.write_file(save_filepath, data_acc, selected_ecoc_name, row_name)

        if np.all(data_sensitivity):
            save_filepath = res_folder_path + 'sensitivity_' + selected_fs_name[k] + '.xls'
            data_sensitivity.append(np.mean(data_sensitivity, axis=0))
            Read_Write_tool.write_file(save_filepath, data_sensitivity, selected_ecoc_name, row_name)

        if np.all(data_specifity):
            save_filepath = res_folder_path + 'specifity_' + selected_fs_name[k] + '.xls'
            data_specifity.append(np.mean(data_specifity, axis=0))
            Read_Write_tool.write_file(save_filepath, data_specifity, selected_ecoc_name, row_name)

        if np.all(data_precision):
            save_filepath = res_folder_path + 'precision_' + selected_fs_name[k] + '.xls'
            data_precision.append(np.mean(data_precision, axis=0))
            Read_Write_tool.write_file(save_filepath, data_precision, selected_ecoc_name, row_name)

        if np.all(data_Fscore):
            save_filepath = res_folder_path + 'Fscore_' + selected_fs_name[k] + '.xls'
            data_Fscore.append(np.mean(data_Fscore, axis=0))
            Read_Write_tool.write_file(save_filepath, data_Fscore, selected_ecoc_name, row_name)
