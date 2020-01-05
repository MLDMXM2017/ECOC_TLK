# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/29 18:33
# file: Evaluation_tool.py
# description: this module provides some methods to evaluate the model

import numpy as np


class Evaluation:
    def __init__(self, true_labels, pre_labels):
        self.true_labels = true_labels
        self.pre_labels = pre_labels
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []
        self.each_accuracy = []
        self.each_senitivity = []
        self.each_specifity = []
        self.each_precision = []
        self.each_Fscore = []
        self.accuracy = 0
        self.senitivity = 0
        self.specifity = 0
        self.precision = 0
        self.Fscore = 0

        if len(np.unique(true_labels)) >= len(np.unique(pre_labels)):
            self.ulabel = list(np.unique(true_labels))
        else:
            self.ulabel = list(np.unique(pre_labels))

    def evaluation(self, **key):
        self.__get_PN_values()
        if self.__check_data() == False:
            return

        operation_name = {'simple_acc': self.evaluate_arruracy_simple, 'accuracy': self.evaluate_arruracy \
            , 'sensitivity': self.evaluate_sensitivity, 'specifity': self.evaluate_specifity
            , 'precision': self.evaluate_precision, 'Fscore': self.evaluate_Fscore}
        res = {}
        for i, each in enumerate(key['option']):
            res[each] = operation_name[each]()
        return res

    def __thansform_labels(self, cls, labels):

        temp_labels = np.zeros(len(labels))
        for i, key in enumerate(labels):
            if key == cls:
                temp_labels[i] = 1
            else:
                temp_labels[i] = -1

        return temp_labels

    def __get_PN_values(self):
        """
         this function compute FP etc. by onevsone strategy
        :return: TP, TN, FP, FN
        """
        true_labels = self.true_labels
        pre_labels = self.pre_labels
        ulabel = self.ulabel

        for each in ulabel:
            temp_true_lables = self.__thansform_labels(each, true_labels)  # [+1,-1]
            temp_pre_lables = self.__thansform_labels(each, pre_labels)  # [+1,-1]

            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(len(true_labels)):
                if temp_true_lables[i] == 1 and temp_pre_lables[i] == 1:
                    TP = TP + 1
                elif temp_true_lables[i] == -1 and temp_pre_lables[i] == -1:
                    TN = TN + 1
                elif temp_true_lables[i] == -1 and temp_pre_lables[i] == 1:
                    FP = FP + 1
                else:
                    FN = FN + 1

            self.TP.append(TP)
            self.FP.append(FP)
            self.TN.append(TN)
            self.FN.append(FN)

    def __check_data(self):
        """
        the samples labels and prediced labels are checked with size
        :param true_labels: the labels for test samples
        :param pre_labels: the predicted labels for test samples
        :return:True or False
        """
        true_labels = self.true_labels
        pre_labels = self.pre_labels

        if len(true_labels) == 0 or len(pre_labels) == 0:
            return False
        if len(true_labels) != len(pre_labels):
            return False

        return True

    def evaluate_arruracy(self):
        ulabel = self.ulabel
        accuracy = []
        for i, each in enumerate(ulabel):
            P = list(self.true_labels).count(each)
            N = len(self.true_labels) - P
            accuracy.append((self.TP[i] + self.TN[i]) / float(P + N))

        self.each_accuracy = accuracy
        return np.mean(accuracy)

    def evaluate_sensitivity(self):
        """

        :return:sensitivity, TP/P
        """
        ulabel = self.ulabel
        sensitivity = []
        for i, each in enumerate(ulabel):
            P = list(self.true_labels).count(each)
            sensitivity.append(self.TP[i] / float(P))

        self.each_senitivity = sensitivity
        return np.mean(sensitivity)

    def evaluate_specifity(self):
        """

        :return: specifity, TN/N
        """
        ulabel = self.ulabel
        specifity = []
        for i, each in enumerate(ulabel):
            P = list(self.true_labels).count(each)
            N = len(self.true_labels) - float(P)
            specifity.append(self.TN[i] / N)

        self.each_specifity = specifity
        return np.mean(specifity)

    def evaluate_precision(self):
        """

        :return: precision, TP/(TP+FP)
        """
        ulabel = self.ulabel
        precision = []
        for i, each in enumerate(ulabel):
            if (float(self.TP[i] + self.FP[i])) == 0:
                precision.append(0)
            else:
                precision.append(self.TP[i] / float(self.TP[i] + self.FP[i]))

        self.each_precision = precision
        return np.mean(precision)

    def evaluate_Fscore(self):
        """

        :return: precision, 2*precision*sensitivity/precision+sensitivity
        """
        ulabel = self.ulabel
        Fscore = []
        for i, each in enumerate(ulabel):
            P = list(self.true_labels).count(each)
            N = len(self.true_labels) - P
            sensitivity = self.TP[i] / float(P)
            if self.TP[i] + self.FP[i] == 0:
                precision = 0
            else:
                precision = self.TP[i] / float(self.TP[i] + self.FP[i])

            if sensitivity + precision != 0:
                Fscore.append((2.0 * precision * sensitivity) / float(precision + sensitivity))
            else:
                Fscore.append(0)
        return np.mean(Fscore)

    def evaluate_arruracy_simple(self):
        true_labels = self.true_labels
        pre_labels = self.pre_labels

        res = []
        for i, each in enumerate(true_labels):
            if true_labels[i] == pre_labels[i]:
                res.append(1)
        res = float(len(res)) / len(true_labels)

        return res

    def __cal_classifier_diversity(self, predicted_res1, predicted_res2):
        dis = 0
        for i in range(len(predicted_res1)):
            if predicted_res1[i] != predicted_res2[i]:
                dis = dis + 1
        diversity1 = dis / float(len(predicted_res1))

        dis = 0
        predicted_res1 = -np.array(predicted_res1)
        predicted_res2 = -np.array(predicted_res2)
        for i in range(predicted_res1.shape[0]):
            if predicted_res1[i] != predicted_res2[i]:
                dis = dis + 1
        diversity2 = dis / float(predicted_res1.shape[0])

        if diversity1 == diversity2:
            return diversity1
        else:
            return np.min(diversity1, diversity2)

    # Kappa statistic
    # diversity
    def evaluate_diversity(self, predicted_vector):
        diversity_matrix = np.zeros((len(predicted_vector[0]), len(predicted_vector[0])))
        for i in range(len(predicted_vector[0])):
            i_column = [row[i] for row in predicted_vector]
            for j in range(i + 1, len(predicted_vector[0])):
                j_column = [row[j] for row in predicted_vector]

                diversity_matrix[i][j] = self.__cal_classifier_diversity(i_column, j_column)
                diversity_matrix[j][i] = diversity_matrix[i][j]
                # logging.info('i:%d j:%d diversity:%f' % (i, j,diversity_matrix[i][j]))
        mean_res = [round(np.mean(row), 3) for row in diversity_matrix]
        return mean_res

    def evaluate_PD(self, m1, m2):
        total_dis = []
        for i in range(m1.shape[1]):
            dis = []
            for j in range(m2.shape[2]):
                c1 = m1[:, i]
                c2 = m2[:, j]
                dis.append(np.sum([1 for k in range(len(c1)) if c1[k] != c2[k]]))
            total_dis.append(min(dis))

        return np.mean(total_dis)

    def evaluate_classifier_accuracy(self, matrix, predicted_vector, true_label):
        class_index = np.unique(true_label)
        accuracy = []
        for i in range(matrix.shape[1]):
            pre_label = [row[i] for row in predicted_vector]

            # find class1 and class2
            c1 = matrix[:, i]
            class1 = []
            class2 = []
            for j in range(len(c1)):
                if c1[j] == 1:
                    class1.append(class_index[j])
                elif c1[j] == -1:
                    class2.append(class_index[j])

            # constructed 1,-1 label vector
            temp_label = []  # construted ture label
            temp_sample_inx = []
            for i, each in enumerate(true_label):
                if each in class1:
                    temp_label.append(1)
                    temp_sample_inx.append(i)
                elif each in class2:
                    temp_label.append(-1)
                    temp_sample_inx.append(i)

            # constructed predicted vector
            temp_predicted_label = []  # part predicted label
            for i in temp_sample_inx:
                temp_predicted_label.append(pre_label[i])

            right_num = np.sum([1 for i in range(len(temp_label)) if temp_predicted_label[i] == temp_label[i]])
            acc = right_num / float(len(temp_label))
            accuracy.append(round(acc, 3))

        return accuracy
