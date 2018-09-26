import numpy as np
import copy
from sklearn.svm import SVC,SVR
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
from itertools import combinations
from scipy.special import comb
import Distance_Toolkit
import Matrix_Toolkit
import SFFS
import Criterion
import CrossValidaion_Toolkit


class BaseECOC:
    def __init__(self, base_estimator=SVC, distance_measure=Distance_Toolkit.euclidean_distance, **estimator_params):
        self.index = []
        self.estimators = []
        self.matrix = None
        self.base_estimator = base_estimator
        self.distance_measure = distance_measure
        self.estimator_params = estimator_params

    def create_matrix(self, data, label):
        return np.array([]), []

    def train_col(self, data, label, col):
        train_data, train_label = Matrix_Toolkit.get_data_from_col(data, label, col, self.index)
        estimator = self.base_estimator(**self.estimator_params).fit(train_data, train_label)
        return estimator

    def train_matrix(self, data, label):
        self.matrix, self.index = self.create_matrix(data, label)
        self.matrix = self.check_matrix()
        self.fillzero(data, label)
        self.estimators = []
        for i in range(self.matrix.shape[1]):
            self.estimators.append(self.train_col(data, label, self.matrix[:, i]))
        return self.matrix, self.estimators

    def fit(self, data, label):
        self.train_matrix(data, label)

    def predict(self, data):
        if not self.estimators:
            raise ValueError('This Model has not been trained!')
        vectors = self.predict_vector(data)
        labels = self.vectors_to_labels(vectors)
        return labels

    def predict_vector(self, data):
        vectors = None
        for estimator in self.estimators:
            if vectors is None:
                vectors = estimator.predict(data).reshape((-1, 1))
            else:
                vectors = np.hstack((vectors, estimator.predict(data).reshape((-1, 1))))
        return vectors

    def vectors_to_labels(self, vectors, weights=None):
        labels = []
        for vector in vectors:
            distance = np.inf
            label = self.index[0]
            for matrix_index in range(len(self.matrix)):
                matrix_row = self.matrix[matrix_index, :]
                distance_temp = self.distance_measure(vector, matrix_row, weights)
                if distance_temp < distance:
                    distance = distance_temp
                    label = self.index[matrix_index]
            labels.append(label)
        return labels

    def check_matrix(self):
        self.matrix, index = np.unique(self.matrix, axis=1, return_index=True)
        # self.estimators = [self.estimators[i] for i in index]
        index_to_delete = []
        for i in range(self.matrix.shape[1]):
            if not Matrix_Toolkit.exist_two_class_for_col(self.matrix[:, i]):
                index_to_delete.append(i)
        self.matrix = np.delete(self.matrix, index_to_delete, axis=1)
        # self.estimators = [self.estimators[i] for i in range(len(self.estimators)) if i not in index_to_delete]
        return self.matrix

    def check_column(self, column):
        column = column.reshape([1,-1])
        sign_column = np.sign(column)
        if (1 not in sign_column) or (-1 not in sign_column):
            return False
        if self.matrix is None:
            return True
        sign_matrix = np.sign(self.matrix)
        for i in range(sign_matrix.shape[1]):
            if np.all([sign_column == sign_matrix[:, i]]) or \
                    np.all([sign_column == sign_matrix[:, i]]):
                return False
        return True

    def fillzero(self, data, label):
        pass

class OVA_ECOC(BaseECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix = np.eye(len(index)) * 2 - 1
        return matrix, index

class OVO_ECOC(BaseECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        groups = combinations(range(len(index)), 2)
        matrix_row = len(index)
        matrix_col = np.int(comb(len(index), 2))
        col_count = 0
        matrix = np.zeros((matrix_row, matrix_col))
        for group in groups:
            class_1_index = group[0]
            class_2_index = group[1]
            matrix[class_1_index, col_count] = 1
            matrix[class_2_index, col_count] = -1
            col_count +=1
        return matrix, index

class Dense_random_ECOC(BaseECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix_row = len(index)
        if np.power(2, matrix_row) > np.int(np.floor(10 * np.log2(matrix_row))):
            matrix_col = np.int(np.floor(10 * np.log2(matrix_row)))
        else:
            matrix_col = np.power(2, matrix_row)
        matrix = np.random.random((matrix_row, matrix_col))
        class_1_index = matrix > 0.5
        class_2_index = matrix <=0.5
        matrix[class_1_index] = 1
        matrix[class_2_index] = -1
        return matrix, index

class Sparse_random_ECOC(BaseECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix_row = len(index)
        if np.power(3, matrix_row) > np.int(np.floor(15 * np.log2(matrix_row))):
            matrix_col = np.int(np.floor(15 * np.log2(matrix_row)))
        else:
            matrix_col = np.power(3, matrix_row)
        matrix = np.random.random((matrix_row,matrix_col))
        class_0_index = np.logical_and(0.25<=matrix, matrix<0.75)
        class_1_index = matrix >= 0.75
        class_2_index = matrix < 0.25
        matrix[class_0_index] = 0
        matrix[class_1_index] = 1
        matrix[class_2_index] = -1
        return matrix, index

class H_ECOC(BaseECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix = None
        labels_to_divide = [list(index)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            data_subset, label_subset = Matrix_Toolkit.get_data_by_label(data, label, label_set)
            class_1_unique_label, class_2_unique_label = SFFS.sffs_divide(data_subset, label_subset)
            new_col = np.zeros((len(index), 1))
            for i in class_1_unique_label:
                new_col[index.index(i)] = 1
            for i in class_2_unique_label:
                new_col[index.index(i)] = -1
            if matrix is None:
                matrix = copy.deepcopy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1_unique_label) > 1:
                labels_to_divide.append(class_1_unique_label)
            if len(class_2_unique_label) > 1:
                labels_to_divide.append(class_2_unique_label)
        return matrix, index
		
class DataBasedECOC(BaseECOC):
    def __init__(self, base_estimator=SVR, distance_measure=Distance_Toolkit.euclidean_distance,
                 coverage='normal', **estimator_params):
        BaseECOC.__init__(self, base_estimator, distance_measure, **estimator_params)
        self.coverage=coverage
        if coverage == 'normal':
            self.cover_rate = Matrix_Toolkit.coverage_rate
        elif coverage == 'distance':
            self.cover_rate = Matrix_Toolkit.distance_coverage_rate
    def check(self, data, label):
        check_matrix = np.zeros(self.matrix.shape)
        weights = np.zeros((1, self.matrix.shape[1]))
        for i in range(self.matrix.shape[1]):
            col = self.matrix[:, i]
            new_label = [col[n] for lab in label for n in range(len(self.index)) if lab == self.index[n]]
            pred_label = self.estimators[i].predict(data)
            mse = mean_squared_error(new_label, pred_label)
            weights[0, i] = np.exp(-mse)
            for k in range(len(self.index)):
                data_temp = np.array([data[j, :] for j in range(data.shape[0]) if label[j] == self.index[k]])
                check_matrix[k, i] = np.mean(self.estimators[i].predict(data_temp))
        # print('before:\n',self.matrix)
        # self.matrix = (self.matrix + check_matrix * weights) / (np.ones(self.matrix.shape) + weights)
        self.matrix = (self.matrix + check_matrix * (weights)) / (np.ones(self.matrix.shape) + (weights))
        # print('after:\n',self.matrix)

    def fit(self, data, label):
        data, label = Matrix_Toolkit.duplicate_single_class(data, label)
        self.train_matrix(data, label)
        self.check(data, label)

    def drop_bad_col(self, validation_data, validation_label):
        index_to_move = []
        for i, estimator in enumerate(self.estimators):
            data_temp, label_temp = Matrix_Toolkit.get_data_from_col(validation_data, validation_label,
                                                                     self.matrix[:, i], self.index, hardcode=True)
            pred_label = np.sign(estimator.predict(data_temp))
            if accuracy_score(label_temp, pred_label) < 0.5:
                index_to_move.append(i)
        self.matrix = np.delete(self.matrix, index_to_move, axis=1)
        self.estimators = [self.estimators[i] for i in range(len(self.estimators))
                           if i not in index_to_move]

    def fill_column_zero(self, data, label, column):
        column = copy.deepcopy(column)
        pos_to_fill = [index for index in range(len(column)) if column[index]==0]
        positive_label = [self.index[index] for index in range(len(column)) if column[index] > 0]
        negative_label = [self.index[index] for index in range(len(column)) if column[index] < 0]
        positive_data, positive_label = Matrix_Toolkit.get_data_by_label(data, label, positive_label)
        negative_data, negative_label = Matrix_Toolkit.get_data_by_label(data, label, negative_label)
        positive_center = np.mean(positive_data, axis=0)
        negative_center = np.mean(negative_data, axis=0)
        for i in pos_to_fill:
            target_label = [self.index[i]]
            data_temp, label_temp =Matrix_Toolkit.get_data_by_label(data, label, target_label)
            if self.coverage == 'normal':
                group = [self.distance_measure(data, positive_center) < self.distance_measure(data, negative_center) for data in data_temp]
                positive_num = group.count(True)
                negative_num = group.count(False)
                if positive_num >negative_num:
                    column[i] = positive_num / (positive_num + negative_num)
                else:
                    column[i] = -negative_num / (positive_num + negative_num)
            elif self.coverage == 'distance':
                positive_distance_mean = np.mean([self.distance_measure(data, positive_center) for data in data_temp])
                negative_distance_mean = np.mean([self.distance_measure(data, negative_center) for data in data_temp])
                if positive_distance_mean < negative_distance_mean:
                    column[i] = 2/(1 + np.exp(-positive_distance_mean)) - 1
                else:
                    column[i] = 2/(1 + np.exp(negative_distance_mean)) - 1
        return column
    # def fill_column_zero(self, data, label, column, column_num):
    #     column = copy.deepcopy(column)
    #     pos_to_fill = [index for index in range(len(column)) if column[index]==0]
    #     for i in pos_to_fill:
    #         target_label = [self.index[i]]
    #         data_temp, label_temp =Matrix_Toolkit.get_data_by_label(data, label, target_label)
    #         column[i] = np.mean(self.estimators[column_num].predict(data_temp))
    #     return column
    def fillzero(self, data, label):
        for i in range(self.matrix.shape[1]):
            self.matrix[:, i] = self.fill_column_zero(data, label, self.matrix[:, i])
        return self.matrix

    def train_col(self, data, label, col):
        train_data, train_label = Matrix_Toolkit.get_data_from_col(data, label, col, self.index, hardcode=False)
        estimator = self.base_estimator(**self.estimator_params).fit(train_data, train_label)
        return estimator

class CS_ECOC(DataBasedECOC):
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix = None
        label_to_divide = [index]
        while len(label_to_divide)>0:
            label_set = label_to_divide.pop(0)
            data_temp, label_temp = Matrix_Toolkit.get_data_by_label(data, label, label_set)
            positive_label_set, negative_label_set = SFFS.sffs_divide(data_temp, label_temp)
            positive_data, positive_label = Matrix_Toolkit.get_data_by_label(data, label, positive_label_set)
            negative_data, negative_label = Matrix_Toolkit.get_data_by_label(data, label, negative_label_set)
            rate = self.cover_rate(positive_data, positive_label, negative_data, negative_label)
            new_column = np.zeros((len(index), 1))
            for key in rate:
                new_column[index.index(key)] = rate[key]
            if matrix is None:
                matrix = new_column
            else:
                matrix = np.hstack((matrix, new_column))
            if len(positive_label_set) > 1:
                label_to_divide.insert(0, positive_label_set)
            if len(negative_label_set) > 1:
                label_to_divide.insert(0, negative_label_set)
        return matrix, index

    def add_columns(self, data, label):
        # matrix_sum = np.sum(self.matrix)
        # matrix_mean = matrix_sum / np.sum(self.matrix != 0)
        # row_mean = np.mean(self.matrix, axis=1)
        # mask = row_mean > matrix_mean
        # while len(mask) > 0:
        while True:
            if end_flag:
                break
            matrix_mean = np.mean(self.matrix[self.matrix.nonzero()])
            row_mean = []
            for i in range(self.matrix.shape[0]):
                row = self.matrix[i, :]
                row_mean.append(np.mean(row[row.nonzero()]))
            row_mean = np.array(row_mean)
            row_index = np.array(self.index)[row_mean < matrix_mean]
            label_to_enforce = [self.index[i] for i in range(len(self.index)) if self.index[i] in row_index]
            if len(label_to_enforce) < 2:
                break
            while True:
                train_data_temp, train_label_temp = Matrix_Toolkit.get_data_by_label(data, label,
                                                                                     label_to_enforce)
                positive_label, negative_label = SFFS.sffs_divide(train_data_temp, train_label_temp)
                data_positive, label_positive = Matrix_Toolkit.get_data_by_label(train_data_temp, train_label_temp,
                                                                                 positive_label)
                data_negative, label_negative = Matrix_Toolkit.get_data_by_label(train_data_temp, train_label_temp,
                                                                                 negative_label)
                rate = self.cover_rate(data_positive, label_positive, data_negative, label_negative,
                                       distance_measure=self.distance_measure)
                new_column = np.zeros((self.matrix.shape[0], 1))
                for label in positive_label:
                    new_column[self.index.index(label)] = rate[label]
                for label in negative_label:
                    new_column[self.index.index(label)] = rate[label]
                if not self.check_column(new_column):
                    label_to_enforce.pop()
                    if len(label_to_enforce) < 2:
                        end_flag = True
                        break
                else:
                    new_estimator = self.train_col(data, label, new_column)
                    self.matrix = np.hstack((self.matrix, new_column))
                    self.estimators.append(new_estimator)
                    break

class ECOC_ONE(BaseECOC):
    def __init__(self, base_estimator=SVC, distance_measure=Distance_Toolkit.y_euclidean_distance, t=10, **estimator_params):
        BaseECOC.__init__(self, base_estimator, distance_measure, **estimator_params)
        self.weights = []
        self.iteration = t

    def train_matrix(self, data, label):
        self.index = list(np.unique(label))
        data, label = Matrix_Toolkit.duplicate_single_class(data, label)
        train_data, validation_data, train_label,validation_label = train_test_split(data, label, test_size=0.3, stratify=label)
        label_to_divide = [self.index]
        while len(label_to_divide) > 0:
            label_set = label_to_divide.pop(0)
            groups = combinations(range(len(label_set)), np.int(np.floor(len(label_set)/2)))
            best_score = -np.inf
            best_estimator = None
            best_positive_unique_label = None
            best_negative_unique_label = None
            new_column = None
            for group in groups:
                positive_unique_label = [label_set[i] for i in group]
                negative_unique_label = [label for label in label_set if label not in positive_unique_label]
                new_column = np.zeros((len(self.index), 1))
                for i in positive_unique_label:
                    new_column[self.index.index(i)] = 1
                for i in negative_unique_label:
                    new_column[self.index.index(i)] = -1
                if not self.check_column(new_column):
                    continue
                train_positive_data, train_positive_label = Matrix_Toolkit.get_data_by_label(train_data, train_label, positive_unique_label)
                train_negative_data, train_negative_label = Matrix_Toolkit.get_data_by_label(train_data, train_label, negative_unique_label)
                train_positive_label = np.ones(len(train_positive_label))
                train_negative_label = -np.ones(len(train_negative_label))
                train_data_temp = np.vstack((train_positive_data, train_negative_data))
                train_label_temp = np.hstack((train_positive_label, train_negative_label))
                estimator_temp = self.base_estimator(**self.estimator_params).fit(train_data_temp, train_label_temp)
                validation_positive_data, validation_positive_label = Matrix_Toolkit.get_data_by_label(validation_data, validation_label, positive_unique_label)
                validation_negative_data, validation_negative_label = Matrix_Toolkit.get_data_by_label(validation_data, validation_label, negative_unique_label)
                validation_positive_label = np.ones(len(validation_positive_label))
                validation_negative_label = -np.ones(len(validation_negative_label))
                validation_data_temp = np.vstack((validation_positive_data, validation_negative_data))
                validation_label_temp = np.hstack((validation_positive_label, validation_negative_label))
                score = estimator_temp.score(validation_data_temp, validation_label_temp)
                if score > best_score:
                    best_score = score
                    best_estimator = copy.deepcopy(estimator_temp)
                    best_column = copy.deepcopy(new_column)
                    best_positive_unique_label = copy.deepcopy(positive_unique_label)
                    best_negative_unique_label = copy.deepcopy(negative_unique_label)
            if best_score == -np.inf:
                continue
            if self.matrix is None:
                self.matrix = best_column
            else:
                self.matrix = np.hstack((self.matrix, best_column))
            self.estimators.append(best_estimator)
            self.weights.append(self.get_weight(1-best_score))
            if len(best_positive_unique_label) > 1:
                label_to_divide.append(best_positive_unique_label)
            if len(best_negative_unique_label) > 1:
                label_to_divide.append(best_negative_unique_label)
        return self.matrix, self.estimators, self.weights

    def get_weight(self, error):
        error = error + 0.001
        return 0.5*np.log1p((1-error)/error)

    def train_col(self, data, label, col):
        data, label = Matrix_Toolkit.duplicate_single_class(data, label)
        data, label = Matrix_Toolkit.get_data_from_col(data, label, col, self.index)
        train_data, validation_data, train_label, validation_label = train_test_split(data, label, test_size=0.3, stratify=label)
        estimator = self.base_estimator(**self.estimator_params).fit(train_data, train_label)
        score = estimator.score(validation_data,validation_label)
        weight = self.get_weight(1-score)
        return estimator, weight

    def fit(self, data, label):
        data, label = Matrix_Toolkit.duplicate_single_class(data, label)
        train_data, validation_data, train_label, validation_label = train_test_split(data, label, test_size=0.3, stratify=label)
        self.matrix = None
        self.estimators = []
        self.weights = []
        self.train_matrix(train_data, train_label)
        t=0
        while t < self.iteration:
            node_label = Matrix_Toolkit.get_node_label(self.matrix, self.index)
            pred_label = self.predict(validation_data)
            upper_confusion_matrix = Matrix_Toolkit.upper_triangular_matrix(confusion_matrix(validation_label, pred_label, self.index))
            target_index = np.argmax(upper_confusion_matrix)
            target_row_index = np.int(np.floor(target_index / upper_confusion_matrix.shape[1]))
            target_column_index = target_index % upper_confusion_matrix.shape[1]
            row_label = self.index[target_row_index]
            column_label = self.index[target_column_index]
            node_contain_row_label = [node for node in node_label if row_label in node and column_label not in node]
            node_contain_column_label = [node for node in node_label if column_label in node and row_label not in node]
            original_matrix = copy.deepcopy(self.matrix)
            original_estimators = copy.deepcopy(self.estimators)
            original_weights = copy.deepcopy(self.weights)
            best_accuracy = accuracy_score(validation_label, pred_label)
            best_matrix = None
            best_estimators = None
            best_weights = None
            for positive_node in node_contain_row_label:
                for negative_node in node_contain_column_label:
                    new_node = sorted(positive_node+negative_node)
                    if new_node == sorted(self.index) or new_node in node_label:
                        continue
                    new_column = np.zeros((len(self.index), 1))
                    for label in positive_node:
                        new_column[self.index.index(label)] = 1
                    for label in negative_node:
                        new_column[self.index.index(label)] = -1
                    estimator, weight = self.train_col(train_data, train_label, new_column)
                    self.matrix = np.hstack((original_matrix, new_column))
                    self.estimators = list(original_estimators) + [estimator]
                    self.weights = list(original_weights) + [weight]
                    pred_label = self.predict(validation_data)
                    accuracy = accuracy_score(validation_label, pred_label)
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        best_matrix = copy.deepcopy(self.matrix)
                        best_estimators = copy.deepcopy(self.estimators)
                        best_weights = copy.deepcopy(self.weights)
            if best_matrix is None:
                self.matrix = copy.deepcopy(original_matrix)
                self.estimators = copy.deepcopy(original_estimators)
                self.weights = copy.deepcopy(original_weights)
                break
            else:
                self.matrix = copy.deepcopy(best_matrix)
                self.estimators = copy.deepcopy(best_estimators)
                self.weights = copy.deepcopy(best_weights)
            t = t + 1

    def vectors_to_labels(self, vectors, weights=None):
        return BaseECOC.vectors_to_labels(self, vectors, weights=self.weights)




