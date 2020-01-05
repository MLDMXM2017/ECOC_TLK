import numpy as np
import pandas as pd
import collections
import random
import copy
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import time
from data_preprocess import custom_preprocess
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, matthews_corrcoef
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings('ignore')
def euclidence_distance(x, y, soft=True):
	if soft:
		x_ = np.array([x[i] for i in range(len(x)) if x[i]!=0 and y[i]!=0])
		y_ = np.array([y[i] for i in range(len(y)) if x[i]!=0 and y[i]!=0])
	else:
		x_ = np.array(x)
		y_ = np.array(y)
	if len(x_)==0:

		raise Exception('len(x_)==0')
	return np.sqrt(np.sum(np.power(x_-y_,2))/(len(x_)+0.0001))

def hamming_distance(x, y, soft=True):
	if soft:
		x_ = np.array([x[i] for i in range(len(x)) if x[i]!=0 and y[i]!=0])
		y_ = np.array([y[i] for i in range(len(y)) if x[i]!=0 and y[i]!=0])
	else:
		x_ = np.array(x)
		y_ = np.array(y)
	if len(x_)==0:
		print('hamming distance')
		print('x:',x)
		print('y:',y)
		raise Exception('len(x_)==0')
	return np.sum(x_!=y_)/(len(x_)+0.0001)

def plot_matrix(save_path, matrix_list, matrix_column_accuracy_list, ylabel=None):
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	for i in range(len(matrix_list)):
		matrix = np.array(matrix_list[i])
		matrix_column_accuracy = np.round(matrix_column_accuracy_list[i],2).tolist()
		file_name = str(len(os.listdir(save_path)) + 1) + '.png'
		file_save_path = os.path.join(save_path, file_name)
		plt.figure(figsize=[matrix.shape[1], matrix.shape[0]])
		sns.heatmap(matrix, cmap='Blues', annot=True, xticklabels=matrix_column_accuracy, yticklabels=ylabel)
		plt.yticks(rotation=360)
		plt.title('Number of columns:' + str(matrix.shape[1]))
		plt.savefig(file_save_path)
		plt.close()

def plot_confusion_matrix(save_path, predicted_label_list, xlabel=None, ylabel=None):
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	true_label = predicted_label_list[0]
	for i in range(1, len(predicted_label_list)):
		conf_matrix = confusion_matrix(true_label, predicted_label_list[i], labels=xlabel)
		file_name = str(len(os.listdir(save_path))+1)+'.png'
		file_save_path = os.path.join(save_path, file_name)
		plt.figure(figsize=[conf_matrix.shape[1], conf_matrix.shape[0]])
		sns.heatmap(conf_matrix, cmap='Blues', annot=True, xticklabels=xlabel, yticklabels=ylabel)
		plt.yticks(rotation=360)
		plt.title('Confusion Matrix')
		plt.savefig(file_save_path)
		plt.close()

def plot_train_test_metric(save_path,  predicted_train_label_list, predicted_test_label_list, metric=accuracy_score, **metric_param):
	train_true_label = predicted_train_label_list[0]
	test_true_label = predicted_test_label_list[0]
	train_metric = [metric(train_true_label, predicted_train_label_list[i], **metric_param) for i in range(1,len(predicted_train_label_list))]
	test_metric = [metric(test_true_label, predicted_test_label_list[i], **metric_param) for i in range(1, len(predicted_test_label_list))]
	plt.figure()
	max_train_accuracy = max(train_metric)
	max_test_accuracy = max(test_metric)
	plt.title('Validation: %.2f Test:%.2f' % (max_train_accuracy, max_test_accuracy))
	plt.plot(train_metric, 'go-')
	plt.plot(test_metric, 'ro-')
	plt.legend(['Validation', 'Test'])
	plt.savefig(save_path)
	plt.close()

def purity(count_positive, count_negative):
	return max([count_positive, count_negative]) / (count_positive + count_negative + 0.00001)

class VL_ECOC:
	def __init__(self, data, label, base_classifier=SVC, feature_ratio=0.8, sample_ratio=0.8,
				 accuracy_threshold=0.6, class_level_matrix_threshold=2, sample_level_matrix_threshold=2, save_path=None):
		self.index = None
		self.base_classifier = base_classifier
		self.matrix = None
		self.feature_subspace_ratio = feature_ratio
		self.sample_ratio = sample_ratio
		self.feature_subspaces = []
		self.classifiers = []
		self.classifier_recall = []
		self.accuracy_threshold = accuracy_threshold
		self.class_level_matrix_threshold = class_level_matrix_threshold
		self.sample_level_matrix_threshold = sample_level_matrix_threshold
		self.train_data = None
		self.train_label = None
		self.test_data = None
		self.test_label = None
		self.validate_data = None
		self.validate_label = None
		self.split_data(data, label)
		self.matrix_record = []
		self.matrix_column_accuracy_record = []
		self.train_predicted_label_record = [self.train_label]
		self.train_predicted_code_record = [self.train_label]
		self.test_predicted_label_record = [self.test_label]
		self.test_predicted_code_record = [self.test_label]
		self.validate_predicted_label_record = [self.validate_label]
		self.validate_predicted_code_record = [self.validate_label]
		self.save_path = save_path
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		self.after_filt_point = -1
		self.class_level_end_point = -1

	def make_record(self):
		self.matrix_record.append(self.matrix)
		matrix_column_accuracy = self.validate_base_classifier(accuracy_score)
		self.matrix_column_accuracy_record.append(matrix_column_accuracy)
		train_predicted_label, train_predicted_code = self.predict(self.train_data, return_code=True)
		self.train_predicted_label_record.append(train_predicted_label)
		self.train_predicted_code_record.append(train_predicted_code)
		test_predicted_label, test_predicted_code = self.predict(self.test_data, return_code=True)
		self.test_predicted_label_record.append(test_predicted_label)
		self.test_predicted_code_record.append(test_predicted_code)
		validate_predicted_label, validate_predicted_code = self.predict(self.validate_data, return_code=True)
		self.validate_predicted_label_record.append(validate_predicted_label)
		self.validate_predicted_code_record.append(validate_predicted_code)

	def persistence_record(self):
		matrix_record_path = os.path.join(self.save_path, 'matrix_record.json')
		with open(matrix_record_path, 'w') as f:
			json.dump(self.matrix_record, f)
		# matrix_column_accuracy_record_path = os.path.join(self.save_path, 'matrix_column_accuracy_record.json')
		# with open(matrix_column_accuracy_record_path, 'w') as f:
		# 	json.dump(self.matrix_column_accuracy_record, f)
		train_predicted_label_record_path = os.path.join(self.save_path, 'train_predicted_label_record.json')
		with open(train_predicted_label_record_path, 'w') as f:
			json.dump(self.train_predicted_label_record, f)
		train_predicted_code_record_path = os.path.join(self.save_path, 'train_predicted_code_record.json')
		with open(train_predicted_code_record_path, 'w') as f:
			json.dump(self.train_predicted_code_record, f)
		test_predicted_label_record_path = os.path.join(self.save_path, 'test_predicted_label_record.json')
		with open(test_predicted_label_record_path, 'w') as f:
			json.dump(self.test_predicted_label_record, f)
		test_predicted_code_record_path = os.path.join(self.save_path, 'test_predicted_code_record.json')
		with open(test_predicted_code_record_path, 'w') as f:
			json.dump(self.test_predicted_code_record, f)
		validate_predicted_label_record_path = os.path.join(self.save_path, 'validate_predicted_label_record.json')
		with open(validate_predicted_label_record_path, 'w') as f:
			json.dump(self.validate_predicted_label_record, f)
		validate_predicted_code_record_path = os.path.join(self.save_path, 'validate_predicted_code_record.json')
		with open(validate_predicted_code_record_path, 'w') as f:
			json.dump(self.validate_predicted_code_record, f)
		# end_point_path = os.path.join(self.save_path, 'end_point.json')
		# with open(end_point_path, 'w') as f:
		# 	json.dump([self.after_filt_point, self.class_level_end_point], f)

	def visualization(self):
		matrix_path = os.path.join(self.save_path, 'Coding Matrix')
		plot_matrix(matrix_path, self.matrix_record, self.matrix_column_accuracy_record, ylabel=self.index)
		train_conf_matrix_path = os.path.join(self.save_path, 'Train Data Confusion Matrix')
		plot_confusion_matrix(train_conf_matrix_path, self.train_predicted_label_record, xlabel=self.index, ylabel=self.index)
		test_conf_matrix_path = os.path.join(self.save_path, 'Test Data Confusion Matrix')
		plot_confusion_matrix(test_conf_matrix_path, self.test_predicted_label_record, xlabel=self.index, ylabel=self.index)
		validate_conf_matrix_path = os.path.join(self.save_path, 'Validate Data Confusion Matrix')
		plot_confusion_matrix(validate_conf_matrix_path, self.validate_predicted_label_record, xlabel=self.index, ylabel=self.index)
		train_test_accuracy_path = os.path.join(self.save_path, 'Validation Test Accuracy.png')
		plot_train_test_metric(train_test_accuracy_path, self.validate_predicted_label_record, self.test_predicted_label_record, metric=accuracy_score)
		train_test_fscore_path = os.path.join(self.save_path, 'Validation Test Fscore.png')
		plot_train_test_metric(train_test_fscore_path, self.validate_predicted_label_record, self.test_predicted_label_record, metric=f1_score, average='macro')
		train_test_MCC_path = os.path.join(self.save_path, 'Validation Test MCC.png')
		plot_train_test_metric(train_test_MCC_path, self.validate_predicted_label_record, self.test_predicted_label_record, metric=matthews_corrcoef)
		# min_hamming_distance_list = [self.min_hamming_distance(matrix = self.matrix_record[i])[0] for i in range(len(self.matrix_record))]
		# plt.figure()
		# plt.plot(list(range(1, len(min_hamming_distance_list)+1)), min_hamming_distance_list, 'o-')
		# plt.title('Min Hamming Distance')
		# plt.savefig(os.path.join(self.save_path, 'Min Hamming Distance'))
		# plt.close()
		# plt.figure()
		# plt.title('Average Hamming Distance')
		# plt.savefig(os.path.join(self.save_path, 'Average Hamming Distance'))
		# plt.close()

	def get_class_data(self, data, label, class_name):
		if class_name not in label:
			return []
		result_data = [data[i] for i in range(len(label)) if label[i]==class_name]
		result_label = [label[i] for i in range(len(label)) if label[i]==class_name]
		return result_data, result_label

	def split_data(self, data, label):
		self.index = np.unique(label).tolist()
		sample_count = collections.Counter(label)
		for cla in sample_count:
			if sample_count[cla] < 4:
				temp_data, temp_label= self.get_class_data(data, label, cla)
				multiple = int(np.ceil(4/sample_count[cla]))
				data += temp_data * multiple
				label += temp_label * multiple
		self.train_data, temp_data, self.train_label, temp_label = train_test_split(data, label, test_size=0.5, stratify=label)
		self.test_data, self.validate_data, self.test_label, self.validate_label = train_test_split(temp_data, temp_label, test_size=0.5, stratify=temp_label)
		cust_preprocess = custom_preprocess(scaler=StandardScaler)
		cust_preprocess.fit(self.train_data)
		self.train_data = cust_preprocess.transform(self.train_data)
		self.test_data = cust_preprocess.transform(self.test_data)
		self.validate_data = cust_preprocess.transform(self.validate_data)

	def stratified_sample(self, data, label):
		temp_data, temp_label = RandomOverSampler().fit_resample(data, label)
		sample_data, _, sample_label, _ = train_test_split(temp_data, temp_label, test_size=1-self.sample_ratio, stratify=temp_label)
		return sample_data.tolist(), sample_label.tolist()

	def train_base_classifier(self, column, feature_subspace, classifier_param={}, custom_data=None, custom_label=None, fill_zero=True):
		if feature_subspace is None:
			feature_size = int(np.ceil(self.feature_subspace_ratio*len(self.train_data[0])))
			feature_subspace = random.sample(list(range(len(self.train_data[0]))), k=feature_size)
		if custom_data is None and custom_label is None:
			train_data = copy.deepcopy(self.train_data)
			train_data = np.array(train_data)[:,feature_subspace].tolist()
			train_label = copy.deepcopy(self.train_label)
		else:
			train_data = copy.deepcopy(custom_data)
			train_data = np.array(train_data)[:,feature_subspace].tolist()
			train_label = copy.deepcopy(custom_label)
		train_data, train_label = self.stratified_sample(train_data, train_label)
		negative_class = []
		positive_class = []
		for index,value in enumerate(column):
			if value == 1:
				positive_class.append(self.index[index])
			elif value == -1:
				negative_class.append(self.index[index])
		data = []
		label = []
		for index,value in enumerate(train_label):
			if value in positive_class:
				data.append(train_data[index])
				label.append(1)
			elif value in negative_class:
				data.append(train_data[index])
				label.append(-1)
		classifier = self.base_classifier(**classifier_param).fit(data, label)
		if fill_zero:
			self.fill_column_zeros(column, classifier, feature_subspace)
		return column, classifier, feature_subspace

	def validate_base_classifier(self, metric=accuracy_score):
		metrics = []
		temp_matrix = np.array(self.matrix)
		for i in range(len(temp_matrix[0])):
			temp_column = temp_matrix[:,i]
			temp_metric = self.validate_single_classifier(metric, temp_column, self.classifiers[i], self.feature_subspaces[i])
			metrics.append(temp_metric)
		return metrics

	def validate_single_classifier(self, metric, column, classifier, feature_subspace, metric_param={}):
		temp_data = np.array(self.validate_data)[:,feature_subspace].tolist()
		temp_label = list(self.validate_label)
		temp_data = [temp_data[i] for i in range(len(temp_label)) if column[self.index.index(temp_label[i])]!=0]
		temp_label = [temp_label[i] for i in range(len(temp_label)) if column[self.index.index(temp_label[i])]!=0]
		temp = []
		for l in temp_label:
			temp.append(column[self.index.index(l)])
		temp_label = temp
		predicted_label = classifier.predict(temp_data)
		return metric(temp_label, predicted_label, **metric_param)

	def predict(self, data, return_code = False):
		row_code = []
		for i,classifier in enumerate(self.classifiers):
			temp_data = data[:,self.feature_subspaces[i]]
			temp_code = classifier.predict(temp_data).tolist()
			for j in range(len(temp_code)):
				if temp_code[j] == -1:
					temp_code[j] = float(temp_code[j]) * self.classifier_recall[i][0] + 0.0001
				else:
					temp_code[j] = float(temp_code[j]) * self.classifier_recall[i][1] + 0.0001
			row_code.append(temp_code)
		row_code = np.array(row_code).T.tolist()
		predicted_label = []
		for code in row_code:
			temp_label = None
			min_distance = np.inf
			for i,matrix_code in enumerate(self.matrix):
				distance = euclidence_distance(code, matrix_code)
				if distance < min_distance:
					min_distance = distance
					temp_label = self.index[i]
			predicted_label.append(temp_label)
		if return_code:
			return predicted_label, row_code
		else:
			return predicted_label

	def test(self, metric=accuracy_score, **metric_param):
		predicted_test_label = self.predict(self.test_data)
		return metric(self.test_label, predicted_test_label, **metric_param)

	def validate(self, metric=accuracy_score, **metric_param):
		predicted_validate_label = self.predict(self.validate_data)
		return metric(self.validate_label, predicted_validate_label, **metric_param)\

	def min_hamming_distance(self, matrix=None):
		if matrix is None:
			matrix = self.matrix
		min_distance = np.inf
		classes = []
		for i in range(len(matrix)):
			for j in range(i+1, len(matrix)):
				hd = hamming_distance(matrix[i], matrix[j])
				if hd < min_distance:
					min_distance = hd
					classes = [self.index[i], self.index[j]]
		return min_distance, classes

	def get_centroid(self, class_name, feature_subspace=None):
		if feature_subspace is None:
			temp_data = copy.deepcopy(self.train_data)
		else:
			temp_data = np.array(self.train_data)[:,feature_subspace].tolist()
		temp_label = copy.deepcopy(self.train_label)
		target_data = [temp_data[i] for i in range(len(temp_label)) if temp_label[i]==class_name]
		return np.average(target_data, axis=0).tolist()

	def make_column_with_centroid_class(self, centroid_classes=(), centroids=None, feature_subspace=None, custom_data=None, custom_label=None):
		column = [0]*len(self.index)
		if len(centroid_classes)>0:
			column[self.index.index(centroid_classes[0])] = 1
			positive_centroid = self.get_centroid(centroid_classes[0], feature_subspace)
			column[self.index.index(centroid_classes[1])] = -1
			negative_centroid = self.get_centroid(centroid_classes[1], feature_subspace)
		else:
			positive_centroid = centroids[0]
			negative_centroid = centroids[1]
		if custom_data is None:
			temp_data = np.array(self.train_data)[:,feature_subspace].tolist()
			temp_label = copy.deepcopy(self.train_label)
			temp_data, temp_label = self.stratified_sample(temp_data, temp_label)
		else:
			temp_data = np.array(custom_data)[:,feature_subspace].tolist()
			temp_label = copy.deepcopy(custom_label)
		for i,c in enumerate(self.index):
			if c not in centroid_classes:
				temp_class_data = copy.deepcopy([temp_data[i] for i in range(len(temp_label)) if temp_label[i]==c])
				positive_count = 0
				negative_count = 0
				for data in temp_class_data:
					if euclidence_distance(data, positive_centroid, soft=False) < euclidence_distance(data, negative_centroid, soft=False):
						positive_count += 1
					else:
						negative_count += 1
				if purity(positive_count, negative_count) >= 0.6:
					column[i] = 1 if positive_count>negative_count else -1
		return column

	def filter_duplicate_columns(self):
		before_filte_accuracy = self.validate()
		temp_matrix = copy.deepcopy(self.matrix)
		temp_classifiers = copy.deepcopy(self.classifiers)
		temp_feature_subspaces = copy.deepcopy(self.feature_subspaces)
		temp_classifers_recall = copy.deepcopy(self.classifier_recall)
		self.matrix, index = np.unique(self.matrix, axis=1, return_index=True)
		self.matrix = self.matrix.tolist()
		self.classifiers = [self.classifiers[i] for i in index]
		self.feature_subspaces = [self.feature_subspaces[i] for i in index]
		self.classifier_recall = [self.classifier_recall[i] for i in index]
		after_filte_accuracy = self.validate()
		if before_filte_accuracy > after_filte_accuracy:
			self.matrix = copy.deepcopy(temp_matrix)
			self.classifiers = copy.deepcopy(temp_classifiers)
			self.feature_subspaces = copy.deepcopy(temp_feature_subspaces)
			self.classifier_recall = copy.deepcopy(temp_classifers_recall)
		self.make_record()

	def get_tough_classes(self, conf_matrix):
		conf_matrix = np.array(conf_matrix)
		row_sum = np.sum(conf_matrix, axis=1).reshape([-1,1])
		conf_matrix_ratio = conf_matrix / row_sum
		for i in range(len(conf_matrix_ratio)):
			conf_matrix_ratio[i][i]=-1
		max_index = np.argmax(conf_matrix_ratio)
		row_index = int(max_index / len(conf_matrix_ratio))
		col_index = max_index % len(conf_matrix_ratio)
		return [self.index[row_index], self.index[col_index]]

	def fill_column_zeros(self, column, classifier, feature_subspace):
		for index, value in enumerate(column):
			if value == 0:
				temp_data, _ = self.get_class_data(self.train_data, self.train_label, self.index[index])
				temp_data = np.array(temp_data)[:,feature_subspace].tolist()
				predicted_label = classifier.predict(temp_data)
				label_count = collections.Counter(predicted_label)
				column[index] = 1 if label_count[1]>label_count[-1] else -1
		return column

	def make_seed_column(self):
		feature_size = int(np.ceil(self.feature_subspace_ratio * len(self.train_data[0])))
		feature_subspace = random.sample(list(range(len(self.train_data[0]))), k=feature_size)
		train_data, _ = self.stratified_sample(self.train_data, self.train_label)
		train_data = np.array(train_data)[:,feature_subspace].tolist()
		kmeans = KMeans(n_clusters=2).fit(train_data)
		seed_column = self.make_column_with_centroid_class(centroids=kmeans.cluster_centers_, feature_subspace=feature_subspace)
		fill_value = 0
		if -1 not in seed_column:
			fill_value = -1
		elif 1 not in seed_column:
			fill_value = 1
		if fill_value != 0:
			for i in range(len(seed_column)):
				if seed_column[i]==0:
					seed_column[i]=fill_value
		if all(np.array(seed_column)==1) or all(np.array(seed_column)==-1):
			seed_column[0] = seed_column[0] * -1
		seed_column, classifier, feature_subspace = self.train_base_classifier(seed_column, feature_subspace)
		seed_column = self.fill_column_zeros(seed_column, classifier, feature_subspace)
		self.matrix = np.array(seed_column).reshape([-1, 1]).tolist()
		self.classifiers.append(classifier)
		self.feature_subspaces.append(feature_subspace)
		self.classifier_recall.append(self.validate_single_classifier(recall_score, seed_column, classifier, feature_subspace, metric_param={'average':None}))
		self.make_record()
		return seed_column, classifier, feature_subspace

	def make_class_level_matrix(self):
		validated_metric_record = []
		best_performance_index = -1
		mark_time = 0
		while mark_time <= self.class_level_matrix_threshold:
			predicted_label = self.predict(self.validate_data)
			conf_matrix = confusion_matrix(self.validate_label, predicted_label)
			tough_classes = self.get_tough_classes(conf_matrix)
			feature_size = int(np.ceil(self.feature_subspace_ratio * len(self.train_data[0])))
			feature_subspace = random.sample(list(range(len(self.train_data[0]))), k=feature_size)
			column = self.make_column_with_centroid_class(centroid_classes=tough_classes, feature_subspace=feature_subspace)
			column, classifier, feature_subspace = self.train_base_classifier(column, feature_subspace)
			if self.validate_single_classifier(accuracy_score, column, classifier, feature_subspace) >= self.accuracy_threshold:
				self.matrix = np.hstack((self.matrix, np.array(column).reshape([-1,1]))).tolist()
				self.classifiers.append(classifier)
				self.feature_subspaces.append(feature_subspace)
				self.classifier_recall.append(self.validate_single_classifier(recall_score, column, classifier, feature_subspace, metric_param={'average':None}))
				self.make_record()
			else:
				mark_time += 1
				continue
			validated_metric = self.validate(accuracy_score)
			validated_metric_record.append(validated_metric)
			best_performance_index = np.argmax(validated_metric_record)
			if len(validated_metric_record)- 1 - best_performance_index >= self.class_level_matrix_threshold or \
					validated_metric_record[best_performance_index]==1:
				break
		reverse_index = best_performance_index - len(validated_metric_record)
		if reverse_index == -1:
			reverse_index = len(self.matrix[0])-1
		self.matrix = np.array(self.matrix)[:, :reverse_index+1].tolist()
		self.classifiers = self.classifiers[:reverse_index+1]
		self.feature_subspaces = self.feature_subspaces[:reverse_index+1]
		self.classifier_recall = self.classifier_recall[:reverse_index+1]
		self.make_record()
		return

	def increase_hamming_distance(self):
		# hamming_threshold = 0.1 * len(self.matrix[0])
		hamming_threshold = 0
		step = 0
		min_distance, classes = self.min_hamming_distance()
		while step < 10 and min_distance <= hamming_threshold:
			step += 1
			feature_size = int(np.ceil(self.feature_subspace_ratio * len(self.train_data[0])))
			feature_subspace = random.sample(list(range(len(self.train_data[0]))), k=feature_size)
			column = self.make_column_with_centroid_class(centroid_classes=classes, feature_subspace=feature_subspace)
			column, classifier, feature_subspace = self.train_base_classifier(column, feature_subspace)
			if self.validate_single_classifier(accuracy_score, column, classifier, feature_subspace) > self.accuracy_threshold:
				self.matrix = np.hstack((self.matrix, np.array(column).reshape([-1,1]))).tolist()
				self.classifiers.append(classifier)
				self.feature_subspaces.append(feature_subspace)
				self.classifier_recall.append(self.validate_single_classifier(recall_score, column, classifier, feature_subspace, metric_param={'average':None}))
				self.make_record()
			else:
				continue
			min_distance, classes = self.min_hamming_distance()

	def get_misclassified_samples(self):
		predicted_label = self.predict(self.validate_data)
		misclassified_data = [self.validate_data[i] for i in range(len(self.validate_label)) if self.validate_label[i]!=predicted_label[i]]
		misclassified_label = [self.validate_label[i] for i in range(len(self.validate_label)) if self.validate_label[i]!=predicted_label[i]]
		misclassified_label_count = collections.Counter(misclassified_label)
		validate_label_count = collections.Counter(self.validate_label)
		misclassified_ratio = {key:misclassified_label_count[key]/validate_label_count[key] for key in misclassified_label}
		min_ratio_threshold = 0.05
		misclassified_data = [misclassified_data[i] for i in range(len(misclassified_label)) if misclassified_ratio[misclassified_label[i]]>=min_ratio_threshold]
		misclassified_label = [misclassified_label[i] for i in range(len(misclassified_label)) if misclassified_ratio[misclassified_label[i]]>=min_ratio_threshold]
		train_data, train_label = self.stratified_sample(self.train_data, self.train_label)
		train_label_count = collections.Counter(train_label)
		res_data = []
		res_label = []
		for l in misclassified_label_count:
			neighbor_number = int(np.ceil(misclassified_ratio[l] * train_label_count[l]))
			neighbor_per_sample = int(np.ceil(neighbor_number / misclassified_label_count[l]))
			temp_train_data = [train_data[i] for i in range(len(train_label)) if train_label[i]==l]
			nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
			nn.fit(temp_train_data)
			for i in range(len(misclassified_label)):
				if misclassified_label[i]==l:
					neighbors = nn.kneighbors([misclassified_data[i]],n_neighbors=neighbor_per_sample, return_distance=False)
					res_data += [temp_train_data[i] for i in neighbors[0]]
					res_label += [l]*len(neighbors[0])
		return res_data, res_label

	def make_sample_level_matrix(self):
		validated_metric_record = []
		best_performance_index = -1
		mark_time = 0
		while mark_time <= self.sample_level_matrix_threshold:
			predicted_label = self.predict(self.validate_data)
			conf_matrix = confusion_matrix(self.validate_label, predicted_label, labels=self.index)
			tough_classes = self.get_tough_classes(conf_matrix)       
			feature_size = int(np.ceil(self.feature_subspace_ratio * len(self.train_data[0])))
			feature_subspace = random.sample(list(range(len(self.train_data[0]))), k=feature_size)
			train_data, train_label = self.get_misclassified_samples()
			tough_class = tough_classes[1]
			if len(train_label)==0:
				break
			if tough_class not in np.unique(train_label):
				temp_train_data, temp_train_label = self.get_class_data(self.train_data, self.train_label, tough_class)
				train_data += random.sample(temp_train_data, k=int(np.ceil(0.5*len(temp_train_data))))
				train_label += random.sample(temp_train_label, k=int(np.ceil(0.5*len(temp_train_label))))
			column = self.make_column_with_centroid_class(centroid_classes=tough_classes, feature_subspace=feature_subspace, custom_data=train_data, custom_label=train_label)
			try:
				column, classifier, feature_subspace = self.train_base_classifier(column, feature_subspace, custom_data=train_data, custom_label=train_label, fill_zero=True)
			except Exception as e:
				break
			if self.validate_single_classifier(accuracy_score, column, classifier,
											   feature_subspace) >= self.accuracy_threshold:
				self.matrix = np.hstack((self.matrix, np.array(column).reshape([-1, 1]))).tolist()
				self.classifiers.append(classifier)
				self.feature_subspaces.append(feature_subspace)
				self.classifier_recall.append(self.validate_single_classifier(recall_score, column, classifier, feature_subspace, metric_param={'average':None}))
			else:
				mark_time += 1
				continue
			validated_metric = self.validate(accuracy_score)
			validated_metric_record.append(validated_metric)
			if len(validated_metric_record)>=2 and validated_metric < validated_metric_record[-2]:
				validated_metric_record.pop(-1)
				self.matrix = np.array(self.matrix)[:,:-1].tolist()
				self.classifiers.pop(-1)
				self.feature_subspaces.pop(-1)
				self.classifier_recall.pop(-1)
				mark_time += 1
				continue
			self.make_record()
			best_performance_index = np.argmax(validated_metric_record)
			if len(validated_metric_record) - 1 - best_performance_index >= self.sample_level_matrix_threshold or \
					validated_metric_record[best_performance_index]==1:
				break
		reverse_index = best_performance_index - len(validated_metric_record)
		if reverse_index == -1:
			reverse_index = len(self.matrix[0]) - 1
		self.matrix = np.array(self.matrix)[:,:reverse_index + 1].tolist()
		self.classifiers = self.classifiers[:reverse_index + 1]
		self.feature_subspaces = self.feature_subspaces[:reverse_index + 1]
		self.classifier_recall = self.classifier_recall[:reverse_index + 1]
		self.make_record()
		return

	def create_matrix(self):
		self.matrix = None
		self.classifiers = []
		self.feature_subspaces = []
		self.classifier_recall = []
		self.make_seed_column()
		self.make_class_level_matrix()
		self.filter_duplicate_columns()
		self.after_filt_point = len(self.train_predicted_label_record)-1
		self.increase_hamming_distance()
		self.class_level_end_point = len(self.train_predicted_label_record)-1
		self.make_sample_level_matrix()


	def score(self):
		start_time = time.time()
		self.create_matrix()
		train_time = time.time()-start_time
		start_time = time.time()
		accuracy = self.test(accuracy_score)
		fscore = self.test(f1_score, average='micro')
		matthews = self.test(matthews_corrcoef)
		predict_time = time.time()-start_time
		print('Accuracy:', accuracy)
		print('F-Score:', fscore)
		print('Matthews:', matthews)
		return accuracy, fscore, matthews, train_time, predict_time
