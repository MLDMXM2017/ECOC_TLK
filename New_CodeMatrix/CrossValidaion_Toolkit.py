import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

def get_least_class(label):
    label = pd.Series(label)
    return min(label.value_counts())

def get_cross_validated_confusion_matrix(data, label, estimator, index, nfolds=10):
    # nfolds = get_least_class(label)
    skf = StratifiedKFold(n_splits=nfolds)
    con_matrix = np.zeros((len(np.unique(label)), len(np.unique(label))))
    for train_index, test_index in skf.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = np.array(label)[train_index], np.array(label)[test_index]
        estimator.train_matrix(train_data, train_label)
        pred_label = estimator.predict(test_data)
        con_matrix = con_matrix + confusion_matrix(test_label, pred_label, labels = index)
    return con_matrix

def get_cross_validated_error(data, label, real_label, estimator, nfolds=10):
    # nfolds = get_least_class(label)
    skf = StratifiedKFold(n_splits=nfolds)
    false_pred = {}
    for i in np.unique(real_label):
        false_pred[i] = 0
    for train_index, test_index in skf.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
        train_real_label, test_real_label = real_label[train_index], real_label[test_index]
        estimator.fit(train_data, train_label)
        pred_label = estimator.predict(test_data)
        # print(len(test_label))
        mask = np.array(pred_label) != np.array(test_label)
        mask_label = test_real_label[mask]
        # print(len(test_real_label),len(mask))
        for i in mask_label:
            false_pred[i] = false_pred[i] + 1
    return false_pred
