import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC, SMOTE
import os
from sklearn.preprocessing import StandardScaler
import random

def data_subspace(data, label, sample_subspace=0.5, feature_subsapce=0.5,):
    # feature_number = int(data.shape[1] * feature_subsapce)
    sample_number = int(data.shape[0] * sample_subspace)
    # feature_index = random.sample(range(data.shape[1]), feature_number)
    # random.seed(7)
    sample_index = random.sample(range(data.shape[0]), sample_number)
    selected_data = data[sample_index, :]
    # selected_data = selected_data[:, feature_index]
    selected_label = label[sample_index]
    return selected_data, selected_label

def custom_counter(seq):
    seq = np.array(seq.flat)
    assert len(seq.shape)==1
    res = {}
    unique_seq = np.unique(seq)
    list_seq = seq.tolist()
    for i in unique_seq:
        res[i]=list_seq.count(i)
    return res

class custom_preprocess:
    def __init__(self, scaler, **params):
        self.scaler = scaler(**params)
        self.continous_feature = None
        self.categorial_feature = None

    def fit(self, data):
        data = np.array(data)
        continous_feature = []
        categorial_feature = []
        for i in range(data.shape[1]):
            if len(np.unique(data[:, i])) != 2:
                continous_feature.append(i)
            else:
                categorial_feature.append(i)
        self.continous_feature = continous_feature
        self.categorial_feature = categorial_feature
        if len(self.continous_feature)>0:
            data_continous = data[:, self.continous_feature]
            self.scaler.fit(data_continous)

    def transform(self, data):
        data = np.array(data)
        if len(self.continous_feature)>0:
            data_continous = data[:, self.continous_feature]
            data_categorial = data[:, self.categorial_feature]
            data_continous = self.scaler.transform(data_continous)
            data = np.hstack((data_continous, data_categorial))
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def categorial_feature(data):
    res = []
    for i in range(data.shape[1]):
        if len(np.unique(data[:, i])) == 2:
            res.append(i)
    return res

def filter_small_sample(df):
    df_groupby = df.groupby(df.shape[1] - 1)
    df_groupby = df_groupby.count()
    delete_row = []
    df_temp = df
    for i in df_groupby.index:
        if df_groupby.loc[i, 0] < 6:
            delete_row = delete_row + np.where(df.loc[:, df.shape[1] - 1] == i)[0].tolist()
            df_temp = df.drop(delete_row, axis=0)
            df_temp = df_temp.reset_index(drop=True)
    return df_temp


if __name__=='__main__':
    # files = [file for file in os.listdir() if file.endswith(r'.csv') and not file.endswith(r'resampled.csv')][7:]
    # print(files)
    #
    # categorial_index = []
    #
    # for file in files:
    #     df = pd.read_csv(file, header=None, index_col=None)
    #     print(file + ' number of features: ', df.shape[1])
    #     index = categorial_feature(df.values[:, :-1])
    #     print('categorial feature: ', len(index))
    #     categorial_index.append(index)
    #     column_names = list(df.columns)
    #     column_names[-1] = 'label'
    #     df.columns = column_names
    #     df['count'] = np.random.random(df.shape[0])
    #     df_groupby = df.groupby('label')
    #     print(df_groupby.count()['count'])
    #     print()
    #
    # for i, file in enumerate(files):
    #     df = pd.read_csv(file, header=None, index_col=None)
    #     df = filter_small_sample(df)
    #     df_groupby = df.groupby(df.shape[1] - 1)
    #     data = df.iloc[:, :-1]
    #     print(data.shape)
    #     label = df.iloc[:, -1]
    #     print(categorial_index[i])
    #     try:
    #         if categorial_index[i] != []:
    #             data_resampled, label_resampled = SMOTENC(categorical_features=categorial_index[i],
    #                                                       k_neighbors=5).fit_resample(data, label)
    #             print(data.shape)
    #         else:
    #             data_resampled, label_resampled = SMOTE().fit_resample(data, label)
    #             print(data.shape)
    #     except Exception as e:
    #         print()
    #         print(file)
    #         # print(e)
    #         print(data.shape)
    #         print("Can't resample!")
    #         print()
    #         continue
    #     print(file, ': ', data_resampled.shape)
    #     df_save = pd.DataFrame(data_resampled)
    #     df_save['label'] = label_resampled
    #     df_save.to_csv(file[:file.find(r'.')] + '_resampled.csv', header=None, index=None)
    cp = custom_preprocess(StandardScaler)
    df = pd.read_csv(r'C:\Users\Feng\Desktop\data\continous_data\page_blocks.csv', header=None, index_col=None)
    data = df.iloc[:,:-1].values
    data = cp.fit_transform(data)
    print(data)



