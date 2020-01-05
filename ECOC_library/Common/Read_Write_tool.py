# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/2/2 11:04
# file: Read_Write_tool.py
# description: this module offers some method for read and write Microarray and UCI datasets

import xlwt
import time
import sys
import xlrd
import numpy as np
import logging
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from imp import reload


def form_style(arr):
    styles = []
    min_v = min(arr)
    max_v = max(arr)
    for i in range(len(arr)):
        if arr[i] == max_v:
            styles.append('font: bold 1, color red;')
        elif arr[i] == min_v:
            styles.append('font: bold 1, color blue;')
        else:
            styles.append('font: bold 0, color black;')
    return styles


def write_file(filepath, row_values, row_titles, col_names):
    xls = xlwt.Workbook()  # 创建工作簿

    sheet = xls.add_sheet(u'sheet1', cell_overwrite_ok=True)

    default_style = xlwt.easyxf('font:bold 1, color black;')

    if isinstance(row_values[0], int):  # 生成第一行
        logging.debug('row values only contain a single number')
        return []

    for i in range(0, len(row_titles)):  # 第一行的标题
        sheet.write(0, i + 1, row_titles[i], default_style)

    for row, row_value in enumerate(row_values):  # 行
        sheet.write(row + 1, 0, col_names[row], default_style)
        for col, col_value in enumerate(row_value):  # 列
            col_styles = form_style(row_value)
            col_value = str(round(col_value, 3))
            style = xlwt.easyxf(col_styles[col])
            sheet.write(row + 1, col + 1, col_value, style)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    sheet.write(len(row_values) + 5, 0, 'file create time:' + time_str, default_style)
    xls.save(filepath)  # 保存文件, 不能出现空格


def write_matrix(filepath, M):
    logging.info('save file:' + filepath)

    xls = xlwt.Workbook(encoding='utf-8')  # 创建工作簿

    sheet = xls.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for row, value in enumerate(M):

        for col, v in enumerate(value):
            sheet.write(row, col, str(M[row][col]))

    xls.save(filepath)  # 保存文件


def read_matirx(filepath):
    reload(sys)

    file = xlrd.open_workbook(filepath)

    sheet = file.sheet_by_index(0)

    nrows = sheet.nrows
    ncols = sheet.ncols

    M = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            M[i][j] = float(str(sheet.cell(i, j).value))

    return M


def write_FS_data(path, data, label):
    Transition_tool.round_list(data)
    af = pd.DataFrame(label).T
    bf = pd.DataFrame(data).T

    predictions = pd.concat([af, bf])
    predictions.to_csv(path, index=False, header=False)


def read_UCI_Dataset(path):
    """
    to read UCI_data data set from file
    :param path: path of file
    :return: data, label
    """
    df = pd.read_csv(path, header=None)
    df_values = df.values
    col_num = df_values.shape[1]
    data = df_values[:, 0:col_num - 1]
    label = df_values[:, col_num - 1]
    return data, label


def read_Microarray_Dataset(path):
    """
    to read Microarray data set from file
    :param path: path of file
    :return:
    """
    pattern = re.compile(r'(\w+)(\.)*.*')
    df = pd.read_csv(path)
    df_columns = np.array([pattern.match(col).group(1) for col in df.axes[1]])
    df_values = df.values
    data = df_values.T
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    label = df_columns
    return data, label


def write_txt(fname, content):
    fobj = open(fname, 'a')  # 没有就创建,在后面追加内容
    fobj.write('\n' + content)  # 这里的\n的意思是在源文件末尾换行，即新加内容另起一行插入。
    fobj.close()  # 特别注意文件操作完毕后要close


def read_ECOC_res(path, ecoc_name, data_name):
    print(path)
    df = pd.read_excel(path)
    df_data_name = df.index.values
    inx = list(df_data_name).index(data_name)
    values = df[ecoc_name].values
    return values[inx]
