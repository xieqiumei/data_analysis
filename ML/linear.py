#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:linear.py
# @Software:PyCharm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score  # 要注意预测评估函数，有回归和分类之分

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression
from feature_analysis.train_config import svr_params_dict
from sklearn.ensemble import RandomForestRegressor
from utils.metrics import get_adjusted_r_squared, get_p_values
from utils.feature_select import backward_regression, select_by_recursive_feature_elimination, select_by_lasso, \
    column_index

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_scale_data(x_train, x_test, y_train, y_test):
    """
    对数据进行缩放，缩放后的数据用scaler.inverse_transform(test_S)还原
    :return:
    """
    scale_x = StandardScaler().fit(x_train)
    x_train = scale_x.transform(x_train)
    x_test = scale_x.transform(x_test)

    scale_y = StandardScaler().fit(y_train)
    y_train = scale_y.transform(y_train).ravel()
    y_test = scale_y.transform(y_test).ravel()

    return x_train, x_test, y_train, y_test, scale_y


def split_and_scale(x_array, y_array, test_size=0.2):
    """
    1.split train and test data
    2. scale train and test data
    :param x_array:
    :param y_array:
    :param test_size:
    :return:
    """
    n_train = int(x_array.shape[0] * (1 - test_size))
    x_train = x_array[:n_train, :]
    x_test = x_array[n_train:, :]
    y_train = y_array[:n_train]
    y_test = y_array[n_train:]

    # 分隔输入X和输出y
    # print("测试前：", y_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # 数据缩放
    # x_train, x_test, y_train, y_test, scale_y = get_scale_data(x_train, x_test, y_train, y_test)
    return get_scale_data(x_train, x_test, y_train, y_test)


def find(arr, min, max):
    """
    用于查找预测数据落在±50mm范围的数据
    :param arr:
    :param min:
    :param max:
    :return:
    """
    pos_min = arr >= min
    pos_max = arr <= max
    pos_rst = pos_min & pos_max
    return np.where(pos_rst == True)  # where的返回值刚好可以用[]来进行元素提取


def model_fit_and_score(x_train_fs, y_train, x_test_fs, y_test, selected_features, scale_y):
    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                             normalize=False)
    if model is None:
        print(f"Currently can't predicate by this model type.")
        return

    model.fit(x_train_fs, y_train)
    # evaluate the model
    y_predicate = model.predict(x_test_fs)
    # importance = model.coef_

    y_test = scale_y.inverse_transform(y_test)
    y_predicate = scale_y.inverse_transform(y_predicate)
    tmp = y_predicate - y_test

    print("误差±50mm的数量和百分比：", len(find(tmp, -50.0, 50.0)[0]), len(tmp), len(find(tmp, -50.0, 50.0)[0]) * 100 / len(tmp))
    # evaluate predictions:explained_variance_score, r2_score,mean_absolute_error
    r2score = r2_score(y_test, y_predicate)
    adjusted_r2squared = get_adjusted_r_squared(x_test_fs, y_test, r2score)
    print(
        f'Test R^2 of selected features {selected_features} is {r2score}, \nand adjusted_R^2 is {adjusted_r2squared}.')


def process(x_array, y_array, x_cloumns, selected_feature, test_size=0.2):
    X_df = pd.DataFrame(data=x_array, columns=x_cloumns)
    feature_ids = column_index(X_df, selected_feature)

    x_train, x_test, y_train, y_test, scale_y = split_and_scale(x_array, y_array, test_size=test_size)

    model_fit_and_score(x_train[:, feature_ids], y_train, x_test[:, feature_ids], y_test,
                        selected_feature, scale_y)


df = pd.read_excel("D:/工作项目/鸿玖/中泰数据 min级汇总/按分钟汇总数据/1min采样/C电极的参数汇总_1min__增加炉底温度.xlsx", sheet_name='Sheet1')
# 剔除电流异常数据
df.drop([894, 895, 896, 897, 898, 899, 900, 901, 903], inplace=True)
x_cloumns = ['焦炭实际配料值', '兰炭实际配料值', '石灰二实际配料值',
             '石灰一实际配料值', '电流_电极电流 KA_IA',
             '电流_电极电流 KA_IB', '电流_电极电流 KA_IC', '电压_电极电压 V_UA', '电压_电极电压 V_UB',
             '电压_电极电压 V_UC', '功率_电极功率_Pa', '功率_电极功率_Pb', '功率_电极功率_Pc',
             '功率_无功功率', '功率_有功功率', '温度_炉   底 ℃_3', '加热元件温度C']
X = df[x_cloumns].values
Y = df['C电极长度mm(计算)'].values

selected_feature = ['功率_有功功率' '温度_炉   底 ℃_3']
process(X, Y, x_cloumns, selected_feature, test_size=0.2)
