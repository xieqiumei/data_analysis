#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:model_analysis.py
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


def select_features(x_train, y_train, x_test, x_cloumns, y_column, select_type, k=None):
    """
    feature selection
    :param x_train:
    :param y_train:
    :param x_test:
    :param x_cloumns:
    :param y_column:
    :param select_type:
    :param k:
    :return:
    """
    if select_type == 'backward':
        features, feature_ids = backward_regression(x_train, y_train, x_cloumns, y_column)

        return x_train[:, feature_ids], x_test[:, feature_ids], features

    if select_type == 'rfe':
        features, feature_ids = select_by_recursive_feature_elimination(x_train, y_train, x_cloumns)
        return x_train[:, feature_ids], x_test[:, feature_ids], features

    if select_type == 'lasso':
        features, feature_ids = select_by_lasso(x_train, y_train, x_cloumns)
        return x_train[:, feature_ids], x_test[:, feature_ids], features

    if select_type in ['mi', 'f']:
        if k is None:
            return None, None, None

        # configure to select a subset of features
        if select_type == 'mi':
            fs = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            fs = SelectKBest(score_func=f_regression, k=k)
        # learn relationship from training dataf_regression
        fs.fit(x_train, y_train)
        # transform train input data
        # print(fs.get_support())
        x_train_fs = fs.transform(x_train)
        # transform test input data
        x_test_fs = fs.transform(x_test)
        # 特征的重要性
        support = fs.get_support()
        index = np.where(support == 1)[0]

        return x_train_fs, x_test_fs, np.array(x_cloumns)[index]

    return None, None, None


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


def get_svr_best_params(x_train, y_train):
    """
    网格搜索寻找最优的svr参数
    :param x_train:
    :param y_train:
    :return:
    """
    # 创建SVR实例
    svr = SVR()
    gscv = GridSearchCV(
        estimator=svr,
        param_grid=svr_params_dict,
        n_jobs=2,
        scoring='r2',
        cv=6)
    gscv.fit(x_train, y_train)  # 寻找最优参数
    best_params = gscv.best_params_
    return best_params


def create_model(x_train, y_train, model_type):
    if model_type == 'svr':
        # 网格搜索寻找最优的svr参数
        best_params = get_svr_best_params(x_train, y_train)
        print(f"Best svr params are {best_params}.")
        kernel = best_params['kernel']
        return SVR(C=best_params['C'], kernel=kernel, gamma=best_params['gamma'], epsilon=best_params['epsilon'])

    if model_type == 'linear':
        return LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                normalize=False)
    elif model_type == 'rf':

        return RandomForestRegressor(n_estimators=50, oob_score=True, random_state=100)
    else:
        return None


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


def model_fit_and_score(x_train_fs, y_train, x_test_fs, y_test, model_type, selected_features, scale_y):
    model = create_model(x_train_fs, y_train, model_type)
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


def svr_predicate_with_full_feature(x_array, y_value, y_name, feature_names, model_type, test_size=0.2):
    """
    svr回归预测
    :param x_array:
    :param y_value:
    :param y_name:
    :param feature_names:
    :param model_type:
    :param test_size:
    :return:
    """
    # 按照比例分成测试语料和训练语料
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_value, test_size=test_size, random_state=999)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    # 数据缩放
    x_train, x_test, y_train, y_test, scale_y = get_scale_data(x_train, x_test, y_train, y_test)

    model = create_model(x_train, y_train, model_type)
    if model is None:
        print(f"Currently can't predicate by this model type.")
        return

    model.fit(x_train, y_train)
    y_predicate = model.predict(x_test)
    # evaluate predictions
    score = r2_score(y_test, y_predicate)
    print(f'Test R^2 of {y_name} by features {feature_names[:-1]} is {score}')


def svr_predicate_with_feature_select(x_array, y_array, x_cloumns, y_column, model_type,
                                      select_type, test_size=0.2, selected_feature=None):
    """
    :param x_array:
    :param y_value:
    :param feature_names:
    :param model_type:
    :param k_num:
    :param test_size:
    :param select_type:
    :return:
    """
    # x_train, x_test, y_train, y_test = train_test_split(x_array, y_value, test_size=test_size)

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
    x_train, x_test, y_train, y_test, scale_y = get_scale_data(x_train, x_test, y_train, y_test)

    if selected_feature is not None:
        X_df = pd.DataFrame(data=x_array, columns=x_cloumns)
        feature_ids = column_index(X_df, selected_feature)

        model_fit_and_score(x_train[:, feature_ids], y_train, x_test[:, feature_ids], y_test, model_type,
                            selected_feature, scale_y)
        return

    if select_type is None or select_type not in ['f', 'mi', 'backward', 'rfe', 'lasso']:
        print("The wrong feature select type!")
        return

    # 特征选择

    if select_type in ['f', 'mi']:
        k_num = [i for i in range(1, x_array.shape[1] + 1)]
        # k_num = [i for i in range(1, 6)]
        # k_num = [x_array.shape[1]]
        for e in k_num:
            x_train_fs, x_test_fs, features = select_features(x_train, y_train, x_test, x_cloumns, y_column,
                                                              select_type, k=e)

            model_fit_and_score(x_train_fs, y_train, x_test_fs, y_test, model_type, features, scale_y)
            print("\n")

    else:
        x_train_fs, x_test_fs, features = select_features(x_train, y_train, x_test, x_cloumns, y_column,
                                                          select_type)

        model_fit_and_score(x_train_fs, y_train, x_test_fs, y_test, model_type, features, scale_y)


def outlier_treatment_by_median(df_data):
    for column in df_data.columns.tolist():
        if np.abs(df_data[column].skew()) > 1:
            # median = df_data[column].quantile(0.5)
            # _95percentile = df_data[column].quantile(0.95)
            # # print(median, _95percentile)
            # df[column] = np.where(df[column] >= _95percentile, median, df[column])
            flooring = df_data[column].quantile(0.1)
            capping = df_data[column].quantile(0.9)
            df_data[column] = np.where(df_data[column] < flooring, flooring, df_data[column])
            df_data[column] = np.where(df_data[column] > capping, capping, df_data[column])
    return df_data


# df = outlier_treatment_by_median(df)

# print(df.corr()['P'].sort_values())

if __name__ == '__main__':

    df = pd.read_excel("D:/工作项目/鸿玖/中泰数据 min级汇总/按分钟汇总数据/1min采样/C电极的参数汇总_1min__增加炉底温度.xlsx", sheet_name='Sheet1')
    # 剔除电流异常数据
    df.drop([894, 895, 896, 897, 898, 899, 900, 901, 903], inplace=True)

    X = df[['焦炭实际配料值', '兰炭实际配料值', '石灰二实际配料值',
            '石灰一实际配料值', '电流_电极电流 KA_IA',
            '电流_电极电流 KA_IB', '电流_电极电流 KA_IC', '电压_电极电压 V_UA', '电压_电极电压 V_UB',
            '电压_电极电压 V_UC', '功率_电极功率_Pa', '功率_电极功率_Pb', '功率_电极功率_Pc',
            '功率_无功功率', '功率_有功功率', '温度_炉   底 ℃_3', '加热元件温度C']].values
    Y = df['C电极长度mm(计算)'].values

    svr_predicate_with_feature_select(X, Y, ['焦炭实际配料值', '兰炭实际配料值', '石灰二实际配料值',
                                             '石灰一实际配料值', '电流_电极电流 KA_IA',
                                             '电流_电极电流 KA_IB', '电流_电极电流 KA_IC', '电压_电极电压 V_UA', '电压_电极电压 V_UB',
                                             '电压_电极电压 V_UC', '功率_电极功率_Pa', '功率_电极功率_Pb', '功率_电极功率_Pc',
                                             '功率_无功功率', '功率_有功功率', '温度_炉   底 ℃_3', '加热元件温度C'], ['C电极长度mm(计算)'],
                                      'rf',
                                      'mi', test_size=0.3, selected_feature=None)

""""
linear预测明显效果优于svr
"""
