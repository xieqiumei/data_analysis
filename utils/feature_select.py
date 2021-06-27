#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:feature_select.py
# @Software:PyCharm


import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression
import warnings

warnings.filterwarnings("ignore")
"""
参考自https://github.com/AakkashVijayakumar/stepwise-regression
1.Stepwise Feature Elimination
"""


def forward_regression(X, y,
                       threshold_in=0.01,
                       verbose=False):
    """
    Forward elimination starts with no features, and the insertion
    of features into the regression model one-by-one.
    :param X:
    :param y:
    :param threshold_in:
    :param verbose:
    :return:
    """
    initial_list = []
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included


def column_index(df_data, query_cols):
    """
    Get column index from column name
    :param df_data:
    :param query_cols:
    :return:
    """
    cols = df_data.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def backward_regression(X, y, X_columns, y_column,
                        threshold_out=0.05,
                        verbose=False):
    X_df = pd.DataFrame(data=X, columns=X_columns)
    print(y)
    print(y_column)
    y_df = pd.DataFrame(data=y, columns=y_column)
    included = list(X_columns)
    while True:
        changed = False
        # model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # pvalues = model.pvalues.iloc[1:]
        # worst_pval = pvalues.max()# null if pvalues is empty

        model = sm.OLS(y_df.values, sm.add_constant(X_df[included].values)).fit()

        # # use all coefs except intercept
        pvalues = model.pvalues[1:]
        index = np.argsort(pvalues)
        worst_pval = pvalues[index[-1]]

        if worst_pval > threshold_out:
            changed = True
            worst_feature = included[index[-1]]
            # worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

    return included, column_index(X_df, included)


def select_by_recursive_feature_elimination(X, y, X_columns):
    """
    A popular feature selection method within sklearn is the Recursive Feature Elimination.
    RFE selects features by considering a smaller and smaller set of regressors.  The starting
    point is the original set of regressors. Less important regressors are recursively pruned
    from the initial set.
    :param X:
    :param y:
    :return:
    """
    # use linear regression as the model
    lin_reg = LinearRegression()
    # This is to select 5 variables: can be changed and checked in model for accuracy
    rfe_mod = RFE(lin_reg, 5, step=1)  # RFECV(lin_reg, step=1, cv=5)
    # This is to select 8 variables: can be changed and checked in model for accuracy
    # rfe_mod = RFECV(lin_reg, step=1, cv=300)  # RFE(lin_reg, 4, step=1)

    myvalues = rfe_mod.fit(X, y)  # to fit
    # myvalues.ranking_  #Selected (i.e., estimated best) features are assigned rank 1.
    feature_idx = np.where(myvalues.ranking_ == 1)[0]
    return np.array(X_columns)[feature_idx], feature_idx


def select_by_lasso(X, y, X_columns):
    # Use L1 penalty
    estimator = LassoCV(cv=3, normalize=True)
    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(estimator, threshold=0.05, prefit=False, norm_order=1, max_features=None)
    sfm.fit(X, y)
    feature_idx = sfm.get_support()
    return np.array(X_columns)[feature_idx], feature_idx


def select_features(x_train, y_train, x_test, select_type, k):
    """
    feature selection
    :param x_train:
    :param y_train:
    :param x_test:
    :param select_type:
    :param k:
    :return:
    """
    # configure to select a subset of features
    if select_type == 'mi':
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
    else:
        fs = SelectKBest(score_func=f_regression, k=k)
    print(fs.scores_)
    print(fs.pvalues_)
    # learn relationship from training dataf_regression
    fs.fit(x_train, y_train)
    # transform train input data
    # print(fs.get_support())
    x_train_fs = fs.transform(x_train)
    # transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


# data = pd.read_csv("D:/工作项目/新和成/sheet8.csv")
# X = data[['A2', 'B2', 'C2', 'D2']].values
# y = data[['Z']].values
# X_columns = ['A2', 'B2', 'C2', 'D2']

# from sklearn.datasets import load_boston
#
# boston = load_boston()
#
# X = boston.data
# y = boston.target
#
# print(X)
# print(y)
#
# X_columns = boston.feature_names
# y_column = ['MEDV']
# boston_features_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
# boston_target_df = pd.DataFrame(data=boston.target, columns=['MEDV'])

# df = pd.read_csv("D:/工作项目/新和成/sheet10.csv")
#
#
# def outlier_treatment_by_median(df_data):
#     for column in df_data.columns.tolist():
#         if np.abs(df_data[column].skew()) > 1:
#             # median = df_data[column].quantile(0.5)
#             # _95percentile = df_data[column].quantile(0.95)
#             # # print(median, _95percentile)
#             # df[column] = np.where(df[column] >= _95percentile, median, df[column])
#             flooring = df_data[column].quantile(0.10)
#             capping = df_data[column].quantile(0.90)
#             df_data[column] = np.where(df_data[column] < flooring, flooring, df_data[column])
#             df_data[column] = np.where(df_data[column] > capping, capping, df_data[column])
#     return df_data
#
#
# #
# df = outlier_treatment_by_median(df)
#
# X = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']].values
# y = df['O'].values
#
# X_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
# y_column = ['O']
# # X = boston_features_df
# # y = boston_target_df
#
# # print(backward_regression(X, y, verbose=True))
#
# print(select_by_recursive_feature_elimination(X, y, X_columns))
#
# print(select_by_lasso(X, y, X_columns))
#
# print(backward_regression(X, y, X_columns, y_column, verbose=True))

# data = pd.read_csv("D:/工作项目/新和成/sheet8.csv")
# # sub_dfs['group1_X'] = df[['A', 'B', 'C', 'D', 'X']]
# # sub_dfs['group1_Z'] = df[['A', 'B', 'C', 'D', 'Z']]
# # sub_dfs['group2_X'] = df[['A2', 'B2', 'C2', 'D2', 'X']]
# sub_data = data[['A2', 'B2', 'C2', 'D2', 'Z']]
#
# corr = sub_data.corr()
# columns = np.full((corr.shape[0],), True, dtype=bool)
