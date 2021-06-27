#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:ts_correlation.py
# @Software:PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import spearmanr, pearsonr
from scipy.stats import probplot, moment
from sklearn.linear_model import Ridge
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU
from feature_analysis.model_analysis import svr_predicate_with_feature_select

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
"""
https://zhuanlan.zhihu.com/p/191211602
"""


# convert series to supervised learning
def series_to_supervised(df_data, n_in=1, n_out=1, dropnan=False):
    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n)
    n_vars = df_data.columns

    # input sequence (t-n, ... t-1)
    # for i in range(0, n_in, 1):
    #     cols.append(df_data.shift(i))
    #     names += [('var%s(t-%d)' % (j, i + 1)) for j in n_vars]

    # input sequence (t-n, ... t-1)
    if type(n_in) is list:
        for i, v in enumerate(n_in):

            for j in v:
                cols.append(df_data[n_vars[i]].shift(j + 1))
                names += [('%s(t-%d)' % (n_vars[i], j + 1))]

                # print([('var%s(t-%d)' % (n_vars[i], j + 1))])

        # for j in range(0, len(n_vars)):
        #     print(n_in[j])
        #     for i in n_in[j]:
        #         cols.append(df_data.shift(i))
        #         names += [('var%s(t-%d)' % (j, i + 1)) for j in n_vars]
    else:
        for i in range(1, n_in + 1, 1):
            cols.append(df_data.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in n_vars]
    for i in range(0, n_out):
        cols.append(df_data.shift(-i))
        if i == 0:
            names += [('%s(t)' % j) for j in n_vars]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in n_vars]

    # put it all together
    agg = pd.concat(cols, axis=1)
    # print(len(agg))
    # print(len(names))
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def time_series_feature_select(df_data):
    # separate into input and output variables
    array = df_data.values
    X = array[:, 1:]
    y = array[:, 0]
    # perform feature selection
    rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), n_features_to_select=4)
    fit = rfe.fit(X, y)
    # report selected features
    print('Selected Features:')
    names = df_data.columns.values[1:]
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            print(names[i])
    # plot feature rank
    names = df_data.columns.values[0:-1]
    ticks = [i for i in range(len(names))]
    pyplot.bar(ticks, fit.ranking_)
    pyplot.xticks(ticks, names, rotation=90)
    pyplot.show()


def polt(df_data):
    """
    画变量关系图
    :param df_data:
    :return:
    """
    # colormap = plt.cm.RdBu
    plt.figure(figsize=(15, 10))
    # plt.title(u'6 hours', y=1.05, size=16)
    corr = df_data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, center=0, annot_kws={"size": 8})
    plt.xticks(rotation=90)
    plt.show()


def plot_acf_and_pacf(df_data, column, lags):
    """
    自相关性检查
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
    ax[0] = plot_acf(df_data[column], ax=ax[0], lags=lags)
    ax[1] = plot_pacf(df_data[column], ax=ax[1], lags=lags)
    fig.suptitle(f"Var_{column} autocorrlation and partial autocorrlation")
    plt.show()


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
    平稳序列检查
    """
    """ Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistics': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    result = [f'Var_{name}', signif, output["test_statistics"], output["n_lags"]]

    result.extend(np.round(np.array(list(r[4].values())), 3))

    if p_value <= signif:
        result.extend([p_value, 'Yes'])
    else:
        result.extend([p_value, 'No'])

    return result


def hurst(ts):
    lags = range(2, 100)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(log(lags), log(tau), 1)
    return poly[0] * 2.0


def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x) / 5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values), 2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


def test_unit_root(df):
    return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=3):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose and r == '9':
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# print(pd.DataFrame(adfuller_result,
#                    columns=['var name', 'Significance level', 'Test Statistics', 'n_lags', 'Critical value 1%',
#                             'Critical value 5%', 'Critical value 10%', 'P-Value', 'Stationary Series']))

# for i in sub_data.columns:#
#     plot_correlogram(sub_data[i], lags=10, title=f'{i}')
#
# plt.show()

# df_new = series_to_supervised(sub_data, [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1], [1, 2], [0, 1]], n_out=1,
#                               dropnan=True)


# Implementing VAR method
def VectorAutoRegression(df_data):
    values = df_data.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    n_train = int(0.8 * (len(values)))
    train = scaled[:n_train, :]
    test = scaled[n_train:, :]
    # fit the VAR model
    model = VAR(endog=train)
    model_fit = model.fit()
    prediction = model_fit.forecast(model_fit.y, steps=len(test))
    mse = mean_squared_error(prediction[:, 0], test[:, 0])
    r2score = r2_score(prediction[:, 0], test[:, 0])

    inv_yhat = scaler.inverse_transform(prediction)
    inv_y = scaler.inverse_transform(test)
    # print(r2score)
    # print(inv_y[:, 0], inv_yhat[:, 0])

    return mse, r2score, inv_y[:, 0], inv_yhat[:, 0]


# # Implementing Lasso Regression method
def RidgeRegression(df_data):
    # Fit Lasso
    values = df_data.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    n_train = int(0.8 * (len(values)))
    train = scaled[:n_train, :]
    test = scaled[n_train:, :]
    # split into input and outputs
    train_x, train_y = train[:, 1:], train[:, 0]
    test_x, test_y = test[:, 1:], test[:, 0]

    model = Ridge()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    r2score = r2_score(prediction, test_y)
    mse = mean_squared_error(prediction, test_y)

    inv_yhat = np.concatenate((prediction.reshape(-1, 1), test_x), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    return mse, r2score, inv_y, inv_yhat


def predicate_by_model(df_data, n_in, n_vars):
    values = df_data.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    n_train = int(0.8 * (len(values)))
    train = scaled[:n_train, :]
    test = scaled[n_train:, :]
    # split into input and outputs
    train_x, train_y = train[:, 1:], train[:, 0]
    test_x, test_y = test[:, 1:], test[:, 0]

    trainX = train_x.reshape((train_x.shape[0], n_in, n_vars))
    testX = test_x.reshape((test_x.shape[0], n_in, n_vars))

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2]), dropout=0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    stop_noimprovement = EarlyStopping(patience=10)

    history = model.fit(trainX, train_y, validation_data=(testX, test_y), epochs=100, verbose=2,
                        callbacks=[stop_noimprovement], shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    yhat = model.predict(testX)
    test_X = testX.reshape((testX.shape[0], -1))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    mse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)
    # print(inv_y, inv_yhat)
    r2score = r2_score(inv_y, inv_yhat)
    # print(r2score)
    # aa = [x for x in range(len(inv_y))]
    # plt.plot(aa, inv_y, marker='.', label="actual")
    # plt.plot(aa, inv_yhat, 'r', label="prediction")
    # plt.ylabel('Global_active_power', size=15)
    # plt.xlabel('Time step', size=15)
    # plt.legend(fontsize=15)
    # plt.show()

    # predicted = model.predict(testX)
    # testXRe = testX.reshape(testX.shape[0], testX.shape[2])
    # predicted = np.concatenate((predicted, testXRe[:, 1:]), axis=1)
    # print(predicted.shape)
    # print(predicted)
    # predicted = scaler.inverse_transform(predicted)
    # testY = test_y.reshape(len(test_y), 1)
    # testY = np.concatenate((testY, testXRe[:, 1:]), axis=1)
    # testY = scaler.inverse_transform(testY)
    # print(np.sqrt(mean_squared_error(testY[:, 0], predicted[:, 0])))
    #
    # result = pd.concat([pd.Series(inv_yhat), pd.Series(testY[:, 0])], axis=1)
    # result.columns = ['thetahat', 'theta']
    # result['diff'] = result['thetahat'] - result['theta']
    # print(result)

    return mse, r2score, inv_y, inv_yhat


n_in = 3
n_vars = 7

df = pd.read_excel("D:/工作项目/鸿玖/中泰数据 min级汇总/按分钟汇总数据/1min采样/C电极的参数汇总_1min__增加炉底温度.xlsx", sheet_name='Sheet1')
# 剔除电流异常数据
df.drop([894, 895, 896, 897, 898, 899, 900, 901, 903], inplace=True)

selected_column_names = ['功率_有功功率', '温度_炉   底 ℃_3', 'C电极长度mm(计算)']

sub_data = df[selected_column_names]

df_new = series_to_supervised(sub_data, n_in, n_out=1, dropnan=True)

# time_series_feature_select(df_new)

names = df_new.columns.values
values = df_new.values
X = values[:, :-1]  #
Y = values[:, -1]
print("Total features:")
print(f"\t{names[:-1]}")
# ['f', 'mi', 'backward', 'rfe', 'lasso']

svr_predicate_with_feature_select(X, Y,
                                  names[:-1], [names[-1]], 'linear', 'mi',
                                  test_size=0.2, selected_feature=None)
