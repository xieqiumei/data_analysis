#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:preliminary_analysis.py
# @Software:PyCharm
"""
相关性分析的重要性：
1.相关性可以帮助从一个属性预测另一个(伟大的方式，填补缺失值)；
2.相关性(有时)可以表示因果关系的存在；
3.相关性被用作许多建模技术的基本量；
4.微弱或高度正相关的特征可以是0.5或0.7；如果存在强而完全的正相关，则用0.9或1的相关分值表示结果。
5.决策树和提升树算法不受多重共线性的影响，其他算法受多重共线性的影响
6.Spearman相关系数度量非线性关系（scipy.stats.spearmanr(X,Y)），Pearson相关数度量线性关系
"""
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_univariate_from_figure(data_x1, data_x2):
    """
    # 图示初判
    # （1）变量之间的线性相关性
    :param data_x1:
    :param data_x2:
    :return:
    """
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(data_x1, data_x2)
    plt.grid()


def get_multivariable_from_figure(file_name):
    """
    # 图示初判
    # （2）散点图矩阵初判多变量间关系
    :param: file_name
    :return:None
    """
    data = pd.read_csv(file_name)
    pd.plotting.scatter_matrix(data, figsize=(8, 8),  # 注意Pandas中的用法与之前不同
                               diagonal='kde',
                               alpha=0.8,
                               range_padding=0.1)
    plt.show()


def get_correlation_heatmap_for_group(df1, df2):
    """
       画分组的变量相关性的热力图
       :return:
       """
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)

    corr = df1.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True, annot_kws={"size": 8})
    plt.xticks(rotation=90)
    ax1.set_xticklabels(df1.columns)
    ax1.set_yticklabels(df1.columns)
    plt.title("group1")

    ax2 = fig.add_subplot(1, 2, 2)
    corr2 = df2.corr()
    sns.heatmap(corr2, cmap='coolwarm', vmin=-1, vmax=1, annot=True)
    plt.xticks(rotation=90)
    ax2.set_xticklabels(df2.columns)
    ax2.set_yticklabels(df2.columns)
    plt.title("group2")
    fig.suptitle('Correlation matrix of XXX data')
    plt.show()


def get_correlation_heatmap(df_data):
    """
       画变量相关性的热力图
       :return:
       """
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    corr = df_data.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True, center=0, annot_kws={"size": 8})
    plt.xticks(rotation=90)
    ax1.set_xticklabels(df_data.columns)
    ax1.set_yticklabels(df_data.columns)

    fig.suptitle('Correlation matrix of XXX data')
    plt.show()


def show_statistics(df_data):
    """
    显示max,min等统计信息
    :return:
    """
    print(df_data.describe().T)


def get_vif_factor(df_data):
    """
    变量多重共线性分析:
    一般VIF值越大，我们认为共线性越强。在实际的使用过程中，一般认为，如果最大的VIF超过10，则变量间存在着严重的共线性。
    解决多种相关性的方法:
    1)向后消除法(Backward elimination)：每次循环，遍历当前还没有剔除的变量，依次计算对应的 VIF，
    再去除最差的那个变量（也就是VIF值最大的变量），一直循环，直至变量数目少于预期个数或者所有的变量VIF值都小于VIF阈值。
    2)PCA降维：PCA降维后，所有提取的主成分间两两独立，所以不会再有共线性。
    :return:
    """
    x = StandardScaler().fit_transform(df_data)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    vif['columns'] = df_data.columns
    print(vif)


type = 'B'
raw_data = pd.read_excel(f"D:/工作项目/鸿玖/电石炉/采样/{type}电极的参数汇总_1min_0.xlsx", sheet_name='Sheet1')
raw_data = raw_data[
    ['B电极长度mm(计算)', '温度_炉   底 ℃_2',
     '功率_电极功率_Pb', 'N2流量（每两台电石炉?', '三楼2co报警', '四楼2co报警', '氢含量报警', '二楼co报警'
     ]]
# get_correlation_heatmap(raw_data)
get_vif_factor(raw_data)

show_statistics(raw_data)
