#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:outliers_process.py
# @Software:PyCharm
"""
outliers can adversely affect the training process of a machine learning algorithm,
resulting in a loss of accuracy.
https://www.pluralsight.com/guides/cleaning-up-data-from-outliers
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def see_the_description_variables(df_data):
    """
    See all the variables' records(查看是否有缺失值)
    :param df_data:
    :return:
    """
    # print(df_data.shape)
    # print(df_data.info())
    print(df_data.describe())


def color_negative_red(value, min_d, max_d):
    """
    Colors elements in a dateframe
    green if positive and red if
    negative. Does not color NaN
    values.
    """

    if value < min_d:
        color = 'red'
    elif value > max_d:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color


def identify_outliers_by_iqr(df_data):
    """
    The interquartile range (IQR) is a measure of statistical dispersion and is calculated
    as the difference between the 75th and 25th percentiles. It is represented by the formula
    IQR = Q3 − Q1.
    Points where the values are 'True' represent the presence of the outlier.
    :param df_data:
    :return:
    """
    q1 = df_data.quantile(0.1)
    q3 = df_data.quantile(0.9)
    iqr = q3 - q1
    min_d = q1 - 1.5 * iqr
    max_d = q3 + 1.5 * iqr
    # df = (df_data < (q1 - 1.5 * iqr)) | (df_data > (q3 + 1.5 * iqr))

    df_data.style.applymap(color_negative_red(min_d, max_d), subset=df_data.columns)
    df_data.to_csv("D:/sheet10.csv")
    print(df_data)


def identify_outliers_by_skewness(df_data):
    """
    Several machine learning algorithms make the assumption that the data follow a normal (or Gaussian)
    distribution. This is easy to check with the skewness value, which explains the extent to which the
    data is normally distributed. Ideally, the skewness value should be between -1 and +1, and any major
    deviation from this range indicates the presence of extreme values.
    For example,the skewness value of 6.5 shows that the variable 'Income' has a right-skewed distribution,
    indicating the presence of extreme higher values. The maximum 'Income' value of USD 108,000 proves this point.
    :param df_data:
    :return:
    """
    # print(df_data[column].skew())
    # print(df_data[column].describe())

    print(df_data.skew())


def identify_outliers_by_visualization(df_data, column, target_column):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(222)
    plt.boxplot(df_data[column])
    plt.title("Histogram")

    plt.subplot(221)
    df_data[column].hist()
    plt.title("Box Plot")

    ax = fig.add_subplot(2, 1, 2)
    ax.scatter(df_data[column], df_data[target_column])
    ax.set_xlabel(f'Independent Variable {column}')
    ax.set_ylabel(f'Dependent Variable {target_column}')
    plt.title("Scatterplot")
    fig.suptitle(f'Outliers Visualization Of Independent Various {column}')
    plt.show()


def outlier_treatment_by_flooring_capping(df_data, column):
    """
    Quantile-based Flooring and Capping:
    we will do the flooring (e.g., the 10th percentile) for the lower values
    and capping (e.g., the 90th percentile) for the higher values.
    :param df_data:
    :param column:
    :return:
    """
    flooring = df_data[column].quantile(0.10)
    capping = df_data[column].quantile(0.90)
    df_data[column] = np.where(df_data[column] < flooring, flooring, df_data[column])
    df_data[column] = np.where(df_data[column] > capping, capping, df_data[column])
    print(df_data[column].skew())


def outlier_treatment_by_trimming(df_data, column):
    """
    In this method, we completely remove data points that are outliers.
    Consider the 'Age' variable, which had a minimum value of 0 and a maximum value of 200.
    """
    index = df_data[(df_data[column] >= 100) | (df_data[column] <= 18)].index
    df_data.drop(index, inplace=True)
    print(df_data[column].describe())


def outlier_treatment_by_iqr(df_data):
    """
    This shows that for our data, a lot of records get deleted if we use the IQR method.
    :param df_data:
    :return:
    """
    q1 = df_data.quantile(0.25)
    q3 = df_data.quantile(0.75)
    iqr = q3 - q1
    df_out = df_data[~((df_data < (q1 - 1.5 * iqr)) | (df_data > (q3 + 1.5 * iqr))).any(axis=1)]
    print(df_out.shape)


def outlier_treatment_by_log(df_data):
    df_data["Log_Loanamt"] = df_data["Loan_amount"].map(lambda i: np.log(i) if i > 0 else 0)
    print(df_data['Loan_amount'].skew())
    print(df_data['Log_Loanamt'].skew())


def outlier_treatment_by_median(df_data):
    print(np.min(df_data['J'].values))
    for column in df_data.columns.tolist():
        if np.abs(df_data[column].skew()) > 1:
            # median = df_data[column].quantile(0.5)
            # _95percentile = df_data[column].quantile(0.95)
            # # print(median, _95percentile)
            # df[column] = np.where(df[column] >= _95percentile, median, df[column])
            flooring = df_data[column].quantile(0.10)
            capping = df_data[column].quantile(0.90)
            df_data[column] = np.where(df_data[column] < flooring, flooring, df_data[column])
            df_data[column] = np.where(df_data[column] > capping, capping, df_data[column])

            # print(df[column].describe())
    print(df_data.skew())
    print(np.min(df_data['J'].values))


# see_the_description_variables(df)
# # identify_outliers_by_iqr(df)
# identify_outliers_by_skewness(df)

# identify_outliers_by_visualization(df, 'J', 'P')
# outlier_treatment_by_flooring_capping(df, 'J')

# outlier_treatment_by_median(df)

# Reading the data
df = pd.read_csv("D:/工作项目/新和成/sheet10.csv")

# see_the_description_variables(df)
# identify_outliers_by_skewness(df)
# identify_outliers_by_visualization(df, 'U', 'Z')
identify_outliers_by_iqr(df)
