#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:check_normality.py
# @Software:PyCharm

import scipy.stats as stats

"""
数据正态分布检测
"""


def check_normality(data):
    # 20<样本数<50用normal test算法检验正态分布性
    size = len(data)
    if 20 < size < 50:
        p_value = stats.normaltest(data)[1]
        if p_value < 0.05:
            print("use normaltest")
            print("data are not normal distributed")
            return False
        else:
            print("use normaltest")
            print("data are normal distributed")
            return True

    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if size < 50:
        p_value = stats.shapiro(data)[1]
        if p_value < 0.05:
            print("use shapiro")
            print("data are not normal distributed")
            return False
        else:
            print("use shapiro")
            print("data are  normal distributed")
            return True

    p_value = stats.kstest(data, 'norm')[1]
    if p_value < 0.05:
        print("use kstest:")
        print("data are not normal distributed")
        return False
    else:
        print("use kstest:")
        print("data are normal distributed")
        return True

# # 对所有样本组进行正态性检验
# def NormalTest(list_groups):
#     for group in list_groups:
#         # 正态性检验
#         status = check_normality(group)
#         if status is False:
#             return False
#
#
# NormalTest(list_groups)
