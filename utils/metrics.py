#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:demo.py
# @Software:PyCharm

# what explained_variance_score really is
# 1-np.cov(np.array(y_true)-np.array(y_pred))/np.cov(y_true)

# what r^2 really is
# 1-((np.array(y_true)-np.array(y_pred))**2).sum()/(4*np.array(y_true).std()**2)


import numpy as np
import statsmodels.api as sm


def get_adjusted_r_squared(x_test, y_test, r_squared):
    """
    若对两个具有不同个数自变量的回归方程进行比较时，不能简单地用R2作为评价回归方程的标准，
    还必须考虑方程所包含的自变量个数的影响，此时应用校正的决定系数（R2-adjusted）：Rc2，
    所谓“最优”回归方程是指Rc2最大者。因此在讨论多重回归的结果时，通常使用Rc2。
    相关系数要在0.7~0.5才有意义，因此，R2应大于0.5*0.5=0.25，所以有种观点认为，在直线回归中应R2大于0.3才有意义。
       adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1),n是测试数据数目，k是因变量个数
    https://www.sohu.com/a/158761950_655168
    https://stackoverflow.com/questions/42033720/python-sklearn-multiple-linear-regression-display-r-squared
    :return:
    """
    # ss_residual = np.sum((y_test - y_predicate) ** 2)
    # ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    # r_squared = 1 - (float(ss_residual)) / ss_total

    n = len(y_test)
    k = x_test.shape[1]
    # adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    return adjusted_r_squared


def get_p_values(x_ndarray, y_ndarray, column):
    """
    计算p_values和t_values
    :param x_ndarray:
    :param y_ndarray:
    :param column:
    :return:
    """
    x2 = sm.add_constant(x_ndarray)
    est = sm.OLS(y_ndarray, x2)
    est2 = est.fit()
    pvalues = est2.pvalues  # significance sign值
    return np.array(column)[np.where(pvalues[1:] > 0.05)[0]]
