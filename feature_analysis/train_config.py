#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:train_config.py
# @Software:PyCharm

"""
模型训练需要的参数
"""
svr_params_dict = {
    'C': [0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 5, 10],
    'gamma': [0.001, 0.01, 0.1, 0.2, 0.5, 0.8],
    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8],
    'kernel': ['rbf', 'linear', 'poly']
}

