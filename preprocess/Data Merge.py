#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:Data Merge.py
# @Software:PyCharm
import os
import pandas as pd

from pandas import to_datetime
import re

# 以下文件没有数据
# blank_file_for_17 = [13, 54, 55, 56, 57, 58, 59, 82, 83, 84, 106, 107, 108, 109, 110, 114, 115]

regex = re.compile("17#炉[0-9]{1,2}#料仓", re.I)  # 特征值列名重命名


def initial_file_process(df_data, resample_rule='1T'):
    """
    1.筛选出每个文件的值和时间两列
    2.列名'值'用点位描述来表示
    3.秒级数据变为分钟采样，df_data_new.resample('5T').mean()
    """
    df_data_new = df_data.iloc[:, 2:4]

    name = df_data['点位描述'][0]
    if "17#18#" in name:
        name = name.split('#')[2][1:]

    elif regex.match(name) is None:
        name = name.split('#')[1][1:]
    else:
        name = name[4:]
    df_data_new.rename(columns={"值": name}, inplace=True)  # 生成新的列名和值
    df_data_new['时间'] = to_datetime(df_data_new['时间'])  # 时间作为index,便于采样

    df_data_new.set_index(["时间"], inplace=True)

    df_data_new = df_data_new.resample(resample_rule).mean().dropna(axis=0)

    return df_data_new


def pole_depth_and_position_merge(depth_file, position_file, target_dir, pole_type, resample_rule):
    """
    将电极A、B、C的入炉深度和位置分别取交集
    A电极：KD_DS17#18#_tagInfo_54.xls和KD_DS17#18#_tagInfo_57.xls
    B电极：KD_DS17#18#_tagInfo_55.xls和KD_DS17#18#_tagInfo_58.xls
    C电极：KD_DS17#18#_tagInfo_56.xls和KD_DS17#18#_tagInfo_59.xls
    :param depth_file:
    :param position_file:
    :return:
    """
    # 读取电极入炉深度数据
    data_1 = pd.read_excel(depth_file, sheet_name='Sheet1')
    # 读取电极位置
    data_2 = pd.read_excel(position_file, sheet_name='Sheet1')

    # 预处理并采样
    data_1_new = initial_file_process(data_1, resample_rule=resample_rule)
    data_2_new = initial_file_process(data_2, resample_rule=resample_rule)

    # 电极入炉深度和位置分别取交集
    result = pd.merge(data_2_new, data_1_new, on='时间', how='inner')

    result.sort_values(by='时间', inplace=True)
    result.reset_index(inplace=True)
    new = pd.DataFrame()
    new['时间'] = result['时间']

    param = '1'
    if pole_type == 'B':
        param = '2'
    elif pole_type == 'C':
        param = '3'

    new[f'{pole_type}电极长度mm(计算)'] = result[f'电极位置{param}'] + result[f'电极入炉深度mm_{param}']
    new[f'{pole_type}电极位置mm'] = result[f'电极位置{param}']
    new.to_excel(os.path.join(target_dir, f'{pole_type}电极工作长度.xlsx'), index=False)
	
file_index_mapping = {
        '料仓料位报警': [12, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
        '料位长时间不加料报警': [4, 16, 18, 20, 24, 26, 28, 30, 32, 34, 36],
        '气体报警': [87, 97, 98, 102, 105],
        '原料参数': [49, 50, 51, 52, 93, 94, 95, 96],
        '电气参数': [46, 47, 48, 60, 61, 62, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78],
        '炉膛压力': [116, 117, 118],
        # 总的温度参数文件的角标： 112,113,88,81,42,53,85,86,1,2,80,111,43,40,44,45,3
        '温度参数A': [111, 85],
        '温度参数B': [112, 86],
        '温度参数C': [113, 88]
    }

def parameter_merge_by_group(input_dir, target_dir, data_summary_type, resample_rule):
    """
    data_summary_type：仓料位报警、料位长时间不加料报警、气体报警、原料参数、电气参数、温度参数
    先按照data_summary_type进行组内合并
    :return:
    """

    if data_summary_type is None or data_summary_type not in file_index_mapping.keys():
        return

    # sdir = "D:/BaiduNetdiskDownload/KD_DS17#18#_12.10-12.25_tagInfo/17#时间去重/"
    file_index = file_index_mapping[data_summary_type]

    data_1 = pd.read_excel(os.path.join(input_dir, f'KD_DS17#18#_tagInfo_{file_index[0]}.xls'), sheet_name='Sheet1')
    data_1 = initial_file_process(data_1, resample_rule=resample_rule)

    result = None
    filelist = os.listdir(input_dir)  # 列出文件夹下所有的目录与文件

    for i in range(1, len(filelist)):  # 从列表中第二个元素开始，依次汇总合并
        index = int(filelist[i].replace(".xls", '').split('_')[-1])
        if index in file_index:
            path = os.path.join(input_dir, filelist[i])
            if os.path.isfile(path):
                data = pd.read_excel(path, sheet_name='Sheet1')
                print(f"正在处理文件{filelist[i]}，其index={index}")
                data_new = initial_file_process(data, resample_rule=resample_rule)
                if result is None:
                    result = pd.merge(data_1, data_new, on='时间', how="inner")
                else:
                    result = pd.merge(result, data_new, on='时间', how='inner')

    result.sort_values(by='时间', inplace=True)
    result.reset_index(inplace=True)
    result.to_excel(os.path.join(target_dir, f'{data_summary_type}.xlsx'), index=False)


def processed_data_merge(target_dir, pole_name, resample_rule):
    # all_merge_file = ['仓料位报警.xlsx', '料位长时间不加料报警.xlsx', '原料参数.xlsx', '电气参数.xlsx',
    #               '温度参数A.xlsx', '温度参数B.xlsx', '温度参数C.xlsx', '炉膛压力.xlsx', '气体报警.xlsx', 'A电极工作长度.xlsx', 'B电极工作长度.xlsx',
    #               'C电极工作长度.xlsx']

    merge_file = ['原料参数.xlsx', '电气参数.xlsx', '炉膛压力.xlsx', '温度参数B.xlsx']
    data_1 = pd.read_excel(os.path.join(target_dir, f"{pole_name}电极工作长度.xlsx"), sheet_name='Sheet1')
    result = None
    for i in merge_file:
        path = os.path.join(target_dir, i)
        if os.path.isfile(path):
            data = pd.read_excel(path, sheet_name='Sheet1')
            print(f"正在处理文件{i}")
            if result is None:
                result = pd.merge(data_1, data, on='时间', how='inner')
            else:
                result = pd.merge(result, data, on='时间', how='inner')
    result.sort_values(by='时间', inplace=True)
    result.reset_index(inplace=True)
    del result['index']
    result.set_index(["时间"], inplace=True)
    result.to_excel(os.path.join(target_dir, f'{pole_name}电极的参数汇总_{resample_rule}.xlsx'))


def get_max_time_for_feature(target_dir):
    """
    仓料位报警、料位长时间不加料报警、气体报警、原料参数、电气参数、温度参数A、温度参数B、温度参数C
    查看特征的最大采样时间，根据时间将参数的时间对齐
    :return:
    """
    merge_file = ['仓料位报警.xlsx', '料位长时间不加料报警.xlsx', '原料参数.xlsx', '电气参数.xlsx',
                  '温度参数A.xlsx', '温度参数B.xlsx', '温度参数C.xlsx', '炉膛压力.xlsx', '气体报警.xlsx', 'A电极工作长度.xlsx', 'B电极工作长度.xlsx',
                  'C电极工作长度.xlsx']
    result = []
    for i in merge_file:
        path = os.path.join(target_dir, i)
        if os.path.isfile(path):
            data = pd.read_excel(path, sheet_name='Sheet1')
            print(f"正在处理文件{i}")
            data['时间'] = to_datetime(data['时间'])
            max(data['时间'])
            result.append([i, max(data['时间'])])
    for e in result:
        print(e)


source_dir = 'D:/BaiduNetdiskDownload/KD_DS17#18#_12.10-12.25_tagInfo/17#时间去重/'
target_dir = 'D:/工作项目/鸿玖/电石炉/采样'
resample_rule = '1T'  # 采样频率
# 1.0 电极工作长度计算
""""
    电极工作长度计算:
    将电极A、B、C的入炉深度和位置分别取交集
    A电极：KD_DS17#18#_tagInfo_54.xls和KD_DS17#18#_tagInfo_57.xls
    B电极：KD_DS17#18#_tagInfo_55.xls和KD_DS17#18#_tagInfo_58.xls
    C电极：KD_DS17#18#_tagInfo_56.xls和KD_DS17#18#_tagInfo_59.xls
"""
pole_mapping = {'A': [54, 57], 'B': [55, 58], 'C': [56, 59]}

for k, v in pole_mapping.items():
    pole_depth_and_position_merge(os.path.join(source_dir, f"KD_DS17#18#_tagInfo_{v[0]}.xls"),
                                  os.path.join(source_dir, f"KD_DS17#18#_tagInfo_{v[1]}.xls"), target_dir, k,
                                  resample_rule)
# 2.0 仓料位报警、料位长时间不加料报警、气体报警、原料参数、电气参数、温度参数A、温度参数B、温度参数C
#data_summary_type = '仓料位报警'
for k in file_index_mapping.keys():
	parameter_merge_by_group(source_dir, target_dir, data_summary_type, resample_rule)
# 3.0 计算每种类型参数的最大采样时间
get_max_time_for_feature(target_dir)

# 4.0 将A/B/C电极与其他的参数进行合并
for e in ['A', 'B', 'C']:
    processed_data_merge(target_dir, e, resample_rule)
