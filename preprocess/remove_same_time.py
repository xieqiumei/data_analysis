#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:count_field.py
# @Software:PyCharm
import pandas as pd
import os
import numpy as np

pd.set_option('display.max_rows', None)  # 显示所有的行信息

# 含以下编号的文件含有重复时间
A = [1, 2, 3, 4, 5, 6, 7, 14, 17, 23, 37, 38, 39, 46, 47, 48, 53, 54, 55, 56, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
     69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 85, 86, 87, 88, 95, 97, 98, 100, 102, 105, 111, 112, 113
    , 116, 117, 118]
# 以下编号的文件不含重复时间，但是转成一样格式的excel，便于程序统一处理
REST_A = [12, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
    , 34, 35, 36, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 57, 58, 79, 82, 83
    , 84, 89, 90, 91, 92, 93, 94, 96, 104, 106, 107, 108, 109, 110, 114, 115]


def convert_timedelta(duration):
    """"
    辅助方法，将程序内置的时间差数据结构转成中文表达形式
    """
    days, seconds = duration.days, duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    s = ""
    if days > 0:
        s = f"{days}天"
    if hours > 0:
        s += f"{hours}小时"

    if minutes > 0:
        s += f"{minutes}分"

    if seconds > 0:
        s += f"{seconds}秒"

    if s is "":
        return "0秒"
    return s


def count_feature_frequency(input_dir, output_dir):
    """
    了解数据每个特征的采样频率
    :return:
    """
    stringcols = ['文件名', '特征', '数据长度', 'interval_max', 'interval_min', 'interval_quantile0.25',
                  'interval_quantile0.5', 'interval_quantile0.75']
    res = list()
    filelist = os.listdir(input_dir)  # 列出文件夹下所有的目录与文件
    none_file = []
    for i in range(0, len(filelist)):
        path = os.path.join(input_dir, filelist[i])
        if os.path.isfile(path):
            data = pd.read_excel(path, sheet_name='Sheet1')
            print(f"正在处理文件{filelist[i]}")
            if len(data) > 0:
                tm_1 = pd.to_datetime(data['时间'])  # 将字符串格式转成时间表达式
                tmp = tm_1.shift(periods=-1, axis=0)[:-1] - tm_1[:-1]  # 相邻两次采样间隔时间

                res.append(
                    [filelist[i], data['点位描述'][0], len(data), convert_timedelta(max(tmp)),
                     convert_timedelta(min(tmp)),

                     convert_timedelta(tmp.quantile(0.25)), convert_timedelta(tmp.quantile(0.5)),
                     convert_timedelta(tmp.quantile(0.75))])
            else:
                none_file.append(filelist[i])

    df_res = pd.DataFrame(res, columns=stringcols)
    df_res.to_csv(f"{output_dir}数据特征采集频率统计.csv", index=False)


def mark_same_position(mark, data):
    """
    辅助方法，相同时间删除策略
    :param mark:
    :param data:
    :return:
    """
    start = int(mark[0])
    time_col = pd.to_datetime(data['时间'])
    index = []
    for i, e in enumerate(mark):
        e_end = mark[i] + 1

        if time_col[e] == time_col[e_end] and (i == len(mark) - 1 or mark[i + 1] - e == 2):
            if i == len(mark) - 1:
                if (e_end - start) > 1:
                    index.extend(np.arange(start, mark[-1] + 2))
                if (e_end - start) == 1:
                    index.extend(np.arange(start + 1, mark[-1] + 2))
            else:
                continue
        else:
            if (e_end - start) > 1:
                index.extend(np.arange(start, e_end + 1))
                start = mark[i + 1]
            if (e_end - start) == 1:
                index.extend(np.arange(start + 1, e_end + 1))
                start = mark[i + 1]
            else:
                start = mark[i + 1]
                continue
        data.drop(data.index[index], inplace=True)
        # 如果index只有一个元素，用 data.drop(index, inplace=True)也可以，多元素则不行
        if len(data) == len(np.unique(data['时间'])):
            return data
        else:
            return None


def remove_same_time(input_dir, output_dir):
    """
    1.如果方特征的最小间隔时间为0，这说明该文件中有重复时间出现
    2.判断出现重复时间的区域
    2.分两种策略删除重复时间
    :return:
    """
    filelist = os.listdir(input_dir)  # 列出文件夹下所有的目录与文件
    none_file = []
    for i in range(0, len(filelist)):
        name = filelist[i].replace(".xls", '')
        index = int(name.split('_')[-1])
        if index in A:  # 含有重复的时间的文件才需要处理
            print(f"当前文件{filelist[i]}中！！！！！！！！！！！！！！")
            path = os.path.join(input_dir, filelist[i])
            data = pd.read_excel(path, sheet_name='sheet1')
            tm_1 = pd.to_datetime(data['时间'])
            tmp = tm_1.shift(periods=-1, axis=0) - tm_1
            tmp = np.array(tmp / np.timedelta64(1, 's'))
            mark = np.where(tmp == 0.0)[0]  # 如果方特征的最小间隔时间为0，这说明该文件中有重复时间出现
            #  找出其所在位置
            data1 = mark_same_position(mark, data)  # 相同时间删除策略

            if data1 is None:
                print(f"文件{filelist[i]}需要人工校验！")
                none_file.append(filelist[i])  # 与设定的删除规则不相符合，需要人工校验，可以输出到文件
            else:
                # data1.to_excel(f"D:/BaiduNetdiskDownload/KD_DS17#18#_12.10-12.25_tagInfo/17#时间去重/{name}.xls",
                #                index=False)

                data1.to_excel(f"{output_dir}{name}.xls",
                               index=False)


def modify_rest_file(input_dir, output_dir):
    """"
    不做任何处理，仅是将剩下的文件转成一样的excel格式
    """
    filelist = os.listdir(input_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(filelist)):
        name = filelist[i].replace(".xls", '')
        index = int(name.split('_')[-1])
        if index in REST_A:
            print(f"当前文件{filelist[i]}中！！！！！！！！！！！！！！")
            path = os.path.join(input_dir, filelist[i])
            data = pd.read_excel(path, sheet_name='sheet1')
            # data.to_excel(f"D:/BaiduNetdiskDownload/KD_DS17#18#_12.10-12.25_tagInfo/17#时间去重/{name}.xls",
            #               index=False)
            data.to_excel(f"{output_dir}{name}.xls",
                          index=False)


# 程序入口
input_file_dir = 'D:/BaiduNetdiskDownload/KD_DS17#18#_12.10-12.25_tagInfo/KD_DS17#18#_12.10-12.25_tagInfo/'
output_dir = "D:/工作项目/鸿玖/电石炉/"
count_feature_frequency(input_file_dir, output_dir)  # input_file_dir根据去重和不去重，指向不同的文件夹

# input_file_dir=XXX
# output_dir=XXX
# remove_same_time(input_dir, output_dir)
# modify_rest_file(input_dir, output_dir)
