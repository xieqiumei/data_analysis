#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:extract_product_acount.py
# @Software:PyCharm
# 参考URL
# https://www.jianshu.com/p/1b0e553e0b81
# https://developer.51cto.com/art/202010/630126.htm
# https://blog.csdn.net/qq_23981335/article/details/108648096

"""
抽取电石出炉相关数据
请注意：中泰给的数据报表里面有中文的冒号、分号等错误信息
"""

from openpyxl import load_workbook
import pandas as pd
import os
import datetime


def not_none(s):
    """
    推导式+filter过滤list中的元素
    :param s:
    :return:
    """
    return s != 'None'


def flat(a: list):
    """
    a = ['a', 'b', 'c']
    m = ['d', 'f', a[1:3]]
    g = ['d', 'f', 'b', 'c']
    :param a:
    :return:
    """
    for _ in a:
        if isinstance(_, list):
            yield from flat(_)
        else:
            yield _


def get_dianshi_output(sheet_, datas, date_str):
    """
     获得电石产量的有关数据
    :param sheet_:
    :return:
    """

    time = sheet_['D10:BP33']  # 获得电石产量和电极压放量的数据域范围

    for row in time:
        tmp = list(filter(not_none, [str(cl.value) for cl in row]))  # None值过滤掉
        tmp = [s for s in tmp if s != '']  # 除了过滤None还得过滤空行
        if len(tmp) == 0:
            continue
        if tmp[1] == '——':  # 当前周期内没有电石出炉，需要将当班时间作为该行record的时间
            s_time = datetime.datetime.strptime(f'{date_str} {tmp[0]}:00', "%Y-%m-%d %H:%M:%S")
            # 时间加0.5小时
            offset = datetime.timedelta(hours=0.5)
            re_date = (s_time + offset).strftime('%Y-%m-%d %H:%M:%S')
            datas.append([_ for _ in flat([s_time.strftime('%Y-%m-%d %H:%M:%S'), re_date, '0', tmp[2:]])])
        else:
            s_e = tmp[1].split('-')
            s_time = datetime.datetime.strptime(f'{date_str} {s_e[0]}:00', "%Y-%m-%d %H:%M:%S")
            if len(s_e) == 2:
                re_date = datetime.datetime.strptime(f'{date_str} {s_e[1]}:00', "%Y-%m-%d %H:%M:%S")
            else:
                offset = datetime.timedelta(hours=0.5)
                re_date = s_time + offset
            datas.append([_ for _ in
                          flat(
                              [s_time.strftime('%Y-%m-%d %H:%M:%S'), re_date.strftime('%Y-%m-%d %H:%M:%S'), '1',
                               tmp[2:]])])


def get_consumption_count(sheet_, datas, date_str):
    # labels = ['日期', '总锅数', '日电耗（度）', '日产量（吨）', '单电耗（度/吨）', '电极糊消耗（吨）']
    amount_total = ''.join(set([str(cell[0].value) for cell in sheet_['J34:K34']]))
    power_total = ''.join(set([str(cell[0].value) for cell in sheet_['O34:S34']]))
    output_total = ''.join(set([str(cell[0].value) for cell in sheet_['AA34:AE34']]))
    power_single = ''.join(set([str(cell[0].value) for cell in sheet_['AK34:AS34']]))
    electrode_paste = ''.join(set([str(cell[0].value) for cell in sheet_['BA34:BE34']]))
    datas.append([date_str, amount_total, power_total, output_total, power_single, electrode_paste])


def get_consumption_by_shifts_1(rows, date_str, datas, shifts):
    """
    辅助方法
    :param sheet_:
    :param date_str:
    :param datas:
    :param shifts:
    :return:
    """
    row1 = list(filter(not_none, [str(cl.value) for cl in rows[0]]))
    result = [date_str, shifts, row1[0], row1[3], row1[5], row1[7], row1[9]]
    row2 = list(filter(not_none, [str(cl.value) for cl in rows[1]]))
    result.extend([row2[1], row2[3], row2[5], row2[7], row2[10]])
    row3 = list(filter(not_none, [str(cl.value) for cl in rows[2]]))
    result.extend([row3[1], row3[3], row3[5], row3[7]])
    datas.append(result)


def get_consumption_by_shifts(sheet_, datas, date_str):
    # labels = ['日期', '班组', '电表度', '锅数', '电石产量（吨）', '总电耗（度）', '电极糊消耗（吨）', '电极压放1#（mm）', '电极压放2#（mm）',
    #           '电极压放3#（mm）', '发气量（L/Kg）', '发气量测量位置', '塌料次数1#', '塌料次数2#', '塌料次数3#', '单电耗（度/吨）']
    rows = sheet_['I43:Y45']
    get_consumption_by_shifts_1(rows, date_str, datas, '早丙班')
    rows = sheet_['AC43:AU45']
    get_consumption_by_shifts_1(rows, date_str, datas, '中乙班')
    rows = sheet_['AY43:BP45']
    get_consumption_by_shifts_1(rows, date_str, datas, '晚甲班')


def get_manual_measurement_length(rows, datas, date_str, type='电极长度'):
    """
    解析人工测量长度的数据，包括电极工作长度和电极糊长度
    :param rows:
    :param datas:
    :return:
    """
    for row in rows:
        tmp = list(filter(not_none, [str(cl.value) for cl in row]))  # None值过滤掉
        if len(tmp) == 0 or tmp[0] == '':
            continue

        s_e = tmp[0].split('-')
        new_time = f'{date_str} {s_e[0].strip()}'
        if len(s_e[0].split(":")) == 2:  # 时间格式不统一
            new_time += ':00'

        s_time = datetime.datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")
        if len(s_e) == 2:
            new_time = f'{date_str} {s_e[1].strip()}'
            if len(s_e[1].split(":")) == 2:  # 时间格式不统一
                new_time += ':00'
            re_date = datetime.datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")
        else:
            re_date = None
        if type == '电极糊高度':
            datas.append([_ for _ in
                          flat(
                              [s_time.strftime('%Y-%m-%d %H:%M:%S'),
                               tmp[1:]])])

        else:
            datas.append([_ for _ in
                          flat(
                              [s_time.strftime('%Y-%m-%d %H:%M:%S'), re_date.strftime('%Y-%m-%d %H:%M:%S'),
                               tmp[1:]])])


def get_manual_electrode_length(sheet_, datas, date_str):
    """
     读取人工测量的电极长度
    :param sheet_:
    :param date_str:
    :return:
    """
    rows = sheet_['D38:N40']
    get_manual_measurement_length(rows, datas, date_str)

    rows = sheet_['AA38:AG40']
    get_manual_measurement_length(rows, datas, date_str)

    rows = sheet_['AW38:BC40']
    get_manual_measurement_length(rows, datas, date_str)


def get_manual_electrode_paste_length(sheet_, datas, date_str):
    rows = sheet_['O38:Y41']
    get_manual_measurement_length(rows, datas, date_str, type='电极糊高度')

    rows = sheet_['AH38:AU41']
    get_manual_measurement_length(rows, datas, date_str, type='电极糊高度')

    rows = sheet_['BD38:BP41']
    get_manual_measurement_length(rows, datas, date_str, type='电极糊高度')


def process(data_count_type, dir_name, output_file):
    """
    :return:
    """
    # 遍历电石炉17#的所有文件
    file_list = os.listdir(dir_name)  # 列出文件夹下所有的目录与文件
    datas = []
    for i in range(0, len(file_list)):
        if file_list[i].startswith('18'):  # 电石炉18暂时不处理
            continue
        path = os.path.join(dir_name, file_list[i])
        print(f"当前正在处理文件{file_list[i]}！！！！")
        workbook = load_workbook(filename=path, read_only=False)
        sheet = workbook.active  # 获取当前页
        date = sheet['BL2:BP2']  # 获得报表日期，比如2020-12-10
        date_str = ''.join(set([str(cell[0].value) for cell in date]))  # 用set简易去重后用,连接，填word表用
        if data_count_type == '电石出炉数据':
            get_dianshi_output(sheet, datas, date_str)
        elif data_count_type == '电石产量和电极消耗日统计':
            get_consumption_count(sheet, datas, date_str)
        elif data_count_type == '电石产量和电极消耗班组统计':
            get_consumption_by_shifts(sheet, datas, date_str)
        elif data_count_type == '电极工作长度测量':
            get_manual_electrode_length(sheet, datas, date_str)
        elif data_count_type == '电极糊高度测量':
            get_manual_electrode_paste_length(sheet, datas, date_str)
        else:
            continue
    for e in datas:
        print(e)
    df = pd.DataFrame(datas, columns=label_mapping[data_count_type])
    df.to_csv(f'{output_file}{data_count_type}.csv', index=False)


# 输出文件的字段
label_mapping = {
    '电石出炉数据': ['出炉开始时间', '出炉结束时间', '电石是否出炉', '炉眼', '锅数', '压放量1#mm', '压放量2#mm', '压放量3#mm', '炉壁℃', '空气压力', '冷却水压力',
               '冷却水进水温度℃',
               '冷却水回水温度℃'],
    '电石产量和电极消耗日统计': ['日期', '总锅数', '日电耗（度）', '日产量（吨）', '单电耗（度/吨）', '电极糊消耗（吨）'],
    '电石产量和电极消耗班组统计': ['日期', '班组', '电表度', '锅数', '电石产量（吨）', '总电耗（度）', '电极糊消耗（吨）', '电极压放1#（mm）', '电极压放2#（mm）',
                      '电极压放3#（mm）', '发气量（L/Kg）', '发气量测量位置', '塌料次数1#', '塌料次数2#', '塌料次数3#', '单电耗（度/吨）'],
    '电极工作长度测量': ['测量开始时间', '测量结束时间', '电极测量长度_1#（mm）', '电极测量长度_2#（mm）', '电极测量长度_3#（mm）'],
    '电极糊高度测量': ['测量开始时间', '电极糊测量高度_1#（mm）', '电极糊测量高度_2#（mm）', '电极糊测量高度_3#（mm）']
}

# 调用入口
input_dir_name = 'D:/BaiduNetdiskDownload/出炉1至15/17/'
output_dir_file = 'D:/出炉1至15/'
for k in label_mapping.keys():
    print(f"当前正在汇总的数据类型为：{k}！！！！")
    process(k, input_dir_name, output_dir_file)
