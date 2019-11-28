#!/usr/bin/env python

'决策树'
import pandas as pd
import math

class Node():

    def __init__(self):
        pass


# 信息增益
def information_gain(x_list, value_of_x, y_list, value_of_y, hy):
    return hy - condition_entropy(x_list, value_of_x, y_list, value_of_y)


# 条件熵, X给定条件下Y的条件概率分布的熵对X的数学期望
def condition_entropy(x_list, value_of_x, y_list, value_of_y):
    hx, value_count = entropy(x_list, value_of_x)
    h = 0
    temp = {x: [] for x in value_of_x}
    for x,y in zip(x_list, y_list):
        temp[x].append(y)
    total_x = len(x_list)
    for x, y_belong_x in temp.items():
        hyx, value_count_y = entropy(y_belong_x, value_of_y)
        h = h + (value_count[x]/total_x)*hyx
    return h

# 计算随机变量的熵
def entropy(value_list, value_of_x):
    value_count = {value:0 for value in value_of_x}
    total = len(value_list)
    for value in value_list:
        value_count[value] = value_count[value] + 1
    h = 0
    for value in value_of_x:
        count = value_count[value]
        p = count/total
        if count == 0: continue
        else: h = h + p*math.log(p,2)
    h = -h
    return h, value_count


# 特征的信息增益
def information_gain_of_features(df, set_x, set_y):
    columns = df.columns.values.tolist()
    columns.remove('Y')
    # 每种特征的信息增益
    fea_infor_gain = {col:0 for col in columns}
    y_list = df['Y'].values.tolist()
    hy, value_count_y = entropy(y_list, set_y)
    for col_index,col in enumerate(columns):
        fea_infor_gain[col] = information_gain(df[col].values.tolist(), set_x[col_index], y_list, set_y, hy)
    print(fea_infor_gain)
    print('Y的熵', hy)
    return hy, fea_infor_gain

def get_data():
    set_x = [('青年','中年', '老年'), ('是', '否'), ('是', '否'), ('一般', '好', '非常好')]
    set_y = ['是', '否']
    age = ['青年','青年','青年','青年','青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年']
    have_job = ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否']
    have_house = ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否']
    credit_detail = ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般']
    y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    df = pd.DataFrame()
    df['X1'] = age
    df['X2'] = have_job
    df['X3'] = have_house
    df['X4'] = credit_detail
    df['Y'] = y
    return df, set_x, set_y


if __name__ == '__main__':
    df, set_x, set_y = get_data()
    information_gain_of_features(df, set_x, set_y)
