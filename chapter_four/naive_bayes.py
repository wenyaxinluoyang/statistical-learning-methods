#!/usr/bin/env python

import pandas as pd
import numpy as np
'''
朴素贝叶斯算法

P(A)是 A 的先验概率，之所以称为“先验”是因为它不考虑任何 B 方面的因素。
P(A|B)是已知 B 发生后 A 的条件概率，也由于得自 B 的取值而被称作 A 的后验概率。
P(B|A)是已知 A 发生后 B 的条件概率，也由于得自 A 的取值而被称作 B 的后验概率。
P(B)是 B 的先验概率，也作标淮化常量（normalizing constant）
'''

# probability
# 计算先验概率
def proba_of_class(y_list):
    total = len(y_list)
    class_unique = list(set(y_list))
    count_of_class = {key:0 for key in class_unique}
    for y in y_list:
        count_of_class[y] = count_of_class[y] + 1
    probability_of_class = {}
    for key, value in count_of_class.items():
        probability_of_class[key] = value/total
    return count_of_class, probability_of_class


# 计算条件概率, 在y=ck时，特征向量的第i个特征有j个取值
# 计算先验概率和条件概率
def condition_proba(x_list, y_list):
    # 在分类为c的条件下，计算第j个特征取每个值的概率
    df = pd.DataFrame()
    length = x_list[0].shape[0]
    array = [[] for i in range(length)]
    for x in x_list:
        for i in range(length):
            array[i].append(x[i][0]) # 第i个特征
    for i in range(length):
        column_name = 'X' + str(i)
        df[column_name] = array[i]
    df['Y'] = y_list
    count_of_class, probability_of_class = proba_of_class(y_list)
    columns = df.columns.values.tolist()
    columns.remove('Y')
    condition_proba_dict = {y: [] for y in list(set(y_list))}
    # 把相同分类的特征向量聚合在一起
    for index, data in df.groupby(['Y']):
        array = []
        for column in columns:
            proba = dict()
            value_count = dict()
            values = data[column].value_counts()
            values = values.reset_index()
            total = 0
            for _, row in values.iterrows():
                value = row['index']
                count = row[column]
                total = total + count
                value_count[value] = count
            for key,value in value_count.items():
                proba[key] = value/total
            array.append(proba)
        condition_proba_dict[index] = array
    return probability_of_class, condition_proba_dict


# 预测
def predict(x, probability_of_class, condition_proba_dict):
    class_list = list(probability_of_class.keys())
    target = None
    max_proba = 0
    for c in class_list:
        mul = 1
        for i in range(x.shape[0]):
            proba = condition_proba_dict[c][i]
            value = x[i][0]
            mul = mul * proba[value]
        mul = mul * probability_of_class[c]
        print(mul)
        if mul > max_proba:
            max_proba = mul
            target = c
    return max_proba, target


# 书本例4.1数据
def get_data():
    size = (2,1)
    x1 = np.array([1, 'S']).reshape(size)
    x2 = np.array([1, 'M']).reshape(size)
    x3 = np.array([1, 'M']).reshape(size)
    x4 = np.array([1, 'S']).reshape(size)
    x5 = np.array([1, 'S']).reshape(size)
    x6 = np.array([2, 'S']).reshape(size)
    x7 = np.array([2, 'M']).reshape(size)
    x8 = np.array([2, 'M']).reshape(size)
    x9 = np.array([2, 'L']).reshape(size)
    x10 = np.array([2, 'L']).reshape(size)
    x11 = np.array([3, 'L']).reshape(size)
    x12 = np.array([3, 'M']).reshape(size)
    x13 = np.array([3, 'M']).reshape(size)
    x14 = np.array([3, 'L']).reshape(size)
    x15 = np.array([3, 'L']).reshape(size)
    x_list = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15]
    y_list = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    return x_list, y_list



if __name__ == '__main__':
    x_list, y_list = get_data()
    probability_of_class, condition_proba_dict =  condition_proba(x_list, y_list)
    class_list = list(probability_of_class.keys())
    for c in class_list:
        print(f'P(Y={c}) =', probability_of_class[c])
    for c in class_list:
        proba_list = condition_proba_dict[c]
        for index,proba in enumerate(proba_list):
            for key,value in proba.items():
                print(f'在分类{c}下, 第{index}个特征，取值为{key}的概率为{value}')
                print('*'*20)
        print('-'*20)
    test_x = np.array([2,'S']).reshape((2,1))
    print(predict(test_x, probability_of_class, condition_proba_dict))
    #condition_proba(x_list, y_list)
    # count_of_class, probability_of_class = proba_of_class(y_list)
    # print(count_of_class, probability_of_class)





















