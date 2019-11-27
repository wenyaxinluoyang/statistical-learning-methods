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
# lambda_value 为 0的时候是极大似然估计，为1时是拉普拉斯平滑
def proba_of_class(y_list, set_y, lambda_value):
    total = len(y_list)
    count_of_class = {key:0 for key in set_y}
    for y in y_list:
        count_of_class[y] = count_of_class[y] + 1
    probability_of_class = {}
    for value, count in count_of_class.items():
        probability_of_class[value] = (count+lambda_value)/(total+len(set_y)*lambda_value)
    return count_of_class, probability_of_class


# 计算条件概率, 在y=ck时，特征向量的第i个特征有j个取值
# 计算先验概率和条件概率
# set_x 每维特征的取值集合
def condition_proba(df, set_x, set_y, lambda_value):
    # 在分类为c的条件下，计算第j个特征取每个值的概率
    count_of_class, probability_of_class = proba_of_class(df['Y'].values.tolist(), set_y, lambda_value)
    columns = df.columns.values.tolist()
    columns.remove('Y')
    condition_proba_dict = {y: [] for y in set_y}
    for class_name in set_y:
        # 把分类为class_name的特征向量聚合在一起
        data = df[df.Y == class_name]
        array = []
        for col_id, column in enumerate(columns):
            proba = dict()
            value_count = {value:0 for value in set_x[col_id]}
            values = data[column].value_counts()
            values = values.reset_index()
            values['index'] = values['index'].astype(type(set_x[col_id][0]))
            total = 0
            for _, row in values.iterrows():
                value = row['index']
                count = row[column]
                total = total + count
                value_count[value] = count
            for value, count in value_count.items():
                proba[value] = (count + lambda_value)/(total+len(set_x[col_id])*lambda_value)
            array.append(proba)
        condition_proba_dict[class_name] = array
    return probability_of_class, condition_proba_dict


# 预测
def predict(x, set_x, probability_of_class, condition_proba_dict):
    class_list = list(probability_of_class.keys())
    target = None
    max_proba = 0
    for c in class_list:
        mul = 1
        for i, value in enumerate(x):
            proba = condition_proba_dict[c][i]
            mul = mul * proba[value]
        mul = mul * probability_of_class[c]
        print(mul)
        if mul > max_proba:
            max_proba = mul
            target = c
    return max_proba, target


# 书本例4.1数据
def get_data():
    df = pd.DataFrame()
    df['X1'] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    df['X2'] = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    df['Y'] = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    return df



if __name__ == '__main__':
    df = get_data()
    set_x = [(1,2,3), ('S', 'M', 'L')]
    set_y = [-1, 1]
    lambda_value = 1
    probability_of_class, condition_proba_dict =  condition_proba(df, set_x, set_y, lambda_value)
    for c in set_y:
        print(f'P(Y={c}) =', probability_of_class[c])
    for c in set_y:
        proba_list = condition_proba_dict[c]
        for index,proba in enumerate(proba_list):
            for key,value in proba.items():
                print(f'在分类{c}下, 第{index+1}个特征，取值为{key}的概率为{value}')
                print('*'*20)
        print('-'*20)
    test_x = [2, 'S']
    print(predict(test_x, set_x, probability_of_class, condition_proba_dict))
    #condition_proba(x_list, y_list)
    # count_of_class, probability_of_class = proba_of_class(y_list)
    # print(count_of_class, probability_of_class)





















