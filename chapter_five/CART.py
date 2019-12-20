#!/usr/bin/env python

import pickle

'''
分类与回归树 (classification and regression tree)
是二叉树
回归树： 用平方误差最小化准则
分类树： 用基尼指数最小化准则
基尼指数：
sum(pk * (1-pk) = 1 - sum(pk*pk)
某个样本点被错误分类的概率.
样本点属于第k类的概率为pk，它被分错的概率为(1-pk)
对于二类分类问题，若样本点属于第1个类的概率为p
Gini = p*(1-p) + (1-p)*[1-(1-p)] = p*(1-p) + p*(1-p) = 2*p*(1-p)
'''

import math
import pandas as pd
import numpy as np

INF = float('inf')

class Node:
    def __init__(self, fea_name, split_value):
        self.fea_name = fea_name
        self.split_value = split_value
        self.left = None
        self.right = None
        self.output = None

# 平方误差
def square(value1, value2):
    return math.pow((value1-value2),2 )

# 构建回归树, 特征是连续型变量，输出也是连续型变量
def build_regression_tree(df, target, thresold):
    if df is None: return None
    if df.empty: return None
    # 样本量小于某个数的时候停止分裂
    if df.shape[0]<=thresold: return None
    columns = df.columns.values.tolist()
    columns.remove(target)
    if columns is None: return None
    min_square_loss = INF # 最小平方损失
    split_col = None # 切分变量
    split_value = None # 切分值
    left_data = None # 左训练集
    right_data = None # 右训练集
    for col in columns:
        values = df[col].values.tolist()
        length = len(values)
        target_values = df[target].values.tolist()
        if len(list(set(values)))==1: continue
        for value in list(set(values)):
            left_target = [target_values[i] for i in range(length) if values[i]<=value]
            right_target = [target_values[i] for i in range(length) if values[i]>value]
            if len(left_target)==0 or len(right_target)==0: continue
            aver1 = np.mean(left_target)
            aver2 = np.mean(right_target)
            sum1 = sum2 = 0
            for y in left_target: sum1 += square(y, aver1)
            for y in right_target: sum2 += square(y, aver2)
            if sum1 + sum2 < min_square_loss:
                split_col = col
                split_value = value
                left_data = df[df[col] <= value]
                right_data = df[df[col] > value]
    if split_col is None: return None
    tree = Node(split_col, split_value)
    print(split_col, split_value)
    # 递归构建左右子树
    tree.output = np.average(df[target].values.tolist())
    tree.left = build_regression_tree(left_data, target, thresold)
    tree.right = build_regression_tree(right_data, target, thresold)
    return tree


# 构建分类树, 特征是离散的
def build_class_tree(df, set_x, target):
    if df is None: return None
    if df.empty: return None
    # 如果某个数据集中，所有分类都为同一分类，则不需再划分,因为对于任何特征，其基尼系数都是0
    target_values = df[target].values.tolist()
    if len(set(target_values)) == 1:
        node = Node('', '')
        node.output = target_values[0]
        return node
    columns = df.columns.values.tolist()
    columns.remove(target)
    min_gini = INF
    fea_name = None
    split_value = None
    left_df = None
    right_df = None
    for col in columns:
        value_of_col = set_x[col]
        values = df[col].values.tolist()
        if len(list(set(values))) == 1:
            continue
        # 枚举特征col可取的每一个值
        for value in value_of_col:
            df1 = df[df[col] == value]
            df2 = df[df[col] != value]
            fm1 = df1.shape[0]*df1.shape[0]
            fm2 = df2.shape[0]*df2.shape[0]
            temp1 = df1[target].value_counts().reset_index()
            temp2 = df2[target].value_counts().reset_index()
            sum = 0
            for _, count in zip(temp1['index'].values.tolist(), temp1[target].values.tolist()):
                sum += (count*count)/fm1
            gini1 = 1 - sum
            sum = 0
            for _, count in zip(temp2['index'].values.tolist(), temp2[target].values.tolist()):
                sum += (count*count)/fm2
            gini2 = 1 - sum
            gini = df1.shape[0]/df.shape[0]*gini1 + df2.shape[0]/df.shape[0]*gini2
            #print(f'特征值={col}; 取值={value}; 基尼指数={gini_df_col}')
            if gini < min_gini:
                min_gini = gini
                fea_name = col
                split_value = value
                left_df = df1
                right_df = df2
    if fea_name is None: return None
    tree = Node(fea_name, split_value)
    tree.output = target_counts = df[target].value_counts()[0]
    tree.left = build_class_tree(left_df, set_x, target)
    tree.right = build_class_tree(right_df, set_x, target)
    return tree

def get_data():
    set_x = {
        "age":('青年','中年', '老年'),
        "have_job":('是', '否'),
        "have_house":('是', '否'),
        "credit_detail":('一般', '好', '非常好')}
    set_y = ['是', '否']
    age = ['青年','青年','青年','青年','青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年']
    have_job = ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否']
    have_house = ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否']
    credit_detail = ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般']
    y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    df = pd.DataFrame()
    df['age'] = age
    df['have_job'] = have_job
    df['have_house'] = have_house
    df['credit_detail'] = credit_detail
    df['Y'] = y
    return df, set_x, set_y

def display(tree, level=1):
     if tree is not None:
        print(f'特征名:{tree.fea_name} 特征划分值:{tree.split_value} 最大分类:{tree.output} 层次:{level}')
        if tree.left is not None:
            display(tree.left, level+1)
        if tree.right is not None:
            display(tree.right, level+1)

def regression_predict(test_data, root):
    predict_target = []
    for index, row in test_data.iterrows():
        temp = root
        output = 0
        while temp is not None:
            value = row[temp.fea_name]
            output = temp.output
            if value <= temp.split_value: temp = temp.left
            else: temp = temp.right
        predict_target.append(output)
    test_data['predict_target'] = predict_target
    return test_data


if __name__ == '__main__':
    df, set_x, set_y = get_data()
    # train_data, test_data = get_data2()
    # tree = build_regression_tree(train_data, target='pm2.5', thresold=50)
    # print('build tree success')
    # f = open('./data/model.pkl', 'wb')
    # pickle.dump(tree, f)
    # display(tree)
    # f = open('./data/model.pkl', 'rb')
    # tree = pickle.load(f)
    # result = regression_predict(test_data, tree)
    # result = regression_predict(test_data, tree)
    # result.to_csv('result.csv', index=False)
    # result.to_csv('./data/result2.csv')
    #df, set_x, set_y = get_data()
    tree = build_class_tree(df, set_x, 'Y')
    display(tree)

