#!/usr/bin/env python

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
def build_regression_tree(df, target):
    if df.empty: return None
    columns = df.columns.values.tolist()
    # 最小平方损失
    min_square_loss = INF
    split_col = None
    split_value = None
    left_data = None
    right_data = None
    for col in columns:
        values = df[col].values.tolist()
        y = df[target].values.tolist()
        for value in values:
            r1 = df[df[col] <= value]
            r2 = df[df[col] > value]
            aver1 = np.average(r1[target].values.tolist())
            aver2 = np.average(r2[target].values.tolist())
            sum1 = 0
            sum2 = 0
            for y in r1[target].values.tolist():
                sum1 += square(y, aver1)
            for y in r2[target].values.tolist():
                sum2 += square(y, aver2)
            total = sum1 + sum2
            if total < min_square_loss:
                split_col = col
                split_value = value
                left_data = r1
                right_data = r2
    tree = Node(split_col, split_value)
    # 递归构建左右子树
    tree.output = np.average(df[target].values.tolist())
    tree.left = build_regression_tree(left_data, target)
    tree.right = build_regression_tree(right_data, target)
    return tree


# 构建分类树, 特征是离散的
def build_class_tree(df, set_x, target):
    if df is None: return None
    if df.empty: return None
    columns = df.columns.values.tolist()
    columns.remove(target)
    min_gini = INF
    fea_name = None
    split_value = None
    left_df = None
    right_df = None
    for col in columns:
        value_of_col = set_x[col]
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
            gini_df_col = df1.shape[0]/df.shape[0]*gini1 + df2.shape[0]/df.shape[0]*gini2
            #print(f'特征值={col}; 取值={value}; 基尼指数={gini_df_col}')
            if gini_df_col < min_gini:
                min_gini = gini_df_col
                fea_name = col
                split_value = value
                left_df = df1
                right_df = df2
        #print('-'*50)
    # if min_gini == 0.0:
    #     return None
    if fea_name is None: return None
    if left_df is not None and left_df.empty == False:
        left_df = left_df.drop([fea_name], axis=1)
    if right_df is not None and right_df.empty == False:
        right_df = right_df.drop([fea_name], axis=1)
    tree = Node(fea_name, split_value)
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
        print(f'特征名:{tree.fea_name}; 特征划分值:{tree.split_value}; 层次:{level}')
        if tree.left is not None:
            display(tree.left, level+1)
        if tree.right is not None:
            display(tree.right, level+1)

if __name__ == '__main__':
    df, set_x, set_y = get_data()
    tree = build_class_tree(df, set_x, 'Y')
    display(tree)

