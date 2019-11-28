#!/usr/bin/env python

'决策树'
import pandas as pd
import math
import copy

class Node():

    def __init__(self, sign, fea_name, fea_value, data):
        self.fea_name = fea_name
        self.sign = sign
        self.fea_value = fea_value
        self.child_list = []
        self.data = data



# 信息增益值
def information_gain(x_list, value_of_x, y_list, value_of_y, hy):
    return hy - condition_entropy(x_list, value_of_x, y_list, value_of_y)

# 信息增益比
def information_gain_ratio(x_list, value_of_x, y_list, value_of_y, hy):
    h = condition_entropy(x_list, value_of_x, y_list, value_of_y)
    return (hy-h)/hy

# 条件熵, X给定条件下Y的条件概率分布的熵对X的数学期望
def condition_entropy(x_list, value_of_x, y_list, value_of_y):
    hx, value_count = entropy(x_list, value_of_x)
    temp = {x: [] for x in value_of_x}
    for x,y in zip(x_list, y_list):
        temp[x].append(y)
    total_x = len(x_list)
    h = 0
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


# 特征的信息增益比
def infor_gain_of_fea(df, set_x, set_y):
    columns = df.columns.values.tolist()
    columns.remove('Y')
    # 每种特征的信息增益
    fea_infor_gain = {col:0 for col in columns}
    y_list = df['Y'].values.tolist()
    hy, value_count_y = entropy(y_list, set_y)
    for col in columns:
        fea_infor_gain[col] = information_gain(df[col].values.tolist(), set_x[col], y_list, set_y, hy)
    print(fea_infor_gain)
    print('Y的熵', hy)
    return fea_infor_gain

def infor_gain_ratio_of_fea(df, set_x, set_y):
    columns = df.columns.values.tolist()
    columns.remove('Y')
    # 每种特征的信息增益
    fea_infor_ratio_gain = {col: 0 for col in columns}
    y_list = df['Y'].values.tolist()
    hy, value_count_y = entropy(y_list, set_y)
    for col in columns:
        fea_infor_ratio_gain[col] = information_gain_ratio(df[col].values.tolist(), set_x[col], y_list, set_y, hy)
    print(fea_infor_ratio_gain)
    print('Y的熵', hy)
    return fea_infor_ratio_gain

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


# 使用ID3构建决策树
def ID3(df, set_x, set_y, epsilon):
    if df.empty: return None
    y_unique = df['Y'].values.tolist()
    # 训练数据集所有实例都属于同一类
    if len(set(y_unique)) == 1:
        tree = Node(y_unique[0], None, None, df)
        return tree
    feature_names = df.columns.values.tolist()
    feature_names.remove('Y')
    temp = df['Y'].value_counts()
    temp = temp.reset_index()
    max_count_y = temp.loc[0]['index']
    # 特征集是空集
    if len(feature_names) == 0:
        tree = Node(max_count_y, None, None, df)
        return tree
    fea_infor_gain = infor_gain_of_fea(df, set_x, set_y)
    max_infor_gain = 0
    chose_fea = None
    for key,value in fea_infor_gain.items():
        if value > max_infor_gain:
            max_infor_gain = value
            chose_fea = key
    if max_infor_gain < epsilon:
        tree = Node(max_count_y, None, None, df)
        return tree
    tree = Node(max_count_y, chose_fea, None, df)
    for index, data in df.groupby([chose_fea]):
        sub_df = data.drop([chose_fea], axis=1)
        child = ID3(sub_df, set_x, set_y, epsilon)
        child.fea_value = index
        tree.child_list.append(child)
    return tree


# 使用C4.5算法构建生成树, 用信息增益比来选择特征
def C45(df, set_x, set_y, epsilon):
    if df.empty: return None
    y_unique = df['Y'].values.tolist()
    # 训练数据集所有实例都属于同一类
    if len(set(y_unique)) == 1:
        tree = Node(y_unique[0], None, None, df)
        return tree
    feature_names = df.columns.values.tolist()
    feature_names.remove('Y')
    temp = df['Y'].value_counts()
    temp = temp.reset_index()
    max_count_y = temp.loc[0]['index']
    # 特征集是空集
    if len(feature_names) == 0:
        tree = Node(max_count_y, None, None, df)
        return tree
    fea_infor_ratio_gain = infor_gain_ratio_of_fea(df, set_x, set_y)
    max_infor_gain = 0
    chose_fea = None
    for key, value in fea_infor_ratio_gain.items():
        if value > max_infor_gain:
            max_infor_gain = value
            chose_fea = key
    if max_infor_gain < epsilon:
        tree = Node(max_count_y, None, None, df)
        return tree
    tree = Node(max_count_y, chose_fea, None, df)
    for index, data in df.groupby([chose_fea]):
        sub_df = data.drop([chose_fea], axis=1)
        child = C45(sub_df, set_x, set_y, epsilon)
        child.fea_value = index
        tree.child_list.append(child)
    return tree


def display(tree, level=1):
    if tree is not None:
        print('第', level, '层')
        print('选取的特征是:', tree.fea_name)
        print('上一层特征值:', tree.fea_value)
        print('该特征下，最大的分类:', tree.sign)
        print('='*50)
        for child in tree.child_list:
            display(child, level+1)

# 剪枝
def cut_branch(tree, alfa, set_y):
    if tree is not None:
        # 计算该节点的经验熵
        data = tree.data
        h, value_count = entropy(data['Y'].values.tolist(), set_y)
        result = data.shape[0]*h
        if len(tree.child_list) != 0:
            sum = 0
            leaf_node_num = 0
            for child in tree.child_list:
                value, is_leaf, node = cut_branch(child, alfa, set_y)
                sum += value
                leaf_node_num += leaf_node_num
            # sum为当前节点统领节点下的叶子节点的
            if result+alfa < sum+alfa*leaf_node_num:
                tree.child_list = [] # 进行减枝
                return result, 1, tree
            else:
                return sum, 0, tree
        else: # 是叶子节点
             return result, 0, tree






if __name__ == '__main__':
    df, set_x, set_y = get_data()
    #tree = ID3(df, set_x, set_y, 0.001)
    tree = C45(df, set_x, set_y, 0.001)
    display(tree)
    cost, is_leaf, tree = cut_branch(tree, 0.5, set_y)
    print('-'*10, '减枝后的树', '-'*10)
    display(tree)

    #infor_gain_ratio_of_fea(df, set_x, set_y)
