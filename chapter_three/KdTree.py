#!/usr/bin/env python

'''
实现kd树算法
'''

import numpy as np
import math

INF = float('inf') # 无穷大

# 节点，里面应该有属于它的样本点，左孩子，右孩子
class Node:
    def __init__(self, x_list, median, index):
        self.__x_list = x_list
        self.__median = median
        self.__index = index
        self.__left = None
        self.__right = None
    def get_median(self):
        return self.__median
    def get_index(self):
        return self.__index
    def get_left_child(self):
        return self.__left
    def get_right_child(self):
        return self.__right
    def get_x_list(self):
        return self.__x_list
    def set_left_child(self, node):
        self.__left = node
    def set_right_child(self, node):
        self.__right = node
    def set_x_list(self, x_list):
        self.__x_list = x_list

# 递归构建kd树
def build_kd_tree(x_list, k, depth=0):
    if len(x_list)==0: return None
    index = depth%k # 选取index维，用垂直index维并过中位数初的超平面作为划分
    values = [x[index][0] for x in x_list]
    values = sorted(values)
    median = values[len(values)//2]
    temp_list = [x for x in x_list if x[index][0]==median]
    left_list = [x for x in x_list if x[index][0]<median]
    right_list = [x for x in x_list if x[index][0]>median]
    root = Node(temp_list, median, index) # 创建根节点, 随后递归构建左右子树
    root.set_left_child(build_kd_tree(left_list, k, depth+1))
    root.set_right_child(build_kd_tree(right_list, k, depth+1))
    return root # 返回当前根节点

# 打印二叉树
def display(root, level=0, note='root'):
    if root is None: return
    print('level =', level, 'note =', note)
    print('对第', root.get_index(), '维，以该坐标轴的中位数为分点，以垂直与该维，并过切分点的超平面做划分')
    print('切分值为:', root.get_median())
    print('落在该超平面上的点有:')
    x_list = root.get_x_list()
    if len(x_list) != 0:
        for x in x_list:
            print(x)
            print('*'*50)
    print('-'*50)
    display(root.get_left_child(), level+1, note='left') # 打印左子树
    display(root.get_right_child(), level+1, note='right') # 打印右子树

# p>=1，当p=2，称为欧氏距离
# p=1, 称为曼哈顿距离
# p趋于正无穷, 它氏各个坐标距离的最大值
def distance(self, x1, x2, p):
    sum = 0
    for vector1, vector2 in zip(x1, x2):
        v1 = vector1[0]
        v2 = vector2[0]
        sum += math.pow(abs(v1-v2), p)
    return math.pow(sum, 1/p) # 开p次方根



# kd树搜索
def kd_tree_search(root, x):
    left_child = root.get_left_child()
    right_child = root.get_right_child()
    if left_child==None and right_child==None:
        nearest_dis = distance(x, root, 2)
        return nearest_dis, root  # 叶子节点，该节点为最近点
    index = root.get_index()
    median = root.get_median()
    value = x[index][0]
    if value < median:
        if left_child is None: nearest_dis, nearest_node = INF, None
        else: nearest_dis, nearest_node = kd_tree_search(left_child, x)
    if value > median:
        if right_child is None: nearest_dis, nearest_node = INF, None
        else: nearest_dis, nearest_node = kd_tree_search(right_child, x)
    x_list = root.get_x_list()
    flag = False
    for item in x_list:
        dis = distance(x, item, 2)
        if dis < now_dis:
            now_dis = dis
            flag = True
    if flag:
        nearest_node = root

    return nearest_dis, nearest_node





# 获取例3.2数据
def example_three_two():
    x1 = np.array([2,3]).reshape((2,1))
    x2 = np.array([5,4]).reshape((2,1))
    x3 = np.array([9,6]).reshape((2,1))
    x4 = np.array([4,7]).reshape((2,1))
    x5 = np.array([8,1]).reshape((2,1))
    x6 = np.array([7,2]).reshape((2,1))
    x_list = [x1, x2, x3, x4, x5, x6]
    return x_list

if __name__ == '__main__':
    x_list = example_three_two()
    #x1 = x_list[0]
    #print(x1[0][0])
    root = build_kd_tree(x_list, k=2)
    display(root)


