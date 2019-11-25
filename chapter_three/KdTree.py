#!/usr/bin/env python

'''
实现kd树算法
'''

import numpy as np
import math
import copy

INF = float('inf') # 无穷大

# 节点，里面应该有属于它的样本点，左孩子，右孩子
class Node:
    def __init__(self, point, split_value, dimension_index):
        self.point = point
        self.split_value = split_value
        self.dimension_index = dimension_index
        self.left = None
        self.right = None

# 递归构建kd树
def build_kd_tree(x_list, k, depth=0):
    if len(x_list)==0: return None
    dimension_index = depth%k  # 维度索引
    x_list.sort(key=lambda x: x[dimension_index][0])
    split_index = len(x_list)//2 # 切分点索引
    point = x_list[split_index] # 切分点
    split_value = point[dimension_index][0] # 切分值
    left_list = x_list[0: split_index]
    right_list = x_list[split_index+1: len(x_list)]
    root = Node(point, split_value, dimension_index) # 创建根节点, 随后递归构建左右子树
    root.left = build_kd_tree(left_list, k, depth+1)
    root.right = build_kd_tree(right_list, k, depth+1)
    return root # 返回当前根节点

# 打印二叉树
def display(root, level=0, note='root'):
    if root is None: return
    print('level =', level, 'note =', note)
    print('对第', root.dimension_index, '维，以该坐标轴的中位数为分点，以垂直与该维，并过切分点的超平面做划分')
    print('切分值为:', root.split_value)
    print('落在该超平面上的点有:')
    point = root.point
    print(point)
    print('*'*50)
    display(root.left, level+1, note='left') # 打印左子树
    display(root.right, level+1, note='right') # 打印右子树

# p>=1，当p=2，称为欧氏距离
# p=1, 称为曼哈顿距离
# p趋于正无穷, 它氏各个坐标距离的最大值
def distance(x1, x2, p):
    sum = 0
    for vector1, vector2 in zip(x1, x2):
        v1 = vector1[0]
        v2 = vector2[0]
        sum += math.pow(abs(v1-v2), p)
    return math.pow(sum, 1/p) # 开p次方根


# 寻找目标叶子节点
def find_target_leaf_node(root, target):
    nearest_dis, nearest_node = INF, None
    node = copy.deepcopy(root)
    path = []
    while node is not None:
        path.append(node)
        point = node.point # 切分空间的超平面上的点
        dis = distance(point, target, 2)
        if dis < nearest_dis:
            nearest_dis = dis
            nearest_node = node
        dimension_index = node.dimension_index
        split_value = node.split_value
        if target[dimension_index][0] <= split_value:
            node = node.left
        else:
            node = node.right
    return nearest_dis, nearest_node, path


def find_target_leaf_node_two(root, target, k=1):
    k_nearest = [(distance(root.point, target, 2), root)]
    node = copy.deepcopy(root)
    path = []
    while node is not None:
        path.append(node)
        k_nearest.sort(key=lambda dis:  dis[0]) # 按照距离从小到大排序
        point = node.point
        dis = distance(point, target, 2)
        if dis < k_nearest[-1][0]:
            if len(k_nearest)<k: k_nearest.append((dis, node))
            else:
                k_nearest.pop()
                k_nearest.append((dis, node))
        dimension_index = node.dimension_index
        split_value = node.split_value
        if target[dimension_index][0]<= split_value:
            node = node.left
        else:
            node = node.right
    return k_nearest, path

# kd树搜索，搜索最近邻
def kd_tree_search_nearest(root, target):
    # 先找到目标叶子节点
    nearest_dis, nearest_node, path = find_target_leaf_node(root, target)
    # 回溯
    while len(path)!=0:
        node = path.pop()
        split_value = node.split_value
        dimension_index = node.dimension_index
        # 目标点与当前最近点距离的半径的圆与其另一边的子空间有交点
        if abs(target[dimension_index][0]-split_value) < nearest_dis:
            # 原来进入的是左子树，则需要搜索右子树
            if target[dimension_index][0]<=split_value:
                temp_node = node.right
            else:
                temp_node = node.left
            if temp_node is not None:
                path.append(temp_node)
                dis = distance(target, temp_node.point, 2)
                if  dis < nearest_dis:
                    nearest_dis = dis
                    nearest_node = temp_node
    return nearest_dis, nearest_node

# 寻找target的k近邻, 比最近邻多了一个排序数组
def kd_tree_search_knearest(root, target, k):
    k_nearest, path = find_target_leaf_node_two(root, target, k)
    while(len(path)) != 0:
        node = path.pop()
        split_value = node.split_value
        dimension_index = node.dimension_index
        k_nearest.sort(key=lambda dis: dis[0])
        if abs(target[dimension_index][0]-split_value) < k_nearest[-1][0]:
            if target[dimension_index][0]<=split_value:
                temp_node = node.right
            else:
                temp_node = node.left
            if temp_node is not None:
                path.append(temp_node)
                dis = distance(target, temp_node.point, 2)
                if dis < k_nearest[-1][0]:
                    if len(k_nearest)<k: k_nearest.append((dis, temp_node))
                    else:
                        k_nearest.pop()
                        k_nearest.append((dis, temp_node))
    return k_nearest

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
    root = build_kd_tree(x_list, k=2)
    # display(root)
    target = np.array([1,5]).reshape((2,1))
    k_nearest = kd_tree_search_knearest(root, target, k=3)
    # 求距离target最近的前k个点
    for item in k_nearest:
        print('distance:', item[0])
        print('point:', item[1].point)
        print('-'*50)


