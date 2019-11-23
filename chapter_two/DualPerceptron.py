#!/usr/bin/env python

import numpy as np
import random
'''
感知机学习算法的对偶形式
考虑每个样本点更新的次数，那么现在次数变成我们的参数
'''

def sign(value):
    if value>0: return 1
    elif value<0: return -1

# 计算xj * xi 备用
def cal_gram_matrix(x_list):
    size = len(x_list)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = np.dot(x_list[i].T, x_list[j])
    return matrix

def func(x_list, x_index, y_list, n_list, learning_rate, matrix):
    result = 0
    temp = 0
    size = len(x_list)
    for j in range(size):
        result = result + n_list[j]*learning_rate*y_list[j]*matrix[j][x_index]
        temp = temp + n_list[j]*learning_rate*y_list[j]  # b的值
    result = result + temp
    return result

# 感知机学习算法的对偶形式
def dual_perceptron(x_list, y_list, n_list, learning_rate, matrix):
    count = 0
    while True:
        size = len(x_list)
        flag = False # 标记是否存在误分类点
        for i in range(size):
            if y_list[i]*func(x_list, i, y_list, n_list, learning_rate, matrix) <= 0:
                n_list[i] += 1
                flag = True
                break
        if flag==False:
            break
        count = count + 1
        print('第', count, '次迭代:')
        print(n_list)
        print('-'*50)
    return n_list

if __name__ == '__main__':
    x1 = np.array([3, 3]).reshape(2, 1)
    x2 = np.array([4, 3]).reshape(2, 1)
    x3 = np.array([1, 1]).reshape(2, 1)
    x_list = [x1, x2, x3]
    y_list = [1, 1, -1]
    n_list = [0, 0, 0]
    learning_rate = 1
    matrix = cal_gram_matrix(x_list)
    n_list = dual_perceptron(x_list, y_list, n_list, learning_rate, matrix)
    print('params (n_list:', n_list, ', learning_rate: ', learning_rate, ')')

