#!/usr/bin/env python

import numpy as np
import random
'''
感知机学习算法的对偶形式
'''

def sign(value):
    if value>0: return 1
    elif value<0: return -1

def cal_gram_matrix(x_list):
    size = len(x_list)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = np.dot(x_list[i].T, x_list[j])
    return matrix


def func(learning_rate, n_list, x_list, y_list):
    temp =np.zeros((len(x_list), 1))
    for n,x,y in zip(n_list, x_list, y_list):
        pass
        #temp = temp +


if __name__ == '__main__':
    x1 = np.array([3, 3]).reshape(2, 1)
    x2 = np.array([4, 3]).reshape(2, 1)
    x3 = np.array([1, 1]).reshape(2, 1)
    x_list = [x1, x2, x3]
    y_list = [1, 1, -1]
    w = np.zeros((2, 1))
    b = 0
    learning_rate = 1
    matrix = cal_gram_matrix(x_list)
    print(matrix)