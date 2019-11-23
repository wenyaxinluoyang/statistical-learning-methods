#!/usr/bin/env python

'''
k近邻算法
'''

import math

# p>=1，当p=2，称为欧氏距离
# p=1, 称为曼哈顿距离
# p趋于正无穷, 它氏各个坐标距离的最大值
def distrance(self, x1, x2, p):
    sum = 0
    for vector1, vector2 in zip(x1, x2):
        v1 = vector1[0]
        v2 = vector2[0]
        sum += math.pow(abs(v1-v2), p)
    return math.pow(sum, 1/p) # 开p次方根


