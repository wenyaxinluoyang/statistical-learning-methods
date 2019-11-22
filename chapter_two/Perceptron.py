#!/usr/bin/env python
import numpy as np
import random
'''
李航：《统计学习方法》 第二章数据理论的代码实现
模型模型:
yi = sign(w*xi + b)
sign(t)  t>0, sign(t)=1; t<0, sign(t)=-1
解决方案：
w*x + b = 0
找到一个超平面，把正实例点和负实例点划分到超平面两侧使得
y(i)=+1  w*x(i)+b>0
y(i)=-1  w*x(i)+b<0
'''


'''
损失函数，误分类点到超平面的距离
对误分类的点来说，-y(i)(w*x(i)+b) > 0 
因此误分类点xi到超平面的距离是
-1/|w|*y(i)*(w*x(i))+b)

M是误分类点的集合
sign(w*x + b)的损失函数为
xi属于M
yi取值-1,1 
L(w,b) = -SUM(yi* (w*xi + b))  

损失函数中w,b未知，我们要找到使损失函数最小的模型参数w,b，即感知机的模型

使用梯度下降不断地极小化目标函数，极小话过程中不是一次使M中所有误分类点的梯度下降，
而是一次随机选取一个误分类点使其梯度下降：
对w求偏导： -SUM(yixi)
对b求偏导： -SUM(yi)

设学习速率为learning_rate（0，1], 随机选择
w + learning*yi*xi) -> w
b + learning*yi) -> b

'''

# f(x) = sign(x)   x>0返回1，x<0返回-1
def sign(value):
    if value>0: return 1
    elif value<0: return -1

# 确定参数后，就有模型函数
def func(x, w, b):
    return sign(np.dot(w.T,x) + b)

# 求解最佳模型参数
def perceptron(x_list, y_list, learning_rate, w, b):
    # w, b = init_params()
    # 有可能这样的平面不存在，循环不会结束，永远不会收敛
    max_time = int(100 * len(x_list))
    count = 0
    while True:
        error_x_list, error_y_list = error_point(x_list, y_list, w, b)
        if len(error_x_list) == 0:
            break
        # 进行梯度下降算法
        w, b = SGD(learning_rate, error_x_list, error_y_list, w, b)
        count = count + 1
        print('第', count , '次梯度下降: w =', w, ', b =', b)
        if count == max_time:
            break
    return w,b

# 获取当前模型的误分类点
def error_point(x_list, y_list, w, b):
    y_predict_list = []
    for x in x_list:
        y_predict_list.append(func(x, w, b))
    error_x_list = []
    error_y_list = []
    for x, y, y_predict in zip(x_list, y_list, y_predict_list):
        if y != y_predict:
            error_x_list.append(x)
            error_y_list.append(y)
    return error_x_list, error_y_list

# 梯度下降, 随机选取一个误判点进行梯度下降
def SGD(learning_rate, error_x_list, error_y_list, w, b):
    index = random.randint(0, len(error_x_list)-1)
    x = error_x_list[index]
    y = error_y_list[index]
    w = w + learning_rate*y*x
    b = b + learning_rate*y
    return w,b

if __name__ == '__main__':
    x1 = np.array([3,3]).reshape(2,1)
    x2 = np.array([4,3]).reshape(2,1)
    x3 = np.array([1,1]).reshape(2,1)
    x_list = [x1, x2, x3]
    y_list = [1, 1, -1]
    w = np.zeros((2,1))
    b = 0
    learning_rate = 1
    w, b = perceptron(x_list, y_list, learning_rate, w, b)
    print(w, b)


'''
算法的解有： 
w = (1,1) b = -3
w = (2,1) b = -5
w = (1,0) b = -2
w = (3,1) b = -5
感知机学习算法由于采用不同的初值或选择不同的误分类点，解可以不同
'''



















