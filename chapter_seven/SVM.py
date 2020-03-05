import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from chapter_seven.Kernel import *


# 随机获取一个不等i的索引号
def get_index_random(i, size):
    j = i
    while i == j:
        j = random.randint(0, size-1)
    return j

# 获取a2_new的上下界
def get_alpha_limit(a1_old, a2_old, y1, y2, c):
    if y1 != y2:
        low, high = max(0, a2_old - a1_old), min(c, c + a2_old - a1_old)
    else:
        low, high = max(0, a2_old + a1_old - c), min(c, a2_old + a1_old)
    return low, high

# 其中g(x) = sum(ai*yi*K(xi,x)) + b
def func_g(alphas, b, fea_mat, label_list, index, kernel_func):
    label_mat = np.array(label_list).reshape(len(label_list), 1)
    alphas_mat = np.array(alphas).reshape(len(alphas), 1)
    temp = alphas_mat * label_mat # alphas_mat 和 label_mat均是列向量，此处向量temp为两向量对应位置想成
    array = []
    current_sample = fea_mat[index]
    for sample in fea_mat:
        t = kernel_func.calculate(current_sample, sample.reshape((-1, 1)))
        array.append(t)
    result = np.dot(np.array(array), temp)[0] + b
    return result

# Ei = g(xi) - yi,
def get_e(alphas, b, fea_mat, label_list, index, kernel_func):
    result = func_g(alphas, b, fea_mat, label_list, index, kernel_func)
    return result - label_list[index]


# 序列最小化算法(sequential minimal optimization, SMO)
def smo(fea_mat, label_mat, c=1, epsilon=1e-5, max_iter=20, kernel_func=Linear_Kernel_Func()):
    label_list = [label[0] for label in label_mat]
    row_num, col_num = fea_mat.shape
    # 初始化拉格朗日乘子
    alphas = np.zeros((row_num, 1)) # row_num 行，1列的数据
    # 初始化偏置
    b = 0
    iter_cnt = 0
    while iter_cnt < max_iter:
        # 外层循环代表选择第一个变量
        for i in range(row_num):
            gi = func_g(alphas, b, fea_mat, label_list, i, kernel_func)
            ei = gi - label_list[i]
            '''
            选取的a1, 应该违反KKT条件（李航统计学习方法，说挑选违反KKT条件最严重的一个）
            停机条件，停机：所有拉格朗日乘子满足KKT条件
            yi*g(xi) >= 1 , ai = 0
            yi*g(xi) = 1 , 0 < ai < c
            yi*g(xi) <= 1, ai = c
            若ai < c, 则 0 <= ai < c, 若满足KKT条件则 yi*g(xi) >= 1, 考虑精度, yi*g(xi)>=1-epsilon
            则需要违反KKT条件，则yi*g(xi)<1-epsilon, 则 yi*g(xi)-1 < -epsilon
            若ai > 0, 则 0 < ai <=c, 若满足KKT条件则 yi*g(xi) <= 1, 考虑精度，yi*g(xi)<=1+epsilon
            则需要违反KKT条件，则yi*g(xi)>1+epsilon, 则 yi*g(xi)-1 > epsilon
            '''
            if (label_list[i]*ei<-epsilon and alphas[i]<c) or (label_list[i]*ei>epsilon and alphas[i]>0):
                '''
                挑选a2, 李航统计学习方法，说挑选 abs(e1 - e2)最大的，在这里是随机挑选的，
                当然也可以计算所有的e, 然后选abs(e1 - e2)最大的作为a2
                '''
                j = get_index_random(i, row_num)
                gj = func_g(alphas, b, fea_mat, label_list, j, kernel_func)
                ej = gj - label_list[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # 计算alpha_j_new的取值范围
                low, high = get_alpha_limit(alpha_i_old, alpha_j_old, label_list[i], label_list[j], c)

                kii = kernel_func.calculate(fea_mat[i], fea_mat[i].reshape((-1,1)))
                kij = kernel_func.calculate(fea_mat[i], fea_mat[j].reshape((-1,1)))
                kjj = kernel_func.calculate(fea_mat[j], fea_mat[j].reshape((-1,1)))

                alpha_j_new = alpha_j_old + label_list[j]*(ei - ej)/(kii + kjj + 2*kij)
                if alpha_j_new < low: alpha_j_new = low
                if alpha_j_new > high: alpha_j_new = high

                if abs(alpha_j_new - alpha_j_old) < 1e-6: continue

                alpha_i_new = alpha_i_old + label_list[i]*label_list[j]*(alpha_j_old - alpha_j_new)


                # 更新两个拉格朗日乘子
                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new

                # 更新b的值
                bi_new = -ei - label_list[i]*kii*(alpha_i_new-alpha_i_old) - label_list[j]*kij*(alpha_j_new-alpha_j_old) + b
                bj_new = -ej - label_list[i]*kij*(alpha_i_new-alpha_i_old) - label_list[j]*kjj*(alpha_j_new-alpha_j_old) + b
                if 0<alpha_i_new<c and 0<alpha_j_new<c:
                    b = bi_new
                else:
                    b = (bi_new + bj_new)/2
        iter_cnt += 1
    # 由w* = sum(ai*yi*xi)
    w = np.zeros(fea_mat[0].shape) # 行向量
    for i in range(row_num):
        w += alphas[i]*label_list[i]*fea_mat[i]
    # 转为列向量
    w = w.reshape((-1,1))
    return alphas, w, b


# 数据集1， 线性可分
def get_data1():
    # 创建40个点
    fea_mat = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    label_mat = np.array([-1]*20 + [1]*20).reshape((40, 1))
    # plt.scatter(fea_mat[:, 0], fea_mat[:, 1], c = [label[0] for label in label_mat])
    # plt.show()
    return fea_mat, label_mat

# 显示数据集1的效果
def show_effect1():
    fea_mat, label_mat = get_data1()
    alphas, w, b = smo(fea_mat, label_mat, max_iter=200)
    print(w)
    print(b)
    plt.scatter(fea_mat[:, 0], fea_mat[:, 1], c=[label[0] for label in label_mat])
    x_list = list(range(-5, 5, 1))
    y_list = [(-w[0][0] * x - b) / w[1][0] for x in x_list]
    plt.plot(x_list, y_list, label='分隔超平面', color='red')
    plt.show()

# 数据集2，近似线性可分
def get_data2():
    fea_mat = np.r_[np.random.randn(20,2), np.random.randn(20,2) + [2,2]]
    label_mat = np.array([-1]*20 + [1]*20).reshape((-1, 1))
    # plt.scatter(fea_mat[:, 0], fea_mat[:, 1], c = [label[0] for label in label_mat])
    # plt.show()
    return fea_mat, label_mat

def show_effect2():
    fea_mat, label_mat = get_data2()
    alphas, w, b = smo(fea_mat, label_mat, max_iter=200)
    print(w)
    print(b)
    plt.scatter(fea_mat[:, 0], fea_mat[:, 1], c=[label[0] for label in label_mat])
    x_list = list(range(-5, 5, 1))
    y_list = [(-w[0][0] * x - b) / w[1][0] for x in x_list]
    plt.plot(x_list, y_list, label='分隔超平面', color='red')
    plt.show()


if __name__ == '__main__':
    # show_effect1()
    # get_data2()
    show_effect2()