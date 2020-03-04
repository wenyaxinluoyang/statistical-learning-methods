import random
import numpy as np
import pandas as pd
import copy


# 随机获取一个不等i的索引号
def get_index_random(i, size):
    j = i
    while i == j:
        j = random.randint(0, size)
    return j

# 获取a2_new的上下界
def get_alpha_limit(a1_old, a2_old, y1, y2, c):
    if y1 != y2:
        low, high = max(0, a2_old - a1_old), min(c, c + a2_old - a1_old)
    else:
        low, high = max(0, a2_old + a1_old - c), min(c, a2_old + a1_old)
    return low, high

# 其中g(x) = sum(ai*yi*K(xi,x)) + b
def func_g(alphas, b, fea_mat, label_mat, index, kernel_func):
    temp = alphas * label_mat
    array = []
    current_sample = fea_mat[index]
    for sample in fea_mat:
        array.append(kernel_func.calculate(current_sample, sample))
    result = temp * np.array(array).transpose() + b
    return result

# Ei = g(xi) - yi,
def get_e(alphas, b, fea_mat, label_mat, index, kernel_func):
    result = func_g(alphas, b, fea_mat, label_mat, index, kernel_func)
    return result - label_mat[index]


# 序列最小化算法(sequential minimal optimization, SMO)
def smo(fea_mat, label_mat, c, epsilon, max_iter, kernel_func):
    row_num, col_num = fea_mat.shape
    # 初始化拉格朗日乘子
    alphas = np.zeros((row_num, 1)) # row_num 行，1列的数据
    # 初始化偏置
    b = 0
    iter_cnt = 0
    while iter_cnt < max_iter:
        # 外层循环代表选择第一个变量
        for i in range(row_num):
            gi = func_g(alphas, b, fea_mat, label_mat, i, kernel_func)
            ei = gi - label_mat[i]
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
            if (label_mat[i]*ei < -epsilon and alphas[i]<c) or (label_mat[i]*ei > epsilon and alphas[i]>0):
                '''
                挑选a2, 李航统计学习方法，说挑选 abs(e1 - e2)最大的，在这里是随机挑选的，
                当然也可以计算所有的e, 然后选abs(e1 - e2)最大的作为a2
                '''
                j = get_index_random(i, row_num)
                gj = func_g(alphas, b, fea_mat, label_mat, j, kernel_func)
                ej = gj - label_mat[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # 计算alpha_j_new的取值范围
                low, high = get_alpha_limit(alpha_i_old, alpha_j_old, label_mat[i], label_mat[j], c)

                temp = kernel_func.calculate(fea_mat[i], fea_mat[i]) + kernel_func.calculate(fea_mat[j], fea_mat[j]) + 2*kernel_func.calculate(fea_mat[i], fea_mat[j])
                alpha_j_new = alpha_j_old + label_mat[j]*(ei - ej)/temp
                # alpha_j_new 只能在low, high 中取值
                if alpha_j_new < low: alpha_j_new = low
                if alpha_j_new > high: alpha_j_new = high

                alpha_i_new = alpha_i_old + label_mat[i]*label_mat[j]*(alpha_j_old - alpha_j_new)

                # 更新两个拉格朗日乘子
                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new


if __name__ == '__main__':
    fea_mat = np.array(np.arange(12).reshape(3,4))
    label_mat = np.array([[1], [-1], [1]])
    alphas = np.array([[2], [3], [4]])
    print(fea_mat)
    print(label_mat)
    print(alphas*label_mat)