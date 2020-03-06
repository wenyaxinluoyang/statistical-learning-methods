import math
import numpy as np


# 基础分类器
def basic_classify_pred(am_list, feature):
    pass


# 分类误差率
def classify_missing_rate(classifier, weights, fea_mat, label_mat, am_list):
    summary = 0
    row_num, col_num = np.shape(fea_mat)
    for i in range(row_num):
        is_equal = 1 if basic_classify_pred(am_list, fea_mat[i]) != label_mat[i] else 0
        summary += weights[i] * is_equal
    return summary

# 计算基础分类器的系数
def get_classify_coefficient(em):
    am = 0.5*math.log((1-em)/em)
    return am

# 计算规范化因子
def get_normal_factor(classifier, weights, am, fea_mat, label_mat, am_list):
    zm = 0
    row_num, col_num = np.shape(fea_mat)
    for i in range(row_num):
        zm += weights[i] * math.exp(-am * label_mat[i] * basic_classify_pred(am_list, fea_mat[i]))
    return zm

# 更新权重
def update_weight(classifier, weights, em, am, zm, fea_mat, label_mat, am_list):
    length = len(weights)
    new_weights = []
    for i in range(length):
        new_weights[i] = (weights[i]/zm) * math.exp(-am * label_mat[i] * basic_classify_pred(am_list, fea_mat[i]))
        weights = new_weights
    return new_weights,


# AdaBoost算法分类
def AdaBoost_Classify(fea_mat, label_mat, m):
    row_num, col_num = np.shape(fea_mat)
    # 初始化训练数据的权重分布，开始每个样本的权重都一样
    weights = np.array([1/row_num]*row_num)
    cnt = 0
    am_list = []
    while cnt < m:
        em = classify_missing_rate(None, weights, fea_mat, label_mat, am_list)
        am = get_classify_coefficient(em)
        am_list.append(am)
        zm = get_normal_factor(None, weights, am, fea_mat, label_mat, am_list)
        weights = update_weight(None, weights, em, am, zm, fea_mat, label_mat, am_list)
        cnt += 1
    return am_list



if __name__ == '__main__':
    pass