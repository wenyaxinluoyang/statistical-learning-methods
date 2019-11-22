#!/usr/bin/env python

'统计学常用的损失函数'


class ZeroOne:
    '''
    0-1，实际值和预测值相等，返回1，否则返回0
    '''
    # 计算代价函数
    def cal_loss(self, y_actual, y_predict):
        cost = 0
        for actual, predict in zip(y_actual, y_predict):
            cost = self.cal(actual, predict)
        return cost

    def cal(self, actual, predict):
        if actual == predict:
            return 0
        else:
            return 1

class Quadratic:
    '''
    平方损失函数
    '''
    def cal_loss(self, y_actual, y_predict):
        cost = 0
        for actual, predict in zip(y_actual, y_predict):
            cost += self.cal(actual, predict)
        return cost

    def cal(self, actual, predict):
        return (actual-predict)*(actual-predict)

class Absolute:
    '''
    绝对损失函数
    '''
    def cal_loss(self, y_actual, y_predict):
        cost = 0
        for actual, predict in zip(y_actual, y_predict):
            cost = cost + self.cal(actual, predict)

    def cal(self, actual, predict):
        return abs(actual-predict)

