import numpy as np
import math

# 核函数
class Kernel_Func():

    # 计算
    def calculate(self, x, z):
        pass

    # p代表范数
    def normalForm(x, y, p):
        dis_value = 0
        for x_value, y_value in zip(x, y):
            dis_value += math.pow(y_value - x_value, p)
        return math.pow(dis_value, 1 / p)

# 多项式核函数
class Polynomial_Kernel_Func(Kernel_Func):

    def __init(self, p):
        self.__p = p

    def calculate(self, x, z,):
        p = kwargs['degree'] # 多项式的次数
        return math.pow(x * z + 1, self.__p)


# 高斯核函数
class Gaussian_Kernel_Func(Kernel_Func):

    def __init__(self, sigma):
        self.__sigma = sigma

    def calculate(self, x, z):
        sigma = kwargs['sigma']
        return math.exp(-self.normalForm(x, z, 2)/(2*self.__sigma*self.__sigma))
