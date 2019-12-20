import math
import matplotlib.pyplot as plt
import numpy as np

# 绘制 f(x) = x*log(x)的曲线

def func(x):
    return -x*math.log(x, 2)

def draw_func():
    x_list = np.linspace(0.00001, 1, 200, endpoint=True)
    y_list = [func(x) for x in x_list]
    plt.plot(x_list, y_list, lw=2.0, label='xlog(x,2)')
    plt.show()


if __name__ == '__main__':
    draw_func()