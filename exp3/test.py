import torch
import torch.nn as nn
import numpy as np
from database import GetDataFromNC
from random import randint
import torch_geometric
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import os
from scipy import stats
from vmdpy import VMD
from scipy.fftpack import fft

areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    area = areaList[0]
    MODEL_PATH = "model/"
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    u, v = GetDataFromNC.getData(r'database/nc/' + area + '.nc')
    time, N, M = u.shape
    u = u[:-80 * 6, int(N / 2), int(M / 2)]
    v = v[:-80 * 6, int(N / 2), int(M / 2)]
    u = GetDataFromNC.normalization(u)
    v = GetDataFromNC.normalization(v)
    print(u.shape)


    index = 1219
    length = 360 * 6 * 3
    x = u[index : index + length : 6]
    y = u[index + 365 * 6 : index + 365 * 6 + length : 6 * 3]
    t = np.arange(1, int(length/6/3) + 1) * 3
    # plt.plot(t, x)
    # plt.show()
    plt.plot(t, y)
    plt.show()



    # print(torch.tensor([1,2])*torch.tensor([3,4]))

    ## x-axis for the plot
    x_data = np.arange(0, 30, 1)

    ## y-axis as the gaussian
    y_data = stats.norm.pdf(x_data, 15, 2)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    v = y_data[15]

    plt.xlabel('时间')
    plt.ylabel('幅度')

    plt.plot(x_data, y_data/v*0.5,label='异常强度0.5')
    plt.plot(x_data, y_data/v*1,label='异常强度1')
    plt.plot(x_data, y_data/v*1.5,label='异常强度1.5')
    plt.plot(x_data, y_data/v*2,label='异常强度2')
    plt.legend()
    plt.show()

    ## plot data
    plt.plot(x_data, y_data)
    plt.show()
    t = np.arange(1, 30 + 1)
    alpha = 2000
    tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
    K = 5  # K 分解模态（IMF）个数
    DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
    init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
    tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
    vmd, vmd_hat, omega = VMD(u[0:30], alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
    plt.plot(t, u[0:30])
    plt.show()
    t = np.arange(1,31)
    plt.subplot(5,1, 1)
    plt.plot(t, vmd[0])
    plt.subplot(5,1, 2)
    plt.plot(t, vmd[1])
    plt.subplot(5,1, 3)
    plt.plot(t, vmd[2])
    plt.subplot(5,1, 4)
    plt.plot(t, vmd[3])
    plt.subplot(5,1, 5)
    plt.plot(t, vmd[4])
    plt.show()

    alpha = 2000
    tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
    K = 5  # K 分解模态（IMF）个数
    DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
    init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
    tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
    f = u[0:30]
    f = f + y_data/v*0.5
    vmd1, vmd_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
    plt.plot(t, u[0:30], label='正常数据')
    plt.plot(t, f,label='异常数据')
    plt.legend()
    plt.show()
    plt.subplot(5, 1, 1)
    plt.plot(t, vmd1[0])
    plt.subplot(5, 1, 2)
    plt.plot(t, vmd1[1])
    plt.subplot(5, 1, 3)
    plt.plot(t, vmd1[2])
    plt.subplot(5, 1, 4)
    plt.plot(t, vmd1[3])
    plt.subplot(5, 1, 5)
    plt.plot(t, vmd1[4])
    plt.show()

    plt.subplot(5, 1, 1)
    plt.plot(t, vmd1[0] - vmd[0])
    plt.subplot(5, 1, 2)
    plt.plot(t, vmd1[1] - vmd[1])
    plt.subplot(5, 1, 3)
    plt.plot(t, vmd1[2] - vmd[2])
    plt.subplot(5, 1, 4)
    plt.plot(t, vmd1[3] - vmd[3])
    plt.subplot(5, 1, 5)
    plt.plot(t, vmd1[4] - vmd[4])
    plt.show()
    plt.plot(t, f - u[0:100])
    plt.show()