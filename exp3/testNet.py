import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

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
from TorchModels import VMDK

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

    # print(torch.tensor([1,2])*torch.tensor([3,4]))

    ## x-axis for the plot
    x_data = np.arange(-20, 20, 1)

    ## y-axis as the gaussian
    y_data = stats.norm.pdf(x_data, 0, 4)

    ## plot data
    plt.plot(x_data, y_data)
    plt.show()

    alpha = 2000
    tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
    K = 5  # K 分解模态（IMF）个数
    DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
    init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
    tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
    vmd, vmd_hat, omega = VMD(u[0:100], alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
    Net = VMDK.VMDK(input_size = 100, VMD_K = 5, kernel_size = 5, hidden_size = 3, DEVICE = DEVICE)
    print(Net)
    out = Net(torch.from_numpy(vmd).float())
    print('out',out)
    K,W = Net.params['K'],Net.params['W']
    print(Net.params['K'])
    print(Net.params['W'])
    optimizer0 = optim.SGD(Net.parameters(),lr=1)
    optimizer0.zero_grad()
    out = Net(torch.from_numpy(vmd).float())
    print(out)
    out.sum().backward()
    optimizer0.step()
    optimizer0.zero_grad()
    out = Net(torch.from_numpy(vmd).float())
    print(out)
    out.sum().backward()
    optimizer0.step()
    for i in range(100):
        optimizer0.zero_grad()
        out = Net(torch.from_numpy(vmd).float())
        print(out)
        out.sum().backward()
        optimizer0.step()
    print(Net.params['K'])
    print(Net.params['K'] - K)
    print(Net.params['W'] - W)
    print(Net.params['K'].grad)
    print(Net.params['W'].grad)

