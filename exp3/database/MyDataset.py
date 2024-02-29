import copy
import random

import numpy as np
import torch
import sys
import os
from scipy.stats import stats
from scipy import stats
from vmdpy import VMD
from database import GetDataFromNC
import torch
import torch.nn as nn


class DataSet(nn.Module):
    def __init__(self, num, size, vmd_size):
        super(DataSet, self).__init__()
        self.num = num
        self.size = size
        self.vmd_size = vmd_size
        self.x = torch.zeros(num, vmd_size * 2, size)
        self.y = torch.zeros(num)


class MyDataset():
    def __init__(self, area, DEVICE, V=0.5):
        self.area = area
        self.V = V
        self.DEVICE = DEVICE
        self.filePath = r'database/nc/' + self.area + '.nc'
        self.modelPath = r'database/data/' + self.area + str(V)

        alpha = 2000
        tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
        K = 5  # K 分解模态（IMF）个数
        DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
        init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
        tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数

        if os.path.exists(self.modelPath):
            self.Data = torch.load(self.modelPath)
        else:
            u, v = GetDataFromNC.getData(self.filePath)

            time, N, M = u.shape
            u = u[:-80 * 6, int(N / 2), int(M / 2)]
            v = v[:-80 * 6, int(N / 2), int(M / 2)]
            u = GetDataFromNC.normalization(u)
            v = GetDataFromNC.normalization(v)

            x_data = np.arange(0, 10, 1)

            y_data = stats.norm.pdf(x_data, 5, 1.4) * self.V / 0.28

            print(u.shape)
            time = u.shape[0]

            num = int((time - 200) * 0.6)
            time_index = [i for i in range(time - 200)]
            choose = random.sample(time_index, num)

            size = 30
            self.Data = DataSet(num, size, 5)
            for i in range(num):
                if i % 1000 == 999:
                    print(area,self.modelPath,'进度：'+str(i+1)+'/'+str(num))
                index = choose[i]
                ux = copy.deepcopy(u[index:index + size])
                vx = copy.deepcopy(v[index:index + size])
                p = random.randint(0, 10)
                if p < 6:
                    self.Data.y[i] = 1
                    q = random.randint(0, size - 10)
                    if p < 3:
                        ux[q:q + 10] = ux[q:q + 10] + y_data
                    else:
                        vx[q:q + 10] = vx[q:q + 10] + y_data
                else:
                    self.Data.y[i] = 0
                vmdu, vmd_hat, omega = VMD(ux, alpha, tau, K, DC, init, tol)
                vmdv, vmd_hat, omega = VMD(vx, alpha, tau, K, DC, init, tol)
                self.Data.x[i, :5, :] = torch.from_numpy(vmdu).float()
                self.Data.x[i, 5:, :] = torch.from_numpy(vmdv).float()
            torch.save(self.Data, self.modelPath)
