import torch
import torch.nn as nn
import numpy as np
from TorchModels import GAN
from database import getDataFromNC
from TorchModels import SortTools
from random import randint
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from TorchModels import Graph
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dx = [0, 0, 1, 1, 1, -1, -1, -1]
dy = [1, -1, 1, -1, 0, 1, -1, 0]



def TrainGAN():
    u, v = getDataFromNC.getData(r'database/jiduo.nc')
    u = u[:-67 * 6, :, :]
    v = v[:-67 * 6, :, :]
    u = SortTools.normalization(u)
    v = SortTools.normalization(v)
    time, N, M = u.shape
    print(time, N, M)
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    discriminator = GAN.Discriminator(50, 10).to(DEVICE)
    generator = GAN.Generator(10, 50, 2).to(DEVICE)
    optimizerD = torch.optim.Adam(discriminator.parameters())  # Adam优化，几乎不用调参
    criterionD = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
    optimizerG = torch.optim.Adam(generator.parameters())  # Adam优化，几乎不用调参
    criterionG = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
    for epoch in range(200):
        lossD = torch.tensor(0.0).to(DEVICE)
        lossG = torch.tensor(0.0).to(DEVICE)

        for i in range(10):
            x, y = generator(torch.randn(1, 10))
            lossG += criterionG(discriminator(x, y), torch.tensor(1.0).to(DEVICE).to(torch.float32))

        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

        for i in range(50):
            x, y = generator(torch.randn(1, 10))
            lossD += criterionD(discriminator(x, y), torch.tensor(0.0).to(DEVICE).to(torch.float32))

        for i in range(50):
            x1 = randint(1, N - 2)
            y1 = randint(1, M - 2)
            direction = randint(0, 7)
            x2 = x1 + dx[direction]
            y2 = y1 + dy[direction]
            t = randint(0, time - 51)
            x = torch.tensor(
                (np.append(u[t:t + 50, x1, y1], v[t:t + 50, x1, y1])).reshape(1, 50, 2).astype(float),
                dtype=torch.float32).to(
                DEVICE)
            y = torch.tensor(
                (np.append(u[t:t + 50, x2, y2], v[t:t + 50, x2, y2])).reshape(1, 50, 2).astype(float),
                dtype=torch.float32).to(
                DEVICE)
            lossD += criterionD(discriminator(x, y), torch.tensor(1.0).to(DEVICE).to(torch.float32))

        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        print(epoch, lossG, lossD)

    torch.save(discriminator, MODEL_PATH + 'D')
    torch.save(generator, MODEL_PATH + 'G')


def TrainD():
    u, v = getDataFromNC.getData(r'database/jiduo.nc')
    u = u[:-67 * 6, :, :]
    v = v[:-67 * 6, :, :]
    u = SortTools.normalization(u)
    v = SortTools.normalization(v)
    time, N, M = u.shape
    print(time, N, M)
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    discriminator = GAN.Discriminator(50, 10).to(DEVICE)
    optimizerD = torch.optim.Adam(discriminator.parameters())  # Adam优化，几乎不用调参
    criterionD = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
    for epoch in range(200):
        lossD = torch.tensor(0.0).to(DEVICE)

        for i in range(50):
            x1 = randint(0, N - 1)
            y1 = randint(0, M - 1)
            x2 = randint(0, N - 1)
            y2 = randint(0, N - 1)
            t = randint(0, time - 51)
            x = torch.tensor(
                (np.append(u[t:t + 50, x1, y1], v[t:t + 50, x1, y1])).reshape(1, 50, 2).astype(float),
                dtype=torch.float32).to(
                DEVICE)
            y = torch.tensor(
                (np.append(u[t:t + 50, x2, y2], v[t:t + 50, x2, y2])).reshape(1, 50, 2).astype(float),
                dtype=torch.float32).to(
                DEVICE)
            lossD += criterionD(discriminator(x, y),
                                torch.tensor(abs(dx[x1 - x2]) + abs(dy[y1 - y2])).to(DEVICE).to(torch.float32))

        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        print(epoch, lossD)

    torch.save(discriminator, MODEL_PATH + 'D1')


def GetGraph(t):
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + 'D'):
        discriminator = torch.load(MODEL_PATH + 'D')
    discriminator.to(DEVICE)
    u, v = getDataFromNC.getData(r'database/jiduo.nc')
    u = u[:-67 * 6, :3, :3]
    v = v[:-67 * 6, :3, :3]
    u = SortTools.normalization(u)
    v = SortTools.normalization(v)
    time, N, M = u.shape
    print(time, N, M)
    graph = Graph.Graph(discriminator,50,DEVICE,u,v,SortTools.Priority_Queue(lambda x, y: x[1] > y[1],
                                          lambda x: '(' + str(x[0][0]) + ',' + str(x[0][1]) + ')-' '(' + str(
                                              x[0][2]) + ',' + str(
                                              x[0][3]) + ')'))
    edges1 = graph.GetGraph(t,int(N*M*1.5),3,1)
    edges0 = graph.GetGraph(t, int(N*M*1.5), 3, 0)
    graph.ShowGraph(edges0)
    graph.ShowGraph(edges1)
    print(graph.GetEdgeList(edges0))

if __name__ == '__main__':
    # TrainGAN()
    for i in range(3):
        t = randint(0, 19348 - 51)
        print(4321 + i)
        GetGraph(4321 + i)
