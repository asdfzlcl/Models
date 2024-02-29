import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.GCNNet import GCNNet
from database.MyDataset import MyOwnDataset


def GCN(area):

    MODEL_PATH = "model/"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + area + 'D'):
        discriminator = torch.load(MODEL_PATH + area + 'D')
    discriminator.to(DEVICE)
    dataSet0 = MyOwnDataset('database', area, discriminator, DEVICE, 0)
    dataSet1 = MyOwnDataset('database', area, discriminator, DEVICE, 1)
    model0 = GCNNet().to(DEVICE)
    model1 = GCNNet().to(DEVICE)

    optimizer0 = optim.SGD(model0.parameters(), lr=0.000001, weight_decay=5e-4)
    print(model0)
    model0.train()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.000001, weight_decay=5e-4)
    print(model1)
    model1.train()
    print(len(dataSet0))
    print(dataSet0[0])
    length = len(dataSet0)
    N1 = int(length * 0.6)
    N2 = length - N1
    for epoch in range(200):
        optimizer0.zero_grad()
        loss0 = torch.tensor(0.0).to(DEVICE)
        for i in range(N1):
            out = model0(dataSet0[i])
            # print(out.shape)
            loss0 += F.mse_loss(out, dataSet0[i].y)
        loss0.backward()
        optimizer0.step()
        print(str(epoch) + 'loss0:', loss0 / N1)

        optimizer1.zero_grad()
        loss1 = torch.tensor(0.0).to(DEVICE)
        for i in range(N1):
            out = model1(dataSet0[i])
            # print(out.shape)
            loss1 += F.mse_loss(out, dataSet0[i].y)
        loss1.backward()
        optimizer1.step()
        print(str(epoch) + 'loss1:', loss1 / N1)
        with open('result/GCN_SGD' + area + '.txt', 'a') as f:
            f.write(str(float(loss0 / N1)) + ',' + str(float(loss1 / N1)) + '\n')
        if epoch % 10 == 9:
            optimizer0.zero_grad()
            loss0 = torch.tensor(0.0).to(DEVICE)
            for i in range(N2):
                out = model0(dataSet0[i + N1])
                # print(out.shape)
                loss0 += F.mse_loss(out, dataSet0[i + N1].y)
            print(str(epoch) + 'test-loss0:', loss0 / N2)

            optimizer1.zero_grad()
            loss1 = torch.tensor(0.0).to(DEVICE)
            for i in range(N2):
                out = model1(dataSet0[i + N1])
                # print(out.shape)
                loss1 += F.mse_loss(out, dataSet0[i + N1].y)
            print(area + str(epoch) + 'test-loss1:', loss1 / N2)
            with open('result/GCN_SGD' + area + '.txt', 'a') as f:
                f.write('test:' + str(float(loss0 / N2)) + ',' + str(float(loss1 / N2)) + '\n')

areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']

if __name__ == '__main__':
    for area in areaList[0:3]:
        GCN(area)
