import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.GCNNet import GCNNet
from database.MyDataset import MyOwnDataset
from TorchModels.MyGRUGCN import MYGRUGCN


def MYNET(area):
    random.seed(1)
    MODEL_PATH = "model/"
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + area + 'D'):
        discriminator = torch.load(MODEL_PATH + area + 'D')
    discriminator.to(DEVICE)
    dataSet0 = MyOwnDataset('database', area, discriminator, DEVICE, 0)
    model0 = MYGRUGCN(DEVICE).to(DEVICE)
    if os.path.exists(MODEL_PATH + area + 'GRUGCN'):
        model0 = torch.load(MODEL_PATH + area + 'GRUGCN')
    model0 = model0.to(DEVICE)
    model0.DEVICE = DEVICE

    optimizer0 = optim.Adam(model0.parameters())
    print(model0)
    model0.train()
    print(len(dataSet0))
    length = len(dataSet0)
    N1 = int(length * 0.6)
    N2 = length - N1
    print(dataSet0[0])
    h = torch.randn(19297, 81, 4).to(DEVICE)
    for epoch in range(500):
        h = h.data
        optimizer0.zero_grad()
        loss0 = torch.tensor(0.0).to(DEVICE)
        for i in range(N1):
            out, h[i] = model0(dataSet0[i], h, i)
            h = h.to(DEVICE)
            # print(out.shape)
            loss0 += F.mse_loss(out, dataSet0[i].y)
        loss0.backward()
        optimizer0.step()
        print(str(epoch) + 'loss0:', loss0 / N1)
        if epoch % 10 == 9:
            loss1 = torch.tensor(0.0).to(DEVICE)
            for i in range(N2):
                out, h[i + N1] = model0(dataSet0[i + N1], h, i + N1)
                h = h.to(DEVICE)
                # print(out.shape)
                loss1 += F.mse_loss(out, dataSet0[i + N1].y)
                print(str(epoch) + 'loss1:', loss1 / N2)
            torch.save(model0, MODEL_PATH + area + 'GRUGCN')
            with open('result/mynet' + area + '.txt', 'a') as f:
                f.write('test' + str(float(loss1 / N2)) + '\n')
        with open('result/mynet' + area + '.txt', 'a') as f:
            f.write(str(float(loss0 / N1)) + '\n')


areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    for area in areaList[0:3]:
        MYNET(area)
