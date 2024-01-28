import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.GCNNet import GCNNet
from database.MyDataset import MyOwnDataset

if __name__ == '__main__':
    random.seed(1)
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + 'D'):
        discriminator = torch.load(MODEL_PATH + 'D')
    discriminator.to(DEVICE)
    dataSet0 = MyOwnDataset('database', 'jiduo', discriminator, DEVICE, 0)
    dataSet1 = MyOwnDataset('database', 'jiduo', discriminator, DEVICE, 1)
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
    for epoch in range(200):
        optimizer0.zero_grad()
        loss0 = torch.tensor(0.0).to(DEVICE)
        for data in dataSet0:
            out = model0(data)
            # print(out.shape)
            loss0 += F.mse_loss(out, data.y)
        loss0.backward()
        optimizer0.step()
        print(str(epoch) + 'loss0:', loss0)

        optimizer1.zero_grad()
        loss1 = torch.tensor(0.0).to(DEVICE)
        for data in dataSet1:
            out = model1(data)
            # print(out.shape)
            loss1 += F.mse_loss(out, data.y)
        loss1.backward()
        optimizer1.step()
        print(str(epoch) + 'loss1:', loss1)

        with open('result/GCN_SGD.txt', 'a') as f:
            f.write(str(float(loss0)) + ',' + str(float(loss1)) + '\n')
