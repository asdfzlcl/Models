import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.GCNNet import GCNNet
from database.MyDataset import MyOwnDataset
from TorchModels.MyGRUGCN import MYGRUGCN

if __name__ == '__main__':
    random.seed(1)
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + 'D'):
        discriminator = torch.load(MODEL_PATH + 'D')
    discriminator.to(DEVICE)
    dataSet0 = MyOwnDataset('database', 'jiduo', discriminator, DEVICE, 0)
    model0 = MYGRUGCN(DEVICE).to(DEVICE)
    if os.path.exists(MODEL_PATH + 'GRUGCN'):
        model0 = torch.load(MODEL_PATH + 'GRUGCN')
    model0 = model0.to(DEVICE)
    model0.DEVICE = DEVICE

    optimizer0 = optim.Adam(model0.parameters())
    print(model0)
    model0.train()
    print(len(dataSet0))
    length = len(dataSet0)
    print(dataSet0[0])
    h = torch.randn(19297, 81, 4).to(DEVICE)
    for epoch in range(500):
        h = h.data
        optimizer0.zero_grad()
        loss0 = torch.tensor(0.0).to(DEVICE)
        for i in range(length):
            out, h[i] = model0(dataSet0[i], h, i)
            h = h.to(DEVICE)
            # print(out.shape)
            loss0 += F.mse_loss(out, dataSet0[i].y)
        loss0.backward()
        optimizer0.step()
        print(str(epoch) + 'loss0:', loss0)
        if epoch % 10 == 9:
            torch.save(model0, MODEL_PATH+ 'GRUGCN')
        with open('result/NEWNET-mytorch8.txt', 'a') as f:
            f.write(str(float(loss0)) + '\n')
