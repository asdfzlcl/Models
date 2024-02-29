import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.GCNNet import GCNNet
from database.MyDataset import MyOwnDataset
from TorchModels.GRUGCN import GRUGCN

if __name__ == '__main__':
    random.seed(1)
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + 'jiduoD'):
        discriminator = torch.load(MODEL_PATH + 'jiduoD')
    discriminator.to(DEVICE)
    dataSet0 = MyOwnDataset('database', 'jiduo', discriminator, DEVICE, 0)
    model0 = GRUGCN(DEVICE).to(DEVICE)

    optimizer0 = optim.Adam(model0.parameters(), lr=0.001, weight_decay=5e-4)
    print(model0)
    model0.train()
    print(len(dataSet0))
    length = len(dataSet0)
    N1 = int(length*0.6)
    N2 = length - N1
    print(dataSet0[0])
    h = torch.randn(19297, 81, 4).to(DEVICE)
    for epoch in range(2000):
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
        print(str(epoch) + 'loss0:', loss0)
        if epoch % 10 == 9:
            torch.save(model0, MODEL_PATH+ 'TORCHGRU')
            loss1 = torch.tensor(0.0).to(DEVICE)
            for i in range(N2):
                out, h[i + N1] = model0(dataSet0[i + N1], h, i + N1)
                h = h.to(DEVICE)
                # print(out.shape)
                loss1 += F.mse_loss(out, dataSet0[i + N1].y)
                print(str(epoch) + 'loss1:', loss1/N2)
                with open('result/NEWNET-torch.txt', 'a') as f:
                    f.write('test:'+str(float(loss1/N2)) + '\n')
        with open('result/NEWNET-torch.txt', 'a') as f:
            f.write(str(float(loss0/N1)) + '\n')
