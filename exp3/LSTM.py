import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.VMDK import VMDK
from database.MyDataset import MyDataset
from TorchModels.MYGRU import GRU

def vmdk(area, V):
    MODEL_PATH = "model/gru" + area + str(V)
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    dataSet = MyDataset(area, DEVICE, V)
    data = dataSet.Data.to(DEVICE)
    data.x = data.x.to(DEVICE)
    data.y = data.y.to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH)
    else:
        model = GRU(input_size=10,
                     hidden_size=16,
                    output_size=5,
                    DEVICE=DEVICE)
    model = model.to(DEVICE)

    print(model)
    optimizer = optim.Adam(model.parameters())
    model.train()
    num = data.num
    N1 = int(num * 0.6)
    N2 = num - N1
    for epoch in range(500):
        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(DEVICE)
        for i in range(N1):
            y = model(data.x[i])
            loss = loss - data.y[i] * torch.log(y) - (1 - data.y[i]) * torch.log(1 - y)
        loss.backward()
        optimizer.step()
        print(str(epoch) + 'loss:', loss / N1)
        with open('result/gru' + area + str(V) + '.txt', 'a') as f:
            f.write(str(float(loss / N1)) + '\n')
        if epoch % 10 == 9:
            TP = 0
            FP = 0
            FN = 0
            for i in range(N2):
                y = model(data.x[i + N1])
                if y > 0.5:
                    if data.y[i + N1] > 0.9:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                else:
                    if data.y[i + N1] > 0.9:
                        FN = FN + 1
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            print(epoch, 'P', P, 'F1', 2 * P * R / (P + R))
            torch.save(model, MODEL_PATH)
            with open('result/gru' + area + str(V) + '.txt', 'a') as f:
                f.write(str(epoch) + 'P:' + str(P) + ' F1:' + str(2 * P * R / (P + R)) + '\n')


areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    for area in areaList:
        for v in [0.5, 1, 1.5, 2]:
            vmdk(area, v)
