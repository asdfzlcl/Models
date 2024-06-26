import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from TorchModels.VMDK import VMDK
from database.MyDataset import MyDataset


def vmdk(area, V):
    MODEL_PATH = "model/" + area + str(V) + 'Q'
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
        model = VMDK(input_size=dataSet.Data.size,
                     VMD_K=dataSet.Data.vmd_size * 2,
                     kernel_size=5,
                     hidden_size=3,
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
            if data.y[i] < 0.5:
                continue
            loss = loss - data.y[i] * torch.log(y) - (1 - data.y[i]) * torch.log(1 - y)
        loss.backward()
        optimizer.step()
        # for i in range(model.Kernel_size):
        #     for j in range(i+1,model.Kernel_size):
        #         loss = loss - torch.sum((model.params['K'][i] - model.params['K'][j]) * (model.params['K'][i] - model.params['K'][j]))
        print(str(epoch) + 'loss:', loss / N1)
        with open('result/mynet-loss' + area + str(V) + 'Q.txt', 'a') as f:
            f.write(str(float(loss / N1)) + '\n')
        if epoch % 10 == 9:
            TP = 1
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
            if TP > 1:
                TP = TP - 1
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            print(epoch, 'P', P, 'F1', 2 * P * R / (P + R))
            print(TP, FP, FN)
            torch.save(model, MODEL_PATH)
            with open('result/mynet' + area + str(V) + 'Q.txt', 'a') as f:
                f.write(str(epoch) + 'P:' + str(P) + ' F1:' + str(2 * P * R / (P + R)) + '\n')
                f.write(str(epoch) + 'TP:' + str(TP) + 'FP:' + str(FP) + 'FN:' + str(FN) + '\n')


areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    for area in areaList:
        for v in [0.5, 1, 1.5, 2]:
            vmdk(area, v)
