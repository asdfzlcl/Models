import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from TorchModels import Transformer
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData
import os

MODEL_PATH = "model/transformer-kalahai"
DATA_PATH = "database/kalahai.txt"
DEVICE_ID = "cuda:2"
LOAD_FLAG = True

torch.set_printoptions(precision=8)
TIME_STEP = 16
INPUT_SIZE = TIME_STEP * 52 * 2
OUTPUT_SIZE = 52 * 2
DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
EPOCHS = 8000  # 总共训练次数

print("DEVICE is " + DEVICE_ID)

model = Transformer.Transformer(INPUT_SIZE, 128 ,OUTPUT_SIZE)

if os.path.exists(MODEL_PATH) and LOAD_FLAG:
    model = torch.load(MODEL_PATH)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())  # Adam优化，几乎不用调参
criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

model.train()
y_data, Range = GetData.GetDataFromTxt(DATA_PATH)

print(MODEL_PATH)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

N = y_data.shape[0]
N1 = int(N * 0.7)
N2 = N - N1

print("N=" + str(N))
print("N1=" + str(N1))
print("N2=" + str(N2))

trainx = y_data[:N1]
trainy = y_data[1:N1]
testx = y_data[N1:N]
testy = y_data[N1 + 1:N]

# 训练
Tx = torch.from_numpy(trainx[np.newaxis, :])  # shape (batch, time_step, input_size)
Ty = torch.from_numpy(trainy[np.newaxis, :])
Tx = Tx.to(torch.float32).to(DEVICE)
Ty = Ty.to(torch.float32).to(DEVICE)

# 测试
x = torch.from_numpy(testx[np.newaxis, :])  # shape (batch, time_step, input_size)
y = torch.from_numpy(testy[np.newaxis, :])
x = x.to(torch.float32).to(DEVICE)
y = y.to(torch.float32).to(DEVICE)

print("训练开始")

for step in range(EPOCHS):
    loss = torch.tensor(0.0).to(DEVICE)
    for i in range(TIME_STEP, N1):
        prediction = model(Tx[0, i - TIME_STEP:i])
        loss += criterion(prediction, Tx[0, i].view(1,-1)) / (N1 - TIME_STEP)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 2 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        loss = torch.tensor(0.0).to(DEVICE)
        for i in range(TIME_STEP, N2):
            prediction = model(x[0, i - TIME_STEP:i])
            loss += criterion(prediction, x[0, i].view(1,-1)) / (N2 - TIME_STEP)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        torch.save(model, MODEL_PATH)
        print(MODEL_PATH + str(step + 1) + ":" + str(loss))
