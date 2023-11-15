import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from TorchModels import RNN
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData
import os

MODEL_PATH = "model/rnn-kalahai"
DATA_PATH = "database/kalahai.txt"
DEVICE_ID = "cuda:1"

torch.set_printoptions(precision=8)
TIME_STEP = 10 # rnn 时序步长数
INPUT_SIZE = 52*2 # rnn 的输入维度
OUTPUT_SIZE = 52*2
DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
H_SIZE = 128 # of rnn 隐藏单元个数
EPOCHS = 8000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)


print("DEVICE is "+ DEVICE_ID)

model = RNN.RNN(INPUT_SIZE,H_SIZE,OUTPUT_SIZE)

if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

model.train()
y_data,Range = GetData.GetDataFromTxt(DATA_PATH)

print(MODEL_PATH)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


N = y_data.shape[0]
N1 = int(N * 0.7)
N2 = N - N1

print("N="+str(N))
print("N1="+str(N1))
print("N2="+str(N2))

trainx = y_data[:N1 - 1]
trainy = y_data[1:N1]
testx = y_data[N1:N - 1]
testy = y_data[N1 + 1:N]


# 训练
Tx = torch.from_numpy(trainx[np.newaxis, :]) # shape (batch, time_step, input_size)
Ty = torch.from_numpy(trainy[np.newaxis, :])
Tx = Tx.to(torch.float32).to(DEVICE)
Ty = Ty.to(torch.float32).to(DEVICE)

# 测试
x = torch.from_numpy(testx[np.newaxis, :])  # shape (batch, time_step, input_size)
y = torch.from_numpy(testy[np.newaxis, :])
x = x.to(torch.float32).to(DEVICE)
y = y.to(torch.float32).to(DEVICE)


for step in range(EPOCHS):
    # x_np = np.sin(steps)
    # y_np = np.cos(steps)
    h_state = h_state.to(DEVICE)
    cell = cell.to(DEVICE)
    prediction, h_state = model(Tx, h_state, N1 - 1)  # rnn output
    # 这一步非常重要
    h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
    loss = criterion(prediction, Ty)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 2 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        h_state = h_state.to(DEVICE)
        cell = cell.to(DEVICE)
        prediction, h_state = model(x, h_state, N2 - 1)  # rnn output
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction, y)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        torch.save(model, MODEL_PATH)
        print(MODEL_PATH + str(step + 1) + ":" + str(loss))