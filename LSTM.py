import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData

torch.set_printoptions(precision=8)
TIME_STEP = 10 # rnn 时序步长数
INPUT_SIZE = 1 # rnn 的输入维度
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
H_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS = 8000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
id = 0
cell = torch.zeros(1,H_SIZE)


print(DEVICE)

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,cell_size,output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate_f = nn.Linear(input_size + hidden_size, cell_size)
        self.gate_i = nn.Linear(input_size + hidden_size, cell_size)
        self.gate_o = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def lstm_work(self,input,hidden,cell):
        combined = torch.cat((input,hidden),1)
        f_gate = self.sigmoid(self.gate_f(combined))
        i_gate = self.sigmoid(self.gate_i(combined))
        o_gate = self.sigmoid(self.gate_o(combined))
        z_state = self.tanh(self.gate_i(combined))
        cellout = torch.add(torch.mul(cell,f_gate),torch.mul(z_state,i_gate))
        hidden = torch.mul(self.tanh(cellout),o_gate)
        output = self.sigmoid(self.output(hidden))
        return output,hidden,cellout

    def forward(self, x, h_state,cell,N):
        outs = []

        for time_step in range(N):
            output,h_state,cell = self.lstm_work(x[:,time_step,:],h_state,cell)
            outs.append(output[:])

        return torch.stack(outs, dim=1), h_state,cell
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state

lstm = LSTM(1,H_SIZE,H_SIZE,1).to(DEVICE)
optimizer = torch.optim.Adam(lstm.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

lstm.train()
x_data,y_data,N = GetData.GetData("database/datau.txt",id)
lstm.train()
N = 300
N1 = int(N * 0.7)
N2 = N - N1
steps = x_data[N1 + 1:N]
trainx = y_data[:N1 - 1]
trainy = y_data[1:N1]
testx = y_data[N1:N - 1]
testy = y_data[N1 + 1:N]


# 训练
Tx = torch.from_numpy(trainx[np.newaxis, :, np.newaxis]) # shape (batch, time_step, input_size)
Ty = torch.from_numpy(trainy[np.newaxis, :, np.newaxis])
Tx = Tx.to(DEVICE)
Ty = Ty.to(DEVICE)

# 测试
x = torch.from_numpy(testx[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
y = torch.from_numpy(testy[np.newaxis, :, np.newaxis])
x = x.to(DEVICE)
y = y.to(DEVICE)

for step in range(EPOCHS):
    # x_np = np.sin(steps)
    # y_np = np.cos(steps)
    h_state = h_state.to(DEVICE)
    cell = cell.to(DEVICE)
    prediction, h_state,cell = lstm(Tx, h_state,cell, N1 - 1)  # rnn output
    # 这一步非常重要
    h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
    cell = torch.zeros(1,H_SIZE)
    loss = criterion(prediction, Ty)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 200 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        h_state = h_state.to(DEVICE)
        cell = cell.to(DEVICE)
        prediction, h_state,cell = lstm(x, h_state,cell, N2 - 1)  # rnn output
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        cell = torch.zeros(1, H_SIZE)
        loss = criterion(prediction, y)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        print(str(step + 1) + ":" + str(loss))