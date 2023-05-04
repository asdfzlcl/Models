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
TIME_STEP = 10 # lstm 时序步长数
INPUT_SIZE = 1 # lstm 的输入维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS = 8000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)
id = 0

print(DEVICE)


class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gate_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def gru_work(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        r_gate = self.sigmoid(self.gate_r(combined))
        z_gate = self.sigmoid(self.gate_z(combined))
        combined01 = torch.cat((input,torch.mul(z_gate,r_gate)),1)
        h1_state = self.tanh(self.gate_h(combined01))

        h_state = torch.add(torch.mul((1-z_gate),hidden),torch.mul(h1_state,z_gate))
        output = self.output(h_state)
        output = self.sigmoid(output)
        return output,hidden

    def forward(self, x, h_state,N):
        outs = []

        for time_step in range(N):
            output,h_state = self.gru_work(x[:,time_step,:],h_state)
            outs.append(output[:])

        return torch.stack(outs, dim=1), h_state
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state

gru = GRU(1,H_SIZE,1).to(DEVICE)
optimizer = torch.optim.Adam(gru.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

gru.train()

x_data,y_data,N = GetData.GetData("database/datau.txt",id)
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

plt.figure(2)
for step in range(EPOCHS):
    # x_np = np.sin(steps)
    # y_np = np.cos(steps)
    h_state = h_state.to(DEVICE)
    prediction, h_state = gru(Tx, h_state, N1 - 1)  # rnn output
    # 这一步非常重要
    h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
    loss = criterion(prediction, Ty)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 200 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        h_state = h_state.to(DEVICE)
        prediction, h_state = gru(x, h_state, N2 - 1)  # rnn output
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction, y)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        print(str(step + 1) + ":" + str(loss))