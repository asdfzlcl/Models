import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData

TIME_STEP = 10 # rnn 时序步长数
INPUT_SIZE = 1 # rnn 的输入维度
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
H_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS = 8000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态

print(DEVICE)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.Hidden_layer = nn.Linear(input_size+hidden_size,hidden_size)
        self.Out_layer1 = nn.Linear(input_size+hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()

    def rnn_work(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.Hidden_layer(combined)
        output = self.sigmoid(self.Out_layer1(combined))
        return output,hidden

    def forward(self, x, h_state,N):
        outs = []

        for time_step in range(N):
            output,h_state = self.rnn_work(x[:,time_step,:],h_state)
            outs.append(output[:])

        return torch.stack(outs, dim=1), h_state
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state

torch.set_printoptions(precision=8)
rnn = RNN(1,H_SIZE,1).to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

rnn.train()
# plt.figure(2)
x_data, y_data, N = GetData.GetData("database/datau.txt", 0)
y_data = (y_data+10)/25
for step in range(EPOCHS):
    # x_np = np.sin(steps)
    # y_np = np.cos(steps)
    # x_np,y_np,N = GetData.GetData("database/datau.txt",0)
    N = 300
    N1 = int(N*0.7)
    N2 = N - N1
    steps = x_data[:N1]
    x_np = y_data[:N1 - 1]
    y_np = y_data[1:N1]
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]) # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x=x.to(DEVICE)
    h_state=h_state.to(DEVICE)
    prediction, h_state = rnn(x, h_state,N1 - 1) # rnn output
    # 这一步非常重要
    h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
    loss = criterion(prediction.cpu(), y)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step+1)%200==0: #每训练20个批次可视化一下效果，并打印一下loss
        x_np = y_data[N1:N - 1]
        y_np = y_data[N1 + 1: N]
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        x = x.to(DEVICE)
        h_state = h_state.to(DEVICE)
        prediction, h_state = rnn(x, h_state, len(x_np))  # rnn output
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction.cpu(), y)
        loss.backward()
        print(str(step+1)+":"+str(loss))