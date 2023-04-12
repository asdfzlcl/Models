import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData

TIME_STEP = 10 # lstm 时序步长数
INPUT_SIZE = 1 # lstm 的输入维度
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 32 # of lstm 隐藏单元个数
EPOCHS = 100000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)
print(DEVICE)
height = 3
id = 0
datamodel = "model"+str(height)+"-"+str(id)+".pkl"
dataresult = "mtm"+str(height)+"-"+str(id)+".txt"

class MTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,tree_height):
        super(MTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tree_height = tree_height
        self.Left = [[]]
        self.Right = [[]]
        self.Other = [[nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)]]
        self.hidden = []
        self.change = []
        for i in range(tree_height):
            self.Left.append([nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE)])
            self.Right.append([nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE)])
            self.Other.append([nn.Linear(2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(2 * hidden_size, hidden_size).to(DEVICE)])

        self.output = nn.Linear(hidden_size * tree_height + input_size,output_size).to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def init_hidden(self):
        self.hidden = []
        for i in range((1<<self.tree_height)+1):
            self.hidden.append(torch.zeros(1,self.hidden_size).to(DEVICE))
        self.change = [0 for i in range((1<<self.tree_height)+1)]

    def update(self,now,L,R,id,height,input):
        if height == self.tree_height:
            return self.hidden[id]
        ls = now << 1
        rs = ls + 1
        M = (L + R) >> 1
        if self.change[now] == 0:
            self.hidden[ls] = self.sigmoid(self.Other[height][0](torch.cat((self.hidden[now], self.hidden[ls]), 1)))
            self.hidden[rs] = self.sigmoid(self.Other[height][1](torch.cat((self.hidden[now], self.hidden[rs]), 1)))
        combine_input = torch.cat((input,self.hidden[now]),1)
        self.change[id] = 0
        if id<=M:
            self.hidden[ls] = self.sigmoid(self.Left[height][0](torch.cat((combine_input, self.hidden[ls]), 1)))
            self.hidden[rs] = self.sigmoid(self.Left[height][1](torch.cat((combine_input, self.hidden[rs]), 1)))
            OUT = torch.cat((self.hidden[id],self.update(ls, L, M, id, height + 1, input)),1)
            self.change[rs] = 1
        else:
            self.hidden[ls] = self.sigmoid(self.Right[height][0](torch.cat((combine_input, self.hidden[ls]), 1)))
            self.hidden[rs] = self.sigmoid(self.Right[height][1](torch.cat((combine_input, self.hidden[rs]), 1)))
            OUT = torch.cat((self.hidden[id], self.update(rs, M + 1, R, id, height + 1, input)), 1)
            self.change[ls] = 1
        return OUT

    def MTM_work(self,input,id):
        combine_input = torch.cat((input,self.hidden[1]),1)
        self.hidden[1] = self.Other[0][0](combine_input)
        OUT = torch.cat((input,self.update(1,1,1<<(self.tree_height-1),id,1,input)),1)
        out = self.sigmoid(self.output(OUT))
        return out

    def forward(self, x,N):
        outs = []
        turn = 1 << (self.tree_height - 1)
        id = 1
        for time_step in range(N):
            output = self.MTM_work(x[:,time_step,:],id)
            id = id + 1
            if id>turn:
                id = id - turn
            outs.append(output[:])

        return torch.stack(outs, dim=1)
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state


mtm = MTM(1,H_SIZE,1,height).to(DEVICE)
optimizer = torch.optim.Adam(mtm.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差
x_data,y_data,N = GetData.GetData("database/datau.txt",id)
y_data = (y_data+10)/25
mtm.train()
N = 300
N1 = int(N * 0.3)
N2 = N - N1
steps = x_data[:N1]
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
    mtm.init_hidden()
    # mtm = mtm.to(DEVICE)
    prediction = mtm(Tx, len(trainy)) # rnn output
    # 这一步非常重要
    # h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
    # cell = cell.data
    loss = criterion(prediction, Ty)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step+1)%200 == 0: #每训练20个批次可视化一下效果，并打印一下loss
        mtm.init_hidden()
        prediction = mtm(x, len(testy))
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction, y)
        loss.backward()
        print("EPOCHS: {},Loss:{:4f}".format(step+1,loss))
        torch.save(mtm,datamodel)
        file = open(dataresult,"a")
        file.write("EPOCHS: {},Loss:{:4f}\n".format(step+1,loss))
        file.flush()
        file.close()
        # plt.plot(steps, y_np.flatten(), 'r-')
        # plt.plot(steps, prediction.cpu().data.numpy().flatten(), 'b-')
        # plt.draw()
        # plt.pause(0.1)