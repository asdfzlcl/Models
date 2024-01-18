import torch
import torch.nn as nn
import numpy as np
import GetData

TIME_STEP = 10 # lstm 时序步长数
INPUT_SIZE = 1 # lstm 的输入维度
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 32 # of lstm 隐藏单元个数
EPOCHS = 100000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)
print(DEVICE)


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
        for i in range(tree_height):
            self.Left.append([nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE)])
            self.Right.append([nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE)])
            self.Other.append([nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE),
                             nn.Linear(input_size + 2 * hidden_size, hidden_size).to(DEVICE)])

        self.output = nn.Linear(hidden_size * tree_height + input_size,output_size).to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def init_hidden(self):
        self.hidden = []
        for i in range((1<<self.tree_height)+1):
            self.hidden.append(torch.zeros(1,self.hidden_size))
            self.hidden[i] = self.hidden[i].to(DEVICE)

    def update(self,now,L,R,id,height,input):
        if height == self.tree_height:
            return self.hidden[now]
        ls = now <<1
        rs = ls + 1
        M = (L + R) >> 1
        combine_input = torch.cat((input,self.hidden[now]),1)
        if id<L or id>R:
            self.hidden[ls]=self.sigmoid(self.Other[height][0](torch.cat((combine_input,self.hidden[ls]),1))).to(DEVICE)
            self.hidden[rs] = self.sigmoid(self.Other[height][1](torch.cat((combine_input,self.hidden[rs]),1))).to(DEVICE)
            self.update(ls,L,M,id,height+1,input)
            self.update(rs, M + 1, R, id, height + 1, input)
        else:
            if id<=M:
                self.hidden[ls] = self.sigmoid(self.Left[height][0](torch.cat((combine_input, self.hidden[ls]), 1))).to(DEVICE)
                self.hidden[rs] = self.sigmoid(self.Left[height][1](torch.cat((combine_input, self.hidden[rs]), 1))).to(DEVICE)
                OUT = torch.cat((self.hidden[now],self.update(ls, L, M, id, height + 1, input)),1)
                self.update(rs, M + 1, R, id, height + 1, input)
            else:
                self.hidden[ls] = self.sigmoid(self.Right[height][0](torch.cat((combine_input, self.hidden[ls]), 1))).to(DEVICE)
                self.hidden[rs] = self.sigmoid(self.Right[height][1](torch.cat((combine_input, self.hidden[rs]), 1))).to(DEVICE)
                self.update(ls, L, R, id, height + 1, input)
                OUT = torch.cat((self.hidden[now], self.update(rs, M + 1, R, id, height + 1, input)), 1)
            return OUT

    def MTM_work(self,input,id):
        combine_input = torch.cat((input.to(DEVICE),self.hidden[1].to(DEVICE)),1)
        self.hidden[1] = self.Other[0][0](combine_input).to(DEVICE)
        OUT = torch.cat((input,self.update(1,1,1<<(self.tree_height-1),id,1,input).to(DEVICE)),1)
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

mtm = MTM(1,H_SIZE,1,4).to(DEVICE)
optimizer = torch.optim.Adam(mtm.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差
x_data,y_data,N = GetData.GetData("database/datau.txt", 0)
y_data = (y_data+10)/25
mtm.train()
# plt.figure(2)
for step in range(EPOCHS):
    # x_np = np.sin(steps)
    # y_np = np.cos(steps)
    # N = 30
    N1 = int(N * 0.3)
    N2 = N - N1
    steps = x_data[:N1]
    x_np = y_data[:N1 - 1]
    y_np = y_data[1:N1]
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]) # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x=x.to(DEVICE)
    mtm.init_hidden()
    # mtm = mtm.to(DEVICE)
    prediction = mtm(x, len(y_np)) # rnn output
    # 这一步非常重要
    # h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
    # cell = cell.data
    loss = criterion(prediction.cpu(), y)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step+1)%200 == 0: #每训练20个批次可视化一下效果，并打印一下loss
        x_np = y_data[N1:N - 1]
        y_np = y_data[N1 + 1: N]
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        x = x.to(DEVICE)
        mtm.init_hidden()
        h_state = h_state.to(DEVICE)
        prediction = mtm(x, len(y_np))
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction.cpu(), y)
        loss.backward()
        print("EPOCHS: {},Loss:{:4f}".format(step+1,loss))
        # plt.plot(steps, y_np.flatten(), 'r-')
        # plt.plot(steps, prediction.cpu().data.numpy().flatten(), 'b-')
        # plt.draw()
        # plt.pause(0.1)