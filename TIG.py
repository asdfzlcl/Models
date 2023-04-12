import GetData
import RNN
import GRU
import LSTM
import MTMv2
import torch
import torch.nn as nn
import numpy as np

INPUT_SIZE = 1 # lstm 的输入维度
DEVICE =  "cuda"
H_SIZE = 32 # of lstm 隐藏单元个数
EPOCHS = 100000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)
torch.set_printoptions(precision=8)
print(DEVICE)

def testRNN(id):
    torch.set_printoptions(precision=8)
    rnn = RNN.RNN(1, H_SIZE, 1).to(DEVICE)
    optimizer = torch.optim.Adam(rnn.parameters())  # Adam优化，几乎不用调参
    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

    rnn.train()
    # plt.figure(2)
    x_data, y_data, N = GetData.GetData("database/datau.txt", id)
    # y_data = (y_data + 10) / 25
    for step in range(EPOCHS):
        # x_np = np.sin(steps)
        # y_np = np.cos(steps)
        # x_np,y_np,N = GetData.GetData("database/datau.txt",0)
        N = 300
        N1 = int(N * 0.3)
        N2 = N - N1
        steps = x_data[:N1]
        x_np = y_data[:N1 - 1]
        y_np = y_data[1:N1]
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        x = x.to(DEVICE)
        h_state = h_state.to(DEVICE)
        prediction, h_state = rnn(x, h_state, N1 - 1)  # rnn output
        # 这一步非常重要
        h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
        loss = criterion(prediction.cpu(), y)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
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
            print(str(step + 1) + ":" + str(loss))


def testMTM(height,id):
    datamodel = "model" + str(height) + "-" + str(id) + ".pkl"
    dataresult = "mtm" + str(height) + "-" + str(id) + ".txt"
    mtm = MTMv2.MTM(1, H_SIZE, 1, height).to(DEVICE)
    optimizer = torch.optim.Adam(mtm.parameters())  # Adam优化，几乎不用调参
    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
    x_data, y_data, N = GetData.GetData("database/datau.txt", id)
    # y_data = (y_data + 10) / 25
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
    Tx = torch.from_numpy(trainx[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
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
        prediction = mtm(Tx, len(trainy))  # rnn output
        # 这一步非常重要
        # h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
        # cell = cell.data
        loss = criterion(prediction, Ty)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
            mtm.init_hidden()
            prediction = mtm(x, len(testy))
            # 这一步非常重要
            h_state = h_state.data  # 重置隐藏层的状态, 切断和前一次迭代的链接
            loss = criterion(prediction, y)
            loss.backward()
            print("EPOCHS: {},Loss:{:4f}".format(step + 1, loss))
            torch.save(mtm, datamodel)
            file = open(dataresult, "a")
            file.write("EPOCHS: {},Loss:{:4f}\n".format(step + 1, loss))
            file.flush()
            file.close()
            # plt.plot(steps, y_np.flatten(), 'r-')
            # plt.plot(steps, prediction.cpu().data.numpy().flatten(), 'b-')
            # plt.draw()
            # plt.pause(0.1)
