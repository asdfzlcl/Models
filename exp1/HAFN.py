import torch
import torch.nn as nn
import numpy as np
from TorchModels import HAFN
import GetData
import os

city = "xinjiapo"
MODEL_PATH = "model/HAFN-"+city+"1"
DATA_PATH = "database/"+city+".txt"
Result_PATH = "result/"+city+".txt"
DEVICE_ID = "cuda:1"
LOAD_FLAG = False

torch.set_printoptions(precision=8)
TIME_STEP = 16
HISTORY_SIZE = 16
RNN_SIZE = 6
RNN_HIDDEN_SIZE = 8
INPUT_SIZE = TIME_STEP * 13
OUTPUT_SIZE = 6
HIDDEN_SIZE = 8
DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
EPOCHS = 8000  # 总共训练次数

print("DEVICE is " + DEVICE_ID)

model = HAFN.HAFN(INPUT_SIZE, HISTORY_SIZE * 2, RNN_SIZE, RNN_HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DEVICE)

if os.path.exists(MODEL_PATH) and LOAD_FLAG:
    model = torch.load(MODEL_PATH)

model.to(DEVICE)

for i in range(model.RNN_size):
    model.RNNList[i] = model.RNNList[i].to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())  # Adam优化，几乎不用调参
criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
criterion2 = nn.L1Loss()

model.train()
y_data, Range = GetData.GetDataFromTxt(DATA_PATH)

print(MODEL_PATH)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

N = y_data.shape[0]
N1 = int(N * 0.8)
N2 = N - N1 - OUTPUT_SIZE

print("N=" + str(N))
print("N1=" + str(N1))
print("N2=" + str(N2))

trainx = y_data[:N]
trainy = y_data[1:N,6]
testx = y_data[N1:N-1]
testy = y_data[N1 + 1:N,6]

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

min_loss = torch.tensor(1.0).to(DEVICE)
test_loss = torch.tensor(0.0).to(DEVICE)

for step in range(EPOCHS):
    loss = torch.tensor(0.0).to(DEVICE)
    hidden_list = []
    cycle = 1
    for i in range(RNN_SIZE):
        hidden = []
        for j in range(cycle):
            hidden.append(torch.zeros(1,RNN_HIDDEN_SIZE).to(DEVICE))
        cycle = cycle * 2
        hidden_list.append(hidden)
    for i in range(TIME_STEP + 365 + HISTORY_SIZE, N1):
        # print(i)
        prediction,hidden_list = model(Tx[0, i - TIME_STEP:i],hidden_list,Tx[0,i - 365 - HISTORY_SIZE:i - 365 + HISTORY_SIZE],i)
        # print(i)
        loss += criterion(prediction, Ty[0, i:i + OUTPUT_SIZE].view(1,-1)) / (N1 - (TIME_STEP + 365 + HISTORY_SIZE))
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if min_loss > loss:  # 每训练20个批次可视化一下效果，并打印一下loss
        min_loss = loss
        test_loss = torch.tensor(0.0).to(DEVICE)
        test_loss2 = torch.tensor(0.0).to(DEVICE)
        hidden_list = []
        cycle = 1
        for i in range(RNN_SIZE):
            hidden = []
            for j in range(cycle):
                hidden.append(torch.zeros(1, RNN_HIDDEN_SIZE).to(DEVICE))
            cycle = cycle * 2
            hidden_list.append(hidden)
        for i in range(N1, N - OUTPUT_SIZE):
            prediction, hidden_list = model(Tx[0, i - TIME_STEP:i], hidden_list,
                                            Tx[0, i - 365 - HISTORY_SIZE:i - 365 + HISTORY_SIZE], i)
            test_loss += criterion(prediction, Ty[0, i:i + OUTPUT_SIZE].view(1, -1)) / N2
            test_loss2 += criterion2(prediction, Ty[0, i:i + OUTPUT_SIZE].view(1, -1)) / N2
        # 这三行写在一起就可以
        optimizer.zero_grad()
        test_loss.backward()
        torch.save(model, MODEL_PATH)
        print("TEST RESULT MSE"+MODEL_PATH + str(step + 1) + ":" + str(test_loss))
        print("TEST RESULT MAE" + MODEL_PATH + str(step + 1) + ":" + str(test_loss2))
        with open(Result_PATH, "a") as file:
            file.write('>>>{}>>>'.format(step)+"\n")
            file.write("TEST RESULT MSE"+MODEL_PATH + str(step + 1) + ":" + str(test_loss)+"\n")
            file.write("TEST RESULT MAE" + MODEL_PATH + str(step + 1) + ":" + str(test_loss2)+"\n")
    print("TRAIN RESULT "+MODEL_PATH + str(step + 1) + ":" + str(loss) + "\n min_loss: "+ str(min_loss))
    print("NOW MIN TEST RESULT :" + MODEL_PATH + str(test_loss))