from database import MyDataset
import torch
import numpy as np


def GetDataNp(area,V):
    DEVICE = "cpu"
    dataSet = MyDataset.MyDataset(area, DEVICE, V)
    data = dataSet.Data.to(DEVICE)
    data.x = data.x.to(DEVICE)
    data.y = data.y.to(DEVICE)
    num = data.num
    x = np.empty([num,60], dtype = float)
    y = np.empty([num, 1], dtype=float)
    for i in range(num):
        for j in range(5):
            x[i,:30] = x[i,:30] + data.x[i][j].numpy()
            x[i, 30:] = x[i, 30:] + data.x[i][j+5].numpy()
        y[i] = data.y[i].numpy()
    return x,y

areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    area = areaList[0]
    v = 0.5
    GetDataNp(area,v)