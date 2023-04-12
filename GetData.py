import numpy as np


def GetData(filename,id):
    data = open(filename, mode='r')
    for i in range(id - 1):
        positon = data.readline().split(",")
        Data = data.readline().split(",")
    positon = data.readline().split(",")
    Data = data.readline().split(",")
    x = Data[:-1:2]
    y = Data[1::2]
    for i in range(len(x)):
        x[i] = float(x[i])
        y[i] = float(y[i])
    maxx = max(y)
    minn = min(y)
    y = np.array(y, dtype=np.float32)
    y = (y-minn)/(maxx-minn)
    return np.array(x,dtype=np.float32),y,len(x)

# print(GetData("database/datau.txt", 15))