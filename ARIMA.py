import torch
import torch.nn as nn
import statsmodels.api as sm
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
import GetData
import Tools

def work(id,f):
    x_np, y_np, N = GetData.GetData("database/datau.txt", id)
    y_np = (y_np + 10) / 25
    N = 300
    N1 = int(N * 0.7)
    N2 = int(N - N1)
    # print(N,N1,N2)
    ans = y_np[N-N2:N]
    predict = [0 for i in range(N2)]
    for i in range(N2):
        arima = sm.tsa.ARIMA(y_np[N1 + 1 + i - 16 : N1+1+i],order = (0,0,1)).fit()
        predict[i] = arima.forecast(1)[0]
    mse = Tools.MSE(ans,predict)
    print(str(id)+":"+str(mse))
    f.write(str(mse)+"\n")

if __name__=='__main__':
    f = open("ARIMA.txt", mode='w')
    for i in range(50):
        work(i,f)
    f.close()