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
from sklearn.svm import SVR

def reshape(x):
    out = []
    for a in x:
        for b in a:
            out.append(b)
    return out

def work(id,f):
    x_np, y_np, N = GetData.GetData("database/datau.txt", id)
    y_np = (y_np + 10) / 25
    N1 = int(N * 0.7)
    N2 = int(N - N1)
    # print(N,N1,N2)
    N = 200
    # print(N)
    trainx = []
    trainy = []
    testx = []
    testy = []
    for i in range(N2):
        testx.append(y_np[N - i - 1 - 16:N - i - 1])
        testy.append(y_np[N - i - 1])
    for i in range(N1 - 16):
        trainx.append(y_np[i:i + 16])
        trainy.append(y_np[i + 16])
    clf = SVR(kernel = 'rbf',C = 3)
    # clf = SVR(kernel='linear', C=3)
    clf.fit(trainx,trainy)
    predict = clf.predict(testx)
    mse = Tools.MSE(testy,predict)
    print(str(id)+":"+str(mse))
    # print(mse)
    f.write(str(mse)+"\n")

if __name__=='__main__':
    f = open("SVR.txt", mode='w')
    for i in range(50):
        work(i,f)
    f.close()