from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,datasets
from sklearn.svm import SVC
import getdata
from sklearn.ensemble import IsolationForest

def GMM(area, V):
    x,y = getdata.GetDataNp(area,V)
    shape = x.shape
    num = shape[0]
    N1 = int(num * 0.6)
    N2 = num - N1
    model = IsolationForest(n_estimators=1000,
                      max_samples=1024)

    T = 0
    for i in range(N1):
        if y[i][0]>0.5:
            T = T + 1
    X = np.empty([T,60], dtype = float)
    num = 0
    for i in range(N1):
        if y[i][0]>0.5:
            X[num] = x[i]
            num = num + 1
    model.fit(X)
    out = model.predict(x[N1:])
    TP = 0
    FP = 0
    FN = 0
    for i in range(N2):
        if out[i] > 0.5:
            if y[i + N1] > 0.9:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if y[i + N1] > 0.9:
                FN = FN + 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print(area + str(V) + 'P', P, 'F1', 2 * P * R / (P + R))
    with open('result/IF.txt', 'a') as f:
        f.write(area + str(V) +'P:' + str(P) + ' F1:' + str(2 * P * R / (P + R)) + '\n')



areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    for area in areaList:
        for v in [0.5, 1, 1.5, 2]:
            GMM(area, v)
