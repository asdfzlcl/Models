import numpy as np

def MSE(x,y):
    N = len(x)
    sum =0.0
    for i in range(N):
        sum = sum + (x[i] - y[i])**2
    return np.sum(sum)/N/len(sum)