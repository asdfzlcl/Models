import numpy as py

def MSE(x,y):
    N = len(x)
    sum =0.0
    for i in range(N):
        sum = sum + (x[i] - y[i])**2
    return sum/N