import torch
import netCDF4 as nc
import matplotlib.pyplot as plt

if __name__=="__main__":
    data = open("database/datav.txt", mode='r')
    positon = ""
    i = 0
    for line in data:
        positon = line.split(",")
        i = i + 1
        if i %2 == 1:
            print(positon[0:2])
