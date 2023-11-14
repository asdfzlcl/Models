import numpy as np
import csv
import pandas as pd

filename = "data/74W_40N_纽约/"

fileList = ["2013.csv","2014.csv","2015.csv","2016.csv","2017.csv","2018.csv","2019.csv","2020.csv","2021.csv"]

data = []

for name in fileList:
    fileData = pd.read_csv(filename+name, sep=',',header='infer',usecols=[4,5])
    data=data + (fileData.values[:,:].tolist())
    print(fileData.values.shape)

# print(data)
print(np.array(data).shape)
print(113880/365/6)

with open('niuyue.txt', 'w') as file:
    for row in data:
        file.write('\t'.join(map(str, row)) + '\n')
