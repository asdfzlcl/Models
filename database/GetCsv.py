import numpy as np
import csv
import pandas as pd

List = [
    "30E_60N_圣彼得堡",
    "74W_40N_纽约",
    "77W_39N_华盛顿",
    "78W_0N_基多",
    "86E_75N_喀拉海",
    "102E_28N_西昌",
    "103E_1N_新加坡",
    "111E_38N_岢岚"
]

filename = "data/103E_1N_新加坡/"

csv_name = "dataCSV/xinjiapo"

fileList = ["2013.csv","2014.csv","2015.csv","2016.csv","2017.csv","2018.csv","2019.csv","2020.csv","2021.csv"]

data = []
u = []
v = []

for name in fileList:
    fileData = pd.read_csv(filename+name, sep=',',header='infer',usecols=[0])
    data = data + (fileData.values[:,:].tolist())
    fileData = pd.read_csv(filename + name, sep=',', header='infer', usecols=[4])
    u = u + (fileData.values[:, :].tolist())
    fileData = pd.read_csv(filename + name, sep=',', header='infer', usecols=[5])
    v = v + (fileData.values[:, :].tolist())

# print(data)
data = [data[i][0].replace('-', '/')[:-3]  for i in range(len(data)) if i % 52 == 0]
data = data[:-1]
print(data[0:10])

u = np.array(u)
u = u.reshape((-1,52))
print(u.shape)

v = np.array(v)
v = v.reshape((-1,52))
print(v.shape)



for i in range(4):
    x = i*13
    y = (i+1)*13
    dataCSV = {'date': data}
    for j in range(13):
        dataCSV['h'+str(j)] = u[:-1,x+j].reshape(-1).tolist()
    dataCSV['OT'] = u[1:, x + 6].reshape(-1).tolist()
    df = pd.DataFrame(dataCSV)
    df.to_csv(csv_name+'u' + str(i)+'.csv', index=False)

    dataCSV = {'date': data}
    for j in range(13):
        dataCSV['h' + str(j)] = v[:-1, x + j].reshape(-1).tolist()
    dataCSV['OT'] = v[1:, x + 6].reshape(-1).tolist()
    df = pd.DataFrame(dataCSV)
    df.to_csv(csv_name + 'v' + str(i)+'.csv', index=False)

