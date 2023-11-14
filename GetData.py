import numpy as np

def GetDataFromTxt(fileName):
    dataList = []

    with open(fileName, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataList.append(line.strip().split('\t'))

    data = np.array(dataList,dtype=np.float32)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    data = (data - min_values)/np.max(max_values - min_values)

    # print(np.max(max_values - min_values))
    # print(min_values,max_values)

    # print(np.array(dataList).shape)
    # print(1024920/365/6/9)
    # print(1024920/52)
    # print(dataList[0:52])
    return data.reshape((-1,52*2)),max_values - min_values

def GetDataRnn(fileName,batch_size):
    dataList = GetDataFromTxt(fileName)
    N = len(dataList)
    datax = []
    datay = []
    for i in range(N-batch_size):
        datax.append(dataList[i:i+batch_size-1])
        datay.append(dataList[i+batch_size-1])
    return datax,datay

if __name__=='__main__':
    data,range = GetDataFromTxt("database/beijing.txt")
    print(data[0],range)