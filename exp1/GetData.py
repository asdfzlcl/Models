import numpy as np

def GetDataFromTxt(fileName):
    dataList = []
    print("开始读取文件")
    with open(fileName, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataList.append(line.strip().split('\t'))
    print("读取文件完成，开始归一化")
    data = np.array(dataList,dtype=np.float32)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    print(np.max(max_values - min_values))
    print(min_values,max_values)
    data = (data - min_values)
    data = data/np.max(max_values - min_values)
    print("归一化完成")

    # print(np.array(dataList).shape)
    # print(1024920/365/6/9)
    # print(1024920/52)
    # print(dataList[0:52])
    data = data.reshape((-1,52*2))
    data = data[:,[i for i in range(13)]]

    return data,max_values - min_values

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
    print(data.shape)