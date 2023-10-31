import numpy as np

def GetDataFromTxt(fileName):
    dataList = []

    with open('beijing.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataList.append(line.strip().split('\t'))

    print(np.array(dataList).shape)
    print(1024920/365/6/9)
    return dataList

if __name__=='__main__':
    GetDataFromTxt("beijing.txt")