import netCDF4 as nc
import numpy as np

filelist = ""
datafloder = "data/"
datalist = ["data1.nc","data2.nc","data3.nc","data4.nc","data5.nc","data6.nc"]

def takeFirst(elem):
    return elem[0]

def GetData():
    dataU = []
    dataV = []
    for i in datalist:
        filename = datafloder + i
        data = nc.Dataset(filename)
        print(data)
        time = data['time'][:]
        latitude = data['latitude'][:]
        longtitude = data['longitude'][:]
        datau = data['u100'][:]
        datav = data['v100'][:]
        # 构建数据库 格式为[[lat1,lon1,[[time1,v],[time2,v2]]],[lat2,lon2,[[time1,v],[time2,v2]]]]
        for x in latitude:
            for y in longtitude:
                f = True
                # 寻找匹配数据
                for item in dataU:
                    if x == item[0] and y == item[1] :
                        f = False
                        break
                if f:
                    dataU.append([x, y, []])
                    dataV.append([x, y, []])
        lenLat = len(latitude)
        lenLon = len(longtitude)
        lenTime = len(time)
        print(datav.shape)
        N = len(dataU)
        for i in range(lenTime):
            for j in range(lenLat):
                for k in range(lenLon):
                    for l in range(N):
                        if dataU[l][0] == latitude[j] and dataU[l][1] == longtitude[k]:
                            #print(i,j,k,datau[i][j][k])
                            if len(datau[0]) == 2:
                                if str(datau[i][1][j][k]) == "--":
                                    dataU[l][2].append([time[i], datau[i][0][j][k]])
                                else:
                                    dataU[l][2].append([time[i], datau[i][1][j][k]])
                                if str(datav[i][1][j][k]) == "--":
                                    dataV[l][2].append([time[i], datav[i][0][j][k]])
                                else:
                                    dataV[l][2].append([time[i], datav[i][1][j][k]])
                                break
                            else:
                                dataU[l][2].append([time[i], datau[i][j][k]])
                                dataV[l][2].append([time[i], datav[i][j][k]])
                                break

    # print(dataU)
    fileu = open('datau.txt', mode='w')
    filev = open('datav.txt', mode='w')
    for item in dataU:
        itemData = item[2]
        fileu.write(str(item[0])+","+str(item[1])+",\n")
        itemData.sort(key=takeFirst,reverse=False)
        for x in itemData:
            fileu.write(str(x[0]) + "," + str(x[1]) + ",")
        fileu.write("\n")
    fileu.close()
    for item in dataV:
        itemData = item[2]
        filev.write(str(item[0])+","+str(item[1])+",\n")
        itemData.sort(key=takeFirst,reverse=False)
        for x in itemData:
            filev.write(str(x[0]) + "," + str(x[1]) + ",")
        filev.write("\n")
    filev.close()








if __name__ == '__main__':
    GetData()
