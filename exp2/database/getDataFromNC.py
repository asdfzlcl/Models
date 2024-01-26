import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

from exp2.TorchModels import SortTools


def getData(filename):
    nf = nc.Dataset(filename, 'r')
    return np.array(nf.variables['u'][:, 0, :, :].tolist()), np.array(nf.variables['v'][:, 0, :, :].tolist())


if __name__ == '__main__':
    # nf = nc.Dataset(r'jiduo.nc', 'r')
    # print(nf.variables.keys())
    # lon = nf.variables['longitude'][:]
    # lat = nf.variables['latitude'][:]
    # expver = nf.variables['expver'][:]
    # time = nf.variables['time'][:]
    # u = nf.variables['u'][:, 0, :, :]
    # print(lon)
    # print(lon.shape)
    # print(lat)
    # print(lat.shape)
    # print(time.shape)
    # print(expver)
    # print(expver.shape)
    # print(u.shape)
    u, v = getData(r'jiduo.nc')

    u = u[:-67 * 6, :, :]
    v = v[:-67 * 6, :, :]
    u = SortTools.normalization(u)
    v = SortTools.normalization(v)
    time, N, M = u.shape
    data = (np.array([u, v])).transpose((1,2,3,0)).reshape((time,-1,2))
    print(data.shape)
    print(data[50,:,:].shape)
