import torch
import torch.nn as nn
import numpy as np
from random import randint
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import os

'''

'''


class Graph:
    dx = [0, 0, 1, 1, 1, -1, -1, -1]
    dy = [1, -1, 1, -1, 0, 1, -1, 0]

    def __init__(self, discriminator, scale, device, u, v, Queue):
        self.discriminator = discriminator
        self.scale = scale
        self.device = device
        self.dataU = u
        self.dataV = v
        self.Time, self.N, self.M = u.shape
        self.Queue = Queue

    def GetID(self, x, y):
        return x * self.M + y

    def GetPosition(self, ID):
        x = ID // self.M
        y = ID - x * self.M
        return x, y

    def checkValue(self, t, x1, y1, x2, y2):
        x = torch.tensor(
            (np.append(self.dataU[t:t + self.scale, x1, y1], self.dataV[t:t + self.scale, x1, y1])).reshape((1,
                                                                                                             self.scale,
                                                                                                             2)).astype(
                float),
            dtype=torch.float32).to(
            self.device)
        y = torch.tensor(
            (np.append(self.dataU[t:t + self.scale, x2, y2], self.dataV[t:t + self.scale, x2, y2])).reshape((1,
                                                                                                             self.scale,
                                                                                                             2)).astype(
                float),
            dtype=torch.float32).to(
            self.device)
        return self.discriminator(x, y)

    def ShowGraph(self, edges):
        for x1 in range(self.N):
            for y1 in range(self.M):
                plt.scatter(x1, y1, s=50, c='k')
        for edge in edges:
            plt.plot([edge[0][0], edge[0][2]], [edge[0][1], edge[0][3]], c='k')
        plt.show()

    def GetEdgeList(self, edges):
        edgeList = [[0 for _ in range(self.M * self.N)] for _ in range(self.M * self.N)]
        for edge in edges:
            id1 = self.GetID(edge[0][0], edge[0][1])
            id2 = self.GetID(edge[0][2], edge[0][3])
            edgeList[id1][id2] = edgeList[id2][id1] = 1
        return edgeList

    def GetGraph(self, time, numEdge, neighborhood, graphType=0):
        if time + self.scale > self.Time:
            print(time, self.scale, self.Time)
            print("time input error")
            return []

        self.Queue.clear()

        for x1 in range(self.N):
            for y1 in range(self.M):
                for dx in range(neighborhood):
                    for dy in range(neighborhood):
                        x2 = x1 + dx
                        y2 = y1 + dy
                        if dx == 0 and dy == 0:
                            continue
                        if x2 >= self.N or y2 >= self.M:
                            continue
                        value = self.checkValue(time, x1, y1, x2, y2)
                        self.Queue.push([[x1, y1, x2, y2], float(value)])

        edges = []

        if graphType % 2 == 0:
            d = [0 for i in range(self.N * self.M)]
            res = []
            nums = self.N * self.M
            while nums > 0:
                top = self.Queue.getTop()
                id1 = self.GetID(top[0][0], top[0][1])
                id2 = self.GetID(top[0][2], top[0][3])
                if d[id1] == 0 or d[id2] == 0:
                    if d[id1] == 0:
                        nums -= 1
                    if d[id2] == 0:
                        nums -= 1
                    numEdge -= 1
                    d[id1] = d[id2] = 1
                    edges.append(top)
                else:
                    res.append(top)
                self.Queue.pop()

            for edge in res:
                self.Queue.push(edge)

        for i in range(numEdge):
            top = self.Queue.getTop()
            edges.append(top)
            self.Queue.pop()

        return edges


if __name__ == '__main__':
    graph = Graph()
