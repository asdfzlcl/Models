import random
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

    def __init__(self, discriminator, scale, device, u, v, queueCandidate, queueEdges, updateRate):
        self.discriminator = discriminator
        self.scale = scale
        self.device = device
        self.dataU = u
        self.dataV = v
        self.Time, self.N, self.M = u.shape
        self.queueCandidate = queueCandidate
        self.queueEdges = queueEdges
        self.updateRate = updateRate
        self.SafeEdges = [[] for _ in range(self.N * self.M)]
        self.d = [0 for _ in range(self.N * self.M)]

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

    def updateValue(self, edge, time):
        x1, y1, x2, y2 = edge[0][0], edge[0][1], edge[0][2], edge[0][3]
        edge[1] = self.checkValue(time, x1, y1, x2, y2)
        return edge

    def updateQueueValue(self, time):
        IDList = [i + 1 for i in range(self.queueEdges.last_index)]
        IDList = random.sample(IDList, int(self.updateRate * self.queueEdges.last_index))
        edges = self.queueEdges.queue[IDList]
        for edge in edges:
            edge = self.updateValue(edge, time)
            self.queueEdges.updateValue(edge)

        IDList = [i + 1 for i in range(self.queueCandidate.last_index)]
        IDList = random.sample(IDList, int(self.updateRate * self.queueCandidate.last_index))
        edges = self.queueCandidate.queue[IDList]
        for edge in edges:
            edge = self.updateValue(edge, time)
            self.queueCandidate.updateValue(edge)

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

    def updateSafe(self, ID):
        if self.d[ID] > 1 and not self.SafeEdges[ID]:
            self.queueEdges.push(self.SafeEdges[ID])
            self.SafeEdges[ID] = []

    def InitGraph(self, time, numEdge, neighborhood, graphType=0):
        if time + self.scale > self.Time:
            print(time, self.scale, self.Time)
            print("time input error")
            return []

        self.queueCandidate.clear()

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
                        self.queueCandidate.push([[x1, y1, x2, y2], float(value)])

        if graphType % 2 == 0:
            self.d = [0 for i in range(self.N * self.M)]
            res = []
            nums = self.N * self.M
            while nums > 0:
                top = self.queueCandidate.getTop()
                id1 = self.GetID(top[0][0], top[0][1])
                id2 = self.GetID(top[0][2], top[0][3])
                if self.d[id1] == 0 or self.d[id2] == 0:
                    if self.d[id1] == 0:
                        nums -= 1
                    if self.d[id2] == 0:
                        nums -= 1
                    numEdge -= 1
                    self.d[id1] += 1
                    self.d[id2] += 1
                    self.queueEdges.push(top)
                else:
                    res.append(top)
                self.queueCandidate.pop()

            for edge in res:
                self.queueCandidate.push(edge)

        for i in range(numEdge):
            top = self.queueCandidate.getTop()
            id1 = self.GetID(top[0][0], top[0][1])
            id2 = self.GetID(top[0][2], top[0][3])
            self.d[id1] += 1
            self.d[id2] += 1
            self.queueEdges.push(top)
            self.queueCandidate.pop()

        return self.queueEdges.queue[1:]

    def updateQueue(self):
        while self.queueEdges.last_index > 0:
            top = self.queueEdges.getTop()
            id1 = self.GetID(top[0][0], top[0][1])
            id2 = self.GetID(top[0][2], top[0][3])
            if self.d[id1] > 1 and self.d[id2] > 1:
                break
            if self.d[id1] == 1:
                self.SafeEdges[id1] = top
            if self.d[id2] == 1:
                self.SafeEdges[id2] = top
            self.queueEdges.pop()
        topEdges = self.queueEdges.getTop()
        topCandidate = self.queueCandidate.getTop()
        if self.queueCandidate.comparator(topCandidate, topEdges):
            id1 = self.GetID(topEdges[0][0], topEdges[0][1])
            id2 = self.GetID(topEdges[0][2], topEdges[0][3])
            self.d[id1] -= 1
            self.d[id2] -= 1
            id1 = self.GetID(topCandidate[0][0], topCandidate[0][1])
            id2 = self.GetID(topCandidate[0][2], topCandidate[0][3])
            self.d[id1] += 1
            self.d[id2] += 1
            self.queueEdges.pop()
            self.queueCandidate.pop()
            self.queueEdges.push(topCandidate)
            self.queueCandidate.push(topEdges)
            self.updateSafe(id1)
            self.updateSafe(id2)
            return True

        id1 = self.GetID(topCandidate[0][0], topCandidate[0][1])
        id2 = self.GetID(topCandidate[0][2], topCandidate[0][3])
        if self.SafeEdges[id1] == [] and self.SafeEdges[id2] == []:
            return False
        candidateID = -1
        if self.SafeEdges[id1] != [] and self.SafeEdges[id2] != []:
            if self.queueCandidate.comparator(self.SafeEdges[id1], self.SafeEdges[id2]):
                candidateID = id1
            else:
                candidateID = id2
        else:
            if self.SafeEdges[id1]:
                candidateID = id1
            else:
                candidateID = id2
        if not self.queueCandidate.comparator(topCandidate, self.SafeEdges[candidateID]):
            return False

        id3 = self.GetID(self.SafeEdges[candidateID][0], self.SafeEdges[candidateID][1])
        id4 = self.GetID(self.SafeEdges[candidateID][2], self.SafeEdges[candidateID][3])
        if self.d[id3] == 1 and self.d[id4] == 1:
            return False
        self.d[id1] += 1
        self.d[id2] += 1
        self.d[id3] += 1
        self.d[id4] += 1
        self.queueCandidate.pop()
        self.queueCandidate.push(self.SafeEdges[candidateID])
        self.queueEdges.delFeature(self.SafeEdges[candidateID])
        self.SafeEdges[candidateID] = topCandidate
        return True

    def UpdateGraph(self, time):
        self.updateQueueValue(time)
        num = 0
        while self.queueCandidate.last_index > 0 and self.updateQueue():
            num += 1
        for i in range(self.N*self.M):
            if self.SafeEdges[i]:
                self.queueEdges.push(self.SafeEdges[i])
                self.SafeEdges[i] = []
        return num


if __name__ == '__main__':
    graph = Graph()
