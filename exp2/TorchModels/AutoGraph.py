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
        self.netEdge, self.netEdges = self.GetEdgeNet()

    def GetEdgeNet(self):
        source_nodes = []
        target_nodes = []
        edges = []
        for x1 in range(self.N):
            for y1 in range(self.M):
                for dx in range(2):
                    for dy in range(-1,2):
                        x2 = x1 + dx
                        y2 = y1 + dy
                        if dx == 0 and dy <= 0:
                            continue
                        if x2 >= self.N or y2 >= self.M or y2 < 0:
                            continue
                        id1 = self.GetID(x1, y1)
                        id2 = self.GetID(x2, y2)
                        source_nodes.append(id1)
                        source_nodes.append(id2)
                        target_nodes.append(id2)
                        target_nodes.append(id1)
                        edges.append([[x1, y1, x2, y2]])
        print(len(source_nodes) / 2)
        return [source_nodes, target_nodes], edges

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
        return float(self.discriminator(x, y))

    def updateValue(self, edge, time):
        x1, y1, x2, y2 = edge[0][0], edge[0][1], edge[0][2], edge[0][3]
        edge[1] = self.checkValue(time, x1, y1, x2, y2)
        return edge

    def updateQueueValue(self, time):
        IDList = [i + 1 for i in range(self.queueEdges.last_index)]
        IDList = random.sample(IDList, int(self.updateRate * self.queueEdges.last_index))
        edges = []
        for index in IDList:
            edges.append(self.queueEdges.queue[index])
        for edge in edges:
            edge = self.updateValue(edge, time)
            self.queueEdges.updateValue(edge)

        IDList = [i + 1 for i in range(self.queueCandidate.last_index)]
        IDList = random.sample(IDList, int(self.updateRate * self.queueCandidate.last_index))
        edges = []
        for index in IDList:
            edges.append(self.queueCandidate.queue[index])
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
        print("edge nums:" + str(len(edges)))

    def GetEdgeList(self, edges):
        edgeList = [[0 for _ in range(self.M * self.N)] for _ in range(self.M * self.N)]
        for edge in edges:
            id1 = self.GetID(edge[0][0], edge[0][1])
            id2 = self.GetID(edge[0][2], edge[0][3])
            edgeList[id1][id2] = edgeList[id2][id1] = 1
        return edgeList

    def GetEdgeIndex(self, edges):
        source_nodes, target_nodes = [], []
        for edge in edges:
            id1 = self.GetID(edge[0][0], edge[0][1])
            id2 = self.GetID(edge[0][2], edge[0][3])
            source_nodes.append(id1)
            source_nodes.append(id2)
            target_nodes.append(id2)
            target_nodes.append(id1)

        return [source_nodes, target_nodes]

    def InitGraph(self, time, numEdge, neighborhood, graphType=0):
        if time + self.scale > self.Time:
            print(time, self.scale, self.Time)
            print("time input error")
            return []

        print(numEdge)

        self.queueCandidate.clear()
        self.queueEdges.clear()

        for x1 in range(self.N):
            for y1 in range(self.M):
                for dx in range(neighborhood):
                    for dy in range(-neighborhood + 1, neighborhood):
                        x2 = x1 + dx
                        y2 = y1 + dy
                        if dx == 0 and dy <= 0:
                            continue
                        if x2 >= self.N or y2 >= self.M or y2 < 0:
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

    def UpdateGraph(self, time):
        self.updateQueueValue(time)
        edgeTop = self.queueEdges.getTop()
        candidateTop = self.queueCandidate.getTop()
        nums = 0
        while self.queueCandidate.last_index > 0 and self.queueCandidate.comparator(candidateTop, edgeTop):
            self.queueCandidate.pop()
            self.queueEdges.push(candidateTop)
            id1 = self.GetID(candidateTop[0][0], candidateTop[0][1])
            id2 = self.GetID(candidateTop[0][2], candidateTop[0][3])
            self.d[id1] += 1
            self.d[id2] += 1
            nums += 1
            candidateTop = self.queueCandidate.getTop()

        print("update nums:" + str(nums))

        edgeNum = 0
        res = []
        while edgeNum < nums:
            edgeTop = self.queueEdges.getTop()
            id1 = self.GetID(edgeTop[0][0], edgeTop[0][1])
            id2 = self.GetID(edgeTop[0][2], edgeTop[0][3])
            if self.d[id1] == 1 or self.d[id2] == 1:
                res.append(edgeTop)
            else:
                self.d[id1] -= 1
                self.d[id2] -= 1
                edgeNum += 1
                self.queueCandidate.push(edgeTop)
            self.queueEdges.pop()

        for edge in res:
            self.queueEdges.push(edge)

        return self.queueEdges.queue[1:]


if __name__ == '__main__':
    graph = Graph()
