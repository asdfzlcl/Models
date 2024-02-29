import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import sys

from exp2.database import getDataFromNC

sys.path.append("..")
from TorchModels import SortTools, AutoGraph
from torch_geometric.data import Data


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, area, discriminator, DEVICE, edgeType=0, transform=None, pre_transform=None):
        self.area = area
        self.discriminator = discriminator
        self.DEVICE = DEVICE
        self.edgeType = edgeType
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.data.to(DEVICE)
        self.slices = self.slices

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.area + str(self.edgeType)

    def download(self):
        pass

    def process(self):

        data_list = []

        u, v = getDataFromNC.getData(r'database/nc/' + self.area + '.nc')

        u = u[:-80 * 6, :, :]
        v = v[:-80 * 6, :, :]
        u = SortTools.normalization(u)
        v = SortTools.normalization(v)
        time, N, M = u.shape

        graph = AutoGraph.Graph(self.discriminator, 50, self.DEVICE, u, v,
                                SortTools.Priority_Queue(lambda x, y: x[1] > y[1],
                                                         lambda x: '(' + str(
                                                             x[0][0]) + ',' + str(
                                                             x[0][1]) + ')-' '(' + str(
                                                             x[0][2]) + ',' + str(
                                                             x[0][3]) + ')'),
                                SortTools.Priority_Queue(lambda x, y: x[1] < y[1],
                                                         lambda x: '(' + str(
                                                             x[0][0]) + ',' + str(
                                                             x[0][1]) + ')-' '(' + str(
                                                             x[0][2]) + ',' + str(
                                                             x[0][3]) + ')'),
                                0.1
                                )

        dataAll = (np.array([u, v], dtype=np.float64)).transpose((1, 2, 3, 0)).reshape((time, -1, 2))
        edges = graph.InitGraph(0, int(N * M * 1.25), 3, 0)
        for t in range(50, time):
            if self.edgeType == 0:
                edges = graph.UpdateGraph(t - 50)
                edge_index = torch.tensor(graph.GetEdgeIndex(edges), dtype=torch.long)

            if self.edgeType == 1:
                edge_index = torch.tensor(graph.netEdge, dtype=torch.long)

            x = torch.from_numpy(dataAll[t - 1, :, :]).to(torch.float32)

            y = torch.FloatTensor(dataAll[t, :, :]).to(torch.float32)

            data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
