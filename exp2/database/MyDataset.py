import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from exp2.TorchModels import SortTools, AutoGraph
from exp2.database import getDataFromNC
from torch_geometric.data import Data


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, area, discriminator, DEVICE, transform=None, pre_transform=None):
        self.area = area
        self.discriminator = discriminator
        self.DEVICE = DEVICE
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.data.to(DEVICE)
        self.slices = self.slices


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.area == 'jiduo':
            return ['jiduo']
        return ['jiduo']

    def download(self):
        pass

    def process(self):

        data_list = []

        if self.area == 'jiduo':
            u, v = getDataFromNC.getData(r'database/jiduo.nc')

        u = u[:-67 * 6, :, :]
        v = v[:-67 * 6, :, :]
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

        dataAll = (np.array([u, v], dtype=np.float)).transpose((1,2,3,0)).reshape((time,-1,2))
        edges = graph.InitGraph(50, int(N * M * 1.5), 3, 0)
        for t in range(51,time):
            edge_index = torch.tensor(graph.GetEdgeIndex(edges), dtype=torch.long)
            x = torch.from_numpy(dataAll[t-1,:,:]).to(torch.float32)

            y = torch.FloatTensor(dataAll[t,:,:]).to(torch.float32)

            data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])