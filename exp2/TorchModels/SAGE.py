import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid


class GraphSAGE(torch.nn.Module):

    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, classes)
        self.Linear = nn.Linear(4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        return self.Linear(x)