import torch
import torch.nn.functional as F
# 导入GCN层、GraphSAGE层和GAT层
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid


class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden*heads, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
        self.Linear = nn.Linear(4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return self.Linear(x)