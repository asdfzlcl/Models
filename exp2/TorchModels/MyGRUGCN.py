import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from TorchModels.MYGRU import GRU
from TorchModels.Attention import MultiHeadAttention


class MYGRUGCN(torch.nn.Module):
    def __init__(self, DEVICE, parameter={}):
        super(MYGRUGCN, self).__init__()
        self.DEVICE = DEVICE
        self.parameter = self.setParameter(parameter)
        self.conv1 = GCNConv(self.parameter['conv1_input_size'], self.parameter['conv1_output_size'])
        self.conv2 = GCNConv(self.parameter['conv2_input_size'], self.parameter['conv2_output_size'])
        self.attention = MultiHeadAttention(
            n_head=self.parameter['attention_head_size'], d_k=self.parameter['attention_input_size'],
            d_v=self.parameter['attention_input_size'], d_o=self.parameter['attention_input_size']).to(DEVICE)
        self.GRU = GRU(input_size=self.parameter['GRU_input_size'], hidden_size=self.parameter['hidden_size'],
                       output_size=self.parameter['GRU_output_size']).to(DEVICE)
        self.Linear1 = nn.Linear(4, 8)
        self.Linear2 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=self.parameter['dropout'])
        self.relu = nn.ReLU(inplace=False)

    def setParameter(self, parameter):
        if not 'node_size' in parameter:
            parameter['node_size'] = 81
        if not 'conv1_input_size' in parameter:
            parameter['conv1_input_size'] = 2
        if not 'conv1_output_size' in parameter:
            parameter['conv1_output_size'] = 4
        if not 'conv2_input_size' in parameter:
            parameter['conv2_input_size'] = 4
        if not 'conv2_output_size' in parameter:
            parameter['conv2_output_size'] = 4
        if not 'GRU_input_size' in parameter:
            parameter['GRU_input_size'] = 4
        if not 'GRU_output_size' in parameter:
            parameter['GRU_output_size'] = 4
        if not 'attention_input_size' in parameter:
            parameter['attention_input_size'] = 4
        if not 'attention_head_size' in parameter:
            parameter['attention_head_size'] = 2
        if not 'hidden_size' in parameter:
            parameter['hidden_size'] = 4
        if not 'hidden_num' in parameter:
            parameter['hidden_num'] = 12
        if not 'dropout' in parameter:
            parameter['dropout'] = 0.2
        return parameter

    def forward(self, data, hidden, time):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        if time > 0:
            att, h = self.attention(x.reshape(1, -1, self.parameter['attention_input_size']),
                                    hidden[max(0, time - self.parameter['hidden_num']):time].reshape(1, -1,
                                                                                                     self.parameter[
                                                                                                         'attention_input_size']).clone(),
                                    hidden[max(0, time - self.parameter['hidden_num']):time].reshape(1, -1,
                                                                                                     self.parameter[
                                                                                                         'attention_input_size']).clone())
            h = torch.squeeze(h, dim=0)
        else:
            h = torch.randn(self.parameter['node_size'], self.parameter['hidden_size']).to(self.DEVICE)
        x, h = self.GRU(x, h)
        x = self.relu(self.Linear1(x))
        x = self.Linear2(x)

        return x, h.clone()
