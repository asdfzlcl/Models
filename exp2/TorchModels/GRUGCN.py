import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GRUGCN(torch.nn.Module):
    def __init__(self, DEVICE, parameter={}):
        super(GRUGCN, self).__init__()
        self.DEVICE = DEVICE
        self.parameter = self.setParameter(parameter)
        self.conv1 = GCNConv(self.parameter['conv1_input_size'], self.parameter['conv1_output_size'])
        self.conv2 = GCNConv(self.parameter['conv2_input_size'], self.parameter['conv2_output_size'])
        self.attention = nn.MultiheadAttention(embed_dim=self.parameter['attention_input_size'],
                                               num_heads=self.parameter['attention_head_size'])
        self.GRU = nn.GRU(input_size=self.parameter['GRU_input_size'], hidden_size=self.parameter['hidden_size'])
        self.Linear1 = nn.Linear(4, 4)
        self.Linear2 = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p=self.parameter['dropout'])
        self.relu = nn.ReLU(inplace=False)
        self.hidden = torch.randn(19297, self.parameter['node_size'],
                                  self.parameter['hidden_size']).to(DEVICE)

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
            h, att = self.attention(x,
                                self.hidden[min(0,time-self.parameter['hidden_num']):time].reshape(-1, self.parameter['attention_input_size']).clone(),
                                self.hidden[min(0,time-self.parameter['hidden_num']):time].reshape(-1, self.parameter['attention_input_size']).clone())
        else:
            h = torch.randn(self.parameter['node_size'],
                                  self.parameter['hidden_size']).to(self.DEVICE)
        x = torch.unsqueeze(x, dim=0)
        h = torch.unsqueeze(h, dim=0)
        x, h = self.GRU(input=x, hx=h.data)
        x = torch.squeeze(x, dim=0)
        h = torch.squeeze(h, dim=0)
        self.hidden[time] = h.clone()
        x = self.relu(self.Linear1(x))
        x = self.Linear2(x)

        return x,h.clone()
