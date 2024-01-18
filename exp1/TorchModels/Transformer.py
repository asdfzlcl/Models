import torch.nn as nn
from exp1.TorchModels import Attention


class Transformer(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, input_size, hidden_size,output_size):
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.attention = Attention.ScaledDotProductAttention(scale=hidden_size)
        self.layerNorm1 = nn.LayerNorm(hidden_size)
        self.layerNorm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.q1 = nn.Linear(input_size, hidden_size)
        self.k1 = nn.Linear(input_size, hidden_size)
        self.v1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.layerNorm3 = nn.LayerNorm(hidden_size)
        self.layerNorm4 = nn.LayerNorm(hidden_size)
        self.q2 = nn.Linear(hidden_size, hidden_size)
        self.k2 = nn.Linear(hidden_size, hidden_size)
        self.v2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.view(1, -1)
        v = self.v1(input)
        _, out = self.attention(self.q1(input).unsqueeze(-1), self.k1(input).unsqueeze(-1),
                                v.unsqueeze(-1))
        out = out.view(1, -1)
        out = self.layerNorm1(out+v)
        out = self.layerNorm2(self.fc1(out) + out)
        _, out2 = self.attention(self.q2(out).unsqueeze(-1), self.k2(out).unsqueeze(-1),
                                 self.v2(out).unsqueeze(-1))
        out2 = out2.view(1, -1)
        out = self.layerNorm3(out + out2)
        out = self.layerNorm4(self.fc2(out) + out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out
