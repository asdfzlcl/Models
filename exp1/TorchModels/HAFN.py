import torch
import torch.nn as nn
from exp1.TorchModels import Attention
from exp1.TorchModels import RNN


class HAFN(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, input_size, history_size, RNN_size, RNN_hidden_size, hidden_size, output_size, DEVICE,
                 feature_size=13):
        super(HAFN, self).__init__()

        self.input_size = input_size
        self.history_size = history_size
        self.RNN_size = RNN_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = RNN_hidden_size
        self.output_size = output_size
        self.DEVICE = DEVICE

        # all
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        # ANN prediction
        self.fc1 = nn.Linear(input_size, output_size)

        # RNN
        self.RNNList = nn.ModuleList([RNN.RNN(feature_size, RNN_hidden_size, RNN_hidden_size) for _ in range(RNN_size)])

        # History attention
        self.attention = Attention.ScaledDotProductAttention(scale=hidden_size)
        self.layerNorm1 = nn.LayerNorm(hidden_size)
        self.layerNorm2 = nn.LayerNorm(hidden_size)
        self.at1 = nn.Linear(hidden_size, hidden_size)
        self.q1 = nn.Linear(input_size, hidden_size)
        self.k1 = nn.Linear(history_size * feature_size, hidden_size)
        self.v1 = nn.Linear(history_size * feature_size, hidden_size)

        self.at2 = nn.Linear(hidden_size, hidden_size)
        self.layerNorm3 = nn.LayerNorm(hidden_size)
        self.layerNorm4 = nn.LayerNorm(hidden_size)
        self.q2 = nn.Linear(hidden_size, hidden_size)
        self.k2 = nn.Linear(RNN_size * RNN_hidden_size, hidden_size)
        self.v2 = nn.Linear(RNN_size * RNN_hidden_size, hidden_size)

        self.at3 = nn.Linear(hidden_size, hidden_size)

        self.fu = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_list, history, ID):
        # RNN
        RNN_out, hidden_list[0][0] = self.RNNList[0](input[-1, :].unsqueeze(0).unsqueeze(0), hidden_list[0][0], 1)
        hidden_list[0][0] = hidden_list[0][0].to(self.DEVICE)
        cycle = 1
        for i in range(1, self.RNN_size):
            cycle = cycle * 2
            out, hidden_list[i][ID % cycle] = self.RNNList[i](input[-1, :].unsqueeze(0).unsqueeze(0),
                                                              hidden_list[i][ID % cycle], 1)
            hidden_list[i][ID % cycle] = hidden_list[i][ID % cycle].to(self.DEVICE)
            RNN_out = torch.cat((RNN_out, out), dim=1)

        input = input.contiguous().view(1, -1)
        history = history.contiguous().view(1, -1)

        # ANN
        input = input.view(1, -1)
        ann_out = self.fc1(input)

        # attention
        v = self.v1(history).unsqueeze(-1)
        _, out = self.attention(self.q1(input).unsqueeze(-1), self.k1(history).unsqueeze(-1),
                                v)
        out = out.view(1, -1)
        v = v.view(1, -1)
        out = self.layerNorm1(out + v)
        out = self.layerNorm2(self.at1(out) + out)

        # print(out.shape)
        RNN_out = RNN_out.view(1, -1)

        v = self.v2(RNN_out).unsqueeze(-1)
        _, out = self.attention(self.q2(out).unsqueeze(-1), self.k2(RNN_out).unsqueeze(-1),
                                v)
        out = out.view(1, -1)
        v = v.view(1, -1)
        out = self.layerNorm1(out + v)
        out = self.layerNorm2(self.at2(out) + out)

        attention_out = self.relu(self.at3(out))

        # fusion
        out = ann_out + self.fu(attention_out)

        return out, hidden_list
