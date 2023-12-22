import torch
import torch.nn as nn
from TorchModels import Attention
from TorchModels import RNN


class LSTNet(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, input_size, RNN_hidden_size, hidden_size, output_size, DEVICE,
                 feature_size=13):
        super(LSTNet, self).__init__()

        self.input_size = input_size
        self.rnn_hidden_size = RNN_hidden_size
        self.output_size = output_size
        self.DEVICE = DEVICE

        # all
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        # ANN prediction
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # RNN
        self.RNN = RNN.RNN(feature_size, RNN_hidden_size, RNN_hidden_size)

        self.fc4 = nn.Linear(RNN_hidden_size, hidden_size)

        self.fc5 = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden_list):
        # RNN
        RNN_out, hidden_list = self.RNN(input[-1, :].unsqueeze(0).unsqueeze(0), hidden_list, 1)
        hidden_list = hidden_list.to(self.DEVICE)

        input = input.contiguous().view(1, -1)

        # ANN
        input = input.view(1, -1)
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        ann_out = self.fc3(self.relu(out))

        RNN_out = RNN_out.view(1, -1)


        # fusion
        out = ann_out + self.fc5(self.relu(self.fc4(RNN_out)))

        return out, hidden_list
