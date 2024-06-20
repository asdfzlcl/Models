import torch
import torch.nn as nn
from torch.autograd import Variable


class VMDK(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, input_size, VMD_K, kernel_size, hidden_size, DEVICE):
        super(VMDK, self).__init__()

        self.input_size = input_size
        self.Kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.VMD_K = VMD_K
        self.DEVICE = DEVICE

        # all
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        self.VMD = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(self.VMD_K)])

        self.params = nn.ParameterDict({
            'K': nn.Parameter(torch.rand(self.Kernel_size, hidden_size * self.VMD_K),requires_grad=True),
            'W': nn.Parameter(torch.rand(hidden_size * self.VMD_K),requires_grad=True)
        })

        # self.K = nn.Parameter(torch.rand(self.Kernel_size, hidden_size * self.VMD_K), requires_grad=True)
        # self.W = nn.Parameter(torch.rand(hidden_size * self.VMD_K), requires_grad=True)

        self.out = nn.Linear(hidden_size * self.VMD_K, 1)

    def forward(self, input):
        out = self.relu(self.VMD[0](input[0,]))

        for i in range(1, self.VMD_K):
            out = torch.cat((out, self.relu(self.VMD[i](input[i,]))), 0)

        dis_feature = torch.zeros(self.Kernel_size, self.hidden_size * self.VMD_K).to(self.DEVICE)
        dis = torch.zeros(self.Kernel_size).to(self.DEVICE)

        for i in range(self.Kernel_size):
            dis_feature[i] = (out - self.params['K'][i]) * self.params['W']
            dis[i] = torch.sum(dis_feature * dis_feature)

        index = torch.argmin(dis)
        return self.sigmoid(self.out(dis_feature[index]))
