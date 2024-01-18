import torch
import torch.nn as nn
import math


class Discriminator(nn.Module):
    def __init__(self, input_size, kernel_size):
        super(Discriminator, self).__init__()
        self.fcx = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size)
        self.fcy = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size)
        self.feature_size = input_size - kernel_size + 1
        self.fc1 = nn.Linear(in_features=self.feature_size * 2, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=1)
        self.Sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()

    def forward(self, x, y):
        x = torch.transpose(x, dim0=1, dim1=2)
        y = torch.transpose(y, dim0=1, dim1=2)
        feature_x = self.Relu(torch.squeeze(self.fcx(x), dim=1))
        feature_y = self.Relu(torch.squeeze(self.fcy(y), dim=1))
        feature = torch.cat((feature_x, feature_y), dim=1)
        out = self.Sigmoid(self.fc2(self.Relu(self.fc1(feature))))
        return torch.squeeze(out, dim=1)


class Generator(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(Generator, self).__init__()
        self.channel_size = channel_size
        self.output_size = output_size
        self.fc = nn.Linear(in_features=input_size, out_features=2 * input_size)
        self.fcx = nn.Linear(in_features=2 * input_size, out_features=output_size * channel_size)
        self.fcy = nn.Linear(in_features=2 * input_size, out_features=output_size * channel_size)
        self.fox = nn.Linear(in_features=output_size * channel_size, out_features=output_size * channel_size)
        self.foy = nn.Linear(in_features=output_size * channel_size, out_features=output_size * channel_size)
        self.Relu = nn.ReLU()

    def forward(self, input):
        feature = self.Relu(self.fc(input))
        feature_x = self.Relu(self.fcx(feature))
        feature_y = self.Relu(self.fcy(feature))
        x, y = self.fox(feature_x), self.foy(feature_y)
        x = torch.reshape(x, (1, self.output_size, self.channel_size))
        y = torch.reshape(y, (1, self.output_size, self.channel_size))
        return x, y


if __name__ == '__main__':
    discriminator = Discriminator(10, 3)
    x = torch.randn(1, 10, 2)
    y = torch.randn(1, 10, 2)
    print(discriminator(x, y))
    generator = Generator(10, 10, 2)
    x, y = generator(torch.randn(1, 10))
    print(x.shape)
    # print(x,y)
    print(discriminator(x, y))
