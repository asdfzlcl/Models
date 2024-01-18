import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.Hidden_layer = nn.Linear(input_size+hidden_size,hidden_size)
        self.Out_layer1 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()

    def rnn_work(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.sigmoid(self.Hidden_layer(combined))
        output = self.Out_layer1(hidden)
        return output,hidden

    def forward(self, x, h_state,N):
        outs = []

        for time_step in range(N):
            output,h_state = self.rnn_work(x[:,time_step,:],h_state)
            outs.append(output[:])

        return torch.stack(outs, dim=1), h_state