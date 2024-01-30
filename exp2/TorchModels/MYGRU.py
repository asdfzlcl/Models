import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gate_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def gru_work(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        r_gate = self.sigmoid(self.gate_r(combined))
        z_gate = self.sigmoid(self.gate_z(combined))
        combined01 = torch.cat((input,torch.mul(z_gate,r_gate)),1)
        h1_state = self.tanh(self.gate_h(combined01))

        h_state = torch.add(torch.mul((1-z_gate),hidden),torch.mul(h1_state,z_gate))
        output = self.output(h_state)
        output = self.sigmoid(output)
        return output,h_state

    def forward(self, x, h_state):
        output,h_state = self.gru_work(x,h_state)

        return output, h_state
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state