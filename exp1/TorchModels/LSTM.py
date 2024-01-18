import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,cell_size,output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate_f = nn.Linear(input_size + hidden_size, cell_size)
        self.gate_i = nn.Linear(input_size + hidden_size, cell_size)
        self.gate_o = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def lstm_work(self,input,hidden,cell):
        combined = torch.cat((input,hidden),dim=1)
        f_gate = self.sigmoid(self.gate_f(combined))
        i_gate = self.sigmoid(self.gate_i(combined))
        o_gate = self.sigmoid(self.gate_o(combined))
        z_state = self.tanh(self.gate_i(combined))
        cellout = torch.add(torch.mul(cell,f_gate),torch.mul(z_state,i_gate))
        hidden = torch.mul(self.tanh(cellout),o_gate)
        output = self.sigmoid(self.output(hidden))
        return output,hidden,cellout

    def forward(self, x, h_state,cell,N):
        outs = []

        for time_step in range(N):
            output,h_state,cell = self.lstm_work(x[:,time_step,:],h_state,cell)
            outs.append(output[:])

        return torch.stack(outs, dim=1), h_state,cell
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state