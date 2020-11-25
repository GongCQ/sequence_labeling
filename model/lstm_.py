import torch
import torch.nn as nn
from utils.config import *
from utils.data import PAD_INDEX

class LSTMCell_(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_, self).__init__()
        self.linear_i_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_g = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_g = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_o = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, input, h_c_0):
        h_0, c_0 = h_c_0
        i = torch.sigmoid(self.linear_i_i(input) + self.linear_h_i(h_0))
        f = torch.sigmoid(self.linear_i_f(input) + self.linear_h_f(h_0))
        g = torch.tanh(self.linear_i_g(input) + self.linear_h_g(h_0))
        o = torch.sigmoid(self.linear_i_o(input) + self.linear_h_o(h_0))
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)

        return (h_1, c_1)

class LSTM_(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0, bidirectional=False):
        super(LSTM_, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_prob = dropout
        self.bidirectional = bidirectional

        self.cell = LSTMCell_(input_size, hidden_size, bias=bias)
        if bidirectional:
            self.cell_back = LSTMCell_(input_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, h_c_0=None):
        batch_size = input.shape[0] if self.batch_first else input.shape[1]
        if h_c_0 is not None:
            h_0, c_0 = h_c_0
        else:
            full_hidden_size = self.hidden_size * (2 if self.bidirectional else 1)
            h_0 = torch.zeros([batch_size, full_hidden_size])
            c_0 = torch.zeros([batch_size, full_hidden_size])
        if self.batch_first:
            input = input.transpose(0, 1)
        if not self.bidirectional:
            h = h_0
            c = c_0
            hidden_list = []
            for s, x in enumerate(input):
                h, c = self.cell(x, (h, c))
                if self.dropout_prob > 0 and s < input.shape[0] - 1:
                    h = self.dropout(h)
                    c = self.dropout(c)
                hidden_list.append(h)
            output = torch.stack(hidden_list)
        else:
            h_forward = h_0[:, : self.hidden_size]
            c_forward = c_0[:, : self.hidden_size]
            h_back = h_0[:, self.hidden_size : ]
            c_back = c_0[:, self.hidden_size : ]
            hidden_list_forward = []
            hidden_list_back = []
            for s in range(input.shape[0]):
                x_forward = input[s]
                x_back = input[input.shape[0] - s - 1]
                h_forward, c_forward = self.cell(x_forward, (h_forward, c_forward))
                h_back, c_back = self.cell(x_back, (h_back, c_back))
                if self.dropout_prob > 0 and s < input.shape[0] - 1:
                    h_forward = self.dropout(h_forward)
                    c_forward = self.dropout(c_forward)
                    h_back = self.dropout(h_back)
                    c_back = self.dropout(c_back)
                hidden_list_forward.append(h_forward)
                hidden_list_back.append(h_back)
            h = torch.cat([h_forward, h_back], dim=1)
            c = torch.cat([c_forward, c_back], dim=1)
            hidden_list_back.reverse()
            output_forward = torch.stack(hidden_list_forward)
            output_back = torch.stack(hidden_list_back)
            output = torch.cat([output_forward, output_back], dim=2)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h, c)




# t1 = torch.Tensor([1,2,3])
# t2 = torch.Tensor([1,2,3])
# i = torch.inverse(t1)
# z = torch.zeros([3,4])
# t = t1 * t2
# dd = 0