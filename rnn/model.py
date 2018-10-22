import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GRU(nn.Module):
    """docstring for GRU."""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(GRU, self).__init__()
        self.__dict__.update(locals())

        self.weightx = torch.Tensor(3*hidden_size, input_size)
        nn.init.kaiming_normal_(self.weightx, nonlinearity='relu')
        self.weightx = Parameter( self.weightx )
        self.weighth = torch.Tensor(3*hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.weighth, nonlinearity='relu')
        self.weighth = Parameter( self.weighth )
        self.biasx = Parameter(torch.zeros(3*hidden_size))
        self.biash = Parameter(torch.zeros(3*hidden_size))


    def forward(self, inp, hidden):
        # print('inp: {}'.format(inp.size()))
        # print('hidden: {}'.format(hidden.size()))
        xs = F.linear(inp, self.weightx, bias=self.biasx)
        hs = F.linear(hidden, self.weighth, bias=self.biash)
        add = (xs[:,:,:-self.hidden_size] + hs[:,:,:-self.hidden_size])
        # print('add: {}'.format(add))
        sig = F.sigmoid(add)
        # print('sig: {}'.format(sig))
        # r = F.sigmoid((xs[:,:,:-self.hidden_size] + hs[:,:,:-self.hidden_size]))
        # z = F.sigmoid((xs[:,:,self.hidden_size:-self.hidden_size] + hs[:,:,self.hidden_size:-self.hidden_size]))
        r = sig[:,:,:self.hidden_size]
        z = sig[:,:,self.hidden_size:]
        # print('r: {}'.format(r))
        # print('z: {}'.format(z))
        # input('waiting')
        n = F.tanh(xs[:,:,-self.hidden_size:] + (r * hs[:,:,-self.hidden_size:]))
        # print('n: {}'.format(n))
        h = (1-z)*n + z*hidden
        # print('h: {}'.format(h))
        return h, h


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GRUNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_char, hidden):
        # print(hidden)
        output = self.embedding(input_char).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # print(output)
        # print(hidden)
        # input('waiting')
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers


        # encode using embedding layer
        # set up GRU passing in number of layers parameter (nn.GRU)
        # decode output

        # pytorch example rnn code
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_char, hidden):
        # by reviewing the documentation, construct a forward function that properly uses the output
        # of the GRU
        # return output and hidden

        # pytorch example rnn code
        input_combined = torch.cat((input_char, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
