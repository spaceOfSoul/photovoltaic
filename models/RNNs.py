import warnings

import torch
from torch import nn as nn

class GRUModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0, num_layers=1):
        super(GRUModule, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=False, dropout=dropout, batch_first=True, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
            
    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
        
class BiGRUModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0, num_layers=1):
        super(BiGRUModule, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
            
    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
        
class LSTMModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0, num_layers=1):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True,
                           dropout=dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent

class BidirectionalLSTMModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0, num_layers=1):
        super(BidirectionalLSTMModule, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden // 2, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
