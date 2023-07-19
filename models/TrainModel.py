import torch
from torch import nn
from models.CNN import CNN
from models.RNN import BidirectionalGRU

class TrainModel(nn.Module):
   def __init__(self, n_in_channel, activ, conv_dropout, kernel_size, padding, stride, nb_filters, pooling, input_dim, hidden_dim, output_dim, dropout, num_layers):
        super(TrainModel, self).__init__()
        self.cnn = CNN(n_in_channel=n_in_channel, activ=activ, conv_dropout=conv_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling).double()
        self.rnn = BidirectionalGRU(n_in=input_dim, n_hidden=hidden_dim, dropout=dropout, num_layers=num_layers).double()
        self.dense = nn.Linear(hidden_dim*2, output_dim)  # hidden_dim*2: BiGRU

        self.dense.weight.data = self.dense.weight.data.double()
        self.dense.bias.data = self.dense.bias.data.double()

   def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

   def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])

   def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

   def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)
        
   def forward(self, x):
        # x.shape:(batch: 23, seqLeng:30, input_dim: 15 (feature 14 with time))
        x = self.cnn(x)  # Pass x through the CNN
        out = self.rnn(x)
        out = out[:, -1, :]
        out = self.dense(out)
        return out

	    
	   
