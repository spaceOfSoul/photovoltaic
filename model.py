import torch
from torch import nn
from models.RNNs import GRUModule, BiGRUModule, LSTMModule
from models.CNN import CNNModule
from torchaudio.models import Conformer as ConformerModule

# single pred model(LSTM, CNN, Conformer)

class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout1=0, num_layers1=1):
        super(LSTM, self).__init__()
        
        self.lstm = LSTMModule(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout1, num_layers=num_layers1)
        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(hidden_dim, output_dim)

    def load_state_dict(self, state_dict, strict=True):
        self.lstm.load_state_dict(state_dict["lstm"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"lstm": self.lstm.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'lstm': self.lstm.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)


    def forward(self, x): # [nBatch, segLeng, input_dim]
        recurrent = self.lstm(x) # [nBatch, segLeng, nHidden1]
        # recurrent = recurrent[:, -1, :] # [nBatch, nHidden1] 

        # segLeng => Weighted Arithmetic Mean 
        sof = self.softmax(recurrent)
        sof = torch.clamp(sof, min=1e-7, max=1)
        recurrent = (recurrent*sof).sum(1)/sof.sum(1) # [nBatch, nb_filters[-1]]
        
        out = self.dense(recurrent) # [nBatch, output_dim]
        return out

class CNN(nn.Module):

    def __init__(self, n_in_channel, activ="Relu", cnn_dropout=0,
                 kernel_size=2*[3], padding=2*[1], stride=2*[1], nb_filters=[64, 128],
                 pooling=2*[1], output_dim=1):
        super(CNN, self).__init__()
                
        self.cnn = CNNModule(n_in_channel=n_in_channel, activ=activ, cnn_dropout=cnn_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
        self.dense = nn.Linear(nb_filters[-1], output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.dense.load_state_dict(state_dict["dense"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x): # [nBatch, segLeng, input_dim]    
        x = x.permute(0, 2, 1) # [nBatch, input_dim, segLeng]
        out = self.cnn(x) # [nBatch, nb_filters[-1], segLeng]
        
        # segLeng => Weighted Arithmetic Mean 
        sof = self.softmax(out)
        sof = torch.clamp(sof, min=1e-7, max=1)
        out = (out*sof).sum(2)/sof.sum(2) # [nBatch, nb_filters[-1]]
        
        out = self.dense(out) # [nBatch, output_dim]
        return out

def get_lengths(x):
    """
    This function calculates the lengths of each batch element using the dimensions of the input x.
    
    x.size(0) represents the batch size, and x.size(1) represents the sequence length.
    
    Therefore, this function returns a tensor with the same length for all elements of the input x.
    The returned lengths have a shape of (B,), representing the number of valid frames for each batch element.
    
    This function can be used under the assumption that each sequence has the same length and there is no padding.
    """
    return torch.full((x.size(0),), x.size(1), dtype=torch.long)

class Conformer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads=4, ffn_dim=32, num_layers=4, depthwise_conv_kernel_size=31):
        super(Conformer, self).__init__()
        
        self.conformer = ConformerModule(input_dim=input_dim, num_heads=num_heads, ffn_dim=ffn_dim, num_layers=num_layers, depthwise_conv_kernel_size=depthwise_conv_kernel_size).double()        
        self.softmax = nn.Softmax(dim=-1).double()
        self.dense = nn.Linear(input_dim, output_dim).double()
        
    def load_state_dict(self, state_dict, strict=True):
        self.conformer.load_state_dict(state_dict["conformer"])
        self.dense.load_state_dict(state_dict["dense"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"conformer": self.conformer.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'conformer': self.conformer.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)
    
    def forward(self, x):
        x = x.double() # [nBatch, segLeng, input_dim]
        lengths = get_lengths(x).double().cuda() # [nBatch]
        output, _ = self.conformer(x, lengths) # [nBatch, seqLeng, input_dim]
        
        # out = outputs[:, -1, :] # [nBatch, output_dim] 

        # => Weighted Arithmetic Mean 
        sof = self.softmax(output)
        sof = torch.clamp(sof, min=1e-7, max=1)
        out = (output*sof).sum(1)/sof.sum(1) # [nBatch, input_dim]
        out = self.dense(out) # [nBatch, output_dim]

        return out

###########################################################################################################        
# hybrid model 
class LSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout1=0, dropout2=0, num_layers1=1, num_layers2=1, activ="Relu", cnn_dropout=0,
                 kernel_size=2*[3], padding=2*[1], stride=2*[1], nb_filters=[64, 128],
                 pooling=2*[1]):
        super(LSTMCNN, self).__init__()
        
        self.lstm1 = LSTMModule(input_dim=input_dim, hidden_dim=hidden_dim1, dropout=dropout1, num_layers=num_layers1)    
        self.lstm2 = LSTMModule(input_dim=hidden_dim1, hidden_dim=hidden_dim2, dropout=dropout2, num_layers=num_layers2)      
        self.cnn = CNNModule(n_in_channel=hidden_dim2, activ=activ, cnn_dropout=cnn_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
        self.softmax = nn.Softmax(dim=-1)        
        self.dense = nn.Linear(nb_filters[-1], output_dim)

    def load_lstm(self, state_dict):
        self.lstm1.load_state_dict(state_dict)
        self.lstm2.load_state_dict(state_dict)
        
    def load_state_dict(self, state_dict, strict=True):
        self.lstm1.load_state_dict(state_dict["lstm1"])
        self.lstm2.load_state_dict(state_dict["lstm2"])
        self.cnn.load_state_dict(state_dict["cnn"])
        self.dense.load_state_dict(state_dict["dense"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"lstm1": self.lstm1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      "lstm2": self.lstm2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'lstm1': self.lstm1.state_dict(), 'lstm2': self.lstm2.state_dict(), 'cnn': self.cnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)
                        
    def forward(self, x): # [nBatch, segLeng, input_dim]
        recurrent = self.lstm1(x) # [nBatch, segLeng, nHidden1]
        recurrent = self.lstm2(recurrent) # [nBatch, segLeng, nHidden2]
        recurrent = recurrent.permute(0, 2, 1) # [nBatch, nHidden2, segLeng]        
        
        out = self.cnn(recurrent) # [nBatch, nb_filters[-1], segLeng]

        # segLeng => Weighted Arithmetic Mean 
        sof = self.softmax(out)
        sof = torch.clamp(sof, min=1e-7, max=1)
        out = (out*sof).sum(2)/sof.sum(2) # [nBatch, nb_filters[-1]]
               
        out = self.dense(out) # [nBatch, output_dim]
        return out

        
class GRUCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout1=0, dropout2=0, num_layers1=1, num_layers2=1, activ="Relu", cnn_dropout=0,
                 kernel_size=2*[3], padding=2*[1], stride=2*[1], nb_filters=[64, 128],
                 pooling=2*[1]):
        super(GRUCNN, self).__init__()
        
        self.gru1 = GRUModule(input_dim=input_dim, hidden_dim=hidden_dim1, dropout=dropout1, num_layers=num_layers1)    
        self.gru2 = GRUModule(input_dim=hidden_dim1, hidden_dim=hidden_dim2, dropout=dropout2, num_layers=num_layers2)         
        self.cnn = CNNModule(n_in_channel=hidden_dim2, activ=activ, cnn_dropout=cnn_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
        self.softmax = nn.Softmax(dim=-1)        
        self.dense = nn.Linear(nb_filters[-1], output_dim)


    def load_gru(self, state_dict):
        self.gru1.load_state_dict(state_dict)
        self.gru2.load_state_dict(state_dict)
                
    def load_state_dict(self, state_dict, strict=True):
        self.gru1.load_state_dict(state_dict["gru1"])
        self.gru2.load_state_dict(state_dict["gru2"])
        self.cnn.load_state_dict(state_dict["cnn"])
        self.dense.load_state_dict(state_dict["dense"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"gru1": self.gru1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      "gru2": self.gru2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      "cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'gru1': self.gru1.state_dict(), 'gru2': self.gru2.state_dict(), 'cnn': self.cnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)
                
    def forward(self, x):
        x = x.float() # [nBatch, segLeng, input_dim]
        recurrent = self.gru1(x) # [nBatch, segLeng, nHidden1]
        recurrent = self.gru2(recurrent) # [nBatch, segLeng, nHidden2]
        recurrent = recurrent.permute(0, 2, 1) # [nBatch, nHidden2, segLeng]        
        
        out = self.cnn(recurrent) # [nBatch, nb_filters[-1], segLeng]

        # segLeng => Weighted Arithmetic Mean 
        sof = self.softmax(out)
        sof = torch.clamp(sof, min=1e-7, max=1)
        out = (out*sof).sum(2)/sof.sum(2) # [nBatch, nb_filters[-1]]
               
        out = self.dense(out) # [nBatch, output_dim]
        return out
        
class CNNBiGRU1(nn.Module): # same as DCASE2020 CRNN
    def __init__(self, input_dim, hidden_dim1, output_dim, dropout1=0, num_layers1=2, activ="Relu", cnn_dropout=0.5,
                 kernel_size=4*[3], padding=4*[1], stride=4*[1], nb_filters=[16, 32, 64, 128],
                 pooling=4*[1]):
        super(CNNBiGRU1, self).__init__()

        self.cnn = CNNModule(n_in_channel=input_dim, activ=activ, cnn_dropout=cnn_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
                
        self.bigru1 = BiGRUModule(input_dim=nb_filters[-1], hidden_dim=hidden_dim1, dropout=dropout1, num_layers=num_layers1)         

        self.softmax = nn.Softmax(dim=-1)        
        self.dense = nn.Linear(hidden_dim1*2, output_dim)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        
    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.bigru1.load_state_dict(state_dict["bigru1"])
        self.dense.load_state_dict(state_dict["dense"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      "bigru1": self.bigru1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'bigru1': self.bigru1.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)
                
    def forward(self, x):
        x = x.permute(0, 2, 1) # [nBatch, input_dim, segLeng]
        x = self.cnn(x) # [nBatch, nb_filters[-1], segLeng]
        x = x.permute(0, 2, 1) # [nBatch, segLeng, nb_filters[-1]]        
        recurrent = self.bigru1(x) # [nBatch, segLeng, nHidden1*2]

        # segLeng => Weighted Arithmetic Mean 
        sof = self.softmax(recurrent)
        sof = torch.clamp(sof, min=1e-7, max=1)
        out = (recurrent*sof).sum(1)/sof.sum(1) # [nBatch, nHidden1*2]
               
        out = self.dense(out) # [nBatch, output_dim]
        return out


