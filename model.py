import torch
from torch import nn
from models.RNNs import GRUModule, BiGRUModule, LSTMModule
from models.CNN import CNNModule
from torchaudio.models import Conformer as ConformerModule
from utility import conv_output_size, pool_output_size

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

class BASIC_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BASIC_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)


    def load_state_dict(self, state_dict):
        self.lstm.load_state_dict(state_dict["lstm"])
        self.fc.load_state_dict(state_dict["fc"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "lstm": self.lstm.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "fc": self.fc.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({"lstm": self.lstm.state_dict()})
        parameters.update({"fc": self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BASIC_LSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(BASIC_LSTM2, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def load_state_dict(self, state_dict):
        self.lstm1.load_state_dict(state_dict["lstm1"])
        self.lstm2.load_state_dict(state_dict["lstm2"])
        self.fc.load_state_dict(state_dict["fc"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "lstm1": self.lstm1.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "lstm2": self.lstm2.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "fc": self.fc.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({"lstm1": self.lstm1.state_dict()})
        parameters.update({"lstm2": self.lstm2.state_dict()})
        parameters.update({"fc": self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, (hidden, cell) = self.lstm1(x)
        out, (hidden, cell) = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

class BASIC_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BASIC_RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def load_state_dict(self, state_dict):
        self.rnn.load_state_dict(state_dict["rnn"])
        self.fc.load_state_dict(state_dict["fc"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "rnn": self.rnn.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "fc": self.fc.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({"rnn": self.rnn.state_dict()})
        parameters.update({"fc": self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BASIC_LSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,seq_len, output_dim):
        super(BASIC_LSTMCNN, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1, batch_first=True,)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True,)
        
        self.conv1 = nn.Conv1d(in_channels=hidden_dim2, out_channels=64, kernel_size=3, stride=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        conv1_output_size = conv_output_size(seq_len, kernel_size=3, stride=1)
        pool1_output_size = pool_output_size(conv1_output_size, kernel_size=2, stride=2)
        conv2_output_size = conv_output_size(pool1_output_size, kernel_size=3, stride=1)
        pool2_output_size = pool_output_size(conv2_output_size, kernel_size=2, stride=2)
        fc1_input_dim = hidden_dim2 * pool2_output_size # 128
        #fc1_input_dim = hidden_dim2 * pool2_output_size // 2 # 256
        #fc1_input_dim = hidden_dim2 * pool2_output_size //4 # 512

        self.fc1 = nn.Linear(fc1_input_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)

    def forward(self, x):

        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        out = out.permute(0, 2, 1) 
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        #assert out.dtype == torch.float32, f"Output dtype is {out.dtype} instead of float32"
        return out

    def load_state_dict(self, state_dict):
        self.lstm1.load_state_dict(state_dict["lstm1"])
        self.lstm2.load_state_dict(state_dict["lstm2"])
        self.conv1.load_state_dict(state_dict["conv1"])
        self.conv2.load_state_dict(state_dict["conv2"])
        self.fc1.load_state_dict(state_dict["fc1"])
        self.fc2.load_state_dict(state_dict["fc2"])
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return {
            "lstm1": self.lstm1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            "lstm2": self.lstm2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            "conv1": self.conv1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            "conv2": self.conv2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            "fc1": self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            "fc2": self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        
    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({"lstm1": self.lstm1.state_dict()})
        parameters.update({"lstm2": self.lstm2.state_dict()})
        parameters.update({"conv1": self.conv1.state_dict()})
        parameters.update({"conv2": self.conv2.state_dict()})
        parameters.update({"fc1": self.fc1.state_dict()})
        parameters.update({"fc2": self.fc2.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def load_state_dict(self, state_dict):
        self.gru.load_state_dict(state_dict["gru"])
        self.fc.load_state_dict(state_dict["fc"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "gru": self.gru.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "fc": self.fc.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({"gru": self.gru.state_dict()})
        parameters.update({"fc": self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, hidden = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out