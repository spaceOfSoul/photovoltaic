import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dtype=torch.double,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

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

class LSTMLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(LSTMLSTM, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
            dtype=torch.double,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
            dtype=torch.double,
        )
        self.fc = nn.Linear(hidden_dim2, output_dim)

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

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


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dtype=torch.double,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

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

class LSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(LSTMCNN, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        
        self.conv1 = nn.Conv1d(in_channels=hidden_dim2, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(768, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        out = out.permute(0, 2, 1) 
        # convlayer input shape : (batch_size, channels, sequence_length)
        # channels is can match hidden dim
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
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
