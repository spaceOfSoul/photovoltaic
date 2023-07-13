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
        super(LSTM, self).__init__()
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
