from model import *

model_classes = {
        "lstm": LSTM, #o
        "cnn": CNN,#o
        "lstm-cnn": LSTMCNN,#o
        #"cnn-lstm": LSTMCNN,
        "gru-cnn": GRUCNN,#o
        "cnn-bigru1": CNNBiGRU1,#o
        "conformer": Conformer,
        "lstm-basic":BASIC_LSTM,#o
        "lstm2-basic":BASIC_LSTM2,#o
        "lstmcnn-basic":BASIC_LSTMCNN,#o
        "gru":GRU,#o
        "rnn":BASIC_RNN,#o
        "attention-lstm":ATTENTION_LSTM,#o
        "transformer" : TRANSFORMER # ?
    }