import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import argparse
import time
import logging
import datetime

from model import *
from data_loader import WPD
from torch.utils.data import DataLoader

from utility import list_up_solar, list_up_weather, print_parameters,count_parameters
from plot_generate import PlotGenerator
from LossDistribution import LossStatistics

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
        "transformer" : TRANSFORMER
    }

def hyper_params():
    # Default setting
    nlayers = 2 # nlayers of CNN 
    model_params = { 
        # Common
        "seqLeng": 60,
        "input_dim": 8, # feature 7 + time 1
        "output_dim": 1, 
        
        # LSTM of single model(LSTM), LSTM1 of hybrid model(LSTM1-LSTM2-CNN), GRU1 of hybrid model(GRU1-GRU2-CNN), BiGRU1 of hybrid model(CNN-BiGRU1), transformer
        "nHidden1": 256, 
        "dropout1": 0, 
        "num_layers1": 8, # 2: BiGRU1 of hybrid model(CNN-BiGRU1), Transformer num Layrer
        
        
        # LSTM2 of hybrid model(LSTM-CNN), GRU2 of hybrid model(GRU1-GRU2-CNN)
        "nHidden2": 256,         
        "dropout2": 0,
        "num_layers2": 1,
                        
        # CNN
        "activ": "glu", # leakyrelu, relu, glu, cg
        "cnn_dropout": 0, 
        "kernel_size": nlayers*[3],
        "padding": nlayers*[1],
        "stride": nlayers*[1], 
        "nb_filters": [64, 128], # length of nb_filters should be equal to nlayers.
        "pooling": nlayers*[1],

        # transformer
        "nhead" : 8,
        "d_model" : 512
    }

    learning_params = {
        "nBatch": 24,
        "lr": 1.0e-3,
        "max_epoch": 1500,
    }

    hparams = {
        "model": model_params,
        "learning": learning_params,
        # system flags
        "loss_plot_flag": True,
        "save_losses": True,
        "save_result": True,
    }

    return hparams


def parse_flags(hparams):
    parser = argparse.ArgumentParser(description="Photovoltaic estimation")
    parser.add_argument("--log_filename", type=str, default="output.txt")

    # Flags common to all modes
    all_modes_group = parser.add_argument_group("Flags common to all modes")
    all_modes_group.add_argument(
       "--mode", type=str, choices=["train", "test"], required=True
    )
    all_modes_group.add_argument(
       "--model", type=str, choices=list(model_classes.keys()), required=True
    ) 

    # Flags for training only
    training_group = parser.add_argument_group("Flags for training only")
    training_group.add_argument("--save_dir", type=str, default="")
    training_group.add_argument("--aws_dir", type=str, default="./dataset/AWS/")
    training_group.add_argument("--asos_dir", type=str, default="./dataset/ASOS/")
    training_group.add_argument(
        "--solar_dir", type=str, default="./dataset/photovoltaic/GWNU_C9/"
    )
    training_group.add_argument("--loc_ID", type=int, default=678)

    # Flags for validation only
    validation_group = parser.add_argument_group("Flags for validation only")
    validation_group.add_argument("--val_aws_dir", type=str, default="./dataset/AWS/")
    validation_group.add_argument("--val_asos_dir", type=str, default="./dataset/ASOS/")
    validation_group.add_argument(
        "--val_solar_dir", type=str, default="./dataset/photovoltaic/GWNU_C3/"
    )

    # Flags for test only
    test_group = parser.add_argument_group("Flags for test only")
    test_group.add_argument("--load_path", type=str, default="")
    test_group.add_argument("--tst_aws_dir", type=str, default="./dataset/AWS/")
    test_group.add_argument("--tst_asos_dir", type=str, default="./dataset/ASOS/")
    test_group.add_argument("--tst_solar_dir", type=str, default="./samcheck/data/")
    test_group.add_argument("--tst_loc_ID", type=int, default=106)

    # Flags for training params
    trn_param_set = parser.add_argument_group("Flags for training paramters")
    trn_param_set.add_argument(
        "--seqLeng", type=int, default=hparams["model"]["seqLeng"]
    )
    trn_param_set.add_argument(
        "--nHidden1", type=int, default=hparams["model"]["nHidden1"]
    )
    trn_param_set.add_argument(
        "--nBatch", type=int, default=hparams["learning"]["nBatch"]
    )
    trn_param_set.add_argument(
        "--max_epoch", type=int, default=hparams["learning"]["max_epoch"]
    )

    flags = parser.parse_args()

    hparams["model"]["seqLeng"] = flags.seqLeng
    hparams["model"]["nHidden1"] = flags.nHidden1
    hparams["learning"]["nBatch"] = flags.nBatch
    hparams["learning"]["max_epoch"] = flags.max_epoch

    # Additional per-mode validation
    try:
        if flags.mode == "train":
            assert flags.save_dir, "Must specify --save_dir"
        elif flags.mode == "test":
            assert flags.load_path, "Must specify --load_path"

    except AssertionError as e:
        print("\nError: ", e, "\n")
        parser.print_help()
        sys.exit(1)

    return flags, hparams, flags.model


def train(hparams, model_type):
    model_params = hparams["model"]
    learning_params = hparams["learning"]

    trnset = WPD(
        hparams["aws_list"],
        hparams["asos_list"],
        hparams["solar_list"],
        hparams["loc_ID"],
        input_dim=hparams["model"]["input_dim"],
    )
    valset = WPD(
        hparams["val_aws_list"],
        hparams["val_asos_list"],
        hparams["val_solar_list"],
        hparams["loc_ID"],
        input_dim=hparams["model"]["input_dim"],
    )

    trnloader = DataLoader(trnset, batch_size=1, shuffle=False, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=True)
           
    seqLeng = model_params["seqLeng"]
    input_dim = model_params["input_dim"] # feature 7 + time 1
    output_dim = model_params["output_dim"] 
        
    hidden_dim1 = model_params["nHidden1"] 
    dropout1 = model_params["dropout1"] 
    num_layers1 = model_params["num_layers1"] 
        
    hidden_dim2 = model_params["nHidden2"]         
    dropout2 = model_params["dropout2"]
    num_layers2 = model_params["num_layers2"]
                        
    activ = model_params["activ"] 
    cnn_dropout = model_params["cnn_dropout"] 
    kernel_size = model_params["kernel_size"]
    padding = model_params["padding"]
    stride = model_params["stride"] 
    nb_filters = model_params["nb_filters"]
    pooling = model_params["pooling"]

    d_model = model_params["d_model"]
    nhead = model_params["nhead"]

    if model_type in ["lstm"]: # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim, dropout1, num_layers1)
    elif model_type in ["cnn"]: # single model, n_in_channel = input_dim
        model = model_classes[model_type](input_dim, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling, output_dim) 
    elif model_type in ["gru-cnn", "lstm-cnn", "cnn-lstm"]: # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, output_dim, dropout1, dropout2, num_layers1, num_layers2, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling)
    elif model_type in ["cnn-bigru1"]: # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim, dropout1, num_layers1, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling) 
    elif model_type in ["conformer"]:
        model = model_classes[model_type](input_dim, output_dim)
    # Up : chg
    # Down : mine
    elif model_type == "lstm-basic": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "lstm2-basic": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, output_dim)
    elif model_type == "rnn": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "lstmcnn-basic": # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, seqLeng, output_dim)
    elif model_type == "gru":
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "attention-lstm":
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "transformer":
        model = model_classes[model_type](input_dim, output_dim, d_model, nhead)
    else:
        print("The provided model type is not recognized.")
        sys.exit(1)

    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_params["lr"])

    max_epoch = learning_params["max_epoch"]
    seqLeng = model_params["seqLeng"]
    nBatch = learning_params["nBatch"]

    losses = []
    val_losses = []

    prev_loss = np.inf
    train_start = time.time()
    for epoch in range(max_epoch):
        if model_type != "transformer":
            model.train()
            loss = 0
            prev_data = torch.zeros([seqLeng, input_dim]).cuda()
            for i, (x, y) in enumerate(trnloader):
                x = x.float()
                y = y.float()
                x = x.squeeze().cuda()
                x = torch.cat((prev_data, x), axis=0)
                prev_data = x[-seqLeng:, :]
                y = y.squeeze().cuda()

                nLeng, nFeat = x.shape
                batch_data = []
                for j in range(nBatch):
                    stridx = j * 60
                    endidx = j * 60 + seqLeng
                    batch_data.append(x[stridx:endidx, :].view(1, seqLeng, nFeat))
                batch_data = torch.cat(batch_data, dim=0)

                output = model(batch_data.cuda())
                loss += criterion(output.squeeze(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            val_loss = 0
            prev_data = torch.zeros([seqLeng, input_dim]).cuda()
            for i, (x, y) in enumerate(valloader):
                x = x.float()
                y = y.float()
                x = x.squeeze().cuda()
                x = torch.cat((prev_data, x), axis=0).cuda()
                prev_data = x[-seqLeng:, :]
                y = y.squeeze().cuda()

                nLeng, nFeat = x.shape
                batch_data = []
                for j in range(nBatch):
                    stridx = j * 60
                    endidx = j * 60 + seqLeng
                    batch_data.append(x[stridx:endidx, :].view(1, seqLeng, nFeat))
                batch_data = torch.cat(batch_data, dim=0)

                output = model(batch_data.cuda())
                val_loss += criterion(output.squeeze(), y)

        else: # transformer
            model.train()
            total_loss = 0
            for i, (x, y) in enumerate(trnloader):
                x = x.float().cuda()
                y = y.float().squeeze().cuda()

                output = model(x)
                loss = criterion(output[:, -1, :].squeeze(), y)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            total_val_loss = 0
            for i, (x, y) in enumerate(valloader):
                x = x.float().cuda()
                y = y.float().squeeze().cuda()

                with torch.no_grad():
                    output = model(x)
                val_loss = criterion(output[:, -1, :].squeeze(), y)
                total_val_loss += val_loss.item()

        if val_loss < prev_loss:
            savePath = os.path.join(hparams["save_dir"], "best_model")
            model_dict = {"kwargs": model_params, "paramSet": model.state_dict()}
            torch.save(model_dict, savePath)
            prev_loss = val_loss

        losses.append(loss.item())
        val_losses.append(val_loss.item())
        logging.info(f"Epoch [{epoch+1}/{max_epoch}], Trn Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    train_end = time.time()
    logging.info("\n")
    logging.info(f'Training time [sec]: {(train_end - train_start):.2f}')

    min_val_loss = min(val_losses)
    min_val_loss_epoch = val_losses.index(min_val_loss)

    if hparams["save_losses"]:
        trn_loss = np.array(losses)
        val_loss = np.array(val_losses)
        np.save(os.path.join(hparams["save_dir"],"trn_loss.npy"), trn_loss)
        np.save(os.path.join(hparams["save_dir"],"val_loss.npy"), val_loss)

    if hparams["loss_plot_flag"]:
        plt.figure()
        plt.plot(range(max_epoch), np.array(losses), "b", label='Training Loss')
        plt.plot(range(max_epoch), np.array(val_losses), "r", label='Validation Loss')
        plt.scatter(min_val_loss_epoch, min_val_loss, color='k', label='Minimun Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss, Min Val Loss: {min_val_loss:.4f} at Epoch {min_val_loss_epoch}")
        plt.legend()
        plt.savefig(os.path.join(hparams["save_dir"],"figure_train.png"))
        logging.info(f"minimum validation loss: {min_val_loss:.4f} at Epoch {min_val_loss_epoch}")

def test(hparams, model_type):
    model_params = hparams['model']
    learning_params = hparams['learning']

    modelPath = hparams['load_path']
    
    try:
        ckpt = torch.load(modelPath)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Loaded object from {modelPath} is not a dictionary.")
        if 'kwargs' not in ckpt or 'paramSet' not in ckpt:
            raise ValueError(f"Dictionary from {modelPath} does not contain expected keys.")
        model_conf = ckpt['kwargs']
        paramSet = ckpt['paramSet']
    except Exception as e:
        print(f"Error occurred while loading model from {modelPath}")
        print(f"Error: {e}")

    seqLeng = model_params["seqLeng"]
    input_dim = model_params["input_dim"] # feature 7 + time 1
    output_dim = model_params["output_dim"] 
        
    hidden_dim1 = model_params["nHidden1"] 
    dropout1 = model_params["dropout1"] 
    num_layers1 = model_params["num_layers1"] 
        
    hidden_dim2 = model_params["nHidden2"]         
    dropout2 = model_params["dropout2"]
    num_layers2 = model_params["num_layers2"]
                        
    activ = model_params["activ"] 
    cnn_dropout = model_params["cnn_dropout"] 
    kernel_size = model_params["kernel_size"]
    padding = model_params["padding"]
    stride = model_params["stride"] 
    nb_filters = model_params["nb_filters"]
    pooling = model_params["pooling"]

    d_model = model_params["d_model"]
    nhead = model_params["nhead"]

    if model_type in ["lstm"]: # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim, dropout1, num_layers1)
    elif model_type in ["cnn"]: # single model, n_in_channel = input_dim
        model = model_classes[model_type](input_dim, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling, output_dim) 
    elif model_type in ["gru-cnn", "lstm-cnn", "cnn-lstm"]: # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, output_dim, dropout1, dropout2, num_layers1, num_layers2, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling)
    elif model_type in ["cnn-bigru1"]: # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim, dropout1, num_layers1, activ, cnn_dropout, kernel_size, padding, stride, nb_filters, pooling) 
    elif model_type in ["conformer"]:
        model = model_classes[model_type](input_dim, output_dim)
    # Up : chg
    # Down : mine
    # I cant explane model made by chg
    elif model_type == "lstm-basic": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "lstm2-basic": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, output_dim)
    elif model_type == "rnn": # single model
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "lstmcnn-basic": # hybrid model
        model = model_classes[model_type](input_dim, hidden_dim1, hidden_dim2, seqLeng, output_dim)
    elif model_type == "gru":
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "attention-lstm":
        model = model_classes[model_type](input_dim, hidden_dim1, output_dim)
    elif model_type == "transformer":
        model = model_classes[model_type](input_dim, output_dim, d_model, nhead)
    else:
        print("The provided model type is not recognized.")
        sys.exit(1)
        
    model.load_state_dict(paramSet)
    model.cuda()
    model.eval()

    tstset  = WPD(hparams['aws_list'], hparams['asos_list'], hparams['solar_list'], hparams['loc_ID'])    
    tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)

    seqLeng = model_params['seqLeng']
    nBatch  = learning_params['nBatch']
    prev_data = torch.zeros([seqLeng, input_dim]).cuda()	# 7 is for featDim

    criterion = torch.nn.MSELoss()
    total_loss = 0
    total_mape = 0
    total_samples = 24*243
    result = []
    y_true=[]
    test_start = time.time()
    for i, (x, y) in enumerate(tstloader):
        x = x.float()
        y = y.float()
        x = x.squeeze().cuda()
        x = torch.cat((prev_data, x), axis=0)
        prev_data = x[-seqLeng:,:]
        y = y.squeeze(1).cuda()

        nLeng, nFeat = x.shape
        batch_data = []
        for j in range(nBatch):
            stridx = j*60
            endidx = j*60 + seqLeng
            batch_data.append(x[stridx:endidx,:].view(1, seqLeng, nFeat))
        batch_data = torch.cat(batch_data, dim=0)

        output = model(batch_data)

        if model_type == "transformer":
            output = output[:, -1, :].squeeze()
        
        result.append(output.detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        loss = criterion(output.squeeze(), y)
        
        total_loss += loss.item()
    test_end = time.time()

    average_loss = total_loss/total_samples
    # print_parameters(model) # print params infomation
    logging.info("\n--------------------Test Mode--------------------")
    logging.info(f'Average Loss: {average_loss:.4f}')
    logging.info(f'The number of parpameter in model : {count_parameters(model)}')
    logging.info(f'Testing time [sec]: {(test_end - test_start):.2f}')
    
    model_dir = os.path.dirname(modelPath)
    
    if hparams['save_result']:
        result_npy = np.array(result)
        np.save(os.path.join(model_dir,"test_result.npy"), result_npy)
    
    image_dir = os.path.join(model_dir, "test_images")
    os.makedirs(image_dir, exist_ok=True)
    chunks = len(y_true) // 12

    monthly_lengths = [31, 28, 31, 30, 31, 30, 31, 31]  # The number of days in each month from 2022.January to August

    y_true_chunks = []
    result_chunks = []

    start_index = 0
    for month_length in monthly_lengths:
        end_index = start_index + month_length
        y_true_chunks.append(y_true[start_index:end_index])
        result_chunks.append(result[start_index:end_index])
        start_index = end_index

    plot_generator = PlotGenerator(image_dir)

    plot_generator.plot_monthly(y_true_chunks, result_chunks)
    plot_generator.plot_annual(y_true, result)

    for i in range(12):
        plt.figure(figsize=(10, 5))
        plt.plot(np.concatenate(y_true_chunks[i]), label='True')
        plt.plot(np.concatenate(result_chunks[i]), label='Predicted')
        plt.legend()
        plt.title(f'month {i+1}')
        plt.savefig(os.path.join(image_dir, f"month_{i+1}.png"))
        plt.close()

def loss_with_npy_file(path_to_predicted_npy_file):
    tst_aws_dir = "./dataset/AWS/"
    tst_asos_dir = "./dataset/ASOS/"
    tst_solar_dir = "./samcheck/data/"
    tst_loc_ID = 106
    seqLeng = 60 
    nBatch = 24  
    solar_list, first_date, last_date = list_up_solar(tst_solar_dir)
    aws_list = list_up_weather(tst_aws_dir, first_date, last_date)
    asos_list = list_up_weather(tst_asos_dir, first_date, last_date)

    tstset  = WPD(aws_list, asos_list, solar_list, tst_loc_ID)    
    tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)


    criterion = torch.nn.MSELoss()
    total_loss = 0
    total_samples = 24*243

    predicted_output_from_npy = np.load(path_to_predicted_npy_file)

    for i, (_, y) in enumerate(tstloader):
        #x = x.float()
        y = y.float()
        #x = x.squeeze().cuda()
        #x = torch.cat((prev_data, x), axis=0)
        #prev_data = x[-seqLeng:,:]
        y = y.squeeze().cuda()

        #nLeng, nFeat = x.shape
        #batch_data = []
        #for j in range(nBatch):
        #    stridx = j*60
        #    endidx = j*60 + seqLeng
        #    batch_data.append(x[stridx:endidx,:].view(1, seqLeng, nFeat))
        #batch_data = torch.cat(batch_data, dim=0)

        output = torch.tensor(predicted_output_from_npy[i], device='cuda').float()

        loss = criterion(output.squeeze(), y)
        
        total_loss += loss.item()

    average_loss = total_loss/total_samples
    print(f'Average Loss: {average_loss:.4f}')


def plot_loss_from_npy(train_loss_npy_path, val_loss_npy_path):
    # Load the npy files
    train_losses = np.load(train_loss_npy_path)
    val_losses = np.load(val_loss_npy_path)

    # Find the index of the minimum validation loss
    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_val_loss = min_val_loss/(205)

    # Plot the losses
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, "b", label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, "r", label='Validation Loss')
    plt.scatter(min_val_loss_idx, min_val_loss, color='k', label='Minimum Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss, Min Val Loss: {min_val_loss:.4f} at Epoch {min_val_loss_idx}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #plot_loss_from_npy("train_models/old/cnn-lstm_30_64-128_3000/trn_loss.npy","train_models/old/cnn-lstm_30_64-128_3000/val_loss.npy")
    #loss_with_npy_file('train_models/old/gru_30_64_3000/test_result.npy')
    #exit()

    hp = hyper_params()
    flags, hp, model_name = parse_flags(hp)
    if flags.mode == "train":
        if not os.path.isdir(flags.save_dir):
            os.makedirs(flags.save_dir)
        # Set up logging
        log_filename = os.path.join(flags.save_dir, flags.log_filename)
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

        # Then replace 'print' with 'logging.info' in your code
        logging.info('This is a log message.\n')

        current_time = datetime.datetime.now()
        logging.info(f"Current time: {current_time}\n")

        # Log hyperparameters and model name
        logging.info('--------------------Hyperparameters--------------------\n')
        for key, value in hp.items():
            logging.info(f"{key}: {value}\n")
        logging.info(f"Model name: {model_name}\n")
        logging.info("\n--------------------Training Mode (Training and Validating)--------------------")
        # =============================== training data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.solar_dir)
        aws_list = list_up_weather(flags.aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.asos_dir, first_date, last_date)
        logging.info(f"Training on the interval from {first_date} to {last_date}.")
        # =============================== validation data list ===================================#
        # build photovoltaic data list
        val_solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        val_aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        val_asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        logging.info(f"Validating on the interval from {first_date} to {last_date}.\n")
        # ========================================================================================#

        hp.update({"aws_list": aws_list})
        hp.update({"val_aws_list": val_aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"val_asos_list": val_asos_list})
        hp.update({"solar_list": solar_list})
        hp.update({"val_solar_list": val_solar_list})
        hp.update({"save_dir": flags.save_dir})
        hp.update({"loc_ID": flags.loc_ID})

        if not os.path.isdir(flags.save_dir):
            os.makedirs(flags.save_dir)

        train(hp, flags.model)

        hp.update({"load_path": os.path.join(flags.save_dir,"best_model")})
        hp.update({"loc_ID": flags.tst_loc_ID})

        solar_list, first_date, last_date = list_up_solar(flags.tst_solar_dir)
        aws_list = list_up_weather(flags.tst_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_asos_dir, first_date, last_date)

        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})

        test(hp, flags.model)

    elif flags.mode == "test":
        model_dir = os.path.dirname(flags.load_path)
        log_filename = os.path.join(model_dir, flags.log_filename)
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    
        # Then replace 'print' with 'logging.info' in your code
        logging.info('This is a log message.\n')
    
        current_time = datetime.datetime.now()
        logging.info(f"Current time: {current_time}\n")
    
        # Log hyperparameters and model name
        logging.info('--------------------Hyperparameters--------------------\n')
        logging.info(f"Model name: {model_name}\n")
        #logging.info("\n--------------------Test Mode--------------------")
        hp.update({"load_path": flags.load_path})
        hp.update({"loc_ID": flags.tst_loc_ID})

        # =============================== test data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.tst_solar_dir)
        aws_list = list_up_weather(flags.tst_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_asos_dir, first_date, last_date)

        # ========================================================================================#

        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})

        test(hp, flags.model)
