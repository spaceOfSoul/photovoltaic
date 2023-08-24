import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import time
import logging
import datetime

from model import *
from data_loader import WPD
from torch.utils.data import DataLoader

from utility import list_up_solar, list_up_weather, print_parameters,count_parameters
from plot_generate import PlotGenerator
from LossDistribution import LossStatistics

from model_classes import *

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