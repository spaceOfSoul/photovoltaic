import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import time
import logging

from model import *
from data_loader import WPD
from torch.utils.data import DataLoader

from utility import list_up_solar, list_up_weather, print_parameters,count_parameters
from plot_generate import PlotGenerator

from model_classes import *

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
        y = y.squeeze().cuda()

        nLeng, nFeat = x.shape
        batch_data = []
        for j in range(nBatch):
            stridx = j*60
            endidx = j*60 + seqLeng
            batch_data.append(x[stridx:endidx,:].view(1, seqLeng, nFeat))
        batch_data = torch.cat(batch_data, dim=0)

        output = model(batch_data)
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