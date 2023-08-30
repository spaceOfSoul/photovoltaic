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

params = {
    "aws_dir":"./dataset/AWS/",
    "asos_dir":"./dataset/ASOS/",
    "solar_dir":"./dataset/photovoltaic/GWNU_C9/",
    "loc_ID":678
}

if __name__ == "__main__":
    print("load start")
    
    solar_list, first_date, last_date = list_up_solar(params["solar_dir"])
    aws_list = list_up_weather(params["aws_dir"], first_date, last_date)
    asos_list = list_up_weather(params["asos_dir"], first_date, last_date)
    print(f"Load on the interval from {first_date} to {last_date}.")
    
    dataset = WPD(
        aws_list,
        asos_list,
        solar_list,
        params["loc_ID"],
        input_dim=8,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    
    for trn_days, (x, y) in enumerate(dataloader):
        print(x.shape)
        print(y.shape)
        for i in range(x.shape[-1]):
            feature_data = x[0, :, i].numpy()

            # Create plot
            plt.figure(figsize=(10, 4))
            plt.plot(feature_data)
            plt.title(f'Day {trn_days + 1} - Feature {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            
            # Save the plot
            plt.savefig(f"test_dataload/Day_{trn_days+1}_Feature_{i + 1}.png")
            plt.close()