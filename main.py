import os
import sys
import argparse
import logging
import datetime

from model import *
from utility import list_up_solar, list_up_weather, print_parameters,count_parameters

from model_classes import *
from train import train
from test import test

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

if __name__ == "__main__":
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
