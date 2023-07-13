import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import argparse

from model import LSTM
from data_loader import WPD
from torch.utils.data import DataLoader
from utility import list_up_solar, list_up_weather


def hyper_params():
    # Default setting
    model_params = {
        "seqLeng": 30,
        "input_dim": 8,
        "nHidden": 64,
        "output_dim": 1,
    }

    learning_params = {
        "nBatch": 24,
        "lr": 1.0e-3,
        "max_epoch": 5000,
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

    # Flags common to all modes
    all_modes_group = parser.add_argument_group("Flags common to all modes")
    all_modes_group.add_argument(
        "--mode", type=str, choices=["train", "test"], required=True
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
    test_group.add_argument("--tst_loc_ID", type=int, default=678)

    # Flags for training params
    trn_param_set = parser.add_argument_group("Flags for training paramters")
    trn_param_set.add_argument(
        "--seqLeng", type=int, default=hparams["model"]["seqLeng"]
    )
    trn_param_set.add_argument(
        "--nHidden", type=int, default=hparams["model"]["nHidden"]
    )
    trn_param_set.add_argument(
        "--nBatch", type=int, default=hparams["learning"]["nBatch"]
    )
    trn_param_set.add_argument(
        "--max_epoch", type=int, default=hparams["learning"]["max_epoch"]
    )

    flags = parser.parse_args()

    hparams["model"]["seqLeng"] = flags.seqLeng
    hparams["model"]["nHidden"] = flags.nHidden
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

    return flags, hparams


def train(hparams):
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

    input_dim = model_params["input_dim"]
    hidden_dim = model_params["nHidden"]
    output_dim = model_params["output_dim"]
    model = LSTM(input_dim, hidden_dim, output_dim)
    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_params["lr"])

    max_epoch = learning_params["max_epoch"]
    seqLeng = model_params["seqLeng"]
    nBatch = learning_params["nBatch"]

    losses = []
    val_losses = []

    prev_loss = np.inf
    for epoch in range(max_epoch):
        model.train()
        loss = 0
        prev_data = torch.zeros([seqLeng, input_dim]).cuda()
        for i, (x, y) in enumerate(trnloader):
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

            output = model(batch_data)
            loss += criterion(output.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = 0
        prev_data = torch.zeros([seqLeng, input_dim]).cuda()
        for i, (x, y) in enumerate(valloader):
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

            output = model(batch_data)
            val_loss += criterion(output.squeeze(), y)

        if val_loss < prev_loss:
            savePath = os.path.join(hparams["save_dir"], "best_model")  # overwrite
            model_dict = {"kwargs": model_params, "paramSet": model.state_dict()}
            torch.save(model_dict, savePath)
            prev_loss = val_loss

        losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(
            f"Epoch [{epoch+1}/{max_epoch}], Trn Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
        )

    if hparams["save_losses"]:
        trn_loss = np.array(losses)
        val_loss = np.array(val_losses)
        np.save("trn_loss.npy", trn_loss)
        np.save("val_loss.npy", val_loss)

    if hparams["loss_plot_flag"]:
        plt.plot(range(max_epoch), losses, "b", range(max_epoch), val_losses, "r")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()


def test(hparams):
    model_params = hparams["model"]
    learning_params = hparams["learning"]

    modelPath = hparams["load_path"]
    try:
        ckpt = torch.load(modelPath)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Loaded object from {modelPath} is not a dictionary.")
        if "kwargs" not in ckpt or "paramSet" not in ckpt:
            raise ValueError(
                f"Dictionary from {modelPath} does not contain expected keys."
            )
        model_conf = ckpt["kwargs"]
        paramSet = ckpt["paramSet"]
    except Exception as e:
        print(f"Error occurred while loading model from {modelPath}")
        print(f"Error: {e}")

    input_dim = model_conf["input_dim"]
    hidden_dim = model_conf["nHidden"]
    output_dim = model_conf["output_dim"]
    model = LSTM(input_dim, hidden_dim, output_dim)
    model.load_state_dict(paramSet)
    model.cuda()
    model.eval()

    tstset = WPD(hparams["weather_list"], hparams["solar_list"], hparams["loc_ID"])
    tstloader = DataLoader(tstset, batch_size=10, shuffle=False, drop_last=True)

    seqLeng = model_params["seqLeng"]
    nBatch = learning_params["nBatch"]
    prev_data = torch.zeros([seqLeng, input_dim]).cuda()  # 14 is for featDim

    criterion = torch.nn.MSELoss()
    loss = 0
    result = []
    for i, (x, y) in enumerate(tstloader):
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

        output = model(batch_data)
        result.append(output.detach().cpu().numpy())
        loss += criterion(output.squeeze(), y)

    print(f"Tsn Loss: {loss.item():.4f}")

    if hparams["save_result"]:
        result_npy = np.array(result)
        np.save("prediction.npy", result_npy)


if __name__ == "__main__":
    hp = hyper_params()
    flags, hp = parse_flags(hp)

    if flags.mode == "train":
        # =============================== training data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.solar_dir)
        aws_list = list_up_weather(flags.aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.asos_dir, first_date, last_date)
        print("Training on the interval from %s to %s." % (first_date, last_date))
        # =============================== validation data list ===================================#
        # build photovoltaic data list
        val_solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        val_aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        val_asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        print("Validating on the interval from %s to %s." % (first_date, last_date))
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

        train(hp)

    elif flags.mode == "test":
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

        test(hp)
