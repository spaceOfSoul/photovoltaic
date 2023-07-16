import os
import pandas as pd
import numpy as np
from datetime import datetime


def path2date(path):
    date_str = path.split('/')[-2] + '/' + \
        path.split('/')[-1].replace('.xlsx', '')
    date = datetime.strptime(date_str, '%Y_%m/%d')
    return date


def list_up_solar(solar_directory):
    # print("Current working directory in list_up_solar:", os.getcwd())
    # build photovoltaic data list
    solar_dir = os.listdir(solar_directory)
    # solar_dir.sort()
    solar_list = []
    for folder in solar_dir:
        mlist = os.listdir(solar_directory+'/'+folder)
        mlist = [file for file in mlist if file.find('xlsx') > 0]
        mlist = sorted(mlist, key=lambda x: int(x.split('.')[0]))
        for file in mlist:
            path = solar_directory + '/' + folder + '/' + file
            solar_list.append(path)

    solar_list.sort(key=path2date)

    # find period
    first_ = solar_list[0].split('.')[1].split('/')
    first_year, first_month = first_[-2].split('_')
    first_day = str("%02d" % int(first_[-1]))
    first_date = first_year+first_month+first_day

    last_ = solar_list[-1].split('.')[1].split('/')
    last_year, last_month = last_[-2].split('_')
    last_day = str("%02d" % int(last_[-1]))
    last_date = last_year+last_month+last_day
    #print('Training with data from %s to %s.'%(first_date, last_date))

    return solar_list, first_date, last_date


def list_up_weather(weather_directory, first_date, last_date):
    # print("Current working directory in list_up_weather:", os.getcwd())
    # build weather data list
    weather_dir = os.listdir(weather_directory)
    weather_dir.sort(key=lambda x: int(x[:-1]) if x.endswith('월') else int(x))
    weather_list = []
    stridx, endidx, cnt = -1, -1, -1
    for folder in weather_dir:
        wlist = os.listdir(weather_directory+'/'+folder)
        wlist = [file for file in wlist if file.find('csv') > 0]
        wlist.sort()
        for file in wlist:
            path = weather_directory + '/' + folder + '/' + file
            weather_list.append(path)
            cnt += 1
            if path.find(first_date) > 0:
                stridx = cnt
            if path.find(last_date) > 0:
                endidx = cnt

    weather_list = weather_list[stridx:endidx+1]

    return weather_list

def accuracy(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true+eps))) * 100

def conv_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1

def pool_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1