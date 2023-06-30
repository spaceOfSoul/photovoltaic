import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import argparse

from model import RNN
from data_loader import WPD
from torch.utils.data import DataLoader

def hyper_params():
	# Default setting
	hparams = {
		# model params
		'seqLeng': 30,
		'input_dim': 15,
		'nHidden': 64,
		'output_dim': 1,

		# training params
		'nBatch': 24,
		'lr'    : 1.0e-3,
		'max_epoch': 5000,

		# plotting flag
		'loss_plot_flag': True,
		'save_losses': True,
	}

	return hparams

def parse_flags():
	parser = argparse.ArgumentParser(description='Photovoltaic estimation')

	# Flags common to all modes
	all_modes_group = parser.add_argument_group('Flags common to all modes')
	all_modes_group.add_argument('--mode', type=str, choices=['train', 'test'], required=True)

	# Flags for training only
	training_group = parser.add_argument_group('Flags for training only')
	training_group.add_argument('--save_dir', type=str, default='')
	training_group.add_argument('--weather_dir', type=str, default='./dataset/AWS/')
	training_group.add_argument('--solar_dir', type=str, default='./dataset/photovoltaic/GWNU_C9/')

	# Flags for validation only
	validation_group = parser.add_argument_group('Flags for validation only')
	validation_group.add_argument('--val_weather_dir', type=str, default='./dataset/AWS/')
	validation_group.add_argument('--val_solar_dir', type=str, default='./dataset/photovoltaic/GWNU_C3/')

	# Flags for test only
	test_group = parser.add_argument_group('Flags for test only')
	test_group.add_argument('--load_path', type=str, default='')
	flags = parser.parse_args()

	# Additional per-mode validation
	try:
		if flags.mode == 'train':
			assert flags.save_dir, 'Must specify --save_dir'
		elif flags.mode == 'test':
			assert flags.load_path, 'Must specify --load_path'

	except AssertionError as e:
		print('\nError: ', e, '\n')
		parser.print_help()
		sys.exit(1)

	return flags


def train(hparams):
	trnset  = WPD(hparams['weather_list'], hparams['solar_list'], 678)
	valset  = WPD(hparams['val_weather_list'], hparams['val_solar_list'], 678)

	trnloader = DataLoader(trnset, batch_size=1, shuffle=False, drop_last=True)
	valloader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=True)

	input_dim  = hparams['input_dim']
	hidden_dim = hparams['nHidden']
	output_dim = hparams['output_dim']
	model = RNN(input_dim, hidden_dim, output_dim)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

	max_epoch = hparams['max_epoch']
	seqLeng = hparams['seqLeng']
	nBatch  = hparams['nBatch']
	prev_data = torch.zeros([seqLeng, input_dim])	# 14 is for featDim

	losses = []
	val_losses = []

	prev_loss = np.inf
	for epoch in range(max_epoch):
		model.train()
		loss = 0
		for i, (x, y) in enumerate(trnloader):
			x = x.squeeze()
			x = torch.cat((prev_data, x), axis=0)
			prev_data = x[-seqLeng:,:]
			y = y.squeeze()

			nLeng, nFeat = x.shape
			batch_data = []
			for j in range(nBatch):
				stridx = j*60
				endidx = j*60 + seqLeng
				batch_data.append(x[stridx:endidx,:].view(1,seqLeng, nFeat))
			batch_data = torch.cat(batch_data, dim=0)

			output = model(batch_data)
			loss += criterion(output.squeeze(), y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		model.eval()
		val_loss = 0
		for i, (x, y) in enumerate(valloader):
			x = x.squeeze()
			x = torch.cat((prev_data, x), axis=0)
			prev_data = x[-seqLeng:,:]
			y = y.squeeze()

			nLeng, nFeat = x.shape
			batch_data = []
			for j in range(nBatch):
				stridx = j*60
				endidx = j*60 + seqLeng
				batch_data.append(x[stridx:endidx,:].view(1,seqLeng, nFeat))
			batch_data = torch.cat(batch_data, dim=0)

			output = model(batch_data)
			val_loss += criterion(output.squeeze(), y)

		if val_loss < prev_loss:
			savePath = os.path.join(hparams['save_dir'], 'best_model')	# overwrite
			paramSet = model.state_dict()
			torch.save(paramSet, savePath)
			prev_loss = val_loss

		losses.append(loss.item())
		val_losses.append(val_loss.item())
		print(f'Epoch [{epoch+1}/{max_epoch}], Trn Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

	if hparams['save_losses']:
		trn_loss = np.array(losses)
		val_loss = np.array(val_losses)
		np.save('trn_loss.npy', trn_loss)
		np.save('val_loss.npy', val_loss)

	if hparams['loss_plot_flag']:
		plt.plot(range(max_epoch), losses, 'b', range(max_epoch), val_losses, 'r')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Training Loss')
		plt.show()


if __name__=='__main__':
	flags = parse_flags()
	hp    = hyper_params()
	
	if flags.mode == 'train':
		#=============================== training data list ====================================#
		# build photovoltaic data list
		solar_dir = os.listdir(flags.solar_dir)
		solar_list = []
		for folder in solar_dir:
			mlist = os.listdir(flags.solar_dir+'/'+folder)
			mlist = [file for file in mlist if file.find('xlsx') > 0]
			mlist = sorted(mlist, key=lambda x:int(x.split('.')[0]))
			for file in mlist:
				path = flags.solar_dir + '/' + folder + '/' + file
				solar_list.append(path)

		# find period
		first_ = solar_list[0].split('.')[1].split('/')
		first_year, first_month = first_[-2].split('_')
		first_day = str("%02d"%int(first_[-1]))
		first_date = first_year+first_month+first_day

		last_ = solar_list[-1].split('.')[1].split('/')
		last_year, last_month = last_[-2].split('_')
		last_day = str("%02d"%int(last_[-1]))
		last_date = last_year+last_month+last_day
		print('Training with data from %s to %s.'%(first_date, last_date))

		# build weather data list
		weather_dir = os.listdir(flags.weather_dir)
		weather_list = []
		stridx, endidx, cnt = -1, -1, -1
		for folder in weather_dir:
			wlist = os.listdir(flags.weather_dir+'/'+folder)
			wlist = [file for file in wlist if file.find('csv') > 0]
			wlist.sort()
			for file in wlist:
				path = flags.weather_dir + '/' + folder + '/' + file	
				weather_list.append(path)
				cnt += 1
				if path.find(first_date) > 0:
					stridx = cnt
				if path.find(last_date) > 0:
					endidx = cnt

		weather_list = weather_list[stridx:endidx+1]
		#========================================================================================#


		#=============================== validation data list ===================================#		
		# build photovoltaic data list
		solar_dir = os.listdir(flags.val_solar_dir)
		val_solar_list = []
		for folder in solar_dir:
			mlist = os.listdir(flags.val_solar_dir+'/'+folder)
			mlist = [file for file in mlist if file.find('xlsx') > 0]
			mlist = sorted(mlist, key=lambda x:int(x.split('.')[0]))
			for file in mlist:
				path = flags.val_solar_dir + '/' + folder + '/' + file
				val_solar_list.append(path)

		# find period
		first_ = val_solar_list[0].split('.')[1].split('/')
		first_year, first_month = first_[-2].split('_')
		first_day = str("%02d"%int(first_[-1]))
		first_date = first_year+first_month+first_day

		last_ = val_solar_list[-1].split('.')[1].split('/')
		last_year, last_month = last_[-2].split('_')
		last_day = str("%02d"%int(last_[-1]))
		last_date = last_year+last_month+last_day
		print('Validating with data from %s to %s.'%(first_date, last_date))

		# build weather data list
		weather_dir = os.listdir(flags.val_weather_dir)
		val_weather_list = []
		stridx = -1
		endidx = -1
		cnt = -1
		for folder in weather_dir:
			wlist = os.listdir(flags.val_weather_dir+'/'+folder)
			wlist = [file for file in wlist if file.find('csv') > 0]
			wlist.sort()
			for file in wlist:
				path = flags.val_weather_dir + '/' + folder + '/' + file
				val_weather_list.append(path)
				cnt += 1
				if path.find(first_date) > 0:
					stridx = cnt
				if path.find(last_date) > 0:
					endidx = cnt

		val_weather_list = val_weather_list[stridx:endidx+1]

		#========================================================================================#

		hp.update({"weather_list": weather_list})
		hp.update({"val_weather_list": val_weather_list})
		hp.update({"solar_list": solar_list})
		hp.update({"val_solar_list": val_solar_list})
		hp.update({"save_dir": flags.save_dir})

		if not os.path.isdir(flags.save_dir):
			os.makedirs(flags.save_dir)
		
		train(hp)
