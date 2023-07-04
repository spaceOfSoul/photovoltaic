import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
import sys
import torch
import argparse

from model import RNN
from data_loader import WPD
from torch.utils.data import DataLoader

def hyper_params():
	# Default setting
	model_params = {
		'seqLeng': 30,
		'input_dim': 15,
		'nHidden': 64,
		'output_dim': 1,
	}

	learning_params = {
		'nBatch': 24,
		'lr'    : 1.0e-3,
		'max_epoch': 10,
	}

	hparams = {
		'model' : model_params,
		'learning' : learning_params,

		# system flags
		'loss_plot_flag': True,
		'save_losses': True,
		'save_result': True,
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
	training_group.add_argument('--loc_ID', type=int, default=678)

	# Flags for validation only
	validation_group = parser.add_argument_group('Flags for validation only')
	validation_group.add_argument('--val_weather_dir', type=str, default='./dataset/AWS/')
	validation_group.add_argument('--val_solar_dir', type=str, default='./dataset/photovoltaic/GWNU_C3/')

	# Flags for test only
	test_group = parser.add_argument_group('Flags for test only')
	test_group.add_argument('--load_path', type=str, default='')
	test_group.add_argument('--test_weather_dir', type=str, default='')
	test_group.add_argument('--test_solar_dir', type=str, default='')
	test_group.add_argument('--test_loc_ID', type=int, default=876)
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
	model_params = hparams['model']
	learning_params = hparams['learning']

	trnset  = WPD(hparams['weather_list'], hparams['solar_list'], hparams['loc_ID'])
	valset  = WPD(hparams['val_weather_list'], hparams['val_solar_list'], hparams['loc_ID'])

	trnloader = DataLoader(trnset, batch_size=1, shuffle=False, drop_last=True)
	valloader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=True)

	input_dim  = model_params['input_dim']
	hidden_dim = model_params['nHidden']
	output_dim = model_params['output_dim']
	model = RNN(input_dim, hidden_dim, output_dim)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_params['lr'])

	max_epoch = learning_params['max_epoch']
	seqLeng = model_params['seqLeng']
	nBatch  = learning_params['nBatch']
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
			model_dict = {
				'kwargs'   : model_params,
				'paramSet' : model.state_dict()
			}
			torch.save(model_dict, savePath)
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


def test(hparams):
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

	model_conf = ckpt['kwargs']
	paramSet = ckpt['paramSet']

	input_dim  = model_conf['input_dim']
	hidden_dim = model_conf['nHidden']
	output_dim = model_conf['output_dim']
	model = RNN(input_dim, hidden_dim, output_dim)
	model.load_state_dict(paramSet)
	model.eval()

	tstset  = WPD(hparams['weather_list'], hparams['solar_list'], hparams['loc_ID'])
	tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)

	seqLeng = model_params['seqLeng']
	nBatch  = learning_params['nBatch']
	prev_data = torch.zeros([seqLeng, input_dim])	# 14 is for featDim
#
	criterion = torch.nn.MSELoss()
	loss = 0
	result = []
	print(len(tstloader))

	for i, (x, y) in enumerate(tstloader):
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
		result.append(output.detach().numpy())
		#print(type(output.squeeze()))
		#print(type(y))
		#print('???')

		loss += criterion(output.squeeze(), y)
	
	print(f'Tsn Loss: {loss.item():.4f}')

	if hparams['save_result']:
		result_npy = np.array(result)
		np.save('prediction.npy', result_npy)

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
		weather_dir.sort(key=lambda x: int(x[:-1]) if x.endswith('월') else int(x))

		# print(weather_dir)
		# weather_dir.sort()
		weather_list = []
		stridx, endidx, cnt = -1, -1, -1
		for folder in weather_dir:
			wlist = os.listdir(flags.weather_dir+'/'+folder)
			# print(wlist)
			wlist = [file for file in wlist if file.find('csv') > 0]
			wlist.sort()
			# print(len(wlist))
			for file in wlist:
				path = flags.weather_dir + '/' + folder + '/' + file	
				weather_list.append(path)
				cnt += 1
				if path.find(first_date) > 0:
					stridx = cnt
					print('stridx', stridx)
				if path.find(last_date) > 0:
					endidx = cnt
					print('endidx', endidx)
		# print(len(weather_list))
		weather_list = weather_list[stridx:endidx+1]
		#========================================================================================#
		# print(len(weather_list))

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
		weather_dir.sort(key=lambda x: int(x[:-1]) if x.endswith('월') else int(x))
		# print(weather_dir)
		val_weather_list = []
		stridx, endidx, cnt = -1, -1, -1
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
		hp.update({"loc_ID": flags.loc_ID})

		if not os.path.isdir(flags.save_dir):
			os.makedirs(flags.save_dir)
		
		train(hp)

	elif flags.mode == 'test':
		hp.update({"load_path": flags.load_path})
		hp.update({"loc_ID": flags.test_loc_ID})

		#=============================== test data list ====================================#
		# build photovoltaic data list
		solar_dir = os.listdir(flags.test_solar_dir)
		solar_list = []
		for folder in solar_dir:
			mlist = os.listdir(flags.test_solar_dir+'/'+folder)
			mlist = [file for file in mlist if file.find('xlsx') > 0]
			mlist = sorted(mlist, key=lambda x:int(x.split('.')[0]))
			for file in mlist:
				path = flags.test_solar_dir + '/' + folder + '/' + file
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
		print('Test with data from %s to %s.'%(first_date, last_date))

		# build weather data list
		weather_dir = os.listdir(flags.test_weather_dir)
		weather_dir.sort(key=lambda x: int(x[:-1]) if x.endswith('월') else int(x))
		weather_list = []
	
		stridx, endidx, cnt = -1, -1, -1
		for folder in weather_dir:
			wlist = os.listdir(flags.test_weather_dir+'/'+folder)
			wlist = [file for file in wlist if file.find('csv') > 0]
			wlist.sort()
			for file in wlist:
				path = flags.test_weather_dir + '/' + folder + '/' + file	
				weather_list.append(path)
				cnt += 1
				if path.find(first_date) > 0:
					stridx = cnt
				if path.find(last_date) > 0:
					endidx = cnt

		weather_list = weather_list[stridx:endidx+1]
		#========================================================================================#

		hp.update({"weather_list": weather_list})
		hp.update({"solar_list": solar_list})

		test(hp)