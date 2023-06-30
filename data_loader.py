import numpy as np
import pandas as pd
import csv
import os

import torch
from torch.utils.data import Dataset

class WPD(Dataset):
	def __init__(self, weather_list, energy_list, region_ID, datapath='./dataset/'):
		self.wlist = weather_list	# all files for weather info
		self.elist = energy_list	# all files for power gener.
		self.rID   = region_ID
		#print(f'wlist : {len(self.wlist)} elist : {len(self.elist)}')
		# when test , wlist : 272, elist : 243

		if not os.path.isdir(datapath):
			os.makedirs(datapath)
		
	def __len__(self):
		return min(len(self.wlist),len(self.elist))

	def __getitem__(self, idx):
		wfilepath = self.wlist[idx]
		efilepath = self.elist[idx]
		
		wfile_npy = wfilepath.replace(".csv", ".npy")
		if os.path.isfile(wfile_npy):
			weather_data = np.load(wfile_npy)
		else:
			############## weather data loading  #################
			# Loading: weather data for all regions written by chy
			csv = pd.read_csv(wfilepath, encoding='CP949')
			csv = csv.drop(['지점명'], axis=1)
			groups = csv.groupby(csv.columns[0])
			weather_by_region = {}
			for i in groups:
				if weather_by_region.get(i[0]) is not None:
					weather_by_region[i[0]].append(list(i))
				else:
					weather_by_region[i[0]] = list(i)

			# Choose region & Time alignment
			rid = self.rID
			region_data = weather_by_region[rid]
			region_data = region_data[1].values
			weather_data= np.zeros([1440, 15])	# hard coding for 1 day, 14 features & time
			timeflag    = np.ones(1440)
			for i in range(len(region_data)):
				timestamp = region_data[i][1]
				date_, time_ = timestamp.split(' ')
				data = region_data[i][2:].astype(float)
				data = np.nan_to_num(data, nan=0)

				hh = int(time_[:2])
				mm = int(time_[-2:])
				idx = hh*60+mm - 1

				weather_data[idx,0] = idx
				weather_data[idx,1:] = data
				timeflag[idx] = 0

			# interpolation for missing data
			idx = np.where(timeflag==1)[0]
			indices, temp = [], []
			if len(idx) == 1:
				indices.append(idx)
			else:
				diff = np.diff(idx)
				for i in range(len(diff)):
					temp.append(idx[i].tolist())
					if diff[i] == 1:
						temp.append(idx[i+1])
					else:
						indices.append(np.unique(temp).tolist())
						temp = []
				if len(temp) > 0:	# add the last block
					indices.append(np.unique(temp).tolist())
					temp = []

			for n in range(len(indices)):
				idx = indices[n]
				maxV, minV = np.max(idx).astype(int), np.min(idx).astype(int)
				if minV > 0:
					prev = weather_data[minV-1,:]
				else:
					prev = None
				if maxV < 1439:
					post = weather_data[maxV+1,:]
				else:
					post = prev
				if prev is None:
					prev = post
						
				nsteps = len(idx)
				for i in range(nsteps):
					weather_data[i+minV] = (nsteps-i)*prev/(nsteps+1) + (i+1)*post/(nsteps+1)
			np.save(wfile_npy, weather_data)

		weather_data = torch.tensor(weather_data)

		efile_npy = efilepath.replace(".xlsx", ".npy")
		if os.path.isfile(efile_npy):
			power_data = np.load(efile_npy)
		else:
			############## Photovoltaic data loading  #################
			# Loading: power generation data written by chy
			xlsx = pd.read_excel(efilepath, engine='openpyxl', skiprows=range(3))
			xlsx = xlsx.iloc[:-1,:]	# row remove
			power = xlsx.to_numpy()
			power = pd.DataFrame(power, columns=['Datetime', 'Power'])
			power_data = power.to_numpy()
			power_data = power_data[:,1].astype(float)

			np.save(efile_npy, power_data)

		power_data = torch.tensor(power_data)

		return weather_data, power_data