import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils import data
import os
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen
import torch.nn.init as init
import gc
import sys
from PIL import Image
from datetime import datetime
import math

device = "cuda"
torch.set_default_tensor_type(torch.cuda.FloatTensor)

random.seed(datetime.now())

def gabor_patch(size, lambda_, theta, sigma, phase, trim=.005):
    # make linear ramp
    X0 = (np.linspace(1, size, size) / size) - .5
    # Set wavelength and phase
    freq = size / float(lambda_)
    phaseRad = phase * 2 * np.pi
    # Make 2D grating
    Xm, Ym = np.meshgrid(X0, X0)
    # Change orientation by adding Xm and Ym together in different proportions
    thetaRad = (theta / 360.) * 2 * np.pi
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    grating = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad)
    # 2D Gaussian distribution
    gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2))
    # Trim
    gauss[gauss < trim] = 0.005
    return grating * gauss

def add_noise_1(arr):
	mini = np.amin(arr)
	maxi = np.amax(arr)

	for i in range(arr.shape[0]):
		choice = random.choices([0, 1], [.6, .4])
		epsi = random.randint(-90,90)
		if(1 in choice):
			if(arr[i] == mini):
				arr[i] = min(maxi, maxi+epsi)

	for i in range(arr.shape[0]):
		choice = random.choices([0, 1], [.6, .4])
		epsi = random.randint(-90,90)
		if(1 in choice):
			if(arr[i] == maxi):
				arr[i] = max(mini, mini+epsi)
	return arr

def add_noise(arr):
	# mini = np.amin(arr)
	# maxi = np.amax(arr)

	for i in range(arr.shape[0]):
		epsi = int(np.random.normal(0.0, 50))
		arr[i] += epsi
		arr[i] = max(arr[i], 0)
		arr[i] = min(arr[i], 255)
	return arr

array_temp = np.empty((8, 64))
for i in range(8):
	choice = i
	if(choice == 0):
		direc = 0
	if(choice == 1):
		direc = 22
	if(choice == 2):
		direc = 45
	if(choice == 3):
		direc = 67
	if(choice == 4):
		direc = 90
	if(choice == 5):
		direc = 112
	if(choice == 6):
		direc = 135
	if(choice == 7):
		direc = 157

	gabor = gabor_patch(8, 2, direc, 150, .24)

	n = 8
	r = 4
	y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
	mask = x*x + y*y <= r*r
	array = np.zeros((n, n))
	array.fill(0.0)
	array[mask] = 1

	gabor = array*gabor

	for j in range(8):
		for k in range(8):
			if(gabor[j,k]>0.3):
				gabor[j,k] = 255
			else:
				gabor[j,k] = 0

	array_temp[i] = gabor.reshape(64)

def add_noise_2(arr):
	temp = array_temp
	for i in range(8):
		noise_coeff = np.random.uniform(0.01, 0.1)
		curr_level = random.randint(0,7)
		temp[i][np.where(arr == np.amax(arr))[0]] = 255 - (16*curr_level)
		temp[i][np.where(arr == np.amin(arr))[0]] = 16*curr_level		
		arr += noise_coeff * temp[i]
	for i in range(64):
		arr[i] = max(arr[i], 0)
		arr[i] = min(arr[i], 255)
	return arr

def generate_data_4outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y1 = np.zeros(seq_l)
	training_datapoint_y2 = np.zeros(seq_l)

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

	change_frame = sorted(change_frame)
	change_frame_contrast = sorted(change_frame_contrast)

	direc = 0
	choice = random.randint(0,7)
	if(choice == 0):
		direc = 0
	if(choice == 1):
		direc = 22
	if(choice == 2):
		direc = 45
	if(choice == 3):
		direc = 67
	if(choice == 4):
		direc = 90
	if(choice == 5):
		direc = 112
	if(choice == 6):
		direc = 135
	if(choice == 7):
		direc = 157

	curr_level = random.randint(0,7)
	previous_level = choice
	previous_level_contrast = curr_level

	gabor1 = gabor_patch(8, 2, direc, 150, .24)

	n = 8
	r = 4
	y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
	mask = x*x + y*y <= r*r
	array = np.zeros((n, n))
	array.fill(0.0)
	array[mask] = 1

	gabor1 = array*gabor1

	for j in range(8):
		for k in range(8):
			if(gabor1[j,k]>0.3):
				if(curr_level == 0):
					gabor1[j,k] = 255
				elif(curr_level == 1):
					gabor1[j,k] = 239
				elif(curr_level == 2):
					gabor1[j,k] = 223
				elif(curr_level == 3):
					gabor1[j,k] = 207
				elif(curr_level == 4):
					gabor1[j,k] = 191
				elif(curr_level == 5):
					gabor1[j,k] = 175
				elif(curr_level == 6):
					gabor1[j,k] = 159
				elif(curr_level == 7):
					gabor1[j,k] = 143
			else:
				if(curr_level == 0):
					gabor1[j,k] = 0
				elif(curr_level == 1):
					gabor1[j,k] = 16
				elif(curr_level == 2):
					gabor1[j,k] = 32
				elif(curr_level == 3):
					gabor1[j,k] = 48
				elif(curr_level == 4):
					gabor1[j,k] = 64
				elif(curr_level == 5):
					gabor1[j,k] = 80
				elif(curr_level == 6):
					gabor1[j,k] = 96
				elif(curr_level == 7):
					gabor1[j,k] = 112

	training_datapoint_x[:] = gabor1.reshape(64)
	training_datapoint_y1[:] = curr_level
	training_datapoint_y2[:] = choice

	for i in range(1,16):
		if i in change_frame:
			choice = random.randint(0,7)
			while(1):	
				if(choice == previous_level):
					choice = random.randint(0,7)
				else:
					break
			previous_level = choice

			direc = 0
			if(choice == 0):
				direc = 0
			if(choice == 1):
				direc = 22.5
			if(choice == 2):
				direc = 45
			if(choice == 3):
				direc = 67.5
			if(choice == 4):
				direc = 90
			if(choice == 5):
				direc = 112.5
			if(choice == 6):
				direc = 135
			if(choice == 7):
				direc = 157.5

			gabor1 = gabor_patch(8, 2, direc, 150, .24)
			n = 8
			r = 4
			y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
			mask = x*x + y*y <= r*r
			array = np.zeros((n, n))
			array.fill(0.0)
			array[mask] = 1

			gabor1 = array*gabor1

			for j in range(8):
				for k in range(8):
					if(gabor1[j,k]>0.3):
						if(curr_level == 0):
							gabor1[j,k] = 255
						elif(curr_level == 1):
							gabor1[j,k] = 239
						elif(curr_level == 2):
							gabor1[j,k] = 223
						elif(curr_level == 3):
							gabor1[j,k] = 207
						elif(curr_level == 4):
							gabor1[j,k] = 191
						elif(curr_level == 5):
							gabor1[j,k] = 175
						elif(curr_level == 6):
							gabor1[j,k] = 159
						elif(curr_level == 7):
							gabor1[j,k] = 143
					else:
						if(curr_level == 0):
							gabor1[j,k] = 0
						elif(curr_level == 1):
							gabor1[j,k] = 16
						elif(curr_level == 2):
							gabor1[j,k] = 32
						elif(curr_level == 3):
							gabor1[j,k] = 48
						elif(curr_level == 4):
							gabor1[j,k] = 64
						elif(curr_level == 5):
							gabor1[j,k] = 80
						elif(curr_level == 6):
							gabor1[j,k] = 96
						elif(curr_level == 7):
							gabor1[j,k] = 112

			training_datapoint_x[i:] = gabor1.reshape(64)
			training_datapoint_y2[i:] = choice

		if i in change_frame_contrast:
			curr_level = random.randint(0,7)
			while(1):	
				if(curr_level == previous_level_contrast):
					curr_level = random.randint(0,7)
				else:
					break
			previous_level_contrast = curr_level

			gabor1 = gabor_patch(8, 2, direc, 150, .24)
			n = 8
			r = 4
			y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
			mask = x*x + y*y <= r*r
			array = np.zeros((n, n))
			array.fill(0.0)
			array[mask] = 1

			gabor1 = array*gabor1

			for j in range(8):
				for k in range(8):
					if(gabor1[j,k]>0.3):
						if(curr_level == 0):
							gabor1[j,k] = 255
						elif(curr_level == 1):
							gabor1[j,k] = 239
						elif(curr_level == 2):
							gabor1[j,k] = 223
						elif(curr_level == 3):
							gabor1[j,k] = 207
						elif(curr_level == 4):
							gabor1[j,k] = 191
						elif(curr_level == 5):
							gabor1[j,k] = 175
						elif(curr_level == 6):
							gabor1[j,k] = 159
						elif(curr_level == 7):
							gabor1[j,k] = 143
					else:
						if(curr_level == 0):
							gabor1[j,k] = 0
						elif(curr_level == 1):
							gabor1[j,k] = 16
						elif(curr_level == 2):
							gabor1[j,k] = 32
						elif(curr_level == 3):
							gabor1[j,k] = 48
						elif(curr_level == 4):
							gabor1[j,k] = 64
						elif(curr_level == 5):
							gabor1[j,k] = 80
						elif(curr_level == 6):
							gabor1[j,k] = 96
						elif(curr_level == 7):
							gabor1[j,k] = 112

			training_datapoint_x[i:] = gabor1.reshape(64)
			training_datapoint_y1[i:] = curr_level

	for i in range(16):
		training_datapoint_x[i] = add_noise(training_datapoint_x[i])
	
	for i in range(16):
		training_datapoint_x[i] = add_noise_2(training_datapoint_x[i])

	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))

	training_datapoint_y1 = training_datapoint_y1.astype(int)
	training_datapoint_y2 = training_datapoint_y2.astype(int)

	return [training_datapoint_x, training_datapoint_y1], [training_datapoint_x, training_datapoint_y2]

# a = generate_data_4outputs()
# for i in range(16):
# 	plt.imshow(a[0][0][i].reshape(8,8))
# 	plt.show()

def generate_data_change():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y1 = np.zeros(seq_l)
	training_datapoint_y2 = np.zeros(seq_l)

	number_of_changes = 2
	change_frame = []
	change_frame.append(random.randint(2,7))
	change_frame.append(random.randint((change_frame[0]+4), 13))

	number_of_changes = 2
	change_frame_contrast = []
	change_frame_contrast.append(random.randint(2,7))
	change_frame_contrast.append(random.randint((change_frame_contrast[0]+4), 13))

	direc = 0
	choice = random.randint(0,7)
	if(choice == 0):
		direc = 0
	if(choice == 1):
		direc = 22.5
	if(choice == 2):
		direc = 45
	if(choice == 3):
		direc = 67.5
	if(choice == 4):
		direc = 90
	if(choice == 5):
		direc = 112.5
	if(choice == 6):
		direc = 135
	if(choice == 7):
		direc = 157.5

	curr_level = random.randint(0,7)
	previous_level = choice
	previous_level_contrast = curr_level

	gabor1 = gabor_patch(8, 2, direc, 150, .24)

	n = 8
	r = 4
	y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
	mask = x*x + y*y <= r*r
	array = np.zeros((n, n))
	array.fill(0.0)
	array[mask] = 1

	gabor1 = array*gabor1

	for j in range(8):
		for k in range(8):
			if(gabor1[j,k]>0.3):
				if(curr_level == 0):
					gabor1[j,k] = 255
				elif(curr_level == 1):
					gabor1[j,k] = 239
				elif(curr_level == 2):
					gabor1[j,k] = 223
				elif(curr_level == 3):
					gabor1[j,k] = 207
				elif(curr_level == 4):
					gabor1[j,k] = 191
				elif(curr_level == 5):
					gabor1[j,k] = 175
				elif(curr_level == 6):
					gabor1[j,k] = 159
				elif(curr_level == 7):
					gabor1[j,k] = 143
			else:
				if(curr_level == 0):
					gabor1[j,k] = 0
				elif(curr_level == 1):
					gabor1[j,k] = 16
				elif(curr_level == 2):
					gabor1[j,k] = 32
				elif(curr_level == 3):
					gabor1[j,k] = 48
				elif(curr_level == 4):
					gabor1[j,k] = 64
				elif(curr_level == 5):
					gabor1[j,k] = 80
				elif(curr_level == 6):
					gabor1[j,k] = 96
				elif(curr_level == 7):
					gabor1[j,k] = 112

	training_datapoint_x[:] = gabor1.reshape(64)
	training_datapoint_y1[:] = 0
	training_datapoint_y2[:] = 0

	for i in range(1,16):
		if i in change_frame:
			choice = random.randint(0,7)
			while(1):	
				if(choice == previous_level):
					choice = random.randint(0,7)
				else:
					break
			previous_level = choice

			direc = 0
			if(choice == 0):
				direc = 0
			if(choice == 1):
				direc = 22.5
			if(choice == 2):
				direc = 45
			if(choice == 3):
				direc = 67.5
			if(choice == 4):
				direc = 90
			if(choice == 5):
				direc = 112.5
			if(choice == 6):
				direc = 135
			if(choice == 7):
				direc = 157.5

			gabor1 = gabor_patch(8, 2, direc, 150, .24)
			n = 8
			r = 4
			y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
			mask = x*x + y*y <= r*r
			array = np.zeros((n, n))
			array.fill(0.0)
			array[mask] = 1

			gabor1 = array*gabor1

			for j in range(8):
				for k in range(8):
					if(gabor1[j,k]>0.3):
						if(curr_level == 0):
							gabor1[j,k] = 255
						elif(curr_level == 1):
							gabor1[j,k] = 239
						elif(curr_level == 2):
							gabor1[j,k] = 223
						elif(curr_level == 3):
							gabor1[j,k] = 207
						elif(curr_level == 4):
							gabor1[j,k] = 191
						elif(curr_level == 5):
							gabor1[j,k] = 175
						elif(curr_level == 6):
							gabor1[j,k] = 159
						elif(curr_level == 7):
							gabor1[j,k] = 143
					else:
						if(curr_level == 0):
							gabor1[j,k] = 0
						elif(curr_level == 1):
							gabor1[j,k] = 16
						elif(curr_level == 2):
							gabor1[j,k] = 32
						elif(curr_level == 3):
							gabor1[j,k] = 48
						elif(curr_level == 4):
							gabor1[j,k] = 64
						elif(curr_level == 5):
							gabor1[j,k] = 80
						elif(curr_level == 6):
							gabor1[j,k] = 96
						elif(curr_level == 7):
							gabor1[j,k] = 112

			training_datapoint_x[i:] = gabor1.reshape(64)
			training_datapoint_y2[i+1] = 1

		if i in change_frame_contrast:
			curr_level = random.randint(0,7)
			while(1):
				if(curr_level == previous_level_contrast):
					curr_level = random.randint(0,7)
				else:
					break
			previous_level_contrast = curr_level

			gabor1 = gabor_patch(8, 2, direc, 150, .24)
			n = 8
			r = 4
			y,x = np.ogrid[-n/2:n/2, -n/2:n/2]
			mask = x*x + y*y <= r*r
			array = np.zeros((n, n))
			array.fill(0.0)
			array[mask] = 1

			gabor1 = array*gabor1

			for j in range(8):
				for k in range(8):
					if(gabor1[j,k]>0.3):
						if(curr_level == 0):
							gabor1[j,k] = 255
						elif(curr_level == 1):
							gabor1[j,k] = 239
						elif(curr_level == 2):
							gabor1[j,k] = 223
						elif(curr_level == 3):
							gabor1[j,k] = 207
						elif(curr_level == 4):
							gabor1[j,k] = 191
						elif(curr_level == 5):
							gabor1[j,k] = 175
						elif(curr_level == 6):
							gabor1[j,k] = 159
						elif(curr_level == 7):
							gabor1[j,k] = 143
					else:
						if(curr_level == 0):
							gabor1[j,k] = 0
						elif(curr_level == 1):
							gabor1[j,k] = 16
						elif(curr_level == 2):
							gabor1[j,k] = 32
						elif(curr_level == 3):
							gabor1[j,k] = 48
						elif(curr_level == 4):
							gabor1[j,k] = 64
						elif(curr_level == 5):
							gabor1[j,k] = 80
						elif(curr_level == 6):
							gabor1[j,k] = 96
						elif(curr_level == 7):
							gabor1[j,k] = 112

			training_datapoint_x[i:] = gabor1.reshape(64)
			training_datapoint_y1[i+1] = 1

	# for i in range(16):
	# 	training_datapoint_x[i] = add_noise(training_datapoint_x[i])

	for i in range(16):
		training_datapoint_x[i] = add_noise_2(training_datapoint_x[i])
			
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))

	training_datapoint_y1 = training_datapoint_y1.astype(int)
	training_datapoint_y2 = training_datapoint_y2.astype(int)

	return [training_datapoint_x, training_datapoint_y1], [training_datapoint_x, training_datapoint_y2]

# a = generate_data_change()
# for i in range(16):
# 	plt.imshow(a[0][0][i].reshape(8,8))
# 	plt.show()

class RNN_arch_2_1(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn = nn.RNN(64, 256, num_layers =1, bias = False, nonlinearity='tanh')
		self.dropout = nn.Dropout(p=.2)
		self.fc1 = nn.Linear(256, 64)
		self.tanh = nn.Tanh()
		self.fc = nn.Linear(64,8)

	def forward(self, x, hc):
		out, hc = self.rnn(x, hc)
		r_output = out.contiguous().view(-1, 256)
		out1 = self.dropout(r_output)
		out2 = self.fc1(out1)
		out2 = self.tanh(out2)
		out3 = self.fc(out2)

		return out3, hc, out2

class RNN_arch_2_final(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size

		self.rnn1_i2h = nn.Linear(64, 256)
		self.rnn1_h2h = nn.Linear(256,256)
		self.tanh1 = nn.Tanh()
		self.feedback1 = nn.Linear(64,256)
		self.tanh2 = nn.Tanh()
		self.rnn1_h2o = nn.Linear(256, 64)
		self.tanh3 = nn.Tanh()

		self.rnn2_i2h = nn.Linear(64, 256)
		self.rnn2_h2h = nn.Linear(256,256)
		self.tanh4 = nn.Tanh()
		self.feedback2 = nn.Linear(64,256)
		self.tanh5 = nn.Tanh()
		self.rnn2_h2o = nn.Linear(256, 64)
		self.tanh6 = nn.Tanh()

		self.rnn3_i2h = nn.Linear(130, 256)
		self.rnn3_h2h = nn.Linear(256,256)
		self.tanh7 = nn.Tanh()
		self.rnn3_h2o = nn.Linear(256, 64)
		self.tanh8 = nn.ReLU()
		self.fc = nn.Linear(64, 2)

	def forward(self, x, cue, hc1, hc2, hc3, hc4):
		output_seq = torch.empty((16, self.batch_size, 2))
		output_seq1 = torch.empty((16, self.batch_size, 64))
		output_seq2 = torch.empty((16, self.batch_size, 64))
		feed1_arr = torch.empty((16, self.batch_size, 256))
		feed2_arr = torch.empty((16, self.batch_size, 256))
		out_f_arr = torch.empty((16, self.batch_size,64))

		cue_arr = torch.empty(16, self.batch_size, 2)
		for i in range(16):
			cue_arr[i, :, 0] = cue*10
			cue_arr[i, :, 1] = 10*abs(cue - 1)
		
			i2h = self.rnn1_i2h(x[i])
			h2h = self.rnn1_h2h(hc1)
			feed = self.feedback1(hc4)
			feed1_arr[i] = feed
			# feed = self.tanh1(feed)
			hc1 = self.tanh2(i2h + h2h + feed)
			out = self.rnn1_h2o(hc1)
			out = self.tanh3(out)
			output_seq1[i] = out

			i2h = self.rnn2_i2h(x[i])
			h2h = self.rnn2_h2h(hc2)
			feed = self.feedback2(hc4)
			feed2_arr[i] = feed
			# feed = self.tanh4(feed)
			hc2 = self.tanh5(i2h + h2h + feed)
			out = self.rnn2_h2o(hc2)
			out = self.tanh6(out)
			output_seq2[i] = out

			mid_inp = torch.cat((cue_arr, output_seq1, output_seq2), 2)

			i2h = self.rnn3_i2h(mid_inp[i])
			h2h = self.rnn3_h2h(hc3)
			hc3 = self.tanh7(i2h + h2h)
			out = self.rnn3_h2o(hc3)
			hc4 = out
			out = self.tanh8(out)
			out_f_arr[i] = out
			out = self.fc(out)
			output_seq[i] = out

		return output_seq, hc1, hc2, hc3, hc4, output_seq1, output_seq2, cue_arr, feed1_arr, feed2_arr, out_f_arr

def weight_init(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight.data)
		init.constant_(m.bias.data, 0.0)

def init_weights(model):
	for m in model.modules():
		if type(m) in [nn.RNN]:
			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					# torch.nn.init.constant_(param.data, 1.0)
					init.xavier_uniform_(param.data)
				elif 'weight_hh' in name:
					# torch.nn.init.constant_(param.data, 0.0)
					init.xavier_uniform_(param.data)
				elif 'bias' in name:
					param.data.fill_(0)

source = ColumnDataSource(data={"epochs": [], "trainlosses": [], "vallosses": [] })

plot = figure()
plot.line(x= "epochs", y="trainlosses", color="green", alpha=0.8, legend="Train loss", line_width=2, source=source)
plot.line(x= "epochs", y="vallosses", color="red", alpha=0.8, legend="Val loss", line_width=2, source=source)

doc = curdoc()
doc.add_root(plot)

@gen.coroutine
def update(new_data):
    source.stream(new_data)

def predict_new():
	seq_l = 16
	
	model = RNN_arch_2_1(1)
	for param in model.parameters():
		param.requires_grad = False
	model.load_state_dict(torch.load("changed_contrast_new_noise.pt", map_location = device))
	
	model.eval()

	hc1 = torch.zeros(1, 1, 256).cuda()
	hc2 = torch.zeros(1, 1, 256).cuda()
	hc3 = torch.zeros(1, 1, 256).cuda()
	hc4 = torch.zeros(1, 1, 64).cuda()
	
	total = 0
	for i in range(2000):
		temp = generate_data_4outputs()
		data = temp[0]

		x = torch.tensor(data[0])
		y = torch.tensor(data[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		y = y.contiguous().view((1,seq_l)).cuda()

		# output_seq, hc1, hc2, hc3, hc4, output_seq1, output_seq2, cue_arr, feed1, feed2, out_f

		# output_seq, _, _, _, _, output_seq1, output_seq2, cue_arr, feed1, feed2, out_f = model(x.float(), 0, hc1.float(), hc2.float(), hc3.float(), hc4.float())
		output_seq, _, _ = model(x.float(), hc1.float())
		correct = 0
		# print(output_seq)
		# print(y)
		for i in range(1,16):
			output_seq[i] = torch.nn.functional.softmax(output_seq[i])
			# print(y[0][i])
			# print(output_seq[i])
			values, indices = output_seq[i].max(0)
			if(indices == y[0][i]):
				correct += 1

		accuracy = (correct/15)*100
		total += accuracy

	print(total/2000)

predict_new()


# contrast_difficult_temp_8outputs_40noise
# motion_difficult_temp_8outputs

def train():
	# model1 = RNN_arch_2_1(200)
	# model1.load_state_dict(torch.load("contrast_difficult_temp_8outputs_40noise.pt", map_location = device))	
	# model1.eval()

	# model2 = RNN_arch_2_1(200)
	# model2.load_state_dict(torch.load("motion_difficult_temp_8outputs.pt", map_location = device))	
	# model2.eval()

	# model = RNN_arch_2_final(200)
	# model.apply(weight_init)
	# model.cuda()

	# for m in model1.modules():
	# 	if type(m) in [nn.RNN]:
	# 		for name, param in m.named_parameters():
	# 			if 'weight_ih' in name:
	# 				model.rnn1_i2h.weight.data = param.data
	# 			elif 'weight_hh' in name:
	# 				model.rnn1_h2h.weight.data = param.data
	# model.rnn1_i2h.bias.data[:] = 0.0
	# model.rnn1_h2h.bias.data[:] = 0.0
	# model.rnn1_h2o.weight.data = model1.fc1.weight.data
	# model.rnn1_h2o.bias.data = model1.fc1.bias.data
	# model.rnn1_i2h.bias.requires_grad = False
	# model.rnn1_h2h.bias.requires_grad = False
	# model.rnn1_i2h.weight.requires_grad = False
	# model.rnn1_h2h.weight.requires_grad = False
	# model.rnn1_h2o.weight.requires_grad = False
	# model.rnn1_h2o.bias.requires_grad = False

	# for m in model2.modules():
	# 	if type(m) in [nn.RNN]:
	# 		for name, param in m.named_parameters():
	# 			if 'weight_ih' in name:
	# 				model.rnn2_i2h.weight.data = param.data
	# 			elif 'weight_hh' in name:
	# 				model.rnn2_h2h.weight.data = param.data
	# model.rnn2_i2h.bias.data[:] = 0.0
	# model.rnn2_h2h.bias.data[:] = 0.0
	# model.rnn2_h2o.weight.data = model1.fc1.weight.data
	# model.rnn2_h2o.bias.data = model1.fc1.bias.data
	# model.rnn2_i2h.bias.requires_grad = False
	# model.rnn2_h2h.bias.requires_grad = False
	# model.rnn2_i2h.weight.requires_grad = False
	# model.rnn2_h2h.weight.requires_grad = False
	# model.rnn2_h2o.weight.requires_grad = False
	# model.rnn2_h2o.bias.requires_grad = False

	model = RNN_arch_2_1(200)
	model.apply(weight_init)
	model.apply(init_weights)
	model.load_state_dict(torch.load("contrast_difficult_temp_8outputs.pt", map_location = device))
	model.cuda()

	criterion = nn.CrossEntropyLoss().cuda()

	a = []
	for i in range(25000):
		temp = generate_data_4outputs()
		a.append(temp[0])

	b = []
	for i in range(5000):
		temp = generate_data_4outputs()
		b.append(temp[0])

	# c = []
	# for i in range(10000):
	# 	temp = generate_data_change()
	# 	c.append(temp[0])

	# d = []
	# for i in range(10000):
	# 	temp = generate_data_change()
	# 	d.append(temp[1])

	count0 = 0

	while(count0<2000):
		if(count0 < 70):
			optimizer = optim.Adam(model.parameters(), lr = 0.0005)
		elif(count0 < 140):
			optimizer = optim.Adam(model.parameters(), lr = 0.0003)
		elif(count0 < 210):
			optimizer = optim.Adam(model.parameters(), lr = 0.0001)
		else:
			optimizer = optim.Adam(model.parameters(), lr = 0.00005)

		# flag = 0
		# if( int(count0/6) % 2 == 0):
		# 	cue = 0
		# else:
		# 	cue = 1

		# # cue = 0

		# if(cue == 0):
		# 	train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
		# 	valid_loader = data.DataLoader(c, batch_size=200, drop_last = True)

		# if(cue == 1):
		# 	train_loader = data.DataLoader(b, batch_size=200, shuffle = True, drop_last = True)
		# 	valid_loader = data.DataLoader(d, batch_size=200, drop_last = True)

		train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
		valid_loader = data.DataLoader(b, batch_size=200, drop_last = True)

		train_loss = 0.0
		valid_loss = 0.0

		count0 += 1
		count = 1
		model.train()

		hc1 = torch.zeros(1, 200, 256).cuda()
		hc2 = torch.zeros(1, 200, 256).cuda()
		hc3 = torch.zeros(1, 200, 256).cuda()
		hc4 = torch.zeros(1, 200, 64).cuda()

		for x,y in train_loader:
			x = x.permute(1,0,2)
			y = y.permute(1,0)
			x, y = x.cuda(), y.cuda()
			y_reshaped = y.contiguous().view(3200).cuda()

			print("Epoch: ", count0, "Step: ", count)
			count += 1

			optimizer.zero_grad()
			# z, _, _, _, _, _, _, _, _, _, _ = model(x.float(), cue, hc1.float(), hc2.float(), hc3.float(), hc4.float())
			z, _, _ = model(x.float(), hc1.float())
			z = z.contiguous().view(3200,-1)
			
			cross_entropy_loss = criterion(z, y_reshaped)
			# all_linear1_params = torch.cat([x.view(-1) for x in model.feedback1.parameters()])
			# all_linear2_params = torch.cat([x.view(-1) for x in model.feedback2.parameters()])
			# all_linear3_params = torch.cat([x.view(-1) for x in model.rnn3_i2h.parameters()])

			# l2_regularization_1 = 0.0
			# l2_regularization_2 = 0.0
			# # l2_regularization_3 = 0.0
			# if(cue == 1):
			# 	l2_regularization_1 = .008 * torch.norm(all_linear1_params, 2)
			# 	l2_regularization_2 = .008 * torch.norm(all_linear2_params, 2)
			# 	# l2_regularization_3 = .006 * torch.norm(all_linear3_params, 2)
			# else:
			# 	l2_regularization_1 = .006 * torch.norm(all_linear1_params, 2)
			# 	l2_regularization_2 = .006 * torch.norm(all_linear2_params, 2)
			# 	l2_regularization_3 = .006 * torch.norm(all_linear3_params, 2)

			loss = cross_entropy_loss
			# loss = cross_entropy_loss + l2_regularization_1 + l2_regularization_2 + l2_regularization_3
			loss.backward()
			
			optimizer.step()
			train_loss = loss.item()


		hc1 = torch.zeros(1, 200, 256).cuda()
		hc2 = torch.zeros(1, 200, 256).cuda()
		hc3 = torch.zeros(1, 200, 256).cuda()
		hc4 = torch.zeros(1, 200, 64).cuda()
		
		model.eval()
		for val_x, val_y in valid_loader:
			val_x = val_x.permute(1,0,2)
			val_y = val_y.permute(1,0)
			val_x, val_y = val_x.cuda(), val_y.cuda()
			val_y_reshaped = val_y.contiguous().view(3200).cuda()

			# val_output, _, _, _, _, _, _, _, _, _, _ = model(val_x.float(), cue, hc1.float(), hc2.float(), hc3.float(), hc4.float())
			val_output, _, _ = model(x.float(), hc1.float())
			val_output = val_output.contiguous().view(3200, -1)
			valid_loss = criterion(val_output, val_y_reshaped)
			valid_loss = valid_loss.item()

		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
		doc.add_next_tick_callback(partial(update, new_data))

		torch.save(model.state_dict(), "changed_contrast_new_noise.pt")

# thread = Thread(target=train)
# thread.start()