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

device = "cuda"

random.seed(42)

def move_left(arr):
	a = []
	for i in range(8):
		a.extend(arr[((i*8)+1):((i+1)*8)])
		a.extend(arr[(i*8):((i*8)+1)])
	return np.array(a)

def move_right(arr):
	a = []
	for i in range(8):
		a.extend(arr[(((i+1)*8)-1):((i+1)*8)])
		a.extend(arr[(i*8):(((i+1)*8)-1)])
	return np.array(a)

def move_up(arr):
	a = []
	a.extend(arr[8:])
	a.extend(arr[:8])
	return np.array(a)

def move_down(arr):
	a = []
	a.extend(arr[-8:])
	a.extend(arr[:-8])
	return np.array(a)

def move_nw(arr):
	arr = move_up(arr)
	arr = move_left(arr)
	return arr

def move_ne(arr):
	arr = move_up(arr)
	arr = move_right(arr)
	return arr

def move_sw(arr):
	arr = move_down(arr)
	arr = move_left(arr)
	return arr

def move_se(arr):
	arr = move_down(arr)
	arr = move_right(arr)
	return arr

def move(arr, choice = 10, previous_direction = 20):
	# if(choice == 10):
	choice = random.randint(0,6)
		# while(1):
		# 	if(choice == previous_direction):
		# 		choice = random.randint(0,7)
		# 	else:
		# 		break
	if(choice == 0):
		arr = move_up(arr)
	elif(choice == 1):
		arr = move_down(arr)
	elif(choice == 2):
		arr = move_left(arr)
	elif(choice == 3):
		arr = move_right(arr)
	elif (choice == 4):
		arr = move_ne(arr)
	elif(choice == 5):
		arr = move_nw(arr)
	elif(choice == 6):
		arr = move_se(arr)
	elif (choice == 7):
		arr = move_sw(arr)

	return (choice, arr)

def change_contrast1(arr, choice = 10, previous_level = 20):
	if(choice == 10):
		choice = random.randint(0,7)
		while(1):	
			if(choice == previous_level):
				choice = random.randint(0,7)
			else:
				break
	if(choice == 0):
		diff_from_mean = 127
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)
	elif(choice == 1):
		diff_from_mean = 127 - 15*1
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 2):
		diff_from_mean = 127 - 15*2
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 3):
		diff_from_mean = 127 - 15*3
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif (choice == 4):
		diff_from_mean = 127 - 15*4
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 5):
		diff_from_mean = 127 - 15*5
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 6):
		diff_from_mean = 127 - 15*6
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif (choice == 7):
		diff_from_mean = 127 - 15*7
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)

	return (choice, arr, diff_from_mean)

def change_contrast(arr, curr_diff):
	diff_from_mean = random.randint(2,127)
	while(1):
		if(diff_from_mean == curr_diff):
			diff_from_mean = random.randint(2,127)
		else:
			break
	arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
	arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)

	return (diff_from_mean, arr)

def generate_data():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y_2 = np.zeros(seq_l)

	number_of_changes = random.randint(2, (seq_l -1))
	# print(number_of_changes)
	# number_of_changes = 4
	change_frame = random.sample(range(1, seq_l), number_of_changes)

	# change_frame = []
	# change_frame.append(random.randint(1,4))
	# change_frame.append(random.randint(5,9))
	# change_frame.append(random.randint(10,15))	

	number_of_changes = random.randint(2, (seq_l -1))
	# number_of_changes = 1
	change_frame_contrast = random.sample(range(1, seq_l), number_of_changes)

	curr_level = random.randint(0,7)
	training_datapoint_y[:] = curr_level
	training_datapoint_x[:][:] = 127.0
	temp = change_contrast1(training_datapoint_x[0], curr_level)
	curr_level = temp[0]
	diff_from_mean = temp[2]
	training_datapoint_x[:][:] = 127.0
	change_location = random.sample(range(0,64), 42)
	training_datapoint_x[:,change_location[:21]] -= float(diff_from_mean)
	training_datapoint_x[:,change_location[21:42]] += float(diff_from_mean)
	curr_direction = random.randint(0,7)
	training_datapoint_y_2[:] = 7

	for i in range(1,seq_l):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			training_datapoint_y_2[i] = temp[0]
			curr_direction = temp[0]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y[i:] = temp1[0]
			diff_from_mean = temp1[2]
	
	for i in range(seq_l):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)
	training_datapoint_y_2 = training_datapoint_y_2.astype(int)

	return [training_datapoint_x, training_datapoint_y], [training_datapoint_x, training_datapoint_y_2]

class RNN_arch_2_1(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn1_i2h = nn.Linear(64, 256)
		self.rnn1_h2h = nn.Linear(256,256)
		self.tanh1 = nn.Tanh()
		self.rnn1_h2o = nn.Linear(256, 64)
		self.tanh2 = nn.Tanh()
		self.fc1 = nn.Linear(64, 8)

	def forward(self, x, hc1):
		output_seq = torch.empty((16, self.batch_size, 8))
		i2h_seq = torch.empty((16, self.batch_size, 256))
		h2h_seq = torch.empty((16, self.batch_size, 256))
		for i in range(16):
			i2h = self.rnn1_i2h(x[i])
			i2h_seq[i] = i2h
			h2h = self.rnn1_h2h(hc1)
			h2h_seq[i] = h2h
			hc1 = self.tanh1(i2h + h2h)
			out = self.rnn1_h2o(hc1)
			out = self.tanh2(out)
			out = self.fc1(out)
			output_seq[i] = out

		return output_seq, hc1, i2h_seq, h2h_seq

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
		self.fcc = nn.Linear(64, 8)

		self.rnn2_i2h = nn.Linear(64, 256)
		self.rnn2_h2h = nn.Linear(256,256)
		self.tanh4 = nn.Tanh()
		self.feedback2 = nn.Linear(64,256)
		self.tanh5 = nn.Tanh()
		self.rnn2_h2o = nn.Linear(256, 64)
		self.tanh6 = nn.Tanh()
		self.fcc2 = nn.Linear(64, 8)

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
		out1 = torch.empty((16, self.batch_size, 8))
		out2 = torch.empty((16, self.batch_size, 8))
		cue_arr = torch.empty(16, self.batch_size, 2)
		output_seq = torch.empty((16, self.batch_size, 8))
		i2h_seq = torch.empty((16, self.batch_size, 256))
		h2h_seq = torch.empty((16, self.batch_size, 256))
		for i in range(16):
			i2h = self.rnn1_i2h(x[i])
			i2h_seq[i] = i2h
			h2h = self.rnn1_h2h(hc1)
			h2h_seq[i] = h2h
			hc1 = self.tanh1(i2h + h2h)
			out = self.rnn1_h2o(hc1)
			out = self.tanh2(out)
			out = self.fcc(out)
			output_seq[i] = out

			# i2h = self.rnn2_i2h(x[i])
			# h2h = self.rnn2_h2h(hc2)
			# feed = self.feedback2(hc4)
			# feed2_arr[i] = feed
			# # feed = self.tanh4(feed)
			# hc2 = self.tanh5(i2h + h2h)
			# outabc = self.rnn2_h2o(hc2)
			# outabc = self.tanh6(outabc)
			# output_seq2[i] = outabc
			# out2[i] = self.fcc2(outabc)

			# mid_inp = torch.cat((cue_arr, output_seq1, output_seq2), 2).cuda()

			# i2h = self.rnn3_i2h(mid_inp[i])
			# h2h = self.rnn3_h2h(hc3)
			# hc3 = self.tanh7(i2h + h2h)
			# outbcd = self.rnn3_h2o(hc3)
			# hc4 = outbcd
			# outbcd = self.tanh8(outbcd)
			# out_f_arr[i] = outbcd
			# outbcdp = self.fc(outbcd)
			# output_seq[i] = outbcdp

		return output_seq, hc1, hc2, hc3, hc4, output_seq1, output_seq2, cue_arr, feed1_arr, feed2_arr, out_f_arr, out1, out2, i2h_seq, h2h_seq

def predict_new():
	seq_l = 16
	
	model = RNN_arch_2_final(1)
	model.load_state_dict(torch.load("balle_balle_ho_gayaa.pt", map_location = device), strict = False)
	# print(model.state_dict().items())

	model1 = RNN_arch_2_1(1)
	model2 = RNN_arch_2_1(1)
	model1.load_state_dict(torch.load("contrast_manual_bitkin.pt", map_location = device))
	model2.load_state_dict(torch.load("motion_manual.pt", map_location = device))

	model.fcc.weight.data = model1.fc1.weight.data
	model.fcc2.weight.data = model2.fc1.weight.data	

	model.fcc.bias.data = model1.fc1.bias.data
	model.rnn1_i2h.bias.data = model1.rnn1_i2h.bias.data
	model.rnn1_h2h.bias.data = model1.rnn1_h2h.bias.data
	model.rnn1_h2o.bias.data = model1.rnn1_h2o.bias.data

	# for key_item_1 in model.state_dict().items():
	# 	print("model")
	# 	print(key_item_1)
	# for key_item_1 in model1.state_dict().items():
	# 	print("model1")
	# 	print(key_item_1)



	# print(torch.equal(model2.rnn1_i2h.weight, model.rnn2_i2h.weight))
	# print(torch.equal(model2.rnn1_h2h.weight, model.rnn2_h2h.weight))
	# print(torch.equal(model2.rnn1_h2o.weight, model.rnn2_h2o.weight))
	# print(torch.equal(model2.fc1.weight, model.fcc2.weight))

	# print(torch.equal(model1.rnn1_i2h.weight, model.rnn1_i2h.weight))
	# print(torch.equal(model1.rnn1_h2h.weight, model.rnn1_h2h.weight))
	# print(torch.equal(model1.rnn1_h2o.weight, model.rnn1_h2o.weight))
	# print(torch.equal(model1.fc1.weight, model.fcc.weight))

	model.eval()
	model1.eval()
	model.cuda()
	model1.cuda()

	hc1 = torch.zeros(1, 1, 256).cuda()
	hc2 = torch.zeros(1, 1, 256).cuda()
	hc3 = torch.zeros(1, 1, 256).cuda()
	hc4 = torch.zeros(1, 1, 64).cuda()
	
	total = 0

	for i in range(1000):
		temp = generate_data()
		data = temp[0]

		# model.feedback1.weight[:,:] = 0.0
		# model.feedback2.weight[:,:] = 0.0

		x = torch.tensor(data[0])
		y = torch.tensor(data[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		y = y.contiguous().view((1,seq_l)).cuda()

		# output_seq, hc1, hc2, hc3, hc4, output_seq1, output_seq2, cue_arr, feed1, feed2, out_f

		# output, _, i2h_seq1, h2h_seq1 = model1(x.float(), hc1.float())
		output, _, _, _, _, _, _, _, _, _, _, _, _, i2h_seq, h2h_seq = model(x.float(), 0, hc1.float(), hc2.float(), hc3.float(), hc4.float())
		# output = model1(output_seq1)
		correct = 0
		# print(output_seq)
		# print(y)

		for i in range(1,16):
			output[i] = torch.nn.functional.softmax(output[i])
			# print(y[0][i])
			# print(output[i])
			values, indices = output[i].max(1)
			# print(indices)
			# print(y[0][i])
			if(indices == y[0][i].cpu()):
				correct += 1

		accuracy = (correct/15)*100
		total += accuracy

	print(total/1000)

predict_new()