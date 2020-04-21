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
torch.set_default_tensor_type(torch.cuda.FloatTensor)

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

def generate_contrast_data_8outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)

	# change_frame = [0, 0]
	# frames_before_change = random.randint(5, 9)
	# change_frame[0] = frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

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
	# change_frame = sorted(change_frame)
	curr_direction = random.randint(0,7)
	# training_datapoint_y[:] = curr_direction

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			curr_direction = temp[0]
		# else:
		# 	temp = move(training_datapoint_x[i-1], choice = curr_direction)
		# 	training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y[i:] = temp1[0]
			diff_from_mean = temp1[2]
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]

def generate_motion_data_8outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)

	# change_frame = [0, 0]
	# frames_before_change = random.randint(5, 9)
	# change_frame[0] = frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

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

	# change_frame = sorted(change_frame)
	curr_direction = random.randint(0,6)
	training_datapoint_y[:] = 7

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i] = temp[1]
			training_datapoint_y[i] = temp[0]
			curr_direction = temp[0]
		else:
			training_datapoint_y[i] = 7

		if i in change_frame_contrast:
			temp1 = change_contrast(training_datapoint_x[i], diff_from_mean)
			training_datapoint_x[i:] = temp1[1]
			diff_from_mean = temp1[0]
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]

def generate_motion_data():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y[:] = 0

	change_frame = [0, 50]
	frames_before_change = random.randint(5, 7)
	change_frame[0] = 0+frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change - 1

	change_frame_contrast = [0, 0]
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[0] = 0+frames_before_change
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[1] = change_frame_contrast[0] + frames_before_change - 1

	diff_from_mean = random.randint(2,127)
	training_datapoint_x[:][:] = 127.0
	change_location = random.sample(range(0,64), 42)
	training_datapoint_x[:,change_location[:21]] -= float(diff_from_mean)
	training_datapoint_x[:,change_location[21:42]] += float(diff_from_mean)

	change_frame = sorted(change_frame)
	curr_direction = random.randint(0,7)

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1], previous_direction = curr_direction)
			training_datapoint_x[i] = temp[1]
			training_datapoint_y[i] = 1

			curr_direction = temp[0]
		else:
			temp = move(training_datapoint_x[i-1], choice = curr_direction)
			training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast(training_datapoint_x[i], diff_from_mean)
			training_datapoint_x[i:] = temp1[1]
			diff_from_mean = temp1[0]

	# training_datapoint_x = training_datapoint_x.astype(np.float32)
	training_datapoint_y = training_datapoint_y.astype(np.long)

	return [training_datapoint_x, training_datapoint_y]

def generate_contrast_data():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y[:] = 0

	change_frame = [0, 0]
	frames_before_change = random.randint(5, 7)
	change_frame[0] = 0+frames_before_change
	frames_before_change = random.randint(5, 7)
	change_frame[1] = change_frame[0] + frames_before_change - 1

	change_frame_contrast = [0, 0]
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[0] = 0+frames_before_change
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[1] = change_frame_contrast[0] + frames_before_change - 1

	diff_from_mean = random.randint(2,127)
	training_datapoint_x[:][:] = 127.0
	change_location = random.sample(range(0,64), 42)
	training_datapoint_x[:,change_location[:21]] -= float(diff_from_mean)
	training_datapoint_x[:,change_location[21:42]] += float(diff_from_mean)

	change_frame = sorted(change_frame)
	curr_direction = random.randint(0,7)

	for i in range(1,16):
		# if i in change_frame:
		# 	temp = move(training_datapoint_x[i-1], previous_direction = curr_direction)
		# 	training_datapoint_x[i] = temp[1]
		# 	curr_direction = temp[0]
		# else:
		# 	temp = move(training_datapoint_x[i-1], choice = curr_direction)
		# 	training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast(training_datapoint_x[i], diff_from_mean)
			training_datapoint_x[i:] = temp1[1]
			diff_from_mean = temp1[0]
			training_datapoint_y[i] = 1

	# training_datapoint_x = training_datapoint_x.astype(np.float32)
	training_datapoint_y = training_datapoint_y.astype(np.long)

	return [training_datapoint_x, training_datapoint_y]

def generate_data():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y_2 = np.zeros(seq_l)

	# change_frame = [0, 0]
	# frames_before_change = random.randint(5, 9)
	# change_frame[0] = frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

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
	# change_frame = sorted(change_frame)
	curr_direction = random.randint(0,7)
	training_datapoint_y_2[:] = 7

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			training_datapoint_y_2[i] = temp[0]
			curr_direction = temp[0]
		# else:
		# 	temp = move(training_datapoint_x[i-1], choice = curr_direction)
		# 	training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y[i:] = temp1[0]
			diff_from_mean = temp1[2]
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)
	training_datapoint_y_2 = training_datapoint_y_2.astype(int)

	return [training_datapoint_x, training_datapoint_y], [training_datapoint_x, training_datapoint_y_2]

class RNN_arch_2_1(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn = nn.RNN(64, 256, num_layers =1, bias = False, nonlinearity='tanh')
		self.dropout = nn.Dropout(p=.2)
		self.fc1 = nn.Linear(256, 64)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(64,8)

	def forward(self, x, hc):
		# output_seq = torch.empty((16, self.batch_size, 4))
		# output_seq1 = torch.empty((16, self.batch_size, 64))
		out, hc = self.rnn(x, hc)
		r_output = out.contiguous().view(-1, 256)
		out1 = self.dropout(r_output)
		out2 = self.fc1(out1)
		out2 = self.tanh(out2)
		out3 = self.fc2(out2)

		# output_seq[t] = self.fc2(out)
		# output_seq1[t] = out
		# return output_seq.view((16*self.batch_size, -1)), output_seq1.view((16,self.batch_size,64))
		return out3, hc, out2

class RNN_arch_2_final(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.fc0 = nn.Linear(130,64)
		# self.rnn = nn.RNN(64, 256, num_layers =1, bias = False, nonlinearity='tanh')
		# self.dropout = nn.Dropout(p=.2)
		self.fc1 = nn.Linear(64, 64)
		self.tanh1 = nn.ReLU()
		self.fc2 = nn.Linear(64, 64)
		self.tanh2 = nn.ReLU()
		self.fc3 = nn.Linear(64, 64)
		self.tanh3 = nn.ReLU()
		self.fc4 = nn.Linear(64, 8)

	def forward(self, x, hc):
		# output_seq = torch.empty((16, self.batch_size, 4))
		# output_seq1 = torch.empty((16, self.batch_size, 64))
		x1 = self.fc0(x)
		# out, hc = self.rnn(x, hc)
		# r_output = out.contiguous().view(-1, 256)
		# out = self.dropout(r_output)
		x2 = self.fc1(x1)
		x2 = self.tanh1(x2)
		x3 = self.fc2(x2)
		x3 = self.tanh2(x3)
		x4 = self.fc3(x3)
		x4 = self.tanh3(x4)
		x5 = self.fc4(x4)


		# output_seq[t] = self.fc2(out)
		# output_seq1[t] = out
		# return output_seq.view((16*self.batch_size, -1)), output_seq1.view((16,self.batch_size,64))
		return x5, x4



def predict():
	model1 = RNN_arch_2_1(1)
	for param in model1.parameters():
		param.requires_grad = False
	# model2 = RNN_arch_2_1(1)
	# for param in model2.parameters():
	# 	param.requires_grad = False
	model1.load_state_dict(torch.load("contrast_state_newest.pt", map_location = device))
	# model2.load_state_dict(torch.load("motion_state_newer.pt", map_location = device))
	model1.eval()
	# model2.eval()

	# model = RNN_arch_2_final(1)
	# model.load_state_dict(torch.load("godspeed_hope.pt", map_location = device))
	# model.eval()
	# for param in model.parameters():
		# param.requires_grad = False	

	# contingency_table = np.zeros((8,8))

	hc1 = torch.zeros(1, 1, 256).cuda()
	# hc2 = torch.zeros(1, 1, 256).cuda()
	# hc3 = torch.zeros(1, 1, 256).cuda()

	total = 0
	# penult_total = np.zeros((4,8,64))
	# divide_arr = np.zeros((4,8,1))
	
	for i in range(2000):
		data = generate_data()
		temp = data[0]

		x = torch.tensor(temp[0])
		y = torch.tensor(temp[1])

		x = x.contiguous().view((16,1,64)).cuda()
		y = y.contiguous().view((1,16)).cuda()

		# final_input = torch.zeros((16,130))

		# _, _, output1 = model1(x.float(), hc1.float())
		# _, _, output2 = model2(x.float(), hc2.float())

		# final_input[:, 2:66] = output1
		# final_input[:, 66:130] = output2
		# final_input[:, 0] = 1
		# final_input[:, 1] = 0
		# final_input = final_input.cuda()

		# test1, penult = model(final_input.float(), hc3.float())
		test1, _, _ = model1(x.float(), hc1.float())
		
		# penult = penult.cpu()
		# penult = penult.numpy()

		correct = 0
		for i in range(1,16):
			test1[i] = torch.nn.functional.softmax(test1[i])
			values, indices = torch.max(test1[i], 0)
			# contingency_table[int(y[0][i]), int(indices)] += 1.0
			# penult_total[0][int(y[0][i])] += penult[i]
			# divide_arr[0][int(y[0][i])] += 1
			if(indices == y[0][i]):
				correct += 1
		accuracy = (correct/15)*100
		total += accuracy

	print(total/2000)




	# 	# final_input = torch.zeros((16,130))

	# 	# _, _, output1 = model1(x.float(), hc1.float())
	# 	# _, _, output2 = model2(x.float(), hc2.float())

	# 	# final_input[:, 2:66] = output1
	# 	# final_input[:, 66:130] = output2
	# 	final_input[:, 0] = 0
	# 	final_input[:, 1] = 1
	# 	final_input = final_input.cuda()

	# 	test1, penult = model(final_input.float(), hc3.float())

	# 	penult = penult.cpu()
	# 	penult = penult.numpy()
		
	# 	correct = 0
	# 	for i in range(1,16):
	# 		test1[i] = torch.nn.functional.softmax(test1[i])
	# 		values, indices = torch.max(test1[i], 0)
	# 		contingency_table[int(y[0][i]), int(indices)] += 1.0
	# 		penult_total[1][int(y[0][i])] += penult[i]
	# 		divide_arr[1][int(y[0][i])] += 1
	# 		if(indices == y[0][i]):
	# 			correct += 1
	# 	accuracy = (correct/15)*100
	# 	total += accuracy		






	# 	# temp = data[1]

	# 	# if(task == 1):
	# 	# 	temp = generate_contrast_data()
	# 	# 	output[:,:,65] = 1.0

	# 	# if(task == 0):
	# 	# 	temp = generate_motion_data()
	# 	# 	output[:,:,64] = 1.0

	# 	x = torch.tensor(temp[0])
	# 	y = torch.tensor(temp[1])

	# 	x = x.contiguous().view((16,1,64)).cuda()
	# 	y = y.contiguous().view((1,16)).cuda()

	# 	final_input = torch.zeros((16,130))

	# 	_, _, output1 = model1(x.float(), hc1.float())
	# 	_, _, output2 = model2(x.float(), hc2.float())

	# 	final_input[:, 2:66] = output1
	# 	final_input[:, 66:130] = output2
	# 	final_input[:, 0] = 0
	# 	final_input[:, 1] = 0
	# 	final_input = final_input.cuda()

	# 	test1, penult = model(final_input.float(), hc3.float())

	# 	penult = penult.cpu()
	# 	penult = penult.numpy()
		
	# 	correct = 0
	# 	for i in range(1,16):
	# 		test1[i] = torch.nn.functional.softmax(test1[i])
	# 		values, indices = torch.max(test1[i], 0)
	# 		contingency_table[int(y[0][i]), int(indices)] += 1.0
	# 		penult_total[2][int(y[0][i])] += penult[i]
	# 		divide_arr[2][int(y[0][i])] += 1
	# 		if(indices == y[0][i]):
	# 			correct += 1
	# 	accuracy = (correct/15)*100
	# 	total += accuracy




	# 	# final_input = torch.zeros((16,130))

	# 	# _, _, output1 = model1(x.float(), hc1.float())
	# 	# _, _, output2 = model2(x.float(), hc2.float())

	# 	# final_input[:, 2:66] = output1
	# 	# final_input[:, 66:130] = output2
	# 	final_input[:, 0] = 1
	# 	final_input[:, 1] = 1
	# 	final_input = final_input.cuda()

	# 	test1, penult = model(final_input.float(), hc3.float())

	# 	penult = penult.cpu()
	# 	penult = penult.numpy()
		
	# 	correct = 0
	# 	for i in range(1,16):
	# 		test1[i] = torch.nn.functional.softmax(test1[i])
	# 		values, indices = torch.max(test1[i], 0)
	# 		contingency_table[int(y[0][i]), int(indices)] += 1.0
	# 		penult_total[3][int(y[0][i])] += penult[i]
	# 		divide_arr[3][int(y[0][i])] += 1
	# 		if(indices == y[0][i]):
	# 			correct += 1
	# 	accuracy = (correct/15)*100
	# 	total += accuracy

	# for i in range(4):
	# 	for j in range(8):
	# 		penult_total[i][j] = penult_total[i][j] / divide_arr[i][j]

	# np.save("4_activation.np", penult_total)
	# np.save("0", model.fc0.weight.data.cpu().numpy())
	# np.save("1", model.fc1.weight.data.cpu().numpy())
	# np.save("2", model.fc2.weight.data.cpu().numpy())
	# np.save("3", model.fc3.weight.data.cpu().numpy())
	# np.save("4", model.fc4.weight.data.cpu().numpy())
	# np.save("1c", model1.fc1.weight.data.cpu().numpy())
	# np.save("1m", model2.fc1.weight.data.cpu().numpy())

	# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	# ax1.imshow(penult_total[0])
	# ax2.imshow(penult_total[1])
	# ax3.imshow(penult_total[2])
	# ax4.imshow(penult_total[3])
	# plt.show()


	# print(total/5000)			
	# contingency_table = np.around(contingency_table, 2)
	# np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
	# for i in range(8):
	# 	print(*contingency_table[i])

predict()




def weight_init(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight.data)
		init.constant_(m.bias.data, 0.0)

def init_weights(model):
	for m in model.modules():
		if type(m) in [nn.RNN]:
			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					torch.nn.init.constant_(param.data, 1.0)
				elif 'weight_hh' in name:
					torch.nn.init.constant_(param.data, 0.0)
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









# "saved_motion_nchange_state_8dir.pt"   works well
# "saved_motion_nchange_state.pt" should work better
# saved_contrast_hope_state

def train():
	model1 = RNN_arch_2_1(200)
	# model2 = RNN_arch_2_1(200)
	# for param in model1.parameters():
		# param.requires_grad = False
	# for param in model2.parameters():
		# param.requires_grad = False
	model1.load_state_dict(torch.load("contrast_state_newest.pt", map_location = device))
	# model2.load_state_dict(torch.load("motion_state_newer.pt", map_location = device))
	# model1.eval()
	# model2.eval()

	# model = RNN_arch_2_final(200)
	# model.apply(weight_init)
	# model.apply(init_weights)
	# model.fc1.weight.data = model2.fc1.weight.data
	# model.fc2.weight.data = model2.fc2.weight.data
	# for [name1 , param1], [name2 , param2] in zip(model.rnn.named_parameters(), model2.rnn.named_parameters()):
	# 		param1.data = param2.data
	# model.load_state_dict(torch.load("godspeed.pt", map_location = device))

	# model.cuda()
	model1.cuda()

	criterion = nn.CrossEntropyLoss().cuda()

	# optimizer = optim.Adam(model.parameters(), lr = 0.00005)

	# if torch.cuda.device_count() > 1:
	#   print("Let's use", torch.cuda.device_count(), "GPUs!")
	#   model = nn.DataParallel(model).cuda()

	a = []
	for i in range(200000):
		temp = generate_contrast_data_8outputs()
		a.append(temp)

	# b = []
	# for i in range(200000):
	# 	temp = generate_motion_data_8outputs()
	# 	b.append(temp)

	c = []
	for i in range(50000):
		temp = generate_contrast_data_8outputs()
		c.append(temp)

	# d = []
	# for i in range(50000):
	# 	temp = generate_motion_data_8outputs()
	# 	d.append(temp)


	count0 = 0
	count = 1
	optimizer = optim.Adam(model1.parameters(), lr = 0.00006)

	while(count0<1000):
		# if(count0 < 80):
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.00005)
		# elif(count0 < 160):
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0003)
		# elif(count0 < 250):
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0001)
		# else:
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.00005)			

		# elif(count0 < 300):
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0005)
		# else:
		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0003)

		# final_input = torch.zeros((3200,130))

		# flag = 0
		# if( int(count0/5) % 2 == 0):
		# 	cue = 0
		# else:
		# 	cue = 1
		

		cue = 0

		if(cue == 0):
			train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
			valid_loader = data.DataLoader(c, batch_size=200, drop_last = True)
			# final_input[:, 0] = 1.0
			# final_input[:, 1] = 0.0

		# if(cue == 1):
		# 	train_loader = data.DataLoader(b, batch_size=200, shuffle = True, drop_last = True)
		# 	valid_loader = data.DataLoader(d, batch_size=200, drop_last = True)
		# 	final_input[:, 0] = 0.0
		# 	final_input[:, 1] = 1.0

		train_loss = 0.0
		valid_loss = 0.0

		count0 += 1
		model1.train()
		hc1 = torch.zeros(1, 200, 256).cuda()
		# hc2 = torch.zeros(1, 200, 256).cuda()
		# hc3 = torch.zeros(1, 200, 256).cuda()

		for x,y in train_loader:
			# final_input = final_input.contiguous().view(3200,-1)


			x = x.permute(1,0,2)
			y = y.permute(1,0)
			x, y = x.cuda(), y.cuda()
			print("Epoch: ", count0, "Step: ", count)
			count += 1

			# _, _, output1 = model1(x.float(), hc1.float())
			# _, _, output2 = model2(x.float(), hc2.float())

			# final_input[:, 2:66] = output1
			# final_input[:, 66:130] = output2

			# final_input = final_input.contiguous().view(16,200,-1)

			optimizer.zero_grad()
			z, _, _ = model1(x.float(), hc1.float())
			y_reshaped = y.contiguous().view(200*16).cuda()
			
			cross_entropy_loss = criterion(z, y_reshaped)
			# all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
			# all_linear2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])

			# if(cue == 1):
			# 	l2_regularization_1 = .006 * torch.norm(all_linear1_params, 2)
			# 	l2_regularization_2 = .006 * torch.norm(all_linear2_params, 2)
			# else:
			# 	l2_regularization_1 = .006 * torch.norm(all_linear1_params, 2)
			# 	l2_regularization_2 = .006 * torch.norm(all_linear2_params, 2)




			# loss = cross_entropy_loss + l2_regularization_1 + l2_regularization_2
			loss = criterion(z, y_reshaped)
			loss.backward()
			
			# nn.utils.clip_grad_norm_(model.parameters(), 3)
			
			optimizer.step()
			train_loss = loss.item()







		# final_input = torch.zeros((3200,130))

		hc1 = torch.zeros(1, 200, 256).cuda()
		# hc2 = torch.zeros(1, 200, 256).cuda()
		hc3 = torch.zeros(1, 200, 256).cuda()
		
		model1.eval()
		for val_x, val_y in valid_loader:
			# final_input = final_input.contiguous().view(3200,-1)

			val_x = val_x.permute(1,0,2)
			val_y = val_y.permute(1,0)
			val_x, val_y = val_x.cuda(), val_y.cuda()

			# _, _, output1 = model1(val_x.float(), hc1.float())
			# _, _, output2 = model2(val_x.float(), hc2.float())

			# final_input[:, 2:66] = output1
			# final_input[:, 66:130] = output2
			
			# final_input = final_input.contiguous().view(16,200,-1)

			val_output, _, _ = model1(val_x.float(), hc3.float())
			val_y_reshaped = val_y.contiguous().view(200*16).cuda()
			valid_loss = criterion(val_output, val_y_reshaped)
			valid_loss = valid_loss.item()

		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
		doc.add_next_tick_callback(partial(update, new_data))

		torch.save(model1.state_dict(), "contrast_state_newest.pt")


# thread = Thread(target=train)
# thread.start()






# def train():
# 	# model1 = RNN_arch_2_1(200)
# 	# model2 = RNN_arch_2_1(200)
# 	# for param in model1.parameters():
# 		# param.requires_grad = False
# 	# for param in model2.parameters():
# 		# param.requires_grad = False
# 	# model1.load_state_dict(torch.load("saved_contrast_new.pt", map_location = device))
# 	# model2.load_state_dict(torch.load("saved_motion_new.pt", map_location = device))
# 	# model1.eval()
# 	# model2.eval()

# 	# model = RNN_arch_2_final(200)
# 	# model.apply(weight_init)
# 	# model.apply(init_weights)
# 	# model.fc1.weight.data = model2.fc1.weight.data
# 	# model.fc2.weight.data = model2.fc2.weight.data
# 	# for [name1 , param1], [name2 , param2] in zip(model.rnn.named_parameters(), model2.rnn.named_parameters()):
# 			# param1.data = param2.data
# 	# model.load_state_dict(torch.load("saved_contrast_hope_state.pt", map_location = device))


# 	model = RNN_arch_2_1(200)
# 	model.load_state_dict(torch.load("saved_contrast_new.pt", map_location = device))
# 	model.cuda()

# 	criterion = nn.CrossEntropyLoss().cuda()

# 	# optimizer = optim.Adam(model.parameters(), lr = 0.00005)

# 	# if torch.cuda.device_count() > 1:
# 	#   print("Let's use", torch.cuda.device_count(), "GPUs!")
# 	#   model = nn.DataParallel(model).cuda()

# 	# a = []
# 	# for i in range(200000):
# 	# 	temp = generate_contrast_data_8outputs()
# 	# 	a.append(temp)

# 	b = []
# 	for i in range(200000):
# 		temp = generate_contrast_data_8outputs()
# 		b.append(temp)

# 	# c = []
# 	# for i in range(50000):
# 	# 	temp = generate_contrast_data_8outputs()
# 	# 	c.append(temp)

# 	d = []
# 	for i in range(50000):
# 		temp = generate_contrast_data_8outputs()
# 		d.append(temp)


# 	count0 = 0
# 	count = 1
# 	# optimizer = optim.Adagrad(model.parameters(), lr = 0.0002)

# 	while(count0<1000):
# 		if(count0 < 90):
# 			optimizer = optim.Adagrad(model.parameters(), lr = 0.0005)
# 		elif(count0 < 180):
# 			optimizer = optim.Adagrad(model.parameters(), lr = 0.0003)
# 		else:
# 			optimizer = optim.Adagrad(model.parameters(), lr = 0.0001)

# 		# elif(count0 < 300):
# 		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0005)
# 		# else:
# 		# 	optimizer = optim.Adagrad(model.parameters(), lr = 0.0003)

# 		# final_input = torch.zeros((3200,130))

# 		# cue = random.randint(0,1)
# 		# if(cue == 0):
# 		# 	train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
# 		# 	valid_loader = data.DataLoader(c, batch_size=200, drop_last = True)
# 		# 	final_input[:, 0] = 1.0
# 		# 	final_input[:, 1] = 0.0

# 		# if(cue == 1):
# 		# 	train_loader = data.DataLoader(b, batch_size=200, shuffle = True, drop_last = True)
# 		# 	valid_loader = data.DataLoader(d, batch_size=200, drop_last = True)
# 		# 	final_input[:, 0] = 0.0
# 		# 	final_input[:, 1] = 1.0

# 		train_loader = data.DataLoader(b, batch_size=200, shuffle = True, drop_last = True)
# 		valid_loader = data.DataLoader(d, batch_size=200, drop_last = True)

# 		train_loss = 0.0
# 		valid_loss = 0.0

# 		count0 += 1
# 		model.train()
# 		hc = torch.zeros(1, 200, 256).cuda()
# 		# hc2 = torch.zeros(1, 200, 256).cuda()
# 		# hc3 = torch.zeros(1, 200, 256).cuda()

# 		for x,y in train_loader:
# 			# final_input = final_input.contiguous().view(3200,-1)


# 			x = x.permute(1,0,2)
# 			y = y.permute(1,0)
# 			x, y = x.cuda(), y.cuda()
# 			print("Epoch: ", count0, "Step: ", count)
# 			count += 1

# 			# _, _, output1 = model1(x.float(), hc1.float())
# 			# _, _, output2 = model2(x.float(), hc2.float())

# 			# final_input[:, 2:66] = output1
# 			# final_input[:, 66:130] = output2

# 			# final_input = final_input.contiguous().view(16,200,-1)

# 			optimizer.zero_grad()
# 			z, _, _ = model(x.float(), hc.float())
# 			y_reshaped = y.contiguous().view(200*16).cuda()
			
# 			cross_entropy_loss = criterion(z, y_reshaped)
# 			all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
# 			all_linear2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])

# 			l2_regularization_1 = .0005 * torch.norm(all_linear1_params, 2)
# 			l2_regularization_2 = .0005 * torch.norm(all_linear2_params, 2)

# 			loss = cross_entropy_loss + l2_regularization_1 + l2_regularization_2
# 			# loss = criterion(z, y_reshaped)
# 			loss.backward()
			
# 			# nn.utils.clip_grad_norm_(model.parameters(), 3)
			
# 			optimizer.step()
# 			train_loss = loss.item()







# 		# final_input = torch.zeros((3200,130))

# 		hc = torch.zeros(1, 200, 256).cuda()
# 		# hc2 = torch.zeros(1, 200, 256).cuda()
# 		# hc3 = torch.zeros(1, 200, 256).cuda()
		
# 		model.eval()
# 		for val_x, val_y in valid_loader:
# 			# final_input = final_input.contiguous().view(3200,-1)

# 			val_x = val_x.permute(1,0,2)
# 			val_y = val_y.permute(1,0)
# 			val_x, val_y = val_x.cuda(), val_y.cuda()

# 			# _, _, output1 = model1(val_x.float(), hc1.float())
# 			# _, _, output2 = model2(val_x.float(), hc2.float())

# 			# final_input[:, 2:66] = output1
# 			# final_input[:, 66:130] = output2
			
# 			# final_input = final_input.contiguous().view(16,200,-1)

# 			val_output, hc, _ = model(val_x.float(), hc.float())
# 			val_y_reshaped = val_y.contiguous().view(200*16).cuda()
# 			valid_loss = criterion(val_output, val_y_reshaped)
# 			valid_loss = valid_loss.item()

# 		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
# 		doc.add_next_tick_callback(partial(update, new_data))

# 		torch.save(model.state_dict(), "contrast_state_newer.pt")














# def train():
# 	model1 = RNN_arch_2(128)
# 	model2 = RNN_arch_2(128)
# 	model1.load_state_dict(torch.load("saved_contrast_model_state.pt", map_location = device))
# 	model2.load_state_dict(torch.load("saved_motion_model_state.pt", map_location = device))
# 	model1.eval()
# 	model2.eval()

# 	model = RNN_arch_2_final(128)
# 	model.apply(weight_init)
# 	# if torch.cuda.device_count() > 1:
# 	#   print("Let's use", torch.cuda.device_count(), "GPUs!")
# 	#   model = nn.DataParallel(model)
# 	model = model.cuda()

# 	criterion = nn.CrossEntropyLoss()
# 	optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 	count0 = 1
# 	while(count0 <= 1000):
# 		hc1 = torch.zeros(128, 256).cuda()
# 		hc2 = torch.zeros(128, 256).cuda()
# 		hc3 = torch.zeros(128, 256).cuda()
# 		output = torch.zeros((16,128,66))

# 		a = []
# 		if(int(count0/25)%2 == 0):
# 			print("1")
# 			for i in range(128):
# 				temp = generate_contrast_data()
# 				output[:,:,65] = 1.0
# 				a.append(temp)

# 		else:
# 			print("0")
# 			for i in range(128):
# 				temp = generate_motion_data()
# 				output[:,:,64] = 1.0
# 				a.append(temp)

# 		train_loader = data.DataLoader(a, batch_size=128, shuffle=True)

# 		train_loss = 0.0
# 		valid_loss = 0.0

# 		count0 += 1
# 		count = 1
# 		model.train()
		
# 		for x,y in train_loader:
# 			x = x.permute(1,0,2)
# 			y = y.permute(1,0)
# 			x, y = x.cuda(), y.cuda()
# 			count += 1

# 			_, output1 = model1(x.float(), hc1.float())
# 			_, output2 = model2(x.float(), hc2.float())

# 			output[:,:,:64] = output1 + output2
# 			output = output.cuda()

# 			optimizer.zero_grad()
# 			z = model(output.float(), hc3.float())
# 			y_reshaped = y.contiguous().view(16*128).cuda()
# 			loss = criterion(z, y_reshaped)
# 			loss.backward()
# 			optimizer.step()
# 			train_loss = loss.item()

# 		if(count0%10 == 1):
# 			hc1 = torch.zeros(128, 256).cuda()
# 			hc2 = torch.zeros(128, 256).cuda()
# 			hc3 = torch.zeros(128, 256).cuda()
# 			output = torch.zeros((16,128,66))
			
# 			b = []
# 			if(int(count0/25)%2 == 0):
# 				print("1")
# 				for i in range(128):
# 					temp = generate_contrast_data()
# 					output[:,:,65] = 1.0
# 					b.append(temp)
# 			else:
# 				print("0")
# 				for i in range(128):
# 					temp = generate_motion_data()
# 					output[:,:,64] = 1.0
# 					b.append(temp)

# 			valid_loader = data.DataLoader(b, batch_size=128, shuffle=True)
			
# 			count = 1
# 			model.eval()
# 			for val_x, val_y in valid_loader:
# 				val_x = val_x.permute(1,0,2)
# 				val_y = val_y.permute(1,0)
# 				val_x, val_y = val_x.cuda(), val_y.cuda()
# 				count+=1

# 				_, output1 = model1(x.float(), hc1.float())
# 				_, output2 = model2(x.float(), hc2.float())
# 				output[:,:,:64] = output1 + output2
# 				output = output.cuda()

# 				hc = torch.zeros(128, 256).cuda()
# 				val_output = model(output.float(), hc3.float())
# 				val_y_reshaped = val_y.contiguous().view(16*128).cuda()
# 				valid_loss = criterion(val_output, val_y_reshaped)
# 				valid_loss = valid_loss.item()

# 		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
# 		doc.add_next_tick_callback(partial(update, new_data))

# 		if(count0%10 == 1):
# 			torch.save(model.state_dict(), "saved_final_model_state.pt")














# def train():
# 	# model1 = RNN_arch_2(128)
# 	model2 = RNN_arch_2_1(156)
# 	# model1.load_state_dict(torch.load("saved_contrast_model_state.pt", map_location = device))
# 	model2.load_state_dict(torch.load("saved_motion8outputs_model_state.pt", map_location = device))
# 	# model1.eval()
# 	model2.eval()

# 	model = RNN_arch_2(156)
# 	model.apply(weight_init)
# 	model.apply(init_weights)
# 	model.fc1.weight.data = model2.fc1.weight.data
# 		if type(m) in [nn.RNN]:
# 			for  name1 , param1 in model2.rnn.named_parameters() and name2 , param2 in model.rnn.named_parameters():
# 				if 'weight_ih' in name1 and 'weight_ih' in name2:
# 					param1.data = param2.data
# 				elif 'weight_hh' in name1 and 'weight_hh' in name2:
# 					param1.data = param2.data

# 	# if torch.cuda.device_count() > 1:
# 	#   print("Let's use", torch.cuda.device_count(), "GPUs!")
# 	#   model = nn.DataParallel(model)
# 	model = model.cuda()

# 	criterion = nn.CrossEntropyLoss()
# 	optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 	count0 = 0
# 	count = 1
# 	while(count0 <= 1000):
# 		hc1 = torch.zeros(128, 256).cuda()
# 		hc2 = torch.zeros(128, 256).cuda()
# 		output = torch.zeros((16,128,64))

# 		a = []
# 		for i in range(200000):
# 			temp = generate_motion_data()
# 			a.append(temp)

# 		train_loader = data.DataLoader(a, batch_size=156, shuffle=True)

# 		train_loss = 0.0
# 		valid_loss = 0.0

# 		count0 += 1
# 		count = 1
# 		model.train()
		
# 		for x,y in train_loader:
# 			x = x.permute(1,0,2)
# 			y = y.permute(1,0)
# 			x, y = x.cuda(), y.cuda()
# 			count += 1

# 			_, output2 = model2(x.float(), hc2.float())

# 			output[:,:,:64] = output1 + output2
# 			output = output.cuda()

# 			optimizer.zero_grad()
# 			z = model(output.float(), hc3.float())
# 			y_reshaped = y.contiguous().view(16*128).cuda()
# 			loss = criterion(z, y_reshaped)
# 			loss.backward()
# 			optimizer.step()
# 			train_loss = loss.item()

# 		if(count0%10 == 1):
# 			hc1 = torch.zeros(128, 256).cuda()
# 			hc2 = torch.zeros(128, 256).cuda()
# 			hc3 = torch.zeros(128, 256).cuda()
# 			output = torch.zeros((16,128,66))
			
# 			b = []
# 			if(int(count0/25)%2 == 0):
# 				print("1")
# 				for i in range(128):
# 					temp = generate_contrast_data()
# 					output[:,:,65] = 1.0
# 					b.append(temp)
# 			else:
# 				print("0")
# 				for i in range(128):
# 					temp = generate_motion_data()
# 					output[:,:,64] = 1.0
# 					b.append(temp)

# 			valid_loader = data.DataLoader(b, batch_size=128, shuffle=True)
			
# 			count = 1
# 			model.eval()
# 			for val_x, val_y in valid_loader:
# 				val_x = val_x.permute(1,0,2)
# 				val_y = val_y.permute(1,0)
# 				val_x, val_y = val_x.cuda(), val_y.cuda()
# 				count+=1

# 				_, output1 = model1(x.float(), hc1.float())
# 				_, output2 = model2(x.float(), hc2.float())
# 				output[:,:,:64] = output1 + output2
# 				output = output.cuda()

# 				hc = torch.zeros(128, 256).cuda()
# 				val_output = model(output.float(), hc3.float())
# 				val_y_reshaped = val_y.contiguous().view(16*128).cuda()
# 				valid_loss = criterion(val_output, val_y_reshaped)
# 				valid_loss = valid_loss.item()

# 		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
# 		doc.add_next_tick_callback(partial(update, new_data))

# 		if(count0%10 == 1):
# 			torch.save(model.state_dict(), "saved_final_model_state.pt")


