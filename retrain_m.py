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

def generate_contrast_data_2outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y1 = np.zeros(seq_l)
	training_datapoint_y2 = np.zeros(seq_l)

	# change_frame = [0, 0]
	# frames_before_change = random.randint(5, 9)
	# change_frame[0] = frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change

	number_of_changes = random.randint(2,15)
	# number_of_changes = 0
	change_frame = random.sample(range(1,16), number_of_changes)

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	# random.randint(2,15)
	
	number_of_changes = 2
	change_frame_contrast = []
	change_frame_contrast.append(random.randint(4,7))
	change_frame_contrast.append(random.randint((change_frame_contrast[0]+4), 12))

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
	training_datapoint_y[:] = 0
	training_datapoint_y1[:] = curr_level
	training_datapoint_y2[:] = 7

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			training_datapoint_y2[i] = temp[0]
			curr_direction = temp[0]
		# else:
		# 	temp = move(training_datapoint_x[i-1], choice = curr_direction)
		# 	training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y[i+1] = 1
			training_datapoint_y1[i:] = temp1[0]
			diff_from_mean = temp1[2]
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)
	training_datapoint_y1 = training_datapoint_y1.astype(int)
	training_datapoint_y2 = training_datapoint_y2.astype(int)

	return [training_datapoint_x, training_datapoint_y], [training_datapoint_y1, training_datapoint_y2]

def generate_motion_data_2outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y1 = np.zeros(seq_l)
	training_datapoint_y2 = np.zeros(seq_l)

	# change_frame = [0, 0]
	# frames_before_change = random.randint(5, 9)
	# change_frame[0] = frames_before_change
	# frames_before_change = random.randint(5, 7)
	# change_frame[1] = change_frame[0] + frames_before_change

	# random.randint(2,15)
	number_of_changes = 2
	change_frame = []
	change_frame.append(random.randint(4,7))
	change_frame.append(random.randint((change_frame[0]+4), 12))

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	number_of_changes = random.randint(5,15)
	# number_of_changes = 0
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
	training_datapoint_y[:] = 0
	training_datapoint_y1[:] = curr_level
	training_datapoint_y2[:] = 7

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			training_datapoint_y[i+1] = 1
			training_datapoint_y2[i] = temp[0]
			curr_direction = temp[0]
		# else:
		# 	training_datapoint_y[i] = 0

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y1[i:] = temp1[0]
			diff_from_mean = temp1[2]

	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)
	training_datapoint_y1 = training_datapoint_y1.astype(int)
	training_datapoint_y2 = training_datapoint_y2.astype(int)

	return [training_datapoint_x, training_datapoint_y], [training_datapoint_y1, training_datapoint_y2]

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

	number_of_changes = random.randint(7,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	# change_location = random.sample(range(1,16), number_of_changes)
	# for i in change_location:
	# 	if(i%2 == 1):
	# 		change_frame.append(i)

	number_of_changes = 2
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
		for i in range(16):
			i2h = self.rnn1_i2h(x[i])
			h2h = self.rnn1_h2h(hc1)
			hc1 = self.tanh1(i2h + h2h)
			out = self.rnn1_h2o(hc1)
			out = self.tanh2(out)
			out = self.fc1(out)
			output_seq[i] = out

		return output_seq, hc1

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
			feed = self.tanh1(feed)
			hc1 = self.tanh2(i2h + h2h + feed)
			out = self.rnn1_h2o(hc1)
			out = self.tanh3(out)
			output_seq1[i] = out

			i2h = self.rnn2_i2h(x[i])
			h2h = self.rnn2_h2h(hc2)
			feed = self.feedback2(hc4)
			feed2_arr[i] = feed
			feed = self.tanh4(feed)
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
		# return output_seq.view((16*self.batch_size, -1)), output_seq1.view((16,self.batch_size,64))

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




class RNN_arch_2(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn = nn.RNN(64, 256, num_layers =1, bias = False, nonlinearity='tanh')
		self.dropout = nn.Dropout(p=.2)
		self.fc1 = nn.Linear(256, 64)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(64,8)

	def forward(self, x, hc):
		out, hc = self.rnn(x, hc)
		r_output = out.contiguous().view(-1, 256)
		out1 = self.dropout(r_output)
		out2 = self.fc1(out1)
		out2 = self.tanh(out2)
		out3 = self.fc2(out2)

		return out3, hc, out2





# "saved_motion_nchange_state_8dir.pt"   works well
# "saved_motion_nchange_state.pt" should work better
# saved_contrast_hope_state

def train():
	model = RNN_arch_2_1(200)
	model.apply(weight_init)

	model1 = RNN_arch_2(200)
	for param in model1.parameters():
		param.requires_grad = False
	model1.load_state_dict(torch.load("motion_state_newer.pt", map_location = device))

	for m in model1.modules():
		if type(m) in [nn.RNN]:
			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					model.rnn1_i2h.weight.data = param.data
				elif 'weight_hh' in name:
					model.rnn1_h2h.weight.data = param.data
	model.rnn1_h2o.weight.data = model1.fc1.weight.data
	model.fc1.weight.data = model1.fc2.weight.data

	model.cuda()

	criterion = nn.CrossEntropyLoss().cuda()

	a = []
	b = []
	for i in range(100000):
		temp = generate_data()
		a.append(temp[0])
		b.append(temp[1])

	c = []
	d = []
	for i in range(25000):
		temp = generate_data()
		c.append(temp[0])
		d.append(temp[1])

	count0 = 0
	# optimizer = optim.Adagrad(model.parameters(), lr = 0.00005)

	while(count0<1000):
		if(count0 < 80):
			optimizer = optim.Adam(model.parameters(), lr = 0.0001)
		elif(count0 < 160):
			optimizer = optim.Adam(model.parameters(), lr = 0.00006)
		elif(count0 < 250):
			optimizer = optim.Adam(model.parameters(), lr = 0.00003)
		else:
			optimizer = optim.Adam(model.parameters(), lr = 0.00001)

		cue = 1

		if(cue == 0):
			train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
			valid_loader = data.DataLoader(c, batch_size=200, drop_last = True)

		if(cue == 1):
			train_loader = data.DataLoader(b, batch_size=200, shuffle = True, drop_last = True)
			valid_loader = data.DataLoader(d, batch_size=200, drop_last = True)

		train_loss = 0.0
		valid_loss = 0.0

		count0 += 1
		count = 1
		model.train()
		hc1 = torch.zeros(1, 200, 256).cuda()

		for x,y in train_loader:
			x = x.permute(1,0,2)
			y = y.permute(1,0)
			x, y = x.cuda(), y.cuda()
			print("Epoch: ", count0, "Step: ", count)
			count += 1

			optimizer.zero_grad()
			z, _ = model(x.float(), hc1.float())
			z = z.contiguous().view(3200,-1)
			y_reshaped = y.contiguous().view(200*16).cuda()
			
			cross_entropy_loss = criterion(z, y_reshaped)

			# all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
			# all_linear2_params = torch.cat([x.view(-1) for x in model.rnn1_h2o.parameters()])

			# if(cue == 1):
			# 	l2_regularization_1 = .006 * torch.norm(all_linear1_params, 2)
			# 	l2_regularization_2 = .006 * torch.norm(all_linear2_params, 2)
			# # # else:
			# # 	l2_regularization_1 = .006 * torch.norm(all_linear1_params, 2)
			# # 	l2_regularization_2 = .006 * torch.norm(all_linear2_params, 2)

			# loss = cross_entropy_loss + l2_regularization_1 + l2_regularization_2

			loss = criterion(z, y_reshaped)
			loss.backward()
			optimizer.step()
			train_loss = loss.item()




		hc1 = torch.zeros(1, 200, 256).cuda()
		model.eval()
		for val_x, val_y in valid_loader:
			val_x = val_x.permute(1,0,2)
			val_y = val_y.permute(1,0)
			val_x, val_y = val_x.cuda(), val_y.cuda()
			val_output, _ = model(val_x.float(), hc1.float())
			val_output = val_output.contiguous().view(3200,-1)
			val_y_reshaped = val_y.contiguous().view(200*16).cuda()
			valid_loss = criterion(val_output, val_y_reshaped)
			valid_loss = valid_loss.item()

		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
		doc.add_next_tick_callback(partial(update, new_data))

		torch.save(model.state_dict(), "motion_manual.pt")

thread = Thread(target=train)
thread.start()






def predict():
	model1 = RNN_arch_2_1(1)
	for param in model1.parameters():
		param.requires_grad = False
	# model2 = RNN_arch_2_1(1)
	# for param in model2.parameters():
	# 	param.requires_grad = False
	model1.load_state_dict(torch.load("motion_manual.pt", map_location = device))
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
	
	for i in range(2):
		data = generate_data()
		temp = data[1]

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
		test1, _,  = model1(x.float(), hc1.float())
		
		# penult = penult.cpu()
		# penult = penult.numpy()

		correct = 0
		for i in range(1,16):
			test1[i] = torch.nn.functional.softmax(test1[i])
			values, indices = torch.max(test1[i], 1)
			print(y[0][i])
			print(test1[i])
			print(" ")
			# contingency_table[int(y[0][i]), int(indices)] += 1.0
			# penult_total[0][int(y[0][i])] += penult[i]
			# divide_arr[0][int(y[0][i])] += 1
			if(indices == y[0][i]):
				correct += 1
		accuracy = (correct/15)*100
		total += accuracy

	print(total/2000)

# predict()