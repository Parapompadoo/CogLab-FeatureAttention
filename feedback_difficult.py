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

device = "cuda"
torch.set_default_tensor_type(torch.cuda.FloatTensor)

random.seed(datetime.now())

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

def move(arr, choice = 10, previous_direction = 20):
	if(choice == 10):
		choice = random.randint(0,3)
		while(1):
			if(choice == previous_direction):
				choice = random.randint(0,3)
			else:
				break
	if(choice == 0):
		arr = move_up(arr)
	elif(choice == 1):
		arr = move_down(arr)
	elif(choice == 2):
		arr = move_left(arr)
	elif(choice == 3):
		arr = move_right(arr)

	return (choice, arr)

def change_contrast1(arr, choice = 10, previous_level = 20):
	if(choice == 10):
		choice = random.randint(0,3)
		while(1):	
			if(choice == previous_level):
				choice = random.randint(0,3)
			else:
				break
	diff_from_mean = 0
	if(choice == 0):
		diff_from_mean = 127
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)
	elif(choice == 1):
		diff_from_mean = 127 - 32*1
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 2):
		diff_from_mean = 127 - 32*2
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	
	elif(choice == 3):
		diff_from_mean = 127 - 32*3
		arr[np.where(arr == np.amin(arr))[0]] = 127.0 - float(diff_from_mean)
		arr[np.where(arr == np.amax(arr))[0]] = 127.0 + float(diff_from_mean)	

	return (choice, arr, diff_from_mean)

def add_noise(arr):
	mini = np.amin(arr)
	maxi = np.amax(arr)

	for i in range(arr.shape[0]):
		choice = random.choices([0, 1, 2], [.7, .15, .15])
		epsi = random.randint(-90,90)
		if(1 in choice):
			if(arr[i] == mini):
				arr[i] = min(maxi,maxi + epsi)
		if(2 in choice):
			if(arr[i] == mini):
				arr[i] = 127 + epsi

	for i in range(arr.shape[0]):
		choice = random.choices([0, 1, 2], [.7, .15, .15])
		epsi = random.randint(-90,90)
		if(1 in choice):
			if(arr[i] == maxi):
				arr[i] = max(mini,mini + epsi)
		if(2 in choice):
			if(arr[i] == maxi):
				arr[i] = 127 + epsi

	for i in range(arr.shape[0]):
		choice = random.choices([0, 1, 2], [.7, .15, .15])
		epsi = random.randint(-90,90)
		if(1 in choice):
			if(arr[i] == 127):
				arr[i] = min(maxi,maxi + epsi)
		if(2 in choice):
			if(arr[i] == 127):
				arr[i] = max(mini,mini + epsi)
			
	return arr

def generate_contrast_data_4outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

	curr_level = random.randint(0,3)
	training_datapoint_y[:] = curr_level
	training_datapoint_x[:][:] = 127.0
	temp = change_contrast1(training_datapoint_x[0], curr_level)
	curr_level = temp[0]
	diff_from_mean = temp[2]
	training_datapoint_x[:][:] = 127.0
	change_location = random.sample(range(0,64), 42)
	training_datapoint_x[:,change_location[:21]] -= float(diff_from_mean)
	training_datapoint_x[:,change_location[21:42]] += float(diff_from_mean)

	curr_direction = random.randint(0,3)
	training_datapoint_y[:] = curr_level

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1], previous_direction = curr_direction)
			training_datapoint_x[i] = temp[1]
			curr_direction = temp[0]
		else:
			temp = move(training_datapoint_x[i-1], choice = curr_direction)
			training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			training_datapoint_y[i:] = temp1[0]
			diff_from_mean = temp1[2]

	for i in range(16):
		training_datapoint_x[i] = add_noise(training_datapoint_x[i])
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]

def generate_motion_data_4outputs():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)

	number_of_changes = random.randint(2,15)
	change_frame = random.sample(range(1,16), number_of_changes)

	number_of_changes = random.randint(2,15)
	change_frame_contrast = random.sample(range(1,16), number_of_changes)

	curr_level = random.randint(0,3)
	training_datapoint_y[:] = curr_level
	training_datapoint_x[:][:] = 127.0
	temp = change_contrast1(training_datapoint_x[0], curr_level)
	curr_level = temp[0]
	diff_from_mean = temp[2]
	training_datapoint_x[:][:] = 127.0
	change_location = random.sample(range(0,64), 42)
	training_datapoint_x[:,change_location[:21]] -= float(diff_from_mean)
	training_datapoint_x[:,change_location[21:42]] += float(diff_from_mean)

	curr_direction = random.randint(0,3)
	training_datapoint_y[:] = curr_direction

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1], previous_direction = curr_direction)
			training_datapoint_x[i] = temp[1]
			training_datapoint_y[i:] = temp[0]
			curr_direction = temp[0]
		else:
			temp = move(training_datapoint_x[i-1], choice = curr_direction)
			training_datapoint_x[i] = temp[1]

		if i in change_frame_contrast:
			temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
			training_datapoint_x[i:] = temp1[1]
			diff_from_mean = temp1[2]
	
	for i in range(16):
		training_datapoint_x[i] = add_noise(training_datapoint_x[i])

	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]


# a = generate_contrast_data_4outputs()
# for i in range(16):
# 	print(a[1][i])
# 	plt.imshow(a[0][i].reshape((8,8)), vmin = -1, vmax = 1)
# 	plt.show()

class RNN_arch_2_1(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn = nn.RNN(64, 256, num_layers =1, bias = False, nonlinearity='tanh')
		self.dropout = nn.Dropout(p=.2)
		self.fc1 = nn.Linear(256, 64)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(64,4)

	def forward(self, x, hc):
		out, hc = self.rnn(x, hc)
		r_output = out.contiguous().view(-1, 256)
		out1 = self.dropout(r_output)
		out2 = self.fc1(out1)
		out2 = self.tanh(out2)
		out3 = self.fc2(out2)

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
			# feed = self.feedback1(hc4)
			# feed1_arr[i] = feed
			# feed = self.tanh1(feed)
			hc1 = self.tanh2(i2h + h2h)
			out = self.rnn1_h2o(hc1)
			out = self.tanh3(out)
			output_seq1[i] = out

			i2h = self.rnn2_i2h(x[i])
			h2h = self.rnn2_h2h(hc2)
			# feed = self.feedback2(hc4)
			# feed2_arr[i] = feed
			# feed = self.tanh4(feed)
			hc2 = self.tanh5(i2h + h2h)
			out = self.rnn2_h2o(hc2)
			out = self.tanh6(out)
			output_seq2[i] = out

			mid_inp = torch.cat((cue_arr, output_seq1, output_seq2), 2)

			i2h = self.rnn3_i2h(mid_inp[i])
			h2h = self.rnn3_h2h(hc3)
			hc3 = self.tanh7(i2h + h2h)
			out = self.rnn3_h2o(hc3)
			# hc4 = out
			out = self.tanh8(out)
			out_f_arr[i] = out
			out = self.fc(out)
			output_seq[i] = out

		return output_seq, hc1, hc2, hc3, hc4, output_seq1, output_seq2, cue_arr, feed1_arr, feed2_arr, out_f_arr
		# return output_seq.view((16*self.batch_size, -1)), output_seq1.view((16,self.batch_size,64))

class RNN_arch_2(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.rnn1_i2h = nn.Linear(64, 256)
		self.rnn1_h2h = nn.Linear(256,256)
		self.tanh1 = nn.Tanh()
		self.rnn1_h2o = nn.Linear(256, 64)
		self.tanh2 = nn.Tanh()
		self.fc1 = nn.Linear(64, 4)

	def forward(self, x, hc1):
		output_seq = torch.empty((16, self.batch_size, 4))
		for i in range(16):
			i2h = self.rnn1_i2h(x[i])
			h2h = self.rnn1_h2h(hc1)
			hc1 = self.tanh1(i2h + h2h)
			out = self.rnn1_h2o(hc1)
			out = self.tanh2(out)
			out = self.fc1(out)
			output_seq[i] = out

		return output_seq, hc1
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
	model.load_state_dict(torch.load("contrast_difficult_noise.pt", map_location = device))
	model.eval()

	# model = RNN_arch_2(1)
	# model.eval()

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
	# model.fc1.weight.data = model1.fc2.weight.data
	# model.fc1.bias.data = model1.fc2.bias.data	

	hc1 = torch.zeros(1, 1, 256).cuda()

	total = 0
	for i in range(2000):
		data = generate_contrast_data_4outputs()

		x = torch.tensor(data[0])
		y = torch.tensor(data[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		y = y.contiguous().view((1,seq_l)).cuda()

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

# predict_new()

def train():
	model = RNN_arch_2_1(200)
	model.apply(weight_init)
	model.apply(init_weights)
	model.load_state_dict(torch.load("contrast_difficult.pt", map_location = device), strict = False)	
	model.cuda()

	criterion = nn.CrossEntropyLoss().cuda()

	a = []
	for i in range(100000):
		temp = generate_contrast_data_4outputs()
		a.append(temp)

	b = []
	for i in range(25000):
		temp = generate_contrast_data_4outputs()
		b.append(temp)

	count0 = 0

	while(count0<100):
		if(count0 < 70):
			optimizer = optim.Adam(model.parameters(), lr = 0.0005)
		elif(count0 < 140):
			optimizer = optim.Adam(model.parameters(), lr = 0.0003)
		elif(count0 < 210):
			optimizer = optim.Adam(model.parameters(), lr = 0.0001)
		else:
			optimizer = optim.Adam(model.parameters(), lr = 0.00005)

		train_loader = data.DataLoader(a, batch_size=200, shuffle = True, drop_last = True)
		valid_loader = data.DataLoader(b, batch_size=200, drop_last = True)

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
			y_reshaped = y.contiguous().view(3200).cuda()

			print("Epoch: ", count0, "Step: ", count)
			count += 1

			optimizer.zero_grad()
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
		
		model.eval()
		for val_x, val_y in valid_loader:
			val_x = val_x.permute(1,0,2)
			val_y = val_y.permute(1,0)
			val_x, val_y = val_x.cuda(), val_y.cuda()
			val_y_reshaped = val_y.contiguous().view(3200).cuda()

			val_output, _, _ = model(val_x.float(), hc1.float())
			val_output = val_output.contiguous().view(3200, -1)
			valid_loss = criterion(val_output, val_y_reshaped)
			valid_loss = valid_loss.item()

		new_data = {"epochs": [count0], "trainlosses": [train_loss], "vallosses": [valid_loss] }
		doc.add_next_tick_callback(partial(update, new_data))

		torch.save(model.state_dict(), "contrast_difficult_noise.pt")

# thread = Thread(target=train)
# thread.start()