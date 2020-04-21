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
# from bokeh.io import curdoc
# from bokeh.layouts import column
# from bokeh.models import ColumnDataSource
# from bokeh.plotting import figure
# from functools import partial
# from threading import Thread
# from tornado import gen
# import torch.nn.init as init
# import gc
# import sys
# from PIL import Image


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




def generate_data():
	seq_l = 16
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	training_datapoint_y_2 = np.zeros(seq_l)

	# number_of_changes = random.randint(2, (seq_l -1))
	number_of_changes = 4
	change_frame = random.sample(range(1, seq_l), number_of_changes)

	# number_of_changes = random.randint(2, (seq_l -1))
	number_of_changes = 5
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


def generate_data_new():
	seq_l = 12000
	input_size = 64
	training_datapoint_x = np.zeros((seq_l, input_size))
	training_datapoint_y = np.zeros(seq_l)
	# training_datapoint_y_2 = np.zeros(seq_l)

	# number_of_changes = random.randint(2, (seq_l -1))
	# number_of_changes = 4
	# change_frame = random.sample(range(1, seq_l), number_of_changes)

	# number_of_changes = random.randint(2, (seq_l -1))
	# number_of_changes = 5
	# change_frame_contrast = random.sample(range(1, seq_l), number_of_changes)

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
	# training_datapoint_y_2[:] = 7

	temp = training_datapoint_x[0].copy()
	temp = change_contrast1(temp)
	training_datapoint_x[1500:3000] = temp[1]
	training_datapoint_x[4500:6000] = temp[1]
	training_datapoint_y[1500:3000] = temp[0]
	training_datapoint_y[4500:6000] = temp[0]

	temp = training_datapoint_x[0].copy()
	temp = move(temp)
	training_datapoint_x[7500:9000] = temp[1]
	training_datapoint_x[10500:12000] = temp[1]



	# for i in range(1,seq_l):
	# 	if i in change_frame:
	# 		temp = move(training_datapoint_x[i-1])
	# 		training_datapoint_x[i:] = temp[1]
	# 		training_datapoint_y_2[i] = temp[0]
	# 		curr_direction = temp[0]

	# 	if i in change_frame_contrast:
	# 		temp1 = change_contrast1(training_datapoint_x[i], previous_level = curr_level)
	# 		training_datapoint_x[i:] = temp1[1]
	# 		training_datapoint_y[i:] = temp1[0]
	# 		diff_from_mean = temp1[2]
	
	noise = np.random.normal(0, 0.05, (1500, 64))

	for i in range(seq_l):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))

	for k in range(8):
		for i in range(1500):
			for j in range(64):
				# if(noise[i][j]+training_datapoint_x[k*1500 + i][j]<=1 and noise[i][j]+training_datapoint_x[k*1500 + i][j]>=-1):
				training_datapoint_x[k*1500 + i, j] += noise[i][j]




	# training_datapoint_y = training_datapoint_y.astype(int)
	# training_datapoint_y_2 = training_datapoint_y_2.astype(int)

	return [training_datapoint_x, training_datapoint_y]


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
		self.fc0 = nn.Linear(130,64)
		self.fc1 = nn.Linear(64, 64)
		self.tanh1 = nn.ReLU()
		self.fc2 = nn.Linear(64, 64)
		self.tanh2 = nn.ReLU()
		self.fc3 = nn.Linear(64, 64)
		self.tanh3 = nn.ReLU()
		self.fc4 = nn.Linear(64, 8)

	def forward(self, x, hc):
		x1 = self.fc0(x)
		x2 = self.fc1(x1)
		x2 = self.tanh1(x2)
		x3 = self.fc2(x2)
		x3 = self.tanh2(x3)
		x4 = self.fc3(x3)
		x4 = self.tanh3(x4)
		x5 = self.fc4(x4)

		return x5, x4, x3, x2

def predict_new():
	seq_l = 3000
	model1 = RNN_arch_2_1(1)
	for param in model1.parameters():
		param.requires_grad = False
	model2 = RNN_arch_2_1(1)
	for param in model2.parameters():
		param.requires_grad = False
	model1.load_state_dict(torch.load("contrast_state_newer.pt", map_location = device))
	model2.load_state_dict(torch.load("motion_state_newer.pt", map_location = device))
	model1.eval()
	model2.eval()

	model = RNN_arch_2_final(1)
	for param in model.parameters():
		param.requires_grad = False
	model.load_state_dict(torch.load("godspeed_hope.pt", map_location = device))
	model.eval()

	hc1 = torch.zeros(1, 1, 256).cuda()
	hc2 = torch.zeros(1, 1, 256).cuda()
	hc3 = torch.zeros(1, 1, 256).cuda()
	temp_x = np.zeros((12000,64))
	temp_x2 = np.zeros((12000,64))
	temp_x3 = np.zeros((12000,64))
	temp_x4 = np.zeros((12000,64))

	for i in range(1):
		data = generate_data_new()
		# temp = data[0]
		temp = data[0, :3000]
		temp1 = data[1, :3000]

		x = torch.tensor(temp)
		# y = torch.tensor(temp[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		# y = y.contiguous().view((1,seq_l)).cuda()

		final_input = torch.zeros((seq_l,130))

		contrast_out, _, output1 = model1(x.float(), hc1.float())
		motion_out, _, output2 = model2(x.float(), hc2.float())

		cue_arr = torch.zeros((seq_l, 2))
		for i in range(seq_l):
			cue = 0
			# if(i>=3000 and i<9000):
				# cue = 1
			# else: 
				# cue = 0
			anti_cue = (1 - cue)
			cue_arr[i, 0] = cue
			cue_arr[i, 1] = anti_cue

		final_input[:, 2:66] = output1
		final_input[:, 66:130] = output2
		final_input[:, 0:2] = cue_arr
		final_input = final_input.cuda()

		x5, x4, x3, x2 = model(final_input.float(), hc3.float())

		x4, x3, x2 = x4.cpu(), x3.cpu(), x2.cpu()
		x4, x3, x2 = x4.numpy(), x3.numpy(), x2.numpy()
		x = x.cpu()
		x = x.view((seq_l, 64))
		x = x.numpy()

		temp_x[:3000] = x
		temp_x2[:3000] = x2
		temp_x3[:3000] = x3
		temp_x4[:3000] = x4







		temp = data[0, 3000:6000]
		temp1 = data[1, 3000:6000]

		x = torch.tensor(temp)
		# y = torch.tensor(temp[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		# y = y.contiguous().view((1,seq_l)).cuda()

		final_input = torch.zeros((seq_l,130))

		contrast_out, _, output1 = model1(x.float(), hc1.float())
		motion_out, _, output2 = model2(x.float(), hc2.float())

		cue_arr = torch.zeros((seq_l, 2))
		for i in range(seq_l):
			cue = 1
			# if(i>=3000 and i<9000):
				# cue = 1
			# else: 
				# cue = 0
			anti_cue = (1 - cue)
			cue_arr[i, 0] = cue
			cue_arr[i, 1] = anti_cue

		final_input[:, 2:66] = output1
		final_input[:, 66:130] = output2
		final_input[:, 0:2] = cue_arr
		final_input = final_input.cuda()

		x5, x4, x3, x2 = model(final_input.float(), hc3.float())


		x4, x3, x2 = x4.cpu(), x3.cpu(), x2.cpu()
		x4, x3, x2 = x4.numpy(), x3.numpy(), x2.numpy()
		x = x.cpu()
		x = x.view((seq_l, 64))
		x = x.numpy()

		temp_x[3000:6000] = x
		temp_x2[3000:6000] = x2
		temp_x3[3000:6000] = x3
		temp_x4[3000:6000] = x4




		temp = data[0, 6000:9000]

		x = torch.tensor(temp)
		# y = torch.tensor(temp[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		# y = y.contiguous().view((1,seq_l)).cuda()

		final_input = torch.zeros((seq_l,130))

		contrast_out, _, output1 = model1(x.float(), hc1.float())
		motion_out, _, output2 = model2(x.float(), hc2.float())

		cue_arr = torch.zeros((seq_l, 2))
		for i in range(seq_l):
			cue = 1
			# if(i>=3000 and i<9000):
				# cue = 1
			# else: 
				# cue = 0
			anti_cue = (1 - cue)
			cue_arr[i, 0] = cue
			cue_arr[i, 1] = anti_cue

		final_input[:, 2:66] = output1
		final_input[:, 66:130] = output2
		final_input[:, 0:2] = cue_arr
		final_input = final_input.cuda()

		x5, x4, x3, x2 = model(final_input.float(), hc3.float())


		x4, x3, x2 = x4.cpu(), x3.cpu(), x2.cpu()
		x4, x3, x2 = x4.numpy(), x3.numpy(), x2.numpy()
		x = x.cpu()
		x = x.view((seq_l, 64))
		x = x.numpy()

		temp_x[6000:9000] = x
		temp_x2[6000:9000] = x2
		temp_x3[6000:9000] = x3
		temp_x4[6000:9000] = x4





		temp = data[0, 9000:12000]

		x = torch.tensor(temp)
		# y = torch.tensor(temp[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		# y = y.contiguous().view((1,seq_l)).cuda()

		final_input = torch.zeros((seq_l,130))

		contrast_out, _, output1 = model1(x.float(), hc1.float())
		motion_out, _, output2 = model2(x.float(), hc2.float())

		cue_arr = torch.zeros((seq_l, 2))
		for i in range(seq_l):
			cue = 0
			# if(i>=3000 and i<9000):
				# cue = 1
			# else: 
				# cue = 0
			anti_cue = (1 - cue)
			cue_arr[i, 0] = cue
			cue_arr[i, 1] = anti_cue

		final_input[:, 2:66] = output1
		final_input[:, 66:130] = output2
		final_input[:, 0:2] = cue_arr
		final_input = final_input.cuda()

		x5, x4, x3, x2 = model(final_input.float(), hc3.float())

		x4, x3, x2 = x4.cpu(), x3.cpu(), x2.cpu()
		x4, x3, x2 = x4.numpy(), x3.numpy(), x2.numpy()
		x = x.cpu()
		x = x.view((seq_l, 64))
		x = x.numpy()

		temp_x[9000:12000] = x
		temp_x2[9000:12000] = x2
		temp_x3[9000:12000] = x3
		temp_x4[9000:12000] = x4

		# x, x4, x3, x2, output1, output2 = x.reshape((seq_l, 8, 8)), x4.reshape((seq_l, 8, 8)), x3.reshape((seq_l, 8, 8)), x2.reshape((seq_l, 8, 8)), output1.reshape((seq_l, 8, 8)), output2.reshape((seq_l, 8, 8))

		print(temp_x[:1500] - temp_x[9000:10500])

		print(temp_x.shape)
		np.save("x", temp_x)
		print(temp_x2.shape)
		np.save("x2", temp_x2)
		print(temp_x3.shape)
		np.save("x3", temp_x3)
		print(temp_x4.shape)
		np.save("x4", temp_x4)


		# fig, axs = plt.subplots(10, 16, sharey=True, gridspec_kw={'hspace': 0})
		# for i in range(16):
		# 	axs[0][i].pcolor(x[i].reshape(8,8))
		# 	axs[0][i].axis('off')

		# 	axs[1][i].pcolor(cue_arr[i].reshape(1,2))
		# 	axs[1][i].axis('off')
			
		# 	axs[2][i].pcolor(output1[i].reshape(8,8))
		# 	axs[2][i].axis('off')
			
		# 	axs[3][i].pcolor(output2[i].reshape(8,8))
		# 	axs[3][i].axis('off')
			
		# 	axs[4][i].pcolor(contrast_out[i].reshape(1,8))
		# 	axs[4][i].axis('off')
			
		# 	axs[5][i].pcolor(motion_out[i].reshape(1,8))
		# 	axs[5][i].axis('off')
			
		# 	axs[6][i].pcolor(x2[i].reshape(8,8))
		# 	axs[6][i].axis('off')
			
		# 	axs[7][i].pcolor(x3[i].reshape(8,8))
		# 	axs[7][i].axis('off')
			
		# 	axs[8][i].pcolor(x4[i].reshape(8,8))
		# 	axs[8][i].axis('off')
			
		# 	axs[9][i].pcolor(x5[i].reshape(1,8))
		# 	axs[9][i].axis('off')

		# plt.show()

		# final_input[:, 0] = 0
		# final_input[:, 1] = 1

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[1] += np.sum(penult, 0)


		
		# final_input[:, 0] = 0
		# final_input[:, 1] = 0

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[2] += np.sum(penult, 0)



		# final_input[:, 0] = 1
		# final_input[:, 1] = 1

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[3] += np.sum(penult, 0)

	# np.save("4_sanchit.npy", penult_total)

predict_new()

def predict():
	seq_l = 16
	model1 = RNN_arch_2_1(1)
	for param in model1.parameters():
		param.requires_grad = False
	model2 = RNN_arch_2_1(1)
	for param in model2.parameters():
		param.requires_grad = False
	model1.load_state_dict(torch.load("contrast_state_newer.pt", map_location = device))
	model2.load_state_dict(torch.load("motion_state_newer.pt", map_location = device))
	model1.eval()
	model2.eval()

	model = RNN_arch_2_final(1)
	for param in model.parameters():
		param.requires_grad = False
	model.load_state_dict(torch.load("godspeed_hope.pt", map_location = device))
	model.eval()

	hc1 = torch.zeros(1, 1, 256).cuda()
	hc2 = torch.zeros(1, 1, 256).cuda()
	hc3 = torch.zeros(1, 1, 256).cuda()

	penult_total = np.zeros((1,64))
	divide_arr = np.zeros((4,8,1))

	for i in range(1):
		data = generate_data()
		temp = data[0]

		x = torch.tensor(temp[0])
		y = torch.tensor(temp[1])
		x = x.contiguous().view((seq_l,1,64)).cuda()
		y = y.contiguous().view((1,seq_l)).cuda()

		final_input = torch.zeros((16,130))

		contrast_out, _, output1 = model1(x.float(), hc1.float())
		motion_out, _, output2 = model2(x.float(), hc2.float())

		cue_arr = torch.zeros((16, 2))
		for i in range(seq_l):
			if(i<8):
				cue = 1
			else: 
				cue = 0
			anti_cue = (1 - cue)
			cue_arr[i, 0] = cue
			cue_arr[i, 1] = anti_cue

		final_input[:, 2:66] = output1
		final_input[:, 66:130] = output2
		final_input[:, 0:2] = cue_arr
		final_input = final_input.cuda()

		x5, x4, x3, x2 = model(final_input.float(), hc3.float())
		x5, x4, x3, x2, contrast_out, motion_out, output1, output2 = x5.cpu(), x4.cpu(), x3.cpu(), x2.cpu(), contrast_out.cpu(), motion_out.cpu(), output1.cpu(), output2.cpu()
		x5, x4, x3, x2, contrast_out, motion_out, output1, output2 = x5.numpy(), x4.numpy(), x3.numpy(), x2.numpy(), contrast_out.numpy(), motion_out.numpy(), output1.numpy(), output2.numpy()
		x = x.cpu()
		x = x.view((seq_l, 64))
		x = x.numpy()
		cue_arr = cue_arr.cpu()
		cue_arr = cue_arr.numpy()

		x, x4, x3, x2, output1, output2 = x.reshape((seq_l, 8, 8)), x4.reshape((seq_l, 8, 8)), x3.reshape((seq_l, 8, 8)), x2.reshape((seq_l, 8, 8)), output1.reshape((seq_l, 8, 8)), output2.reshape((seq_l, 8, 8))

		print(cue_arr.shape)
		np.save("cue_arr", cue_arr)
		print(x.shape)
		np.save("x", x)
		print(output1.shape)
		np.save("output1", output1)
		print(output2.shape)
		np.save("output2", output2)
		print(contrast_out.shape)
		np.save("contrast_out", contrast_out)
		print(motion_out.shape)
		np.save("motion_out", motion_out)
		print(x2.shape)
		np.save("x2", x2)
		print(x3.shape)
		np.save("x3", x3)
		print(x4.shape)
		np.save("x4", x4)
		print(x5.shape)
		np.save("x5", x5)

		fig, axs = plt.subplots(10, 16, sharey=True, gridspec_kw={'hspace': 0})
		for i in range(16):
			axs[0][i].pcolor(x[i].reshape(8,8))
			axs[0][i].axis('off')

			axs[1][i].pcolor(cue_arr[i].reshape(1,2))
			axs[1][i].axis('off')
			
			axs[2][i].pcolor(output1[i].reshape(8,8))
			axs[2][i].axis('off')
			
			axs[3][i].pcolor(output2[i].reshape(8,8))
			axs[3][i].axis('off')
			
			axs[4][i].pcolor(contrast_out[i].reshape(1,8))
			axs[4][i].axis('off')
			
			axs[5][i].pcolor(motion_out[i].reshape(1,8))
			axs[5][i].axis('off')
			
			axs[6][i].pcolor(x2[i].reshape(8,8))
			axs[6][i].axis('off')
			
			axs[7][i].pcolor(x3[i].reshape(8,8))
			axs[7][i].axis('off')
			
			axs[8][i].pcolor(x4[i].reshape(8,8))
			axs[8][i].axis('off')
			
			axs[9][i].pcolor(x5[i].reshape(1,8))
			axs[9][i].axis('off')

		plt.show()

		# final_input[:, 0] = 0
		# final_input[:, 1] = 1

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[1] += np.sum(penult, 0)


		
		# final_input[:, 0] = 0
		# final_input[:, 1] = 0

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[2] += np.sum(penult, 0)



		# final_input[:, 0] = 1
		# final_input[:, 1] = 1

		# hc3 = torch.zeros(1, 1, 256).cuda()
		# _, penult = model(final_input.float(), hc3.float())
		# penult = penult.cpu()
		# penult = penult.numpy()
		
		# penult_total[3] += np.sum(penult, 0)

	# np.save("4_sanchit.npy", penult_total)

# predict()