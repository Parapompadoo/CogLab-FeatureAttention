import numpy as np
import random
import matplotlib.pyplot as plt

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

def generate_motion_data():
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
		if i in change_frame:
			temp = move(training_datapoint_x[i-1], previous_direction = curr_direction)
			training_datapoint_x[i] = temp[1]
			training_datapoint_y[i] = 1

			curr_direction = temp[0]
		else:
			temp = move(training_datapoint_x[i-1], choice = curr_direction)
			training_datapoint_x[i] = temp[1]

		# if i in change_frame_contrast:
		# 	temp1 = change_contrast(training_datapoint_x[i], diff_from_mean)
		# 	training_datapoint_x[i:] = temp1[1]
		# 	diff_from_mean = temp1[0]

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
	change_frame[1] = change_frame[0] + frames_before_change

	change_frame_contrast = [0, 0]
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[0] = 0+frames_before_change
	frames_before_change = random.randint(5, 7)
	change_frame_contrast[1] = change_frame_contrast[0] + frames_before_change

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
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]

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
	
	# for i in range(16):
		# training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
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
	
	# for i in range(16):
		# training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]


def generate_contrast_data_2outputs():
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
			# training_datapoint_y[i] = 1
			training_datapoint_y[i+1] = 1
			diff_from_mean = temp1[2]
	
	for i in range(16):
		training_datapoint_x[i] = np.interp(training_datapoint_x[i], (0, 255), (-1, 1))
	
	training_datapoint_y = training_datapoint_y.astype(int)

	return [training_datapoint_x, training_datapoint_y]

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
	training_datapoint_y[:] = 0
	training_datapoint_y1[:] = curr_level
	training_datapoint_y2[:] = 7

	for i in range(1,16):
		if i in change_frame:
			temp = move(training_datapoint_x[i-1])
			training_datapoint_x[i:] = temp[1]
			# training_datapoint_y[i] = 1
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

	return [training_datapoint_x, training_datapoint_y]

a = generate_motion_data_2outputs()
for i in range(16):
	print(a[1][i])
	print(a[0][i].reshape((8,8)))
	# plt.show()