import csv
import cv2
import numpy as np
import random

#random.seed(42)

lines = []
data_path = '../P3_data/my_data/'
with open(data_path + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)		
	
images = []
measurements = []

side_correction = 0.15 # this is a parameter to tune

def add_image(data_path, source_path, steering, images, measurements):	
	source_path = line[0]
	filename = source_path.split('/')[-1]
	
	current_path = data_path + 'IMG/' + filename
	image = cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2YUV)
	images.append(image)
	measurements.append(steering)
	
	# flip image
	images.append(cv2.flip(image, 1))
	measurements.append(-steering)	
	

for line in lines:
	steering_center = float(line[3])
	path_center = line[0]
	#if abs(steering_center) >= 0.1 or random.random() < 0.3:
	add_image(data_path, path_center, steering_center, images, measurements)

	# side images
	steering_left = steering_center + side_correction
	path_left = line[1]
	add_image(data_path, path_left, steering_left, images, measurements)

	steering_right = steering_center - side_correction
	path_right = line[2]	
	add_image(data_path, path_right, steering_right, images, measurements)	


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((69,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


'''
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''


'''
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))
'''


model.compile(loss='mse', optimizer='adam')

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, batch_size=256)

model.save('model.h5')