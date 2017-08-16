import csv
import cv2
import numpy as np
import sklearn

from random import shuffle
from sklearn.model_selection import train_test_split

lines = []

alternations_per_sample = 6
data_path = '../P3_data/track_2/'
with open(data_path + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader) # skip first line
	for line in reader:
		lines.append(line)		

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def add_image(data_path, source_path, steering, images, measurements):	
    filename = source_path.split('/')[-1]

    current_path = data_path + 'IMG/' + filename
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2YUV)
    height, width = image.shape[:2]
    image_resized = cv2.resize(image, (200, height), interpolation = cv2.INTER_CUBIC)
    images.append(image_resized)
    measurements.append(steering)

    # flip image
    images.append(cv2.flip(image_resized, 1))
    measurements.append(-steering)	

# returns batch_size * alternations_per_sample images
def generator(samples, batch_size=32):    
    global data_path
    side_correction = 0.15
	
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:			
                steering_center = float(line[3])
                path_center = line[0]	
                add_image(data_path, path_center, steering_center, images, angles)

                # side images
                steering_left = steering_center + side_correction
                path_left = line[1]
                add_image(data_path, path_left, steering_left, images, angles)

                steering_right = steering_center - side_correction
                path_right = line[2]	
                add_image(data_path, path_right, steering_right, images, angles)				

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#for i in range(0, 100):
    #x, y = next(train_generator)
    #print(x[0])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,200,3)))
model.add(Cropping2D(cropping=((69,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dense(1152, activation='relu')) # ??
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5)) # ??
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

model.fit_generator(train_generator,
            samples_per_epoch=len(train_samples) * alternations_per_sample, 
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples) * alternations_per_sample,            
            nb_epoch=5)

model.save('model.h5')