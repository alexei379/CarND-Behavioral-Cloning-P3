import csv
import cv2
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Lists to store CSV parsing results
image_paths = []
steering_angles = []

data_path = '../P3_data/combined/'
# keep 30% of the input data images
input_data_drop_factor = 0.7

# Steering angle factor applied to side images
side_correction = 0.15
# Number of random alternations per input image.
# Total number of training images would be (number_of_images * alternations_per_sample)
alternations_per_sample = 5

number_of_epoch = 10
validation_size = 0.2

with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip first line
    
    for line in reader:
        steering_center = float(line[3])
        # adjust steering for left/right images
        steering_left = steering_center + side_correction
        steering_right = steering_center - side_correction
        
        steering_angles.extend([steering_center, steering_left, steering_right])
        
        path_center = line[0]	
        path_left = line[1]
        path_right = line[2]
        
        for path in [path_center, path_left, path_right]:
            filename = path.split('/')[-1]
            current_path = data_path + 'IMG/' + filename            
            image_paths.append(current_path)                		

# Shuffle and split input data into training/validation set - 80%/20%            
image_paths_train, image_paths_test, steering_angles_train, steering_angles_test = \
    train_test_split(image_paths, steering_angles, test_size=validation_size)

del_from_train = int(len(image_paths_train) * input_data_drop_factor)
del_from_validation = int(len(image_paths_test) * input_data_drop_factor)

del image_paths_train[-del_from_train:]
del steering_angles_train[-del_from_train:]
del image_paths_test[-del_from_validation:]
del steering_angles_test[-del_from_validation:]    

# Functions to alter input images
def shadow(image_HSV, top_col, bottom_col, side, factor = 0.5):
    height, width, chanels = image_HSV.shape
        
    pts = np.array([[0, 0], [top_col, 0], [bottom_col, height], [0, height], [0, 0]], dtype=np.int32)
    mask = np.zeros((height, width))    
    cv2.fillConvexPoly(mask, pts, 1)    
    image_HSV[:,:,2] = np.where(mask[:,:] - side == 0, image_HSV[:,:,2], image_HSV[:,:,2] * factor)
    
    return image_HSV
    
def vertical_shift(image, pixels):
    rows, cols, c = image.shape
    M = np.float32([[1, 0, 0], [0, 1, pixels]])
    return cv2.warpAffine(image, M, (cols, rows))

def change_brightness(image_HSV, factor):
    image_HSV[:,:,2] = np.where(image_HSV[:,:,2] * factor > 255, 255, image_HSV[:,:,2] * factor)
    return image_HSV

# prepocessing function for random images alternations & conversion to YUV
def preprocess_image(image_path, angle):
    # read image
    image_BGR = cv2.imread(image_path)
    
    # flip image
    if np.random.rand() < 0.5:
        image_BGR = cv2.flip(image_BGR, 1)
        angle = -angle
    
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    
    # alter brightness
    brightness_factor = 0.5 + np.random.uniform()
    bright_image_HSV = change_brightness(image_HSV, brightness_factor)
    
    # add shaddow
    shadow_factor = np.random.uniform(low=0.3, high=0.7)
    shadow_side = np.random.randint(2)
    shadow_top = np.random.randint(320)
    shadow_bottom = np.random.randint(320)
    shadow_image_HSV = shadow(bright_image_HSV, shadow_top, shadow_bottom, shadow_side, shadow_factor)
    
    # convert to YUV
    image_BGR = cv2.cvtColor(shadow_image_HSV, cv2.COLOR_HSV2BGR)
    shadow_image_YUV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2YUV)
    
    # vertical shift
    shift_factor = 25 - np.random.uniform() * 50
    shift_image = vertical_shift(shadow_image_YUV, shift_factor)
    
    return shift_image, angle

# Generator to be used by Keras model
def generator(X, y, batch_size=32):    
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            batch_samples = X[offset:offset+batch_size]
            batch_values = y[offset:offset+batch_size]
            
            images = []
            angles = []
            for path, angle in zip(batch_samples, batch_values):
                image, angle = preprocess_image(path, angle) 
                images.append(image)
                angles.append(angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(image_paths_train, steering_angles_train, batch_size=256)
validation_generator = generator(image_paths_test, steering_angles_test, batch_size=256)

# Creating CNN model similar to NVIDIA model
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import CSVLogger

model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Cropping "sky" (top) and "car hood" (bottom)
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Fully connected layers
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Mean Squared Error / Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Generate model viszualization
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png', show_shapes=True)

csv_logger = CSVLogger('training.log')

# Train model and save to model.h5
model.fit_generator(train_generator,
            samples_per_epoch=len(image_paths_train) * alternations_per_sample, 
            validation_data=validation_generator,
            nb_val_samples=len(image_paths_test) * alternations_per_sample,            
            nb_epoch=number_of_epoch, callbacks=[csv_logger])

model.save('model.h5')
