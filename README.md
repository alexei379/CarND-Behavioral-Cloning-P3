# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I've built deep convolutional neural network to clone driving behavior. 
[Project structure](https://github.com/alexei379/CarND-Behavioral-Cloning-P3) and [simulator](https://github.com/udacity/self-driving-car-sim) are provided by [Udacity - Self-Driving Car NanoDegree](http://www.udacity.com/drive).

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (this file) summarizing the results
* Videos (YouTube links)
    * Track 1 <br/>
    [![Track 1](https://img.youtube.com/vi/PVLBKvBOblQ/0.jpg)](https://www.youtube.com/watch?v=PVLBKvBOblQ)
    * Track 2 (Jungle)<br/>
    [![Track 2](https://img.youtube.com/vi/xCddK6yXX2Q/0.jpg)](https://www.youtube.com/watch?v=xCddK6yXX2Q)


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

I updated the drive.py file to feed YUV images to the model(https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/1d5843ea3d1446f7a218e277ed5c09a2cb1a065b/drive.py#L65). Also I made the speed [depent on the steering angle](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/1d5843ea3d1446f7a218e277ed5c09a2cb1a065b/drive.py#L68) and make throttling a bit [more aggressive](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/1d5843ea3d1446f7a218e277ed5c09a2cb1a065b/drive.py#L48).

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture is similar to [NVIDIA's CNN architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

My model consists of 5 convolutional layers and 3 fully connected layers (model.py [lines 135-163](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L135)). RELU used as activation fuction to introduce nonlinearity. The input image is split into YUV planes and passed to the network, where first layers perform normalization and cropping.

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines [152](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L152) and [159](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L159)). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py [line 48](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L159)).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py [line 166](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L166)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used data provided by Udacity for the "Track 1" as a starting point. I extended it by recording extra data for going back and forth through turns that the model could not handle and recording "recovery" only going from the side of the road to the center. Unfortunately, using just data from "Track 1" even with multiple alternations was not sufficient to generalize to drive "Track 2" (Jungle) autonomously, so I ended up repeating the same procedure for it.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try a simple model to verify that a pipline works as expected and then move on to experementing with a model described by NVIDIA.

To verify that a pipline works I implemented a single "Flatten" layer model and used images provided by Udacity to train this model. After the training I verified that I can connect driving.py with the simulator to drive the car. The steering was far from ideal but the pipline was working. 

After verifying that the pipline works, I moved on to implement the the NVIDIA's architecture. The trained model was able to drive around "Track 1", but had issues with couple of turns. To improve the driving behaviour I collected additional data by driving through these turns and doing "recovery" driving from the border to the center. Using this additional data I was able to train the model to drive around "Track 1".  

Simulator outputs images of size 320x160. NVIDIA's architecture uses 200x66 images. I tried scaling image width down in model.py/drive.py, but the resulting model was making more zig-zags, so I decided to keep 320x160 image sizes.

I tried this trained model on "Track 2" and it was not able to control the car appropriately. I thought that alternating data from "Track 1" by shifting, adding shadows, changing the brightnes, and addind dropout layers will help to generalize model to drive around "Track 2" without collecting the data from "Track 2". The trained model was able to drive better on "Track 2", but it still was not enougth to drive safely. 

I ended up collecting driving data from "Track 2" and repeating the same process with fixing couple of turns that were not working correctly.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py [lines 135-163](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L135)) consisted of normalization lambda and cropping layers, followed by three three convolutional layers with a 2×2 stride and a 5×5 kernel, dropout layer with keep probability = 0.5, followed by two non-strided convolutional layers with a 3×3 kernel size, followed by a set of fully connected layers with a single steering output. Visualization of the architecture below contains the layer sizes.

| NVIDIA's architecture         		|     My architecture	        					| 
|:---------------------:|:---------------------------------------------:| 
| ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/nvidia-cnn-architecture.png) | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/keras_model.png) |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used sample training images provided by Udacity for the "Track 1" as a starting point. The data contains views from 3 "cameras": center, left and right. Here are sample images of center lane driving:

| Left | Center | Right | 
|:----:|:------:|:-----:|
| ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/left_2016_12_01_13_30_48_287.jpg) | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/center_2016_12_01_13_30_48_287.jpg) | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/right_2016_12_01_13_30_48_287.jpg) |

I used left and right images as additional trainig data adjusting steering angle by 0.15 (model.py [lines 31-36](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L31)

After training the model and identifying the turns on "Track 1" where the model was not steering well, I recorded additional data on going through the turns and recovering from the left side and right sides of the road back to center:

![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/recovery_1.gif)
![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/recovery_2.gif)

My first attempt was to augment data set from "Track 1" so that the model would be able to generalize for "Track 2". I implemented the following augmentations, parameters for which were randomly chosed (model.py [lines 80-110](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L80)) and applied to the original images:

| Augmentation | Sample Output |
|:--------:|:------:|
| Original | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/original.png) |
| [flip](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L84) | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/flipped.png) |
| [change_brightness](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L75) | factor=1.4 <br/> ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/lighter_1.4.png) |
| [shadow](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L60) | top_col=32, bottom_col=219, side=0, factor=0.33 <br/>![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/shadow_0.33_0_32_219.png) |
| [vertical_shift](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/f5dee4a10da2428d16013460d91b2a80bb7af0f3/model.py#L70) | pixels=15 <br/> ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/shift_15.8.png) |
| Result | ![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/combined_alternations.png) |

Unfortunately, it was not enougth so I repeated this process on track two in order to get more data points. Another technique I used on "Track 2" to train the model in difficult turns was going through them in both directions several times:
![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/recovery_3.gif)
![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/recovery_4.gif)

After the collection process, the number of sample images I had was 34362 for "Track 1" and 36756 for "Track 2" . 
I ended up having a lot of training data so I used gerarator (model.py [line 113](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L113)) to feed it into the model instead of loading everything in the memory.

I introduced an [alternations_per_sample = 5](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L175) factor, to generate multiple image alternations when configuring samples_per_epoch (model.py [line 175](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L175)). 

This results in over 355000 images for training/validation set. I randomly shuffle the data set and put 20% into avalidation set (model.py [line 48](https://github.com/alexei379/CarND-Behavioral-Cloning-P3/blob/4b0481294dd795bca64a8b178efb2dd38a26665b/model.py#L48)). Also I decided to randomly drop 70% of data to reduce the training time.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss plateauing.

![](https://raw.githubusercontent.com/alexei379/CarND-Behavioral-Cloning-P3/master/report_images/train_val_loss.PNG)

I used an adam optimizer so that manually tuning the learning rate wasn't necessary.
