# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:27:09 2018

@author: V510
"""

import numpy as np 
import tensorflow as tf 
import random as rn 

import os 
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(36)

rn.seed(777)

tf.set_random_seed(89)

from keras import backend as K 

sess = tf.Session(graph = tf.get_default_graph())
K.set_session(sess)



from keras.models import Sequential 
from keras.layers import Convolution2D # 2D is for images unlike videos that are 3D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


classifier  = Sequential()

# Step one convolution 

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))

# Step two the pooling step 

classifier.add(MaxPool2D(pool_size = (2, 2)))

# Adding more convolutional layer because I want to increase the accuracy 

classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))

#Add second max pooling 

classifier.add(MaxPool2D(pool_size = (2, 2)))

# Step three Flattening 

classifier.add(Flatten())

# dense connecting layers 
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compile the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Image augmentation -- used typically when we have low amount of images  extracted from the image generator function 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale = 1./255, 
                shear_range = 0.2, 
                zoom_range = 0.2, 
                horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                color_mode = 'grayscale', 
                                                target_size = (64, 64), 
                                                batch_size = 8, 
                                                class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                             color_mode = 'grayscale',
                                                target_size = (64, 64), 
                                                batch_size = 8, 
                                                class_mode = 'binary')

classifier.fit_generator(training_set, 
                        samples_per_epoch = 5216, 
                        nb_epoch = 50, 
                        validation_data = test_set, 
                        nb_val_samples = 624)



# save the model 

from keras.models import load_model

classifier.save('xray_prediction.h5')


# check the labeling of the test set

test_set.class_indices



