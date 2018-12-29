#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:59:52 2018

@author: adminuser
"""


# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64)) # this will be updated if size of image is to be changed
test_image = image.img_to_array(test_image) # to put our image into 3dimension of colors
test_image = np.expand_dims(test_image, axis = 0) #extra dimension is need because of epochs
result = classifier.predict(test_image) # save result into variable
training_set.class_indices 
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'