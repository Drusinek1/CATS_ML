  
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""

import numpy as np
import neural_nets
from matplotlib import pyplot as plt
import pdb

from lidar import get_a_color_map


def add_pad_to_vertical(small_array, target_height):
    print('Size of input array is: ', small_array.shape)
    # mall_array_width = small_array.shape[0]
    small_array_height = small_array.shape[0]
    pads_to_add = target_height - small_array_height
    padding = np.zeros(small_array.shape)
    if len(padding.shape) == 4:
        # four dimensional array
        padding = padding[0:pads_to_add, :, :, :]
    else:
        if len(padding.shape) == 3:
            # three dimensional array
            padding = padding[0:pads_to_add, :, :]
        else:
            # two dimensional array
            padding = padding[0:pads_to_add, :]

    sized_array = np.append(padding, small_array, axis=0)
    print('Size of new array is ', sized_array.shape)
    return sized_array


def image_plot(im_array, scale_min, scale_max):
    plt.imshow(im_array, cmap=get_a_color_map(), vmin=scale_min, vmax=scale_max)
    return None


"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

# loading .npy files created by make_training_set.py
X = np.load('questions.npy', allow_pickle=True)
# X = np.transpose(X, (0, 3, 2, 1))
Y = np.load('answers.npy', allow_pickle=True)

Xt = np.load('test_questions.npy', allow_pickle=True)

Yt = np.load('test_answers.npy', allow_pickle=True)
# Squeeze is no longer necessary 5/28/2020 after update of make_training_set.py
# Yt = np.squeeze(Yt)
# Yt[Yt < 0] = 0
# Yt = np.argmax(Yt, axis=2)
# UNet will expect this input shape for all training data
new_X = np.transpose(X, (3, 1, 2, 0))
full_X = add_pad_to_vertical(new_X, 512)
# (500, 256, 3, 362)
new_Y = np.transpose(Y, (2, 1, 0))
full_Y = add_pad_to_vertical(new_Y, 512)
# (500, 256, 362)
model_shape = (512, 256, 1)       # changed to 256 to match image sizes and dimensions to match their descriptors
# model_shape = (512, 16384, 3)
# features is number of categories in the L2 data
CNN = neural_nets.UNet_binary2(model_shape, features=1, filters=16)
# CNN = neural_nets.UNet_binary2(model_shape, features=6, filters=16)       Used for when we do typing!!
# Move the number of samples to the first dimension
full_X = np.transpose(full_X, [3, 0, 1, 2])
full_Y = np.transpose(full_Y, [2, 0, 1])
# Separate out the channels in the test group individually
channel_1 = full_X[:, :, :, 0]
channel_2 = full_X[:, :, :, 1]
channel_3 = full_X[:, :, :, 2]
# Add fourth dimension to answer if it is not present
full_Y = np.reshape(full_Y, (full_Y.shape[0], full_Y.shape[1], full_Y.shape[2], 1))
# Add fourth dimension to individual channels as required by the model
channel_1 = np.reshape(channel_1, (channel_1.shape[0], channel_1.shape[1], channel_1.shape[2], 1))
channel_2 = np.reshape(channel_2, (channel_2.shape[0], channel_2.shape[1], channel_2.shape[2], 1))
channel_3 = np.reshape(channel_3, (channel_3.shape[0], channel_3.shape[1], channel_3.shape[2], 1))
pdb.set_trace()
# Use Channel 11 to train the model
CNN.model.fit(channel_1, full_Y, epochs=3, verbose=1, validation_split=0.2)
# Use Channel 12 to train the model
CNN.model.fit(channel_2, full_Y, epochs=3, verbose=1, validation_split=0.2)
# Use the data which is the product of the two channels to train the model
CNN.model.fit(channel_3, full_Y, epochs=3, verbose=1, validation_split=0.2)
# No longer run line below
# CNN.model.fit(full_X, full_Y, epochs=3, verbose=1, validation_split=0.2)

new_Xt = np.transpose(Xt, (3, 1, 2, 0))
full_Xt = add_pad_to_vertical(new_Xt, 512)
new_Yt = np.transpose(Yt, (2, 1, 0))
full_Yt = add_pad_to_vertical(new_Yt, 512)

# Still need to rearrange and separate channels for validation set

pred = CNN.model.predict(full_Xt)
pdb.set_trace()
pred = np.squeeze(pred)

arr = np.argmax(pred, axis=2)
arr = arr.astype(int)
image_plot(arr, 0, 100)
