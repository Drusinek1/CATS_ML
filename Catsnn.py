  
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""
import numpy as np
import neural_nets
from matplotlib import pyplot as plt
import tensorflow as tf
import pdb

import lidar
import initializations


def plot(arr):
    plt.imshow(arr, cmap=lidar.get_a_color_map(),vmin=0, vmax=100)
    return None


"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

#loading .npy files created by make_training_set.py
X = np.load('questions.npy', allow_pickle=True)
#X = np.transpose(X, (0,3,2,1))
Y = np.load('answers.npy', allow_pickle=True)


Xt = np.load('test_questions.npy', allow_pickle=True)[:,:,:,:]


Yt = np.load('test_answers.npy', allow_pickle=True)
Yt = np.squeeze(Yt)
Yt[Yt < 0] = 0
Yt = np.argmax(Yt, axis=2)

#UNet will expect this input shape for all training data
model_shape = (512,16384,3)
#features is number of categories in the L2 data 
CNN = neural_nets.UNet(model_shape, features=6, filters=16)
CNN.model.fit(X, Y, epochs=3, verbose=1, validation_split=0.2)


pred = CNN.model.predict(Xt)
pred = np.squeeze(pred)

arr = np.argmax(pred, axis=2)
arr = arr.astype(int)
plt.imshow(arr, cmap=lidar.get_a_color_map(),vmin=0,vmax=100)
#plt.imshow(Yt, cmap=lidar.get_a_color_map(),vmin=0,vmax=100)