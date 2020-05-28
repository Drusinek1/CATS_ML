  
# -*- coding: utf-8 -*-

"""
Created on Mon Apr  6 08:35:13 2020
@author: Daniel Rusinek

This code imports pre-made training and testing data,and runs it through a
binary classification convolutional neural network.


Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""

from datetime import datetime
import numpy as np
import neural_nets
from matplotlib import pyplot as plt
import tensorflow as tf
import pdb

import lidar
import initializations
import cv2



def plot(X,dims):
    """
    Parameters
    ----------
    X : ndarray
        Image to plot
    dims: number of dimensions in the image to be plotted
    Note when plotting L0 data use format plot(L0_data[0],3). Calling
    plot(L0_data,3) will result in an error.
    
    Returns
    -------
    None

    """
    if dims == 3:
        plt.figure()
        plt.title("X_train Channel 1")
        plt.imshow(X[:,:,0].T, aspect='auto', cmap=lidar.get_a_color_map(), vmin=0, vmax=100)
        plt.figure()
        plt.title("X_train Channel 2")
        plt.imshow(X[:,:,1].T, aspect='auto', cmap=lidar.get_a_color_map(), vmin=0, vmax=100)
   
        plt.figure()
        plt.title("X_train Channel 3")
        plt.imshow(X[:,:,2].T, aspect='auto', cmap=lidar.get_a_color_map(), vmin=0, vmax=100)
        plt.show()
    
    else:
        plt.figure()
        plt.title("Predicted")
        plt.imshow(X[:,:].T, aspect='auto', cmap=lidar.get_a_color_map(), vmin=0, vmax=100)
    
    return None



start = datetime.now()

# """
# Formatting Input Training Data
# """
# path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

# #loading .npy files created by make_training_set.py


start = datetime.now()

"""
Formatting Input Training Data
"""


path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

"""
loading .npy files created by make_training_set.py
"""

#load training input
X = np.load('questions.npy')

Y = np.load('answers.npy')

#load X test sample
Xt = np.load('test_questions.npy')[:,:,:,:]


Yt = np.load('test_answers.npy')

#remove extra 1 dimension

path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

# loading .npy files created by make_training_set.py
X = np.load('questions.npy', allow_pickle=True)
# X = np.transpose(X, (0,3,2,1))
Y = np.load('answers.npy', allow_pickle=True)


Xt = np.load('test_questions.npy', allow_pickle=True)[:, :, :, :]


Yt = np.load('test_answers.npy', allow_pickle=True)

Yt = np.squeeze(Yt)





#UNet will expect this input shape for all training data
model_shape = (1024,512,3)

#Keep features as 1 for now. Features parameter is leftover from multi-class CNN.
CNN = neural_nets.UNet_binary(model_shape, features=1, filters=16)

#Train the model
history = CNN.model.fit(X, Y, epochs=5, verbose=1, validation_split=0.3, batch_size=1)


CNN.model.save('my_model.h5')

CNN = tf.keras.models.load_model('my_model.h5')

#predict on test X
pred = CNN.predict(Xt)
pred = np.squeeze(pred)

#Set classification threshold
pred[pred > 0.18] = 1
pred[pred < 0.18] = 0 
pred = pred.astype(int)


# """
# Calculate Recall, Precision, and F1 Score
# """

# #True Positives
TP = np.count_nonzero(pred * Yt)


# #True Negatives 
TN = np.count_nonzero((pred - 1) * (Yt - 1))

# #False Positives
FP = np.count_nonzero(pred * (Yt - 1))

# #False Negatives

#UNet will expect this input shape for all training data
model_shape = (512,1024,3)


CNN = neural_nets.UNet_binary(model_shape, features=1, filters=16)
history = CNN.model.fit(X, Y, epochs=15, verbose=1, validation_split=0.3, batch_size=1)

pred = CNN.model.predict(Xt)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0 
pred = np.squeeze(pred)

"""
Calculate Recall, Precision, and F1 Score
"""

# True Positives
TP = np.count_nonzero(pred * Yt)

# True Negatives
TN = np.count_nonzero((pred - 1) * (Yt - 1))

# False Positives
FP = np.count_nonzero(pred * (Yt - 1))


# False Negatives

FN = np.count_nonzero((pred - 1) * Yt) 

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("Recall: {}\nPrecision:{}\nF1_Score:{}".format(recall, precision, f1))


plt.figure()
training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

#Visualize prediciton 


plt.figure()
fig, ax = plt.subplots()

ax.imshow(pred, aspect='auto')



time = datetime.now() - start

print(time)

pred = cv2.resize(pred, dsize=(512,2048), interpolation=cv2.INTER_CUBIC)

#pred = cv2.resize(pred, dsize=(512,2048), interpolation=cv2.INTER_CUBIC)



plt.figure()
training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.figure()
fig, ax = plt.subplots()

ax.imshow(pred, aspect='auto')

time = datetime.now() - start

print(time)


time = datetime.now() - start

print(time)
