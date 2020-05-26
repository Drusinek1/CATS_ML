  
# -*- coding: utf-8 -*-
"""
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


def plot(arr):
    plt.imshow(arr,)
    return None


start = datetime.now()

"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"

# loading .npy files created by make_training_set.py
X = np.load('questions.npy', allow_pickle=True)
# X = np.transpose(X, (0,3,2,1))
Y = np.load('answers.npy', allow_pickle=True)


Xt = np.load('test_questions.npy', allow_pickle=True)[:, :, :, :]


Yt = np.load('test_answers.npy', allow_pickle=True)
Yt = np.squeeze(Yt)

# UNet will expect this input shape for all training data
model_shape = (512, 1024, 3)


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

# pred = cv2.resize(pred, dsize=(512,2048), interpolation=cv2.INTER_CUBIC)

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
