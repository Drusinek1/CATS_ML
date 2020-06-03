  
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
import csv



start = datetime.now()

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

def get_eval_metrics(pred, actual, classes, image_export_path):
    """
    This function calculates a confusion matrix for the inputed 
    predicted and actual arrays, plots the confusion matrix and outputs 
    accuracy metrics.
    
    Parameters
    ----------
    pred : M x N ndarray
        the predicted array
    actual : M x N ndarray
        the target array
    classes : list[String]
        a list of the class names for the data
    image_export_path : String
        Path where to save plotted confusion matrix
    Returns
    -------
    out : ndarray
              The computed confusion matrix
          dictionary[int]
              Dictionary containin TP, TN, FP, FN, precision, recall, f1 score
    """
    

    # True Positives
    TP = np.count_nonzero(pred * actual)
    
    # True Negatives
    TN = np.count_nonzero((pred - 1) * (actual - 1))
    
    # False Positives
    FP = np.count_nonzero(pred * (actual - 1))
    
    
    # False Negatives
    
    FN = np.count_nonzero((pred - 1) * actual) 
    
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    
    metrics_dict = {'True_Positives':TP,
                    'True_Negatives':TP,
                    'False_Positives':TP,
                    'False_Negatives':TP,
                    'Precision':precision,
                    'Recall':recall,
                    'F1_Score':f1}
    
    confusion_matrix = np.array([[TP, FP],
                                 [FN, TN]])
    
    #Plotting the confusion matrix
    fig, ax = plt.subplots()
    fig.canvas.draw()
    
    #Replacing number labels with text
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ylabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels[1], xlabels[2] = "{} Predicted".format(classes[0]), "{} Predicted".format(classes[1])
    ylabels[1], ylabels[2] = "{} Actual".format(classes[0]), "{} Actual".format(classes[1])
    
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels, rotation=90, verticalalignment="center")

    ax.matshow(confusion_matrix,cmap='Blues', aspect='auto')

    for i in range(0, confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            c = confusion_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
            
    #save the figure to the specified export path        
    plt.savefig(fig, image_export_path)
    
    return confusion_matrix, metrics_dict


def export_eval_metrics(eval_metrics, sample_number):
  """
    This function calculates a confusion matrix for the inputed 
    predicted and actual arrays, plots the confusion matrix and outputs 
    accuracy metrics.
    
    Parameters
    ----------
    eval_metrics : dictionary
        a dictionary containing eval metrics
    sample_number : int
        number of the sample that correspond to the inputed eval_metrics

    Returns
    -------
    None
    """
    

    csv_columns = ['Precision','Recall','F1_Score']
    

    csv_file = "eval_{}.csv".format(sample_number)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    
    return None


def export_eval_metrics(eval_metrics, sample_number):
    """
    This function calculates a confusion matrix for the inputed 
    predicted and actual arrays, plots the confusion matrix and outputs 
    accuracy metrics.
    
    Parameters
    ----------
    eval_metrics : dictionary
        a dictionary containing eval metrics
    sample_number : int
        number of the sample that correspond to the inputed eval_metrics

    Returns
    -------
    None
    """

    with open('eval.csv', 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in eval_metrics.items():
            writer.writerow([key, value])
    
    return None

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

"""
Start of Script
"""

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



model_shape = (512,1024,3)


CNN = neural_nets.UNet_binary(model_shape, features=1, filters=16)
history = CNN.model.fit(X, Y, epochs=15, verbose=1, validation_split=0.3, batch_size=1)

model = CNN.model
model.save("test_model")

# pred = CNN.model.predict(Xt)
# pred[pred >= 0.5] = 1
# pred[pred < 0.5] = 0 
# pred = np.squeeze(pred)

"""
Calculate Recall, Precision, and F1 Score
"""






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
