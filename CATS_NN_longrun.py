
"""
Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""
import sys
import numpy as np
import neural_nets
import pdb
import matplotlib as mpl
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from lidar import get_a_color_map

# Location of where you would like the output to be located
output_directory = "C:\\Users\\akupchoc.NDC\\Documents\\Work\\Jetson TX2\\CATS_ML\\Output_Files\\"
# Location of where the questions, answers, test questions, and test answers are stored
save_directory = "C:\\Users\\akupchoc.NDC\\PycharmProjects\\CATS_ML\\questions_and_answers\\"


def image_plot(im_array, scale_min, scale_max):
    plt.imshow(im_array, cmap=get_a_color_map(), vmin=scale_min, vmax=scale_max)
    return None


def image_2d_data_contcb(arr, cbmin, cbmax, wid, hgt, outfile):
    """ Make & save an image plot of a 2D array of data.
        Use a continuous color bar.

        Please put the horizontal dimension as the first dimension;
        vertical as the 2nd.
    """

    fig, ax = plt.subplots(1, 1, figsize=(wid, hgt))
    ax.imshow(arr, cmap=get_a_color_map(), vmin=cbmin, vmax=cbmax, aspect='auto')
    plt.savefig(outfile)
    plt.close(fig)


def image_2d_data_dsrtcb(arr, clist, cbounds, keystring, wid, hgt, outfile):
    """ Make & save an image plot of a 2D array of data.
        Use a discrete color bar.

        Please put the horizontal dimension as the first dimension;
        vertical as the 2nd.
    """

    # INPUTS:
    # arr   -> 2D array of input data
    # clist -> list of colors for ListedColormap
    # cbounds -> bounds. must have one more element than # of discrete colors
    # keystring -> plot title, but i'd use as a key for the numeric categories
    # wid -> width of figure in inches
    # hgt -> height of figure in inches
    # outfile -> full path output image file
    #
    # OUPUT: None (returns None)

    # NOTES:
    # bounds (aka bins) for values go [<= x, <=y, <=z]
    # for example bounds [0,1,2]

    cmap = mpl.colors.ListedColormap(clist)
    norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(wid, hgt))
    img = ax.imshow(arr, cmap=cmap, vmin=min(cbounds), vmax=max(cbounds), aspect='auto')
    ax.set_title(keystring)
    cbar = plt.colorbar(img, ax=ax)
    cbticks = cbar.ax.get_yticks()
    newticklabels = []
    for tk in cbticks:
        if tk in cbounds:
            newticklabels.append(str(int(tk)))
        else:
            newticklabels.append('')
    newticklabels[:1] = newticklabels[1:]  # move back to center on color
    cbar.ax.set_yticklabels(newticklabels)
    plt.savefig(outfile)
    plt.close(fig)


# ****************** Start of Main ****************** #
"""
Formatting Input Training Data
"""
# loading .npy files created by make_training_set.py
X = np.load(save_directory+'questions.npy', allow_pickle=True)
Y = np.load(save_directory+'answers.npy', allow_pickle=True)
image_key = np.load(save_directory+'image_key.npy', allow_pickle=True)

Xt = np.load(save_directory+'test_questions.npy', allow_pickle=True)
Yt = np.load(save_directory+'test_answers.npy', allow_pickle=True)
test_image_key = np.load(save_directory+'test_image_key.npy', allow_pickle=True)

# Move the number of samples to the first dimension
full_X = np.transpose(X, [0, 1, 3, 2])
full_Y = Y
# Do the same for the test data
full_Xt = np.transpose(Xt, [0, 1, 3, 2])
full_Yt = Yt

# Separate out the channels in the test group individually
channel_1 = full_X[:, :, :, 0]
channel_2 = full_X[:, :, :, 1]
channel_3 = full_X[:, :, :, 2]
# Do the same for the test data
channel_1_test = full_Xt[:, :, :, 0]
channel_2_test = full_Xt[:, :, :, 1]
channel_3_test = full_Xt[:, :, :, 2]
# Adding in fourth dimension for input to model
channel_1 = np.reshape(channel_1, (channel_1.shape[0], channel_1.shape[1], channel_1.shape[2], 1))
channel_2 = np.reshape(channel_2, (channel_2.shape[0], channel_2.shape[1], channel_2.shape[2], 1))
channel_3 = np.reshape(channel_3, (channel_3.shape[0], channel_3.shape[1], channel_3.shape[2], 1))
# Do the same for the test data
channel_1_test = np.reshape(channel_1_test, (channel_1_test.shape[0], channel_1_test.shape[1], channel_1_test.shape[2], 1))
channel_2_test = np.reshape(channel_2_test, (channel_2_test.shape[0], channel_2_test.shape[1], channel_2_test.shape[2], 1))
channel_3_test = np.reshape(channel_3_test, (channel_3_test.shape[0], channel_3_test.shape[1], channel_3_test.shape[2], 1))
# Add fourth dimension to answer if it is not present
full_Y = np.reshape(full_Y, (full_Y.shape[0], full_Y.shape[1], full_Y.shape[2], 1))
# Do the same for the test data
full_Yt = np.reshape(full_Yt, (full_Yt.shape[0], full_Yt.shape[1], full_Yt.shape[2], 1))

print('At this point all of the data should be in the correct format to be used on the neural network')

# ************ MODEL INPUTS TO BE MODIFIED ************ #
# Features is number of categories in the L2 data (1 for binary)
# Dropout is the dropout between steps in the UNet
# Filters are an input variable
model_shape = (512, 256, 1)       # changed to 256 to match image sizes and dimensions to match their descriptors
features = 1
filter_choice = [16]     # [1, 4, 16]
dropout_choice = [0.2]   # [0.1, 0.2]
epoch_choice = [1]      # [1, 2, 3]
channel_selection = [(1, 0, 0)]  # [(1, 1, 1), (1, 1, 0), (0, 0, 1)]      # selector for which channels to train on

# Put for loops in here:

# Where all the magic happens
CNN = neural_nets.UNet_binary2(model_shape, features=1, filters=16, dropout=0.2)


# verbose set to 1 will give you a progress bar if desired
# CNN.model.fit(channel_1, full_Y, epochs=1, verbose=2, validation_split=0.2)
for channel_element in np.arange(0, len(channel_selection)):
    if channel_selection[channel_element][0] == 1:
        # Use Channel 11 to train the model
        print('Training on Channel 11')
        CNN.model.fit(channel_1, full_Y, epochs=1, verbose=1, validation_data=(channel_1_test, full_Yt))
    if channel_selection[channel_element][1] == 1:
        # Use Channel 12 to train the model
        print('Training on Channel 12')
        CNN.model.fit(channel_2, full_Y, epochs=1, verbose=1, validation_data=(channel_1_test, full_Yt))
    if channel_selection[channel_element][2] == 1:
        # Use the Sum of 11 and 12 to train the model
        print('Training on Channel 11 and 12 Sum')
        CNN.model.fit(channel_3, full_Y, epochs=1, verbose=1, validation_data=(channel_1_test, full_Yt))

# del CNN

'''
pred = CNN.model.predict(channel_1_test)
arr = pred[0, :, :, 0]
round_predicts = int(pred.round())
round_predicts = np.reshape(round_predicts, (1, np.prod(round_predicts[:].shape)))
round_predicts_mask = np.where(round_predicts == 1, 1, 0)
full_Yt = full_Yt.round()
full_Yt = np.reshape(full_Yt, (1, np.prod(full_Yt[:].shape)))
full_Yt_mask = np.where(full_Yt == 1, 1, 0)
pdb.set_trace()
scores = f1_score(full_Yt_mask, round_predicts_mask, average='binary', pos_label=1)
# pdb.set_trace()
# pred = np.squeeze(pred)
arr = np.argmax(pred, axis=2)
arr = arr.astype(int)
image_plot(arr, 0, 100)
'''

pred = CNN.model.predict(channel_1_test)

# Continuous color bar example
cbmin = 0
cbmax = 1
wid = 8
hgt = 16
outfile = output_directory + 'image_' + str(filter_choice[0]) + '_' + str(epoch_choice[0]) + '_' + str(int(dropout_choice[0]*100)) + '_' + 'test_contcb.png'
arr = pred[0, :, :, 0]
print('Saving scaled imaged to file')
image_2d_data_contcb(arr, cbmin, cbmax, wid, hgt, outfile)

# Discrete color bar example
arr_round = np.round(arr)
# arr[(arr > 10) & (arr <= 50)] = 1
# arr[arr > 50] = 2
clist = ['blue', 'yellow', 'red', 'green']
# 0 blue, 1 yellow, 2 red, 3 green
cbounds = [0, 1, 2, 3, 4]
keystring = '0 - zeros, 1 = ones, 2 - twos, 3 - threes'
wid = 8
hgt = 16
outfile = output_directory + 'image_' + str(filter_choice[0]) + '_' + str(epoch_choice[0]) + '_' + str(int(dropout_choice[0]*100)) + '_' + 'test_dsrtcb.png'
print('Saving 4-color image to file')
image_2d_data_dsrtcb(arr_round, clist, cbounds, keystring, wid, hgt, outfile)
pdb.set_trace()
