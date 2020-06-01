
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

    fig2, ax2 = plt.subplots(1, 1, figsize=(wid, hgt))

    img2 = ax2.imshow(arr, cmap=get_a_color_map(), vmin=cbmin, vmax=cbmax, aspect='auto')
    plt.colorbar(img2, ax=ax2)
    plt.savefig(outfile)
    plt.close(fig2)


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


def get_eval_metrics_binary(predicted_array, actual_array, classes, image_export_path):
    """
    This function calculates a confusion matrix for the input
    predicted and actual arrays, plots the confusion matrix and outputs
    accuracy metrics.

    Parameters
    ----------
    predicted_array : M x N ndarray
        the predicted array
    actual_array : M x N ndarray
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
    # Added 5/30/2020 in case the input array has not be placed into step function
    predicted_array = np.round(predicted_array)
    # True Positives
    TP = np.count_nonzero(predicted_array * actual_array)
    # True Negatives
    TN = np.count_nonzero((predicted_array - 1) * (actual_array - 1))
    # False Positives
    FP = np.count_nonzero(predicted_array * (actual_array - 1))
    # False Negatives
    FN = np.count_nonzero((predicted_array - 1) * actual_array)
    div_by_zero = 0
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 'N/A - divide by 0'
        div_by_zero = 1

    if (TP + FN) != 0 and not div_by_zero:
        recall = TP / (TP + FN)
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        recall = 'N/A - divide by 0'
        f1 = 'N/A - divide by 0'
    population = np.prod(predicted_array.shape)

    metrics_dict = {'True_Positives': TP,
                    'True_Negatives': TN,
                    'False_Positives': FP,
                    'False_Negatives': FN,
                    'Population': population,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1}

    confusion_matrix = np.array([[TP/population, FP/population], [FN/population, TN/population]])

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    fig.canvas.draw()

    # Replacing number labels with text
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ylabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels[1], xlabels[2] = "{} Predicted".format(classes[0]), "{} Predicted".format(classes[1])
    ylabels[1], ylabels[2] = "{} Actual".format(classes[0]), "{} Actual".format(classes[1])

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels, rotation=90, verticalalignment="center")

    ax.matshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    for i in range(0, confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            c = confusion_matrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center')

    # save the figure to the specified export path
    plt.savefig(image_export_path)
    plt.close(fig)
    return confusion_matrix, metrics_dict


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
filter_choice = [6, 8, 20]         # [4, 16]
dropout_choice = [0.2]              # [0.1, 0.2]
epoch_choice = [2]                  # [1, 2]
channel_selection = [(1, 1, 1), (1, 1, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]   # selector for which channels to train on

try:
    # if list exists already just append to the list
    metrics_dict_list = np.load(save_directory + "metrics_list.npy", allow_pickle=True)
    metrics_dict_list = metrics_dict_list.tolist()
except:
    # if no list exists, create an empty list
    metrics_dict_list = []      # List used to retain the dictionary for each run
classes = ['Layer', 'No Layer']
# Put for loops in here:

for filters_used in np.arange(0, len(filter_choice)):
    for dropout_used in np.arange(0, len(dropout_choice)):
        for epoch_used in np.arange(0, len(epoch_choice)):
            for channel_element in np.arange(0, len(channel_selection)):
                # Where all the magic happens
                CNN = neural_nets.UNet_binary2(model_shape, features=1, filters=filter_choice[filters_used],
                                               dropout=dropout_choice[dropout_used])
                # verbose set to 1 will give you a progress bar if desired
                # CNN.model.fit(channel_1, full_Y, epochs=1, verbose=2, validation_split=0.2)
                if channel_selection[channel_element][0] == 1:
                    # Use Channel 11 to train the model
                    print('Training on Channel 11')
                    CNN.model.fit(channel_1, full_Y, epochs=epoch_choice[epoch_used], verbose=2,
                                  validation_data=(channel_1_test, full_Yt))
                if channel_selection[channel_element][1] == 1:
                    # Use Channel 12 to train the model
                    print('Training on Channel 12')
                    CNN.model.fit(channel_2, full_Y, epochs=epoch_choice[epoch_used], verbose=2,
                                  validation_data=(channel_1_test, full_Yt))
                if channel_selection[channel_element][2] == 1:
                    # Use the Sum of 11 and 12 to train the model
                    print('Training on Channel 11 and 12 Sum')
                    CNN.model.fit(channel_3, full_Y, epochs=epoch_choice[epoch_used], verbose=2,
                                  validation_data=(channel_1_test, full_Yt))
                chan_select_string = str(channel_selection[channel_element][0])+\
                                     str(channel_selection[channel_element][1])+\
                                     str(channel_selection[channel_element][2])
                pred = CNN.model.predict(channel_1_test)
                # In the future you can stitch arrays together here to make comparison confusion matrices and
                # images for larger sets of data
                # predicted_arr is the two dimensional array of data to be plotted and analyzed
                predicted_arr = pred[0, :, :, 0]
                # solution_arry is the corresponding two dimensional array from the input the to predict function
                solution_arry = np.squeeze(full_Yt[0, :, :])
                # Continuous color bar
                cbmin = 0
                cbmax = 1
                wid = 8
                hgt = 16
                outfile = output_directory + 'image_' + str(filter_choice[filters_used]) + '_' + \
                      str(epoch_choice[epoch_used]) + '_' + str(int(dropout_choice[dropout_used]*100)) + '_' + \
                      chan_select_string + '_' + 'predicted.png'
                print('Saving scaled imaged to file')
                image_2d_data_contcb(predicted_arr, cbmin, cbmax, wid, hgt, outfile)
                # Section of code used to separate into discrete values
                predicted_arr_discrete = np.round(predicted_arr)        # Using round creates a separation level of 0.5
                # Determine Metrics and generate confusion matrix
                image_export_path = output_directory + 'image_' + str(filter_choice[filters_used]) + '_' + \
                                str(epoch_choice[epoch_used]) + '_' + str(int(dropout_choice[dropout_used]*100)) + '_' \
                                + chan_select_string + '_' + 'Confusion.png'
                con_matrix, met_dict = get_eval_metrics_binary(predicted_arr_discrete, solution_arry, classes, image_export_path)
                # Discrete color bar plot - You must change the predicted array to integer outputs before this routine
                # TN = 0, FN = 1, FP = 2, TP, = 3
                comparison_plot_array = np.zeros(predicted_arr_discrete.shape)
                # True Positives
                comparison_plot_array += (np.abs(predicted_arr_discrete * solution_arry)*3)
                # True Negatives
                # Currently 0 so they can be skipped but left code below for future updates
                # comparison_plot_array += (np.abs((predicted_arr_discrete - 1) * (solution_arry - 1))*0)
                # False Positives
                comparison_plot_array += (np.abs(predicted_arr_discrete * (solution_arry - 1))*2)
                # False Negatives
                comparison_plot_array += (np.abs((predicted_arr_discrete - 1) * solution_arry)*1)
                clist = ['indigo', 'turquoise', 'red', 'ivory']
                cbounds = [0, 1, 2, 3, 4]
                keystring = '1 - TN, 2 - FN, 3 - FP, 4 - TP'
                wid = 8
                hgt = 16
                outfile = output_directory + 'image_' + str(filter_choice[filters_used]) + '_' + \
                      str(epoch_choice[epoch_used]) + '_' + str(int(dropout_choice[dropout_used]*100)) + '_' + \
                      chan_select_string + '_' + 'comp_plt.png'
                print('Saving 4-color image to file')
                image_2d_data_dsrtcb(comparison_plot_array, clist, cbounds, keystring, wid, hgt, outfile)

                del CNN
                met_dict["Epochs"] = epoch_choice[epoch_used]
                met_dict["Dropout"] = dropout_choice[dropout_used]
                met_dict["Filters"] = filter_choice[filters_used]
                met_dict["Channels"] = chan_select_string
                met_dict["Con_Mat"] = con_matrix
                metrics_dict_list.append(met_dict)

np.save(save_directory+"metrics_list", metrics_dict_list)
print('Completed Runs of CNN configurations')
pdb.set_trace()


''' This commented code is currently garbage - was using to figure out f1_score function input maybe it can be revived
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