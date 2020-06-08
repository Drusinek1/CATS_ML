
"""
Created on June 1st 2020
Created together by Andrew Kupchock, Daniel Rusinek, and Patrick Selmer

Items to be updated per run:
output_directory
save_directory
file_number_to_use
image_subset
vertical_dimension_of_output
features
filter_choice
dropout_choice
epoch_choice
channel_selection

Version 2 changes the input to the neural network to be a variable number of channels.
The validation set and output generation is used with the same channel mask as the fit function
"""
import numpy as np
import neural_nets
import pdb
import matplotlib as mpl
from matplotlib import pyplot as plt
from lidar import get_a_color_map

# Location of where you would like the output to be located
#output_directory = "C:\\Users\\akupchoc.NDC\\Documents\\Work\\Jetson TX2\\CATS_ML\\Output_Files\\"
output_directory = './output2/'
# Location of where the questions, answers, test questions, and test answers are stored
#save_directory = "C:\\Users\\akupchoc.NDC\\PycharmProjects\\CATS_ML\\questions_and_answers\\"
save_directory = './questions_and_answers/'


def image_2d_data_contcb(arr, cbmin, cbmax, wid, hgt, requested_title, outfile):
    """ Make & save an image plot of a 2D array of data.
        Use a continuous color bar.

        Please put the horizontal dimension as the first dimension;
        vertical as the 2nd.
    """

    fig2, ax2 = plt.subplots(1, 1, figsize=(wid, hgt))

    img2 = ax2.imshow(arr, cmap=get_a_color_map(), vmin=cbmin, vmax=cbmax, aspect='auto')
    plt.colorbar(img2, ax=ax2)
    ax2.set_title(requested_title)
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
    cbar.set_ticks(np.arange(min(cbounds), max(cbounds) + 0.5, .5))     # Forces existence of .5 ticks
    cbticks = cbar.ax.get_yticks()
    newticklabels = []
    for tk in cbticks:
        if tk in cbounds:
            newticklabels.append(str(int(tk)))
        else:
            newticklabels.append('')
    newticklabels[:1] = newticklabels[1:]             # move back to center on color
    cbar.ax.set_yticklabels(newticklabels)
    plt.savefig(outfile)
    plt.close(fig)


def neural_net_binary_plotter(test_channel_mask, title_for_input_data, input_data_arry, confusion_matrix,
                              title_for_predicted_data, continuous_predicted_array, confusion_plot_array, classes,
                              clist, cbounds, keystring, wid, hgt, outfile, metrics_dictionary):
    """ Make & save an image plot of a 2D array of data.
        Use a discrete color bar.

        Please put the horizontal dimension as the first dimension;
        vertical as the 2nd.

    Parameters
    ----------
    test_channel_mask : str
        Mask of current channels being used for training and validation
    title_for_input_data : str
        Title requested for input data array plot
    input_data_arry : M x N array
        array used for input into the neural network
    confusion_matrix : 2 x 2 array
        values from confusion matrix calculation of entire dataset
    title_for_predicted_data : str
        Title request for predicted data array plot
    continuous_predicted_array : M x N array
        array from prediction of neural network
    confusion_plot_array : M x N array
        array of discrete values showing TP, TN, FP, and FN
    classes : list
        list of strings with the names of the classes to be used
    clist : list
        list of corresponding colors to be used in confusion visualization plot
    cbounds : list
        list of integers to be used for confusion visualization plot
    keystring : str
        Title of confusion visualization plot
    wid : float
        width of the plot in inches for each curtain plot
    hgt : float
        height of the plot in inches for each curtain plot
    outfile : str
        file where the image generated should be saved including file path
    Returns
    -------
    None
    """
    complete_fig = plt.figure(constrained_layout=True, figsize=(wid + hgt, hgt*3))
    widths = [wid, hgt]
    heights = [hgt, hgt, hgt]
    specs = complete_fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)
    ax1 = complete_fig.add_subplot(specs[0, 0])
    ax2 = complete_fig.add_subplot(specs[1, 0])
    ax3 = complete_fig.add_subplot(specs[2, 0])
    ax4 = complete_fig.add_subplot(specs[1, 1])
    # Below section adds annotation of model performance
    ax5 = complete_fig.add_subplot(specs[2, 1])
    # Removes tick marks
    ax5.set_xticks([])
    ax5.set_yticks([])
    # Removes visible axes
    for vis_spine in ax5.spines.values():
        vis_spine.set_visible(False)
    try:
        label = 'Precision: {:.6f}\n\nRecall:      {:.6f}\n\nF1 Score:  {:.6f}'.format(metrics_dictionary["Precision"],
                                                             metrics_dictionary["Recall"],
                                                             metrics_dictionary["F1_Score"])
    except:
        label = 'Precision: N/A\n\nRecall:      N/A\n\nF1 Score:  N/A'

    ax5.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')



    ax1.set_title(title_for_input_data)
    ax2.set_title(title_for_predicted_data)

    # Plot of input data using rainbow colormap
    img2 = ax1.imshow(input_data_arry, cmap=get_a_color_map(), vmin=0, vmax=100, aspect='auto')
    plt.colorbar(img2, ax=ax1)

    # Plot of predicted matrix using limits of 0 to 1
    img3 = ax2.imshow(continuous_predicted_array, cmap=get_a_color_map(), vmin=0, vmax=1, aspect='auto')
    plt.colorbar(img3, ax=ax2)

    # Plotting the confusion matrix
    # Forcing the correct number of ticks
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    # Replacing number labels with text
    xlabels = [item.get_text() for item in ax4.get_xticklabels()]
    ylabels = [item.get_text() for item in ax4.get_xticklabels()]
    xlabels[1], xlabels[2] = "{} Predicted".format(classes[0]), "{} Predicted".format(classes[1])
    ylabels[1], ylabels[2] = "{} Actual".format(classes[0]), "{} Actual".format(classes[1])

    ax4.set_xticklabels(xlabels, fontsize=6)
    ax4.set_yticklabels(ylabels, rotation=90, verticalalignment="center", fontsize=6)

    ax4.matshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    for i in range(0, confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            c = confusion_matrix[j, i]
            ax4.text(i, j, "{:.4%}".format(c), va='center', ha='center', fontsize=8)

    # Confusion matrix plot
    cmap = mpl.colors.ListedColormap(clist)
    norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)
    img4 = ax3.imshow(confusion_plot_array, cmap=cmap, vmin=min(cbounds), vmax=max(cbounds), aspect='auto')
    ax3.set_title(keystring)
    cbar_together = plt.colorbar(img4, ax=ax3)
    cbar_together.set_ticks(np.arange(min(cbounds), max(cbounds) + 0.5, .5))
    cbticks_together = cbar_together.ax.get_yticks()
    newticklabels = []
    for tk in cbticks_together:
        if tk in cbounds:
            newticklabels.append(str(int(tk)))
        else:
            newticklabels.append('')
    newticklabels[:1] = newticklabels[1:]  # move back to center on color
    cbar_together.ax.set_yticklabels(newticklabels)

    plt.savefig(outfile)
    plt.close(complete_fig)


def stitch_array(input_array, key_for_images, key_used, subset_to_stitch):
    # Function takes input array and stitches images together for future plotting
    prior_image_nums = 0
    for element in np.arange(0, key_used):
        prior_image_nums = prior_image_nums + int(key_for_images[element][1])
    if subset_to_stitch[0] != subset_to_stitch[1]:
        stitched_array = input_array[(prior_image_nums + subset_to_stitch[0]), :, :, :]
        for images in np.arange(subset_to_stitch[0]+1, subset_to_stitch[1]+1):
            stitched_array = np.concatenate((stitched_array, input_array[(prior_image_nums + images), :, :, :]), axis=1)
    else:
        stitched_array = input_array[0, :, :]
    return np.squeeze(stitched_array)


def remove_pad_from_vertical(large_array, target_height):
    # Second dimension needs to be the height
    large_array_height = large_array.shape[1]
    pads_to_remove = large_array_height - target_height
    if len(large_array.shape) == 4:
        # four dimensional array
        large_array = large_array[:, pads_to_remove:, :, :]
    else:
        if len(large_array.shape) == 3:
            # three dimensional array
            large_array = large_array[:, pads_to_remove:, :]
        else:
            # two dimensional array
            large_array = large_array[:, pads_to_remove:]
    return large_array


def get_eval_metrics_binary(predicted_array, actual_array, classes, image_export_path, channel_mask_used, metrics_dict):
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
    image_export_path : str
        Path where to save plotted confusion matrix
    channel_mask_used : str
        Current channel data input into prediction
    metrics_dict
        dictionary to be added to with metric information
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

    metrics_dict['True_Positives'] = TP
    metrics_dict['True_Negatives'] = TN
    metrics_dict['False_Positives'] = FP
    metrics_dict['False_Negatives'] = FN
    metrics_dict['Population'] = population
    metrics_dict['Precision'] = precision
    metrics_dict['Recall'] = recall
    metrics_dict['F1_Score'] = f1
    metrics_dict['Channel_Mask'] = channel_mask_used
    confusion_matrix = np.array([[TP/population, FP/population], [FN/population, TN/population]])

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    fig.canvas.draw()

    # Replacing number labels with text
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ylabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels[1], xlabels[2] = "{} Predicted".format(classes[0]), "{} Predicted".format(classes[1])
    ylabels[1], ylabels[2] = "{} Actual".format(classes[0]), "{} Actual".format(classes[1])

    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_yticklabels(ylabels, rotation=90, verticalalignment="center", fontsize=12)

    ax.matshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    for i in range(0, confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            c = confusion_matrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center', fontsize=12)

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
# The file number below must be less than the number of files the test questions and answers are made from
file_number_to_use = 0
# Sets the top dimension for the images. Note this also changes the number of elements compared for the confusion
# matrices and can change the statistics by manipulated the population of true negatives
# Typically 500 to remove padding added for neural network
vertical_dimension_of_output = 500
# Tuple which defines the start and stop limits for stitching images together
image_subset = (0, int(test_image_key[file_number_to_use][1])-1)
granule_to_plot = test_image_key[file_number_to_use][0].split('/')[-1].split('.')[0] + \
                  test_image_key[file_number_to_use][0].split('/')[-1].split('.')[1]
print('The Granule used to make these plots is ', granule_to_plot)
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
L2_answers = remove_pad_from_vertical(full_Yt, vertical_dimension_of_output)
solution_arry = stitch_array(L2_answers, test_image_key, file_number_to_use, image_subset)
print('At this point all of the data should be in the correct format to be used on the neural network')
# ************ MODEL INPUTS TO BE MODIFIED ************ #
# Features is number of categories in the L2 data (1 for binary)
# Dropout is the dropout between steps in the UNet
# Filters are an input variable
model_shape = (512, 256, 1)       # changed to 256 to match image sizes and dimensions to match their descriptors
features = 1
filter_choice = [8, 16]         # [4, 16]
dropout_choice = [0.1, 0.2, 0.3]#[0.1, 0.2, 0.3]              # [0.1, 0.2]
epoch_choice = [2, 3]                  # [1, 2]
channel_selection = [(1,1,0), (1,1,1)]#[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)]
# ***************************************************** #
try:
    # if list exists already just append to the list
    metrics_dict_list = np.load(save_directory + "metrics_list_v2.npy", allow_pickle=True)
    metrics_dict_list = metrics_dict_list.tolist()
except:
    # if no list exists, create an empty list
    metrics_dict_list = []      # List used to retain the dictionary for each run
class_labels = ['Layer', 'No Layer']
# Put for loops in here:
for filters_used in np.arange(0, len(filter_choice)):
    for dropout_used in np.arange(0, len(dropout_choice)):
        for epoch_used in np.arange(0, len(epoch_choice)):
            for channel_element in np.arange(0, len(channel_selection)):
                # Where all the magic happens
                # model_shape is determined by the number of channels used for input in channel selection
                model_shape = (512, 256, sum(channel_selection[channel_element]))
                CNN = neural_nets.UNetBinary2(model_shape, features=1, filters=filter_choice[filters_used],
                                              dropout=dropout_choice[dropout_used])
                # verbose set to 1 will give you a progress bar if desired
                if channel_selection[channel_element][0] == 1:
                    # Use Channel 11 to train the model
                    print('Training with Channel 11')
                    train_array = channel_1
                    prediction_input_array = channel_1_test
                    L2_questions = remove_pad_from_vertical(channel_1_test, vertical_dimension_of_output)
                    question_title = 'Channel 11 from ' + granule_to_plot
                    predicted_title = "Predicted Output using only Channel 11"
                    if channel_selection[channel_element][1] == 1:
                        # Use channel 11 and also use Channel 12 to train the model
                        print('Also Training with Channel 12')
                        train_array = np.concatenate((train_array, channel_2), axis=3)
                        prediction_input_array = np.concatenate((prediction_input_array, channel_2_test), axis=3)
                        predicted_title = "Predicted Output using Channel 11 and Channel 12"
                    if channel_selection[channel_element][2] == 1:
                        # Also include the sum with training
                        print('Also Training with Sum of 11 and 12')
                        train_array = np.concatenate((train_array, channel_3), axis=3)
                        prediction_input_array = np.concatenate((prediction_input_array, channel_3_test), axis=3)
                        if not channel_selection[channel_element][1] == 1:
                            predicted_title = "Predicted Output using Channel 11 and the Sum of 11 and 12"
                        else:
                            predicted_title = "Predicted Output using Channel 11, 12, and their Sum"
                else:
                    if channel_selection[channel_element][1] == 1:
                        # Do not use 11 but use Channel 12 to train the model
                        print('Training with Channel 12')
                        train_array = channel_2
                        prediction_input_array = channel_2_test
                        L2_questions = remove_pad_from_vertical(channel_2_test, vertical_dimension_of_output)
                        question_title = 'Channel 12 from ' + granule_to_plot
                        predicted_title = "Predicted Output using only Channel 12"
                        if channel_selection[channel_element][2] == 1:
                            # Also include the sum with training
                            train_array = np.concatenate((train_array, channel_3), axis=3)
                            prediction_input_array = np.concatenate((prediction_input_array, channel_3_test), axis=3)
                            predicted_title = "Predicted Output using Channel 12 and the Sum of 11 and 12"
                    else:
                        if channel_selection[channel_element][2] == 1:
                            # Use only Sum of 11 and 12 to train the model
                            print('Training only with sum of Channel 11 and 12')
                            train_array = channel_3
                            prediction_input_array = channel_3_test
                            L2_questions = remove_pad_from_vertical(channel_3_test, vertical_dimension_of_output)
                            question_title = 'Sum of Channel 11 and 12 from ' + granule_to_plot
                            predicted_title = "Predicted Output using only the sum of Channel 11 and 12"
                CNN.model.fit(train_array, full_Y, epochs=epoch_choice[epoch_used], verbose=2,
                              validation_data=(prediction_input_array, full_Yt))

                chan_select_string = str(channel_selection[channel_element][0]) + \
                                     str(channel_selection[channel_element][1]) + \
                                     str(channel_selection[channel_element][2])
                configuration_string = 'image_' + str(filter_choice[filters_used]) + '_' + str(epoch_choice[epoch_used])\
                                       + '_' + str(int(dropout_choice[dropout_used] * 100)) + '_' + chan_select_string + '_'
                met_dict = {}
                pred = CNN.model.predict(prediction_input_array)
                # Question array has the L0 data from the first channel listed in the channel mask
                question_arry = stitch_array(L2_questions, test_image_key, file_number_to_use, image_subset)
                # removing the top padding that was added for the UNet neural network
                pred = remove_pad_from_vertical(pred, vertical_dimension_of_output)
                # predicted_arr is the two dimensional array of data to be plotted and analyzed
                # solution_arry is the corresponding two dimensional array from the input the to predict function
                # The solution_array is created before the for loops since it does not change
                # Stitch arrays together to make comparison confusion matrices and images for larger sets of data
                predicted_arr = stitch_array(pred, test_image_key, file_number_to_use, image_subset)
                # Continuous color bar
                cbmin = 0
                cbmax = 1
                wid = 20
                hgt = (predicted_arr.shape[0]/predicted_arr.shape[1]) * wid
                outfile = output_directory + configuration_string + 'predicted.png'
                print('Saving scaled imaged to file')
                predict_title = "Predicted Output data from " + granule_to_plot
                image_2d_data_contcb(predicted_arr, cbmin, cbmax, wid, hgt, predict_title, outfile)
                # Section of code used to separate into discrete values
                # After the predicted continuous array is created, modify the prediction to discrete values
                pred_discrete = np.round(pred)         # Using round creates a separation level of 0.5
                predicted_arr_discrete = np.round(predicted_arr)    # Used for image, not calculation
                # Determine Metrics and generate confusion matrix
                image_export_pathname = output_directory + configuration_string + 'Confusion.png'
                con_matrix, met_dict = get_eval_metrics_binary(pred_discrete, L2_answers, class_labels,
                                                               image_export_pathname, chan_select_string, met_dict)
                # Discrete color bar plot - Requires discrete matrix inputs
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
                wid = 16
                hgt = (comparison_plot_array.shape[0]/comparison_plot_array.shape[1]) * wid
                outfile = output_directory + configuration_string + 'comp_plt.png'
                print('Saving 4-color image to file')
                image_2d_data_dsrtcb(comparison_plot_array, clist, cbounds, keystring, wid, hgt, outfile)
                met_dict["Con_Mat_" + chan_select_string] = con_matrix
                # Plotting flashy plot with all the fun stuff here
                outfile = output_directory + configuration_string + 'together.png'
                print('Saving comparison data image to file')
                wid = 16
                hgt = (comparison_plot_array.shape[0]/comparison_plot_array.shape[1]) * wid
                neural_net_binary_plotter(chan_select_string, question_title, question_arry, con_matrix,
                                          predicted_title, predicted_arr, comparison_plot_array, class_labels, clist,
                                          cbounds, keystring, wid, hgt, outfile, met_dict)
                del CNN
                met_dict["Epochs"] = epoch_choice[epoch_used]
                met_dict["Dropout"] = dropout_choice[dropout_used]
                met_dict["Filters"] = filter_choice[filters_used]
                met_dict["Channels"] = chan_select_string
                metrics_dict_list.append(met_dict)

np.save(save_directory+"metrics_list_v2", metrics_dict_list)
print('Completed Runs of CNN configurations')
