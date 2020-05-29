# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""
import numpy as np
import cv2
import glob
import h5py
import pdb
from matplotlib import pyplot as plt

# import imutils

from PIL import Image

import avg_data
from lidar import *
from read_routines import *


"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"


n_channels = 12
l0_n_bins = 480
# For L0 frame, bg bins are 400-480
# For L2 frame, bg bins are 506-533 {end}
bg_st_bin = 400
bg_ed_bin = 480
width_to_separate = 300  # Number of records we would like each image to be


def crop(img, windowsize_r, windowsize_c):
    """
    This function splits one image into a list of several small images

    Parameters
    ----------
    img : ndarray
        image to be split
    windowsize_r : int
    windowsize_c : int
    NOTE: As of now, keep windowsize_r and windowsize_c as even divisors of the
    width and height of the image.
    i.e. for a 100 x 100 image, use windowsize_r = 10 or 5  and window_size = 10 or 5 etc.
    Failure to do so will result in the last image in the output list to crop off end data and 
    will be smaller than the others.
    desired sized of each image window

    Returns
    -------
    out : list[ndarray]
        list of split images

    """
    
    holder = []
    for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
            window = img[r:r+windowsize_r,c:c+windowsize_c]
            holder.append(window)
    
    return holder

def merge(img_lst, orig_shape):
    """
    This function takes a list of ndarrays and merges them back into the original image.
    
    Parameters
    ----------
    img_lst : list[ndarray]
        list containing images to be merged back together
    orig_shape : tuple
        shape of the original image before it was split.

    Returns
    -------
    out : ndarray
        recombined image
    """
    merged_lst = np.concatenate(img_lst,axis = 0)
    merged_lst = merged_lst.reshape((orig_shape))

def plot(X):
    """
    This is a plotting routine for visualizing training and test data
   
    Note first dimension of images represents the sample number so plotting
    plot(X) will result in an error. Use plot(X[sample_number]) sample number starts
    at zero
    """

def separate_images(full_image, width_of_separation):
    """
    This function splits one image into a list of several small images

    Parameters
    ----------
    full_image : ndarray of images spanning the first and third indices
        image to be split
    width_of_separation : int
    desired width of each image window

    Returns
    -------
    out : list[ndarray]
        list of split images
    """

    holder = []
    for r in range(0, full_image.shape[0] - width_of_separation, width_of_separation):
        window = full_image[r:r + width_of_separation, :]
        holder.append(window)
    window = full_image[-(width_of_separation+1):-1, :]
    holder.append(window)
    return holder


def plot(image_array):
    """
    This is a plotting routine for visualizing training and test data

    Note first dimension of images represents the sample number so plotting
    plot(X) will result in an error. Use plot(X[sample_number]) sample number starts
    at zero
    """
    plt.imshow(image_array[0, :, :].T, aspect='auto', cmap=get_a_color_map())
    plt.show()
    plt.imshow(image_array[1, :, :].T, aspect='auto', cmap=get_a_color_map())
    plt.show()
    
    return None


def remove_background_radiation(img):

    # Sample a region of input array that you know is
    # safely below the surface - for every profile.
    bg_reg_subset = img[:, :, bg_st_bin:bg_ed_bin]
    # Take the mean in the vertical, bins, dimension, thereby
    # creating an array that is shape (records, chans). This array
    # will be the solar background signal in each channel for each record.
    bg = np.mean(bg_reg_subset, axis=2)  # axis 0 is chans, axis 1 is bins, axis 2 is profiles
    # Create a 3D array of background to avoid looping. This 3D array's values will ONLY vary
    # with channel and record, NOT with bin. Shape will be (bin, records, chans)

    bg3D = np.tile(bg, (480, 1, 1))
    # Transpose shape to match input, 'img' array. (records, chans, bins)
    
    bg3D = np.transpose(bg3D, (1, 2, 0))
    # print('Shape of bg3D: ', bg3D.shape)
    # print('Shape of img: ', img.shape)

    # PERFORM REMOVAL OF BACKGROUND SIGNAL (background radiation)
    img = img - bg3D

    return img


# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy @Divakar
def onehot(a):

    """
    This function one hot codes an an array with N classification categories

    Parameters
    ----------
    a : ndarray
        Array to be one hot encoded.

    Returns
    -------
    out : ndarray
        One hot encoded array.

    """
    ncols = a.max() + 1

    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def get_input(l0_file, fixed_frame, original_frame, top_bin_limit, bot_bin_limit):
    # Added fixed_frame and original_frame inputs [5/22/20]
    raw_l0data = read_in_cats_l0_data(l0_file, n_channels, l0_n_bins)['chan'][:, -2:, :]
    print('Input array is shape : ', raw_l0data.shape)
    raw_l0data = avg_data.drop(raw_l0data)
    l0_fixed_frame = np.zeros((raw_l0data.shape[0], 2, fixed_frame.shape[0]))
    # Crop bottom off of L0 data because it's vertical frame actually
    # extends significantly below the extent of the L2 vertical frame.
    # For "split_bins..." function to work, the fixed frame extent needs
    # to encompass the entirety of the actual frame extent.
    crop_beyond = np.fabs(original_frame - fixed_frame[-1]).argmin() - 1
    # Note the 5 at the end of the line above is arbitrary to make up for ONA adjustment of bin altitude
    bg_sub_l0data = remove_background_radiation(raw_l0data)
    bg_sub_l0data = bg_sub_l0data[:, :, :crop_beyond]
    original_frame = original_frame[:crop_beyond]
    # Not sure this can done outside a loop w/o a lot of work. Hopefully it's fast enough.
    for profile in range(0, bg_sub_l0data.shape[0]):
        l0_fixed_frame[profile, :, :] = split_bins_onto_fixed_frame(fixed_frame, original_frame, bg_sub_l0data[profile, :, :])

    averaged_l0_fixed_frame = avg_data.avg_profs(l0_fixed_frame)
    # Crop off top, "no data" area
    averaged_l0_fixed_frame = averaged_l0_fixed_frame[:, :, top_bin_limit:bot_bin_limit]

    """
    Adding extra channel
    """

    # chan1 = averaged_l0_fixed_frame[:, :, 0]
    # chan2 = averaged_l0_fixed_frame[:, :, 1]
    chan1 = averaged_l0_fixed_frame[:, 0, :]
    chan2 = averaged_l0_fixed_frame[:, 1, :]

    # prod = np.transpose(np.array([chan1 * chan2]), (1, 2, 0))
    prod = chan1 * chan2
    prod = np.reshape(prod, (prod.shape[0], 1, prod.shape[1]))
    full_array = np.concatenate([averaged_l0_fixed_frame, prod], axis=1)
    print('Full_array shape: ', full_array.shape)

    cropped_full = separate_images(full_array, width_to_separate)
    return cropped_full


def get_targets(file, top_bin_limit, bot_bin_limit):
    """
    This function receives an hdf5 file and returns
    an array of the retrieved L2 data with all classes converted
    to 0 (nothing) or 1 (cloud or aerosol)
    Parameters
    ----------
    file : hdf5 file
        an L2 file
    Returns
    -------
    out : ndarray
        L2 data
    """
    hdf5 = h5py.File(file, 'r')
    target = np.asarray(hdf5['profile/Feature_Type_Fore_FOV'])
    hdf5.close()  # Close the hdf5 file
    # Resize Image
    target = target[top_bin_limit:bot_bin_limit, :]
    cropped_full = separate_images(np.transpose(target), width_to_separate)
    # target_r = cv2.resize(target, dsize=(1024, 512), interpolation=cv2.INTER_AREA)

    for target_element in np.arange(0, len(cropped_full)):
        cropped_full[target_element][np.where(cropped_full[target_element] > 0)] = 1
    # cropped_full[cropped_full != 0] = 1
    # target_one_hot = onehot(target_r)
    return cropped_full


# ********** Beginning of Main ********** #
# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"
# directory = "C:\\Users\\pselmer\\Documents\\CATS_ISS\\Data_samples\\NN_train\\"
directory = r"C:/Users/drusi/OneDrive/Desktop/CPL/train"

# Get fixed (L2) and original (L0) vertical bin frames
# Updated to array instead of file read on 5/27/2020
L0_bin_alt = np.linspace(27908.3, -9443.19, 480)
# Look at the top bin across all L2 files. It shouldn't change much.
L2_file_list = glob.glob('{}/*.hdf5'.format(directory))
for L2_file in L2_file_list:
    hdf5 = h5py.File(L2_file, 'r')
    try:
        # Will fail first time and then does something useless for remainder of reads
        print(L2_bin_alt.dtype)
    except:L2_bin_alt = np.asarray(hdf5['metadata_parameters/Bin_Altitude_Array']) * 1e3  # in km and changing to meters
    try:
        top_bin = np.concatenate((top_bin, np.asarray(hdf5['geolocation/Index_Top_Bin_Fore_FOV'])))
    except:
        top_bin = np.asarray(hdf5['geolocation/Index_Top_Bin_Fore_FOV'])
    hdf5.close()

print('\nL2 Top Bin stats')
print('min  :', top_bin.min())
print('mean: ', top_bin.mean())
print('stdv: ', top_bin.std())
print('max:  ', top_bin.max(), '\n')
# Overwrite top_bin array with single value to use for top bin
# On 5/27/2020 added .round() below to make sure selected top bin is closest integer to mean
top_bin = int(top_bin.mean().round())
# Since the L0 alt frame extends below L2 frame, bot_bin will be last bin
bot_bin = L2_bin_alt.shape[0]
print('Only use data between vertical bins {0:d} and {1:d}'.format(top_bin, bot_bin))

x_lst = []
nn = 1

for input_file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, input_file))
    l0_img = get_input(input_file, L2_bin_alt, L0_bin_alt, top_bin, bot_bin)
    for element in np.arange(0, len(l0_img)):
        x_lst.append(l0_img[element])
    nn += 1

np.save('questions', x_lst)

print("Questions completed, moving onto Answers!")
# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"
# directory = "C:\\Users\\pselmer\\Documents\\CATS_ISS\\Data_samples\\NN_train\\"
directory = "C:\\Users\\akupchoc.NDC\\Documents\\Work\\Jetson TX2\\CATS_ML\\NN_train\\"

nn = 1
t_lst = []
for target_file in glob.glob('{}/*.hdf5'.format(directory)):
    print("Reading {} {}...".format(nn, target_file))
    l2_img = get_targets(target_file, top_bin, bot_bin)
    for image_sub in np.arange(0, len(l2_img)):
        t_lst.append(l2_img[image_sub])
    nn += 1
# Need to check with Daniel to see if this needs to be 3 dimensional

np.save('answers', t_lst)
print("Answers completed!")

directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"

x_lst = []
nn = 1
for input_file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, input_file))
    img = get_input(inut_file, L2_bin_alt, L0_bin_alt, top_bin, bot_bin)        
    x_lst.append(img)
    nn+=1
    
np.save('test_questions', x_lst)


directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"

t_lst = []
for file in glob.glob('{}/*.hdf5'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_targets(file) 
    t_lst.append(img)
    nn+=1
    
np.save('test_answers', t_lst)
