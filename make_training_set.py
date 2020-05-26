# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:35:13 2020
@author: drusi
"""
import numpy as np
import read_routines
import cv2
import glob
import h5py
import pdb
import avg_data
import lidar
from matplotlib import pyplot as plt
import imutils
from PIL import Image



"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"


nchans = 12
nbins = 480
# For L0 frame, bg bins are 400-480
# For L2 frame, bg bins are 506-533 {end}
bg_st_bin = 506  # 400
bg_ed_bin = 533  # 480


def crop(im, d): 
    # Channels first
    im = np.transpose(im, (1, 2, 0))
    cnt = 0
    stopw = im.shape[2] // d
    print("stopw = {}".format(stopw))
    # calculate p
    p = im.shape[2] // stopw

    tmp_lst = []
    for i in range(0, stopw):
        window = im[:, 0+cnt:cnt+d, 0+cnt:cnt+p]
        tmp_lst.append(window)
        cnt += 1
      
    print("Cropped {} samples".format(cnt))
    return np.asarray(tmp_lst)


def plot(X):
    plt.imshow(X[0, :, :].T, aspect='auto', cmap=lidar.get_a_color_map())
    plt.show()
    plt.imshow(X[1, :, :].T, aspect='auto', cmap=lidar.get_a_color_map())
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

    # bg3D = np.tile(bg, (480, 1, 1))
    bg3D = np.tile(bg, (533, 1, 1))
    # Transpose shape to match input, 'img' array. (records, chans, bins)
    
    bg3D = np.transpose(bg3D, (1, 2, 0))
    # print('Shape of bg3D: ', bg3D.shape)
    # print('Shape of img: ', img.shape)

    # PERFORM REMOVAL OF BACKGROUND SIGNAL (background radiation)
    img = img - bg3D

    return img


# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy @Divakar
def onehot(a):
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def get_input(file, fixed_frame, original_frame, top_bin, bot_bin):
    # Added fixed_frame and original_frame inputs [5/22/20]

    X = read_routines.read_in_cats_l0_data(file, nchans, nbins)['chan'][:, -2:, :]
    X = avg_data.drop(X)
    X_fixed = np.zeros((X.shape[0], 2, fixed_frame.shape[0]))
    # Crop bottom off of L0 data because it's vertical frame actually
    # extends significantly below the extent of the L2 vertical frame.
    # For "split_bins..." function to work, the fixed frame extent needs
    # to encompass the entirety of the actual frame extent.
    crop_beyond = np.fabs(original_frame - fixed_frame.min()).argmin() - 5
    X = X[:, :, :crop_beyond]
    original_frame = original_frame[:crop_beyond]
    # Not sure this can done outside a loop w/o a lot of work. Hopefully it's fast enough.
    for i in range(0, X.shape[0]):
        X_fixed[i, :, :] = lidar.split_bins_onto_fixed_frame(fixed_frame, original_frame, X[i, :, :])
    X = remove_background_radiation(X_fixed)
    X = avg_data.avg_profs(X)

    # Crop off top, "no data" area
    X = X[:, :, top_bin:bot_bin]

    """
    Adding extra channel
    """
    
    chan1 = X[:, :, 0]
    chan2 = X[:, :, 1]

    prod = np.transpose(np.array([chan1 * chan2]), (1, 2, 0))
    full = np.concatenate([X, prod], axis=2) 
    cropped_full = crop(full, 25)

    return cropped_full


def get_targets(file, top_bin, bot_bin):

    hdf5 = h5py.File(file, 'r')
    target = np.asarray(hdf5['profile/Feature_Type_Fore_FOV'])
    hdf5.close()  # Close the hdf5 file
    # Resize Image
    target = target[top_bin:bot_bin, :]
    target_r = cv2.resize(target, dsize=(1024, 512), interpolation=cv2.INTER_AREA)
    target_r[target_r != 0] = 1
    # target_one_hot = onehot(target_r)
    return target_r


directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"
directory = "C:\\Users\\pselmer\\Documents\\CATS_ISS\\Data_samples\\NN_train\\"

# Get fixed (L2) and original (L0) vertical bin frames
L0_bin_alt_file = "C:\\Users\\pselmer\\Documents\\CATS_ISS\\codes\\unix_codes_backup\\representative_altitude_bin_array_aft_FOV_wrt_instrument_coord_system.txt"
with open(L0_bin_alt_file, 'r') as f_obj:
    lines = f_obj.readlines()
    L0_bin_alt = np.array([float(line) for line in lines])
# Look at the top bin across all L2 files. It shouldn't change much.
L2_file_list = glob.glob('{}/*.hdf5'.format(directory))
for L2_file in L2_file_list:
    hdf5 = h5py.File(L2_file, 'r')
    try:
        print(L2_bin_alt.dtype)
    except:
        L2_bin_alt = np.asarray(hdf5['metadata_parameters/Bin_Altitude_Array']) * 1e3  # in m
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
top_bin = int(top_bin.mean())
# Since the L0 alt frame extends below L2 frame, bot_bin will be last bin
bot_bin = L2_bin_alt.shape[0]
print('Only use data between vertical bins {0:d} and {1:d}'.format(top_bin, bot_bin))

x_lst = []
nn = 1
for file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_input(file, L2_bin_alt, L0_bin_alt, top_bin, bot_bin)
    x_lst.append(img)
    nn += 1

"""
TODO:

"""
for i in x_lst:
    print(i.shape)  
np.save('questions', x_lst)


# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"
directory = "C:\\Users\\pselmer\\Documents\\CATS_ISS\\Data_samples\\NN_train\\"

nn = 1

t_lst = []
for file in glob.glob('{}/*.hdf5'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_targets(file)
    t_lst.append(img)
    nn += 1
    
for i in t_lst:
    print(i.shape)

np.save('answers', t_lst)
pdb.set_trace()

# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"

# x_lst = []
# nn = 1
# for file in glob.glob('{}/*.dat'.format(directory)):
#     print("Reading {} {}...".format(nn, file))
#     img = get_input(file)        
#     x_lst.append(img)
#     nn+=1
    
# np.save('test_questions', x_lst)


# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"

# t_lst = []
# for file in glob.glob('{}/*.hdf5'.format(directory)):
#     print("Reading {} {}...".format(nn, file))
#     img = get_targets(file) 
#     t_lst.append(img)
#     nn+=1
    
# np.save('test_answers', t_lst)



