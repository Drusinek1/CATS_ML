# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:03:33 2020

@author: drusi
"""
import numpy as np
import read_routines
import cv2
from matplotlib import pyplot as plt
import glob
import h5py
import pdb
from matplotlib import pyplot as plt
import lidar


"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS/train/CATS_ISS_L0_D-M7.2-2017-10-01T18-40-01-T18-53-31.dat"


nchans=12
nbins=480
bg_st_bin = 400
bg_ed_bin = 480



def remove_background_radiation(img):
     print("img.shape = ", img.shape)
     # Sample a region of input array that you know is
     # safely below the surface - for every profile.
     bg_reg_subset = img[:, :, bg_st_bin:bg_ed_bin]
     # Take the mean in the vertical, bins, dimension, thereby
     # creating an array that is shape (records, chans). This array
     # will be the solar background signal in each channel for each record.
     bg = np.mean(bg_reg_subset, axis=2) # axis 0 is chans, axis 1 is bins, axis 2 is profiles
     # Create a 3D array of background to avoid looping. This 3D array's values will ONLY vary
     # with channel and record, NOT with bin. Shape will be (bin, records, chans)
     bg3D = np.tile(bg, (480,1,1))
     # Transpose shape to match input, 'img' array. (records, chans, bins)
   
     bg3D = np.transpose(bg3D, (1,2,0))
     # print('Shape of bg3D: ', bg3D.shape)
     # print('Shape of img: ', img.shape)
     # PERFORM REMOVAL OF BACKGROUND SIGNAL (background radiation)
     img = img - bg3D
     return img

def get_input(file):
    print(file)
    X = read_routines.read_in_cats_l0_data(file, nchans, nbins)['chan'][:,-2:,:]
    full = remove_background_radiation(X)
    full = np.transpose(full, (0,2,1))
    # print("blah")
    # lst = []
    # for chan in range(0,full.shape[1]-1):   #<-------------- ERROR Runs out of RAM
    #     tmp = cv2.resize(full[:,chan,:], dsize=(512,2048), interpolation=cv2.INTER_CUBIC)
    #     lst.append(tmp)
    pdb.set_trace()
    return full

X = get_input(path)

X_r = cv2.resize(X, dsize=(512,4096), interpolation=cv2.INTER_CUBIC)
cmap = lidar.get_a_color_map()
plt.imshow(X_r[:,:,0], cmap)
