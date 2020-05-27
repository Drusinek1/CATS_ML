# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:10:46 2020

@author: drusinek
"""
import read_routines
import numpy as np
import pdb
from matplotlib import pyplot as plt
import lidar
import glob 



nchans=12
nbins=480
bg_st_bin = 400
bg_ed_bin = 480



def remove_background_radiation(img):

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





def drop(A):
    for prof in A:
        if A[0][0][0] == 2**14:
            A = np.delete(A,0,0)
    # A = 
    # tmp = np.copy(A)
    # tmp = np.delete(tmp,idxs,axis=2) 
    return A

    
def avg_profs(X):

    X = np.transpose(X, (0,2,1))
    width = X.shape[0]
    #channels first
    #target 2048 width
    split_constant = width // 14
    split_X = np.array_split(X, split_constant, axis=0)
    tmp = []
    for chunk in split_X:
        n_chunk = np.average(chunk, axis=0)
        tmp.append(n_chunk)
    combined = np.stack(tmp, axis=0)

    return combined
    
        
        
   

