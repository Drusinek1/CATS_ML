# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:10:46 2020

@author: drusinek


This script contains several functions for manipulating L0 CATS data
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





def drop_flagged_profiles(A):
    """
    
    Parameters
    ----------
    A : ndarray
        numpy array of raw data.

    Returns
    -------
    A : ndarray
        data with flagged profiles removed

    """
    for prof in A:
        if A[0][0][0] == 2**14:
            A = np.delete(A,0,0)

    return A

    
def avg_profs(X):
    """
    This function averages together every 14 profiles of 
    X and returns the resulting ndarray

    Parameters
    ----------
    X : ndarray
        Photon count array.

    Returns
    -------
    combined : ndarray
        ndarray with every 14 profiles averages together.

    """

    #channels first
    X = np.transpose(X, (0,2,1))


    width = X.shape[0]

    #Determine how many chunks to create
    split_constant = width // 14 
    
    #split into chunks of 14
    split_X = np.array_split(X, split_constant, axis=0)
    holder = []
    
    #Average together each chunk and append onto holder
    for chunk in split_X:
        n_chunk = np.average(chunk, axis=0)
        holder.append(n_chunk)
    
    #Combine all arrays in holder into a numpy array
    combined = np.stack(holder, axis=0)

    return combined
    





        
        
   

