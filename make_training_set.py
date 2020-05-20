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




"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"


nchans=12
nbins=480
bg_st_bin = 400
bg_ed_bin = 480


lst = []

def window(X, shape):
    pdb.set_trace()
    X = np.arange(10000).reshape((100,2,50))
    X = np.transpose(X,(0,2,1))
    X = X.reshape(-1,10,5,2)
    return None
    

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


   
def get_input(file):
    X = read_routines.read_in_cats_l0_data(file, nchans, nbins)['chan'][:,-2:,:]
    X = remove_background_radiation(X)
    #X = np.transpose(X, (1,2,0))
   
    X = avg_data.drop(X)

    X = avg_data.avg_profs(X)

    """
    Adding extra channel
    """
    
    chan1 = X[:,:,0]
    chan2 = X[:,:,1]

    prod = np.transpose(np.array([chan1 * chan2]), (1,2,0))
    full = np.concatenate([X, prod], axis=2) 

    full = cv2.resize(full, dsize=(1024,512), interpolation=cv2.INTER_CUBIC)

    plt.imshow(X[:,:,0], aspect='auto', cmap=lidar.get_a_color_map())
    plt.show()
    return full

def get_targets(file):

    hdf5 = h5py.File(file, 'r')
    target = np.asarray(hdf5['profile/Feature_Type_Fore_FOV'])
    #Resize Image

    target_r = cv2.resize(target, dsize=(1024,512), interpolation=cv2.INTER_CUBIC)
    target_r[target_r != 0] = 1
    #target_one_hot = onehot(target_r)
    return target_r



directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"

x_lst = []
nn = 1
for file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_input(file)
    print("BOLAY")
    pdb.set_trace()
    img = window(img)   
    x_lst.append(img)
    nn+=1
 
#for i in x_lst:
#    print(i.shape)  
#np.save('questions', x_lst)


# directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"

# nn = 1

# t_lst = []
# for file in glob.glob('{}/*.hdf5'.format(directory)):
#     print("Reading {} {}...".format(nn, file))
#     img = get_targets(file)
#     t_lst.append(img)
#     nn+=1
    
# for i in t_lst:
#     print(i.shape)

# np.save('answers', t_lst)


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



