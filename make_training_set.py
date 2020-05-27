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


from PIL import Image



"""
Formatting Input Training Data
"""
path = r"C:/Users/drusi/OneDrive/Desktop/CPL/CATS"


nchans=12
nbins=480
bg_st_bin = 400
bg_ed_bin = 480


def crop(img, windowsize_r, windowsize_c):
    """
    This function splits one image into a list of several small images
    
    Parameters
    ----------
    img : ndarray
        image to be split
    windowsize_r : int
    windowsize_c : int
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

def plot(X):
    """
    This is a plotting routine for visualizing training and test data
   
    Note first dimension of images represents the sample number so plotting
    plot(X) will result in an error. Use plot(X[sample_number]) sample number starts
    at zero
    """



        

def plot(X):
    plt.imshow(X[0,:,:].T, aspect='auto', cmap=lidar.get_a_color_map())
    plt.show()
    plt.imshow(X[1,:,:].T, aspect='auto', cmap=lidar.get_a_color_map())
    plt.show()
    
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

    """
    This function one hot codes an an array with N classifcation categories

    Parameters
    ----------
    a : ndarray
        Array to be one hot encoded.

    Returns
    -------
    out : ndarray
        One hot encoded array.

    """
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out



# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)



#Path to directory containin training .dat files and .hdf5
directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"

x_lst = []

#counter for for loop
nn = 1

#Format each dat file in directory with get_input and append onto x_lst

def get_input(X):
    X = read_routines.read_in_cats_l0_data(file, nchans, nbins)['chan'][:,-2:,:]
    X = remove_background_radiation(X)
 

    X = avg_data.drop_flagged_profiles(X)

    X = avg_data.avg_profs(X)


    """
    Adding extra channel
    """
    
    chan1 = X[:,:,0]
    chan2 = X[:,:,1]

    prod = np.transpose(np.array([chan1 * chan2]), (1,2,0))
    full = np.concatenate([X, prod], axis=2) 
    cropped_full = crop(full,10,10)
    cropped_full = np.stack(cropped_full, axis=0)
    
    return cropped_full

def get_targets(file):

    hdf5 = h5py.File(file, 'r')
    target = np.asarray(hdf5['profile/Feature_Type_Fore_FOV'])
    target = crop(target,10,10)
    target = np.stack(target, axis=0)

    # target_r = cv2.resize(target, dsize=(1024,512), interpolation=cv2.INTER_AREA)
    # target_r[target_r != 0] = 1
    #target_one_hot = onehot(target_r)
    
    return target



directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\train"

x_lst = []
nn = 1

for file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_input(file)

    x_lst.append(img)
    nn+=1


np.save('questions', x_lst)


nn = 1

#Format each hdf5 file in directory with get_target and append onto t_lst

t_lst = []
for file in glob.glob('{}/*.hdf5'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_targets(file)
    t_lst.append(img)
    nn+=1

#save formatted dataset as a binary .npy file for later use
np.save('answers', t_lst)

#directory containing test dat and hdf5 data
directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"

x_lst = []
nn = 1

#Same spiel as above but for training data
for file in glob.glob('{}/*.dat'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_input(file)        
    x_lst.append(img)
    nn+=1
    
np.save('test_questions', x_lst)


directory = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\test"
nn = 1
t_lst = []
for file in glob.glob('{}/*.hdf5'.format(directory)):
    print("Reading {} {}...".format(nn, file))
    img = get_targets(file) 
    t_lst.append(img)
    nn+=1
    
np.save('test_answers', t_lst)




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



