# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:34:15 2020

@author: drusi
"""
import numpy as np
from matplotlib import pyplot as plt
import lidar
import pdb
from PIL import Image
from matplotlib import pyplot as plt
from scipy import misc

impath = r"C:/Users/drusi/OneDrive/Desktop/CPL/george_washington.jpg"
test_image = Image.open(impath)
test_image = np.array(test_image)

orig_shape = test_image.shape

plt.figure()
plt.imshow(test_image)
plt.title("Before Windowing")





# Define the window size
windowsize_r = 5
windowsize_c = 5

def crop(img, windowsize_r, windowsize_c):
    holder = []
    for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
            window = test_image[r:r+windowsize_r,c:c+windowsize_c]
            holder.append(window)
    #convert list containing smaller images into an ndarray
    #of size (x,windowsize_r, windowsize_c) where x is the number of total images
    
    return np.stack(holder, axis=0)





cropped = crop(test_image,100,100)

for idx, img in enumerate(cropped):
    plt.figure()
    plt.imshow(img)
    plt.title("Window # {}".format(idx))

