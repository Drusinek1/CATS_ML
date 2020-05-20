# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:14:59 2020

@author: drusi
"""
from datetime import datetime
import numpy as np
import pdb
import read_routines
from matplotlib import pyplot as plt
import lidar
import cv2

now = datetime.now()

nchans=12
nbins=480
file = "C:\\Users\\drusi\\OneDrive\\Desktop\\CPL\\CATS\\train\\CATS_ISS_L0_D-M7.2-2017-10-01T18-40-01-T18-53-31.dat"

def remove_background_radiation(img):
    img = img.astype(float)
    n_img = np.zeros_like(img)
    for chan in range(0,img.shape[0]-1):
        B = img[chan,:,-50:]
        B_avg = (np.mean(B, axis=1))
        #pdb.set_trace()
        for row in range(0,img.shape[1]-1):
            for col in range(0,img.shape[2]-1):
                n_img[chan][row][col] = img[chan][row][col] - B_avg[row]
                #print("replaced {} with {}".format(img[chan][row][col], img[chan][row][col]-B_avg[col]))
    return n_img


X = read_routines.read_in_cats_l0_data(file, nchans, nbins)['chan']
X = np.transpose(X, (1,0,2))
Xf = remove_background_radiation(X)


re = np.zeros((12,2048,512))
for chan in range(0, Xf.shape[0]-1):
    tmp = cv2.resize(Xf[chan], dsize=(512,2048), interpolation=cv2.INTER_CUBIC)
    re[chan] = tmp

cmap = lidar.get_a_color_map()
# plt.imshow(re[11], cmap=cmap)
print(datetime.now()-now)