# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:28:30 2020

@author: Dani
"""
import numpy as np
import h5py 
import mutable_data_structs as mds
import immutable_data_structs as ids
import tensorflow as tf
import pdb
import neural_nets
import cpl2tensor
from skimage import transform
from matplotlib import pyplot as plt

print("Start")
CLS_meta = mds.define_CLS_meta_struct(256)

nbins = 833
nchans = 4

CLS_struct = mds.define_CLS_structure(nchans, nbins, CLS_meta)
input_shape = (2560, 1440, 4)


day1 = cpl2tensor.extractData(r"C:\Users\drusi\CPLCNN\Training_data\20-012")
day2 = cpl2tensor.extractData(r"C:\Users\drusi\CPLCNN\Training_data\20-013")
day3 = cpl2tensor.extractData(r"C:\Users\drusi\CPLCNN\Training_data\20-014")
day4 = cpl2tensor.extractData(r"C:\Users\drusi\CPLCNN\Training_data\20-015")

day1_r = transform.resize(day1, input_shape)
day2_r = transform.resize(day2, input_shape)
day3_r = transform.rescale(day3, input_shape)
day4_r = transform.rescale(day4, input_shape)

target_shape = (2560, 1440, 1)
filename = r"C:/Users/drusi/CPLCNN/Training_targets/CPL_L2_V1-02_01kmPro_20012_16dec19.hdf5"
hdf5 = h5py.File(filename, 'r')
target1 = np.array(hdf5['profile/Feature_Type'][:,:])

filename = r"CPL_L2_V1-02_01kmPro_20013_15jan20.hdf5"
hdf5 = h5py.File(filename, 'r')
target2 = np.array(hdf5['profile/Feature_Type'][:,:])

# filename = r"C:/Users/drusi/CPLCNN/Training_targets/CPL_L2_V1-02_01kmPro_20014_18jan20.hdf5"
# hdf5 = h5py.File(filename, 'r')
# target3 = np.array(hdf5['profile/Feature_Type'][:,:])

# filename = r"C:/Users/drusi/CPLCNN/Training_targets/CPL_L2_V1-02_01kmPro_20015_25jan20.hdf5"
# hdf5 = h5py.File(filename, 'r')
# target4 = np.array(hdf5['profile/Feature_Type'][:,:])


t1 = np.transpose(target1, (1,0))
t1_r = transform.resize(t1, target_shape)

t2 = np.transpose(target2, (1,0))
t2_r = transform.resize(t2, target_shape)

# t3 = np.transpose(target3, (1,0))
# t3_r = transform.resize(t3, target_shape)


# t4 = np.transpose(target4, (1,0))
# t4_r = transform.resize(t4, target_shape)


CNN = neural_nets.UNet((2560, 1440, 4))

inputs = np.array([day1_r, day2_r])
targets = np.array([t1_r, t2_r])

CNN.model.fit(inputs, targets, epochs=10, verbose=1)
prediction = CNN.model.predict(np.array([day4_r]))

print("End")
