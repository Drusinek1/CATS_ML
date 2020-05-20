# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:24:07 2020

@author: drusi

This script reads in CPL .cls files and L2 files and converts them to 
tensors for use with Tensorflow.
"""
import numpy as np
import collections
import os
import mutable_data_structs 
import immutable_data_structs
import pdb
import mutable_data_structs 
import immutable_data_structs 
import glob


def extractData(directory):
    CLS_meta = mutable_data_structs.define_CLS_meta_struct(256)
    
    nbins = 833
    nchans = 4
    
    CLS_struct = mutable_data_structs.define_CLS_structure(nchans, nbins, CLS_meta)
    
    #directory = r"C:\Users\drusi\CPLCNN\Training_data\20-012"
    dic = collections.OrderedDict()
    
    for file in glob.glob('{}/*.cls'.format(directory)):
        
        raw_data = np.fromfile(file, dtype=CLS_struct)
        file_num = os.path.basename(os.path.normpath(file))[:-4]
        dic["photon_count_{}".format(file_num)] = raw_data['counts']
 

    input_tensor = np.transpose(np.concatenate([dic[x] for x in dic], 0), axes = (1, 0, 2))
    return input_tensor
