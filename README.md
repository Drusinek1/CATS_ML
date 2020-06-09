# CATS_ML

## Descripton:
This repository contains code for preprocessing CATS data and training a convolutional neural network to 
classify cloud and aerosol layers.


## System Requirements
- Python 3.X
- Numpy 1.14 or later
- Runs 64 bit windows


##Notes:
[6/9/2020]
As of today, code to format L0 and L2 data for training as .npy files is located in make_training_set.py.  
Be sure to change name of directory on line 252 to the location of the L0 and L2 data files and save_directory 
on line 253. Code must be run twice, once for training data and once for testing data. For each run be sure to change
name of output files on line 302, 303 and 323.

