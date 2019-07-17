#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Shantanu Neema
Date: July 14th, 2019

The purpose of this file is to run the main program for geological image similarity problem
"""

import os
import sys
import pathlib
import pickle
import pandas as pd
import numpy as np
import shutil

'''
Data Science Case Study
Codes to download and  extract the data for "geological_similarity" problem
link provided in file: 'Data Science Case Study Options 1.0.pdf' 
Option 2: Geological Image Similarity
'''

# Make use of fastai library for computer vision applications
from fastai.vision import *
from fastai.callbacks.hooks import *
from PIL import Image
# from tqdm import tqdm_notebook

import GI_config as conf_lib
import GI_load as load_lib

#fine grain classification problem
# Desired data folders as train, test and validation set for each label image

working_folder = str(pathlib.Path.cwd().parent) + '/src'
os.chdir(working_folder)

import split_data_folder 
import GI_config as conf_lib

print('--------------------------------------------------')
print('1. CLEAN AND CONFIGURE THE INPUT GEOLOGICAL IMAGE DATA')


# to let images flip in horizontally and vertically

geo_data = load_lib.load_and_restruct_data(dir_path = conf_lib.dir_path,
                                           image_folder = conf_lib.image_folder,
                                           image_size = conf_lib.image_size,
                                           ratio_train = conf_lib.ratio_train,
                                           ratio_test = conf_lib.ratio_test,
                                           ratio_valid = conf_lib.ratio_valid,
                                           data_folder_list = conf_lib.data_folder_list,
                                           class_folder_list = conf_lib.class_folder_list,
                                           aws_url = conf_lib.aws_url,
                                           seed=42)

print('geological image data loaded')
print('{0} classes are:'.format(geo_data.c))
print('     ',geo_data.classes)
print('Train dataset size: {0}'.format(len(geo_data.train_ds.x)),'images')
print('Test dataset size: {0}'.format(len(geo_data.valid_ds.x)),'images')
