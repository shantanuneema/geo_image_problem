#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Shantanu Neema
Date: July 14th, 2019

The purpose of this file is to check if all data folders are available and restore the data as needed
"""

import io
import os
import pathlib
import shutil
import requests
import zipfile
import sys
from sys import platform, argv

from fastai.vision import *
from fastai.callbacks.hooks import *

import split_data_folder 

'''
Data Science Case Study
Codes to download and  extract the data for "geological_similarity" problem
link provided in file: 'Data Science Case Study Options 1.0.pdf' 
Option 2: Geological Image Similarity
'''

def load_and_restruct_data(dir_path,
                           image_folder,
                           image_size,
                           ratio_train,
                           ratio_test,
                           ratio_valid,
                           data_folder_list,
                           class_folder_list,
                           aws_url,
                           seed=None):

    cpt = sum([len(files) for r, d, files in os.walk(image_folder)])

    if os.path.isdir(image_folder) and cpt > 25000 and all([os.path.isdir(image_folder+f) \
                                                           for f in class_folder_list+data_folder_list]):
        print('Removing original folders')
        for f in class_folder_list:
            shutil.rmtree(os.path.join(image_folder,f))            
        print('Data is stored as train, test & validation folders in', dir_path)
    elif os.path.isdir(image_folder) and cpt > 25000 and all([os.path.isdir(image_folder+f) \
                                                           for f in data_folder_list]):
        print('Data is stored as train, test & validation folders in', dir_path)
    else:
        try:
            # unzipped the data from the handler
            print('Downloading and storing data in ', dir_path,'/geological_similarity...',sep='')
            res = requests.get(aws_url)
            zipped_geo_data = zipfile.ZipFile(io.BytesIO(res.content))
            zipped_geo_data.extractall(dir_path)
            print('Data is stored in', dir_path)
            print('Rearraging data in test, train and valid folders')
            split_data_folder.ratio(image_folder,
                                    output=image_folder,
                                    seed=seed, 
                                    ratio =(ratio_train,ratio_test,ratio_valid))
        except:
            sys.exit('Error downloading the data from the given url', aws_url)

    current_folders = [f for f in os.listdir(image_folder) if not f.startswith('.')]

    if not set(data_folder_list).issubset(set(current_folders)):
        sys.exit('Input data is invalid, the input folder structure is incorrect')


    # Following method to remove the original folders if needed
    if os.path.isdir(image_folder):
        folder_list = [f for f in os.listdir(image_folder) if not f.startswith('.') and f not in data_folder_list]
        for f in folder_list:
            if f:
                shutil.rmtree(os.path.join(image_folder, f))
                print('   removing folder', f, 'from the directory') 

    tfms = get_transforms(do_flip=True, flip_vert=True)
    geo_data = ImageDataBunch.from_folder(image_folder, ds_tfms = tfms, size = image_size)
    geo_data.normalize(imagenet_stats)
    
    return geo_data, tfms