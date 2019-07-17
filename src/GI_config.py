
"""
Date: July 14th, 2019
Author: Shantanu Neema

The purpose of this file is to set up global configuration parameters. These parameters are far less likely to change between runs
"""

import pathlib

# Amazon's url for case study, option 2
aws_url = 'http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip'

# file path to download the data from given link
notebook_path = pathlib.Path.cwd()
dir_path = notebook_path.parent
image_folder = str(dir_path) + '/geological_similarity/'

# Desired data folders as train, test and validation set for each label image
# Expected folder structure to let fastai library work
data_folder_list = ['test','train','valid']
# Expected folder structure for a 6 image type classification
class_folder_list = ['andesite','gneiss','marble','quartzite','rhyolite','schist']

# ratio of data going in train, test and validation folders respectively:
ratio_train = 0.8
ratio_test = 0.1
ratio_valid = 0.1

# image size input
image_size = 28
