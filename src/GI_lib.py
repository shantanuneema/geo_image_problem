#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Shantanu Neema
Date: July 14th, 2019

The purpose of this file is to create any additional functions or classes needed to develop the GI-Similarity tool
"""

import decimal
from fastai.vision import *
from annoy import AnnoyIndex
from PIL import Image

# Function to format decimals to scientific notations
def format_decimal(x, prec=2):
    x = decimal.Decimal(x)
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    return '{sign}{int}.{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)

# Function to get accuracy from learn object tensor
def get_accuracy(out_list):
    return round(out_list[1].item()*100,2)

# Functions to get output, input and module names (to facilitate image similarity class hook)
def get_output(module, input_value, output):
    return output.flatten(1)

def get_input(module, input_value, output):
    return list(input_value)[0]

def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

# Function to get similar images using Annoy
def get_similar_images_annoy(img_index, img_repr_df, inference_data, dist_type, k):
    
    feature_dim = len(img_repr_df['img_repr'][0])
    t = AnnoyIndex(feature_dim, metric=dist_type)

    for i, vector in enumerate(img_repr_df['img_repr']):
        t.add_item(i, vector)

    _  = t.build(inference_data.c)
    
    base_img_id, base_vector, base_label  = img_repr_df.iloc[img_index, [0, 1, 2]]
    similar_img_ids = t.get_nns_by_item(img_index, k+1)
    return base_img_id, base_label, img_repr_df.iloc[similar_img_ids[1:]]

# Function to show similar images based on given input
def show_similar_images(df, learn):
    images = [open_image(img_id) for img_id in df['img_id']]
    categories = [learn.data.train_ds.y.reconstruct(y) for y in df['label_id']]
    return learn.data.show_xys(images, categories)

