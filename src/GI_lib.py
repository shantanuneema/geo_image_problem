#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Shantanu Neema
Date: July 14th, 2019

The purpose of this file is to create any additional functions or classes needed to develop the GI-Similarity tool
"""

import decimal

def format_decimal(x, prec=2):
    x = decimal.Decimal(x)
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    return '{sign}{int}.{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)

def get_accuracy(out_list):
    return round(out_list[1].item()*100,2)