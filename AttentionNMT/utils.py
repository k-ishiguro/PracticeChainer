# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        utils.py
# Purpose:     A Chainer implementation of Attention NMT: some small functions
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     13/02/2018 (DD/MM/YY)
# Last update: 13/02/2018 (DD/MM/YY)
#-------------------------------------------------------------------------------

import io
import sys
import os
import argparse
# import commands
# import time
# import csv
# import glob
# import collections
# from collections import Counte
import pickle

import numpy as np
# import scipy
# from scipy import sparse

# import numba

import matplotlib

#matplotlib.use('Agg')  # do it background
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from matplotlib import rcParams

#rcParams['font.family'] = 'IPAGothic'

# import pandas as pd
# from pandas import Series, DataFrame

# import boto3

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import cuda

import encoder, decoder, attention, generator

def checkVariables(var):
    """
    report the status of variabe, var

    param: var -- some python variable
    """

    print("hs is: ")
    print("type=" + type(var))
    if len(var):
        print("len=" + len(var))        
    # end np.len-if
    if np.shape(var):
        print("shape" + np.shape(var))
        shapes = np.shape(var)
        if len(var) > 1: # go recursive
            checkVariables(var[0])
        # end 
    # end np.shape-if

# end checkVariables-def