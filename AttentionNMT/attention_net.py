# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        net.py
# Purpose:     
#
#              inputs: 
#
#              outputs: 
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     08/01/2018 (DD/MM/YY)
# Last update: 08/01/2018 (DD/MM/YY)
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

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()

        # init scope for layer (modules) WITH parameters <-- to detect by backward/optimizer/updater?
        with self.init_scope():
            self.l1 = L.Linear(None, n_units) # None means input size is automatically infereed
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
        # end with
    # end init

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
    # end def

# end MLP-classs


