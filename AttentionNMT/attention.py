# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        attention.py
# Purpose:     A Chainer implementation of Attention NMT: Attention layer
#              Compute the context vector, augment the decoder output w/ the context vector and return it.
#
#              inputs: sequence of encoder hidden states(of the top layer)
#                      one decoder hidden state @ time t
#              outputs: augmented decoder hidden state vector @ time t
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     13/01/2018 (DD/MM/YY)
# Last update: 11/02/2018 (DD/MM/YY)
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
from chainer import cuda

class GlobalAttention(chainer.Chain):
    """
    Luong+ (EMNLP 2015)-type "simple" attention, as adopted in OpenNMT and OpenNMT-py
    """

    def __init__(self, lstm_dim, gpu):
        """
        initializer.

        :param lstm_dim: dimension of the hidden states,
        :param gpu: gpu id
        :return: no returns.
        """
        super(GlobalAttention, self).__init__()

        self.lstm_dim = lstm_dim
        global xp
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np

        # init scope for layer (modules) WITH parameters <-- to detect by backward/optimizer/updater?
        with self.init_scope():
            self.W = L.Linear(in_size=lstm_dim, out_size=lstm_dim, nobias=True) # coefficient matrix for "general" score
            self.C = L.Linear(in_size=lstm_dim*2, out_size=lstm_dim*2, nobias=True) # coefficient matrix for decoder augmentation
        # end with
    # end init

    def __call__(self, xs, h):
        """
        compute the context vector.then augment the decoder hidden state with the context vector

        :param xs: chainer variable, B by max_seq_len by D numpy array, B-list of sequences of D-dim encoder hidden states, B is the minibatch size
        :param h: chaienr Variable, consists of B by D numpy array, B-list of the decoder hidden state of the focused time step
        :return: B-list of augmented decoder hidden sate w/ context vector. B by D*2-dim numpy array,
        """
        B = len(xs)

        Wh = self.W(h) # B by lstm_dim
        Wh = Wh[:, :, xp.newaxis]
        Wh.data.astype(xp.float32) # B by lstm_dim by 1

        # xs is B by max_seq_len by lstm_dim
        xWh = F.matmul(xs, Wh) # should be B by max_seq_len 
        attention = F.softmax(xWh) # should be B by max_seq_len 
        
        # weighted sum of xs.
        context_vec = F.sum( F.scale( xs, attention, axis=0 ), axis=1)
                
        # concat and nonlinear transform
        concated_vec = F.hstack( (h, context_vec) ) # should be B by 2*lstm_dim
        augmented_vec = F.tanh( self.C( concated_vec )) # should be B by 2*lstm_dim
        
        print("####### for DEBUG: Computing Attention weight done.######")
        print("h is:")
        print(h.dtype)
        print(np.shape(h))
        print(h[0, 0:5])

        print("reshaped Wh is:")
        print(Wh.dtype)
        print(np.shape(Wh))
        print(Wh[0, 0:5, 0])

        print("xs is:")
        print(xs.dtype)
        print(np.shape(xs))
        print(xs[0, 0, 0:5])

        print("xWh is :")
        print(xWh.dtype)
        print(np.shape(xWh))
        print(xWh[0, 0:5, 0])

        print("attention is :")
        print(attention.dtype)
        print(np.shape(attention))
        print(attention[0, 0:5, 0])

        print("context_vec is")
        print(context_vec.dtype)
        print(np.shape(context_vec))
        print(context_vec[0, 0:5])

        print("concated_vec is")
        print(concated_vec.dtype)
        print(np.shape(concated_vec))
        print(np.shape(concated_vec[0]))

        print("augmented_vec is")
        print(augmented_vec.dtype)
        print(np.shape(augmented_vec))
        print("###############################")


        return augmented_vec
    # end def

# end MLP-classs


