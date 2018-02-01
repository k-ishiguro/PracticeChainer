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
# Last update: 24/01/2018 (DD/MM/YY)
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

        :param xs: chainer Variable consists of B-list of N by D numpy array, B-list of sequences (list, array) of encoder hidden states, B is the minibatch size
        :param h: chaienr Variable, consists of B by D numpy array, B-list of the decoder hidden state of the focused time step
        :return: B-list of augmented decoder hidden sate w/ context vector. B by D*2-dim numpy array,
        """
        B = len(xs)
        augmented_dec = xp.zeros( (B, self.lstm_dim) )
        for b in B:

            # compute the attention similarities between xs and h
            Wh = self.W(h[b])
            xWh = xp.dot(xs[b], Wh) # should be (N x 1)
            attention = F.softmax(xWh) # should be (N x 1)

            # weighted sum of xs.
            context_vec = xp.reshape(xp.dot(xs[b].T, attention), (self.lstm_dim) ) # should be (D)

            # concat and nonlinear transform
            concated_vec = xp.hstack(h[b], context_vec)

            augmented_dec[b, :] = F.tanh( self.C(concated_vec) )

        return augmented_dec
    # end def

# end MLP-classs


