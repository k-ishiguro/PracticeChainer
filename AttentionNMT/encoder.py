# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        encoder.py
# Purpose:     A Chainer implementation of Attention NMT: simple (bi-)directional LSTM encdoer + word-embedding           .
#
#              inputs: sequence of word-embedding vectors
#
#              outputs: sequences(s) of n-unit dimensional cont. vectors (hidden vectors) of the top layer.
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     08/01/2018 (DD/MM/YY)
# Last update: 11/01/2018 (DD/MM/YY)
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

class Encoder(chainer.Chain):
    # A simple stacked LSTM encoder.

    def __init__(self, n_layers=2, vocab_size=50000, w_vec_dim=500, lstm_dim=500):
        '''
        initializer with parameters

        :param n_layers: numboer of LSTM layers. should be larger than 1
        :param vocab_size: vocabulary size
        :param w_vec_dim: dimension of word embedding
        :param lstm_dim: dimension (# of units) of the LSTM hidden vectors
        :return:
        '''
        self.name = "Encoder"
        super(Encoder, self).__init__()

        assert(n_layers > 0) # do not allow one-LSTM layer only.

        # init scope for layer (modules) WITH parameters
        with self.init_scope():

            self.lstm_layers = L.NStepLSTM(n_layers, w_vec_dim, lstm_dim)

        # end with
    # end init

    # wrapped the forward
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        '''
        forward computation of the encoder

        :param x:sequence(s) of word-embedded bectors
        :return:sequence(s) of lstm_dim-dimensional hidden vectors at the top layer of LSTMs.
        '''
        h = self.lstm_layers(None, None, x) # LSTM stacks. initial states are zero.
        return h
    # end def

# end Encoder-classs

class BrnnEncoder(chainer.Chain):
    """"""
    # Stacked LSTM + bi-directional at the first layer

    """Constructor for BrnnEncoder"""
    def __init__(self, ):
        self.name = "Encoder"
        super(BrnnEncoder, self).__init__()

        # TODO: copy rom Encoder, modify it
    # end init

    # wrapped the forward
    def __call__(self, w_seq):
        return self.forward(w_seq)

    def forward(self, w_seq):
        '''
        forward computation of the encoder

        :param w_seq:sequence(s) of one-hot vectors
        :return:sequence(s) of lstm_dim-dimensional hidden vectors at the top layer of LSTMs.
        '''
        # TODO: copy rom Encoder, modify it
    # end def


# Bidirectional RNN for the first layer only
