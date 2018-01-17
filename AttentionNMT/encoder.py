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
# Last update: 14/01/2018 (DD/MM/YY)
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

    n_layers = 2
    vocab_size = 50000
    w_vec_dim = 500
    lstm_dim = 500
    dropout = 0.3

    def __init__(self, n_layers=2, vocab_size=50000, w_vec_dim=500, lstm_dim=500, dropout=0.3):
        '''
        initializer with parameters

        :param n_layers: numboer of LSTM layers. should be larger than 1
        :param vocab_size: vocabulary size
        :param w_vec_dim: dimension of word embedding
        :param lstm_dim: dimension (# of units) of the LSTM hidden vectors
        :param drop_out: dropout ratio
        :return:
        '''
        self.name = "Encoder"
        super(Encoder, self).__init__()

        assert(n_layers > 0) # do not allow one-LSTM layer only.

        # initialize size info
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.w_vec_dim = w_vec_dim
        self.lstm_dim = lstm_dim

        # init scope for layer (modules) WITH parameters
        with self.init_scope():

            self.word_embed = L.EmbedID(in_size=vocab_size, out_size=w_vec_dim)
            self.lstm_layers = L.NStepLSTM(n_layers, w_vec_dim, lstm_dim, dropout=dropout)

        # end with
    # end init

    def reset_state(self):
        self.lstm_layers.reset_state()

    # wrapped the forward
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        '''
        forward computation of the encoder

        :param x: B-list of word index sequence, where B is the minibatch size
        :return:  tuple of (h, c, y)
                   h; all layer's hidden states at the of the sequence. B-list of n_layers by lstm_dim
                   c: all layer's internal cell states at the end of the sequence. B-list of n_layers by lstm_dim
                   y: top layer's hidden state sequence. B-list of seq_length x lstm_dim numpy array
        '''

        x_embed = self.word_embed(x)
        h, c, y = self.lstm_layers(None, None, x_embed) # LSTM stacks. initial states are zero.

        # h; all layer's hidden states at the of the sequence
        # c: all layer's internal cell states at the end of the sqeucne
        # y: top layer's hidden state sequence
        return h, c, y
    # end def

# end Encoder-classs
