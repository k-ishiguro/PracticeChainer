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
# Last update: 01/03/2018 (DD/MM/YY)
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
from chainer import training
from chainer.training import extensions

class Encoder(chainer.Chain):
    # A simple stacked LSTM encoder.

    def __init__(self, n_layers=2, vocab_size=50000, w_vec_dim=500, lstm_dim=500, encoder_type='rnn', dropout=0.3, gpu=0):
        '''
        initializer with parameters

        :param n_layers: numboer of LSTM layers. should be larger than 1
        :param vocab_size: vocabulary size
        :param w_vec_dim: dimension of word embedding
        :param lstm_dim: dimension (# of units) of the LSTM hidden vectors
        :param encoder_type: 'rnn' for uni-directional encoder, 'brnn' for bi-directional encoder
        :param drop_out: dropout ratio
        :param gpu: gpu index. if 0, use cpu
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
        self.encoder_type = encoder_type
        
        global xp
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np
        # end if-else
        
        # init scope for layer (modules) WITH parameters
        with self.init_scope():

            self.word_embed = L.EmbedID(in_size=vocab_size, out_size=w_vec_dim, ignore_label=-1)
            self.lstm_layers = L.NStepLSTM(n_layers, w_vec_dim, lstm_dim, dropout=dropout)
            if encoder_type=='brnn':
                self.lstm_layers = L.NStepBiLSTM(n_layers, w_vec_dim, lstm_dim//2, dropout=dropout)
            # end-if

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

        :param x: a Chainer Variable, B by max_len_seq numpy arrays, is a B-list of word ID sequences of source inputs (padded with -1), B is a mini-batch size
        :return:  tuple of (h, c, y)
                   h; all layer's hidden states at the of the sequence. B-list of n_layers by lstm_dim
                   c: all layer's internal cell states at the end of the sequence. B-list of n_layers by lstm_dim
                   y: top layer's hidden state sequence. B-list of seq_length x lstm_dim lstm_dim (if brnn) numpy array
        '''

        #x_embed = self.word_embed(x)
        x_embed = [ self.word_embed(x_s) for x_s in x  ]

        # x_embed must be a list of Variable, where each Variable corresponds to a sequence ( of embedded vectors)
        h, c, y = self.lstm_layers(None, None, x_embed) # LSTM stacks. initial states are zero.

        # h; all layer's hidden states at the of the sequence
        # c: all layer's internal cell states at the end of the sqeucne
        # y: top layer's hidden state sequence
        
        return h, c, y
    # end def

# end Encoder-classs
