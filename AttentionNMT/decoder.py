# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        decoder.py
# Purpose:     A Chainer implementation of Attention NMT: simple LSTM decoder
#
#              ToDo: write a input-feed version of Decoder
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     08/01/2018 (DD/MM/YY)
# Last update: 05/02/2018 (DD/MM/YY)
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

class Decoder(chainer.Chain):
    # A simple Stacked-LSTM attention decoder

    n_layers = 2
    vocab_size = 50000
    w_vec_dim = 500
    lstm_dim = 500
    dropout = 0.3

    def __init__(self, n_layers=2, vocab_size=50000, w_vec_dim=500, lstm_dim=500, gpu=0):
        '''
        initializer with parameters

        :param n_layers: numboer of LSTM layers. should be larger than 1
        :param vocab_size: vocabulary size
        :param w_vec_dim: dimension of word embedding
        :param lstm_dim: dimension (# of units) of the LSTM hidden vectors
        :param dropout: dropout ratio for LSTM
        :param gpu: gpu id, if 0, CPU use, 
        :return:
        '''
        self.name = "Decoder"
        super(Decoder, self).__init__()

        assert(n_layers > 0) # do not allow one-LSTM layer only.

        # initialize size info
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.w_vec_dim = w_vec_dim
        self.lstm_dim = lstm_dim

        global xp
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np


        # init scope for layer (modules) WITH parameters
        with self.init_scope():

            self.word_embed = L.EmbedID(in_size=vocab_size, out_size=w_vec_dim)

            self.lstm_layers = []

            # first layer; embedding to hidden state
            self.lstm_layers.append(L.LSTM(in_size=w_vec_dim, out_size=lstm_dim))
            # rest of the stacks
            if n_layers > 1:
                for l in range(n_layers-1):
                    self.lstm_layers.append(L.LSTM(in_size=lstm_dim, out_size=lstm_dim))
                # end l-for
            # end n_layers-if

        # end with
    # end __init__-def

    def reset_state(self):
        for l in range(self.n_layers):
            self.lstm_layers[l].reset_state()

    # wrapped the forward: we prefer to call "decoder_forward"
    def __call__(self, y):
        return self.onestep_forward(y)
    # end __call__-def

    def decoder_init(self, c, h):
        """
        set the encoder's final hidden state as the decoder's init

        :param c: chainer Variable, is a n_layer by B by lstm_dim numpy array. all layer's cell state initial values where B is the size of minibatch
        :param h: chainer Variable, is a n_layer by B by lstm_dim numpy array, all layer's hidden state initial values
        :return: no return
        """

        assert(len(h) == self.n_layers)
        assert(len(c) == self.n_layers)

        B = len(h[0])
        assert(B == len(c[0]))

        for l in range(self.n_layers): # for each layer
            self.lstm_layers[l].set_state(c[l], h[l])
    # end decoder_init-def

    def onestep_forward(self, y):
        '''
        forward computation of the encoder

        :param y: B-dim numpy array. B-list of word ID indices, where B is the size of minibatch
        :return: h: hidden state of the top LSTM layer, B by lstm_dim-dim numpy array
        '''

        # embedding
        x = self.word_embed(y) # return Variable

        # x should be a Variable, consists of B by lstm_dim numpy array. 

        # hidden states of each layer
        for l in range(self.n_layers):
            h = self.lstm_layers[l](x)
            x = h
        # end l-for

        return h
    # end def

# end Decoder-classs


