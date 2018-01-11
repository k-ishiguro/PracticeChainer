# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        decoder.py
# Purpose:     A Chainer implementation of Attention NMT: simple LSTM decoder + attention +  soft-max.
#
#              ToDo: write a input-feed version of Decoder
#
#              inputs: sequences(s) of n-unit dimensional vectors. encoder's top layer output.
#
#              outputs: sequcnes(s) of V-dimensional vectors: soft-max probabilities
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     08/01/2018 (DD/MM/YY)
# Last update: 10/01/2018 (DD/MM/YY)
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

class Decoder(chainer.Chain):
    # A simple Stacked-LSTM attention decoder

    def __init__(self, n_layers=2, vocab_size=50000, w_vec_dim=500, lstm_dim=500):
        '''
        initializer with parameters

        :param n_layers: numboer of LSTM layers. should be larger than 1
        :param vocab_size: vocabulary size
        :param w_vec_dim: dimension of word embedding
        :param lstm_dim: dimension (# of units) of the LSTM hidden vectors
        :return:
        '''
        self.name = "Decoder"
        super(Decoder, self).__init__()

        assert(n_layers > 0) # do not allow one-LSTM layer only.

        # init scope for layer (modules) WITH parameters
        with self.init_scope():

            self.lstm_layers = []
            for l in range(n_layers):
                self.lstm_layers.append(L.LSTM(w_vec_dim, lstm_dim))

            # attention to return context vector
            self.attention_net = Attention()

            # soft-max layer (+ linear trans.)
            self.generator = Generator()

        # end with
    # end init

    # wrapped the forward
    def __call__(self, w_seq):
        return self.forward(w_seq)

    def decoder_forward(self, w_seq):
        '''
        forward computation of the encoder

        :param w_seq:sequence(s) of one-hot vectors
        :return:sequence(s) of lstm_dim-dimensional hidden vectors at the top layer of LSTMs.
        '''
        x = self.embed(w_seq) # embedding
        h = self.lstm_layers(x) # LSTM stacks
        return h
    # end def

# end Decoder-classs


