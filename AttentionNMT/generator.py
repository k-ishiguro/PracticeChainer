# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        generator.py
# Purpose:     A Chainer implementation of Attention NMT: generator ( linear transform + soft-max for p(y | X, Y)
#
#              inputs: possibly augmented hidden state @ time, incoming from decoder and attention
#              outputs: V-dimensional sum-to-one array, p(y_t = w | X, Y[1:t])
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
from chainer import training
from chainer.training import extensions

class Generator(chainer.Chain):

    def __init__(self, hidden_dim=1000, vocab_size=50000, dropout=0.3):
        """
        initializer.

        :param hidden_dim: dimension of the incoming hidden states, possibly lstm_dim * 2
        :param vocab_size: vocabulary size
        :return: no returns.
        """
        super(Generator, self).__init__()

        # initialize size info
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # init scope for layer (modules) WITH parameters <-- to detect by backward/optimizer/updater?
        with self.init_scope():
            self.W = L.Linear(in_size=hidden_dim, out_size=vocab_size, nobias=False)
        # end with
    # end init

    def __call__(self, h):
        """
        compute the log probability of soft-max vector.

        :param h: B by hidden_dim-dim numpy array, An incoming (possibly augmented) decoder hidden state wehre B is the size of minibatch
        :return: B-list of log p(y_t = w). B by vocab_size-dim numpy array.
        """

        Wh = self.W(h)
        p_yt = F.log_softmax(Wh) # should be (B x V)

        return p_yt
    # end def

# end MLP-classs


