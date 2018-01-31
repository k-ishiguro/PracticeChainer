# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        nmt_model.py
# Purpose:     A Chainer implementation of Attention NMT: An attentional Encoder-Decoder NMT network as a whole.
#              Combines several sub-networks. Defines interfaces and logics for train/translate.
#
#              ToDo: write the beam search decoding
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     14/01/2018 (DD/MM/YY)
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
from chainer import training
from chainer.training import extensions
from chainer import cuda

from . import encoder, decoder, attention, generator

class SimpleAttentionNMT(chainer.Chain):
    """
    Luong+ (EMNLP 2015)-type "simple" attention NMT model, as adopted in OpenNMT and OpenNMT-py
    No bidirectional LSTM in encoder, no input-feed in decoder.
    """

    def __init__(self, n_layers=2, src_vocab_size=50000, tgt_vocab_size=50000, w_vec_dim=500, lstm_dim=500, dropout=0.1, gpu=0):
        """
        Instantiate and initialize the entire NMT network.

        :param n_layers: number of LSTM stacked layers, shared among encoder and decoder
        :param src_vocab_size: number of unique tokens in src.
        :param tgt_vocab_size: number of unique tokens in tgt.
        :param w_vec_dim: word embedding dimension. shared among encoder and decoder.
        :param lstm_dim: lstm hidden vector dimension. shared among encoder and decoder
        :param dropout: dropout ratio
        :param gpu: gpu id. if > 0, it is GPU-used
        :return:
        """
        super(SimpleAttentionNMT, self).__init__()

        # initialize size info
        self.n_layers = n_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.w_vec_dim = w_vec_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout

        global xp
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np

        # init scope for layer (modules) WITH parameters <-- to detect by backward/optimizer/updater?
        with self.init_scope():
            # ID sequences are fed into the encoder, hidden vector sequences are emitted.
            self.encoder = encoder.Encoder(n_layers, src_vocab_size, w_vec_dim, lstm_dim, dropout)

            # ID sequences and encoder hidden vector sequences are fed into the decoder, hidden vector sequences are emitted
            self.decoder = decoder.Decoder(n_layers, tgt_vocab_size, w_vec_dim, lstm_dim)

            # encdoer/decoder hidden vector sequences are fed into the Attention network, augmented decoder hiddens state are emited
            self.global_attention = attention.GlobalAttention(lstm_dim, gpu)

            # augmented decoder hidden vector sequences are fed into the Generator network, log p(y_t|X, Y_t-1) is emitted.
            self.generator = generator.Generator(lstm_dim * 2, tgt_vocab_size, dropout)
        # end with

        print("nmt_model.SimpleAttentionNMT is initialized")
        print("Encoder=Stacked LSTM, Decoder=Stacked LSTM, Attention=GlobalAttention, Generator=Generator")
        print("number of stacked LSTM layers(src)=" + str(n_layers) )
        print("number of stacked LSTM layers(tgt)=" + str(n_layers) )
        print("word_embedding dimension=" + str(w_vec_dim) + ", lstm hiddne unit dimension=" + str(lstm_dim) )

    # end init

    def __call__(self, src, tgt):
        """
        wrap the forward_train for training the model.

        """
        return self.forward_train(src, tgt)

    def forward_train(self, src, tgt, BOSID):
        """
        forward computation for training. given B pairs of source seq. and target seq,
        compute the log likelihood of the tgt sequence, then return cross entropy loss.

        :param src: chainer Variable consists of B-list of ID sequences (numpy array) of source inputs, where B is the minibatch size.
        :param tgt: chainer Variable consists of B-list of ID sequences (numpy array) of corresponding target inputs.
                     Lengths of sequences must be sorted in descending order (for F.LSTM in decoder)
        :param BOSID: integer, token ID of Target's BOS token
        :return: the cross entropy loss on p(Y | X)
        """

        # forward the encoder with the entire sequence
        self.encoder.reset_state()
        hs, cs, xs = self.encoder.forward(src)

        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        self.decoder.reset_state()
        self.decoder.decoder_init(cs, hs)

        # forward the decoder+attention+generator for each time(token step)
        B = len(tgt)
        max_len_seq = len(tgt[0])
        loss = 0

        # for loop-ing w.r.t. time step t.
        for t in range(max_len_seq):
            # retrieve the (at most, B) tokens of time-t as an numpy array
            input_tokens_at_t_list = []
            tgt_tokens_at_t_list = []
            b = 0
            while (len(tgt[b]) > t) and (b < B):
                if t > 0:
                    input_tokens_at_t_list.append(tgt[b][t-1])
                else:
                    input_tokens_at_t_list.append(BOSID)
                # end if-else
                tgt_tokens_at_t_list.append(tgt[b][t])
                b = b + 1
            # end while

            # fed into the decoder, attention, and generator.
            input_tokens_at_t = chainer.Variable(xp.array(input_tokens_at_t_list, dtype=xp.float32))
            tgt_tokens_at_t = chainer.Variable(xp.array(tgt_tokens_at_t_list, dtype=xp.float32))
            h = self.decoder.onestep_forward(input_tokens_at_t)
            augmented_dec = self.global_attention(xs, h)
            pY_t = self.generator(augmented_dec)

            # add the cross entropy loss
            loss += F.softmax_cross_entropy(pY_t, tgt_tokens_at_t)
        # end tgt-for

        return loss
    # end def

    def decode_translate_greedy(self, src, max_tgt_length, BOSID, EOSID):
        """
        1-best Greedy search for decoding (translation) given a src ID sequence.

        :param src: an ID sequences of source inputs (numpy array)
        :param max_tgt_length: an maximum number of tokens for a translation
        :param BOSID: integer, token ID of Traget's BOS token
        :param EOSID: integer, token ID of Target's EOS token
        :return: an ID sequences of the predicted target outputs, len(??)-dim numpy array
                  log likelihood of the sequence
        """

        # forward the encoder with the entire sequence
        self.encoder.reset_state()
        hs, cs, xs = self.encoder.forward(src)

        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        self.decoder.reset_state()
        self.decoder.decoder_init(cs, hs)

        # forward the decoder+attention+generator for each time(token step)
        # for loop-ing w.r.t. time step t.
        input_token_at_t = BOSID
        tgt_token_at_t = None
        pred_y = []
        log_lk = 0.0

        for t in range(max_tgt_length):

            # input the previously emitted target word
            if t > 0:
                input_token_at_t = tgt_token_at_t
            # end if

            # fed into the decoder, attention, and generator.
            h = self.decoder.onestep_forward(np.array(input_token_at_t, dtype=np.float32))
            augmented_dec = self.global_attention(xs, h)
            pY_t = self.generator(augmented_dec)

            # simple 1-best greedy search
            tgt_token_at_t = np.argmax(pY_t)
            log_lk = log_lk + pY_t[tgt_token_at_t]

            # add the emitted word to the decoding sequence
            pred_y.append(tgt_token_at_t)

            # end if EOS
            if tgt_token_at_t == EOSID:
                return np.array(pred_y)
        # end tgt-for

        # no EOS, reached the maximum length of the target decoding length
        return pred_y, log_lk



# end MLP-classs


