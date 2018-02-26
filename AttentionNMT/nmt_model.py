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
# Last update: 26/02/2018 (DD/MM/YY)
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

import encoder, decoder, attention, generator

class SimpleAttentionNMT(chainer.Chain):
    """
    Luong+ (EMNLP 2015)-type "simple" attention NMT model, as adopted in OpenNMT and OpenNMT-py
    No bidirectional LSTM in encoder, no input-feed in decoder.
    """

    def getSrcID(self, token_str):
        """
        return the token ID of the input token_str, in the src_vocab_dict dictionary
        
        :param token_str: target key
        :return: inrteer, ID of the token str
        """

        return self.src_vocab_dict[token_str]
    # end getSrcID

    def getTgtID(self, token_str):
        """
        return the token ID of the input token_str, in the tgt_vocab_dict dictionary
        
        :param token_str: target key
        :return: inrteer, ID of the token str
        """

        return self.tgt_vocab_dict[token_str]
    # end getTgtID


    def __init__(self, n_layers=2, src_vocab_dict=None, tgt_vocab_dict=None, w_vec_dim=500, lstm_dim=500, dropout=0.1, gpu=0):
        """
        Instantiate and initialize the entire NMT network.

        :param n_layers: number of LSTM stacked layers, shared among encoder and decoder
        :param src_vocab_dict: dictionary of source vocabulary
        :param tgt_vocab_dict: dictionary of target vocabulary
        :param w_vec_dim: word embedding dimension. shared among encoder and decoder.
        :param lstm_dim: lstm hidden vector dimension. shared among encoder and decoder
        :param dropout: dropout ratio
        :param gpu: gpu id. if > 0, it is GPU-used
        :return:
        """
        super(SimpleAttentionNMT, self).__init__()

        # initialize size info
        self.n_layers = n_layers
        self.src_vocab_dict = src_vocab_dict
        self.tgt_vocab_dict = tgt_vocab_dict
        self.w_vec_dim = w_vec_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout

        global xp
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np
        # end if-else

        if src_vocab_dict and tgt_vocab_dict:
            self.src_vocab_size = len(src_vocab_dict)
            self.tgt_vocab_size = len(tgt_vocab_dict)
        else:
            self.src_vocab_size = 1
            self.tgt_vocab_size = 1

        # init scope for layer (modules) WITH parameters <-- to detect by backward/optimizer/updater?
        with self.init_scope():
            # ID sequences are fed into the encoder, hidden vector sequences are emitted.
            self.encoder = encoder.Encoder(n_layers, self.src_vocab_size, w_vec_dim, lstm_dim, dropout, gpu)

            # ID sequences and encoder hidden vector sequences are fed into the decoder, hidden vector sequences are emitted
            self.decoder = decoder.Decoder(n_layers, self.tgt_vocab_size, w_vec_dim, lstm_dim, gpu)

            # encdoer/decoder hidden vector sequences are fed into the Attention network, augmented decoder hiddens state are emited
            self.global_attention = attention.GlobalAttention(lstm_dim, gpu)

            # augmented decoder hidden vector sequences are fed into the Generator network, log p(y_t|X, Y_t-1) is emitted.
            self.generator = generator.Generator(lstm_dim * 2, self.tgt_vocab_size, dropout)
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

    def forward_train(self, src, tgt):
        """
        forward computation for training. given B pairs of source seq. and target seq,
        compute the log likelihood of the tgt sequence, then return cross entropy loss.

        :param src: B-list of (len-seq) numpy array (dtype=int), is a B-list of ID sequences of source inputs, where B is the minibatch size.
        :param tgt: B-list of (len-seq) numpy array (dtype=int), is a B-list of ID sequences (numpy array) of corresponding target inputs.
                     Lengths of sequences must be sorted in descending order (for F.LSTM in decoder)
        :return: the cross entropy loss on p(Y | X)
        """

        # padding the source sentences
        padded_src = F.pad_sequence(src, None, -1)

        # forward the encoder with the entire sequence
        hs, cs, xs = self.encoder.forward(padded_src)       
        # convert xs into a matrix (Variable)
        xs_mat = F.stack(xs)
        
        ### THIS IS A BAD WAY (super slow!!###
        #(B, src_len, ls_dim) = np.shape(xs)
        #xs_mat = xp.zeros( (B, src_len, ls_dim), dtype=xp.float32 )
        #for b in range(B):
        #    xs_mat[b, :, :] = xp.reshape(xs[b].data, (1, src_len, ls_dim))
        ## end b-for

        ### for debug ###
        #print("####### for DEBUG: Encoder forwarding done.######")
        #print("hs is: ")
        #print(hs.dtype)
        #print(np.shape(hs))
        #print("cs is; ")
        #print(cs.dtype)
        #print(np.shape(cs))
        #print("xs is: ")
        #print(type(xs))
        #print(len(xs))
        #print(xs[0].dtype)
        #print(len(xs[0]))
        #print(len(xs[0][0]))
        #print("xs_mat is: ")
        #print(xs_mat.dtype)
        #print(len(xs_mat))
        #print(xs_mat[0].dtype)
        #print(len(xs_mat[0]))
        #print(len(xs_mat[0][0]))
        #print(Xp.sum(xs[0][0, :]))
        #print(Xp.sum(xs_mat[0, 0, :] ) )
        #print("####################")

        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        self.decoder.reset_state()
        self.decoder.decoder_init(cs, hs)
        
        # forward the decoder+attention+generator for each time(token step)
        B = len(tgt)
        max_len_seq = len(tgt[0])
        loss = 0

        # permute the axes of target[sample][time] --> [time][sample]
        transposed_tgt = F.transpose_sequence(tgt)

        # for loop-ing w.r.t. time step t.
        for t in range(max_len_seq):
            tgt_tokens_at_t = transposed_tgt[t]
            #tgt_tokens_at_t = xp.array(transposed_tgt[t].data.astype(xp.int32))

            tgt_batch_size = len(tgt_tokens_at_t)

            if t==0:
                BOSID_array = xp.ones(tgt_batch_size) * self.getTgtID("<BOS>")
                input_tokens_at_t = xp.array(BOSID_array.astype(xp.int32))
            else:
                tgt_t1 = transposed_tgt[t-1].data.astype(xp.int32)
                input_tokens_at_t = xp.array(tgt_t1[0:tgt_batch_size])
            # end if-else

            ### for debug ###
            #print("####### For DEBUG: Decoder input setup done.######")
            #print("transposed_tgt is:")
            #print(type(transposed_tgt))
            #print(len(transposed_tgt))        
            #print(type(transposed_tgt[0]))
            #print(len(transposed_tgt[0]))
            #print(len(transposed_tgt[-1]))
            #print("input_tokens_at_t is: ")
            #print(type(input_tokens_at_t))
            #print(np.shape(input_tokens_at_t))
            #print(input_tokens_at_t[0])
            #print(input_tokens_at_t[-1])
            #print("tgt_tokens_at_t is; ")
            #print(type(tgt_tokens_at_t))
            #print(np.shape(tgt_tokens_at_t))
            #print(tgt_tokens_at_t[0])
            #print(tgt_tokens_at_t[-1])
            #print("####################")            

            # fed into the decoder, attention, and generator.
            h = self.decoder.onestep_forward(input_tokens_at_t)

            #print("####### For DEBUG: Decoder forward done.######")
            #print("h is: ")
            #print(h.dtype)
            #print(np.shape(h))
            #print("####################")            

            augmented_vec = self.global_attention(xs_mat[0:tgt_batch_size], h)
            pY_t = self.generator(augmented_vec)
            #print("####### For DEBUG: attention and generator forward done.######")
            #print("augmented_vec is: ")
            #print(augmented_vec.dtype)
            #print(np.shape(augmented_vec))

            #print("pY_t is: ")
            #print(pY_t.dtype)
            #print(np.shape(pY_t))
            #print("####################")            

            # add the cross entropy loss
            loss_now = F.softmax_cross_entropy(pY_t, tgt_tokens_at_t)            
            loss += loss_now
            #print("####### For DEBUG: loss computed.######")
            #print("loss_now is: ")
            #print(loss_now)

            #print("loss is: ")
            #print(loss)
            #print("####################")            

        # end tgt-for

        return loss
    # end def

    def decode_translate_greedy(self, src, max_tgt_length):
        """
        1-best Greedy search for decoding (translation) given a src ID sequence.

        :param src: an ID sequence of source inputs (int32 nump/cuda array)
        :param max_tgt_length: an maximum number of tokens for a translation
        :return: an ID sequences of the predicted target outputs, len(??)-dim numpy array
                  log likelihood of the sequence
        """

        # forward the encoder with the entire sequence
        #self.encoder.reset_state()
        hs, cs, xs = self.encoder.forward(src)
        # convert xs into a matrix (Variable)
        xs_mat = F.stack(xs)
        #(b, src_len, ls_dim) = np.shape(xs) # b should be 1
        #assert(b == 1)
        #xs_mat = F.reshape(xs[0].data, (1, src_len, ls_dim))
                
        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        #self.decoder.reset_state()
        self.decoder.decoder_init(cs, hs)

        # forward the decoder+attention+generator for each time(token step)
        # for loop-ing w.r.t. time step t.
        input_token_at_t = xp.asarray( [self.getTgtID("<BOS>")] ).astype(xp.int32)

        tgt_token_at_t = None
        pred_y = []
        log_lk = 0.0
        
        for t in range(max_tgt_length):
            
            # input the previously emitted target word
            if t > 0:
                temp = tgt_token_at_t
                input_token_at_t = xp.array([temp], dtype=xp.int32)
            # end if
            
            #print("####### For DEBUG: check decoder input.######")            
            #print("input_token_at_t is;")
            #print(type(input_token_at_t))
            #print(len(input_token_at_t))
            #print(input_token_at_t)
            #print(type(input_token_at_t[0]))
            #print(input_token_at_t[0])
            #print("#############")
            
            # fed into the decoder, attention, and generator.
            h = self.decoder.onestep_forward(input_token_at_t)
            augmented_vec = self.global_attention(xs_mat, h)
            pY_t = self.generator(augmented_vec)
            
            #print("####### For DEBUG: check generator output.######")
            #print("pY_t is;")
            #print(type(pY_t))
            #print(len(pY_t))
            #print(pY_t)
            #print(pY_t.data)
            #print("#############")
            
            # simple 1-best greedy search            
            tgt_token_at_t = xp.argmax(pY_t.data)
            log_lk = log_lk + pY_t.data[0][tgt_token_at_t]
            
            #print("####### For DEBUG: check decoded emission.######")
            #print("tgt_token_at_t is:")
            #print(type(tgt_token_at_t))
            #print(tgt_token_at_t)
            #print(tgt_token_at_t.data)
            #print(type(tgt_token_at_t.data))
            #print("pY_t is:")
            #print(type(pY_t))
            #print(len(pY_t))
            #print(type(pY_t[0]))
            #print(len(pY_t[0]))
            #print(type(pY_t[0][tgt_token_at_t]))
            #print(pY_t[0][tgt_token_at_t])            
            #print("log_lk is:")
            #print(type(log_lk))
            #print(len(pY_t))
            #print(log_lk)            
            #print("#############")
            
            # add the emitted word to the decoding sequence
            pred_y.append(tgt_token_at_t)
            
            # end if EOS            
            if tgt_token_at_t == self.getTgtID("<EOS>"):
                return pred_y, log_lk
        # end tgt-for

        # no EOS, reached the maximum length of the target decoding length
        return pred_y, log_lk

# end MLP-classs


