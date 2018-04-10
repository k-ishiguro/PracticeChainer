# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        nmt_model.py
# Purpose:     A Chainer implementation of Attention NMT: An attentional Encoder-Decoder NMT network as a whole.
#              Combines several sub-networks. Defines interfaces and logics for train/translate.
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     14/01/2018 (DD/MM/YY)
# Last update: 08/03/2018 (DD/MM/YY)
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
import copy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import cuda

import encoder, decoder, attention, generator


class DecoderHypothesis():
    """
    Decoder hypothesis for beam search. 
    Maintaining the decoder network, the last emitted token (id), the entire token emissions (ids), accumulated log likelihood.
    """

    def __init__(self, decoder):
        """
        Instantiate and initialize the decoding hypothesis

        :param decoder: deep copied from the nmt_model. 
        :return:
        """
        
        # initialize 
        self.decoder = decoder
        self.tgt_token_at_t = None # latest token emission id
        self.log_lk = 0.0 # accumulated log likelihood
        self.pred_y = [] # path of translation (list of token ids)
    # end __init__

    def copy(self):
        """ 
        deep copy the class instance
        """
        out = DecoderHypothesis(self.decoder)
        
        out.decoder = copy.deepcopy(self.decoder)
        out.tgt_token_at_t = copy.deepcopy(self.tgt_token_at_t)
        out.log_lk = self.log_lk
        out.pred_y = copy.deepcopy(self.pred_y)

        return out
    # end copy
        
    def extend(self, tgt_token_at_t, log_lk):
        """
        extend the translation path, update the log likelihood
        """
        self.tgt_token_at_t = tgt_token_at_t
        self.log_lk = log_lk
        self.pred_y.append(tgt_token_at_t)
    # end extend-def
    
        
# end DecoderHypothesis-def

class SimpleAttentionNMT(chainer.Chain):
    """
    Luong+ (EMNLP 2015)-type "simple" attention NMT model, as adopted in OpenNMT and OpenNMT-py
    No bidirectional LSTM in encoder, no input-feed in decoder.
    """

    def sortHypotheses(self, hyps, best_k):
        """
        Sort the decoding hypothesis based on the accumulated log likelihood
        
        param: hyps: list of decoding hypothesis
        param: best_k: number of selection
        return: best_k-lenght list of hypothesis
        """

        # get the scores
        log_lks = np.zeros(len(hyps))
        for i, h in enumerate(hyps):
            log_lks[i] = h.log_lk / max( (len(h.pred_y)-1), 1 )
        
        # sort by scores
        sorted_idx = np.argsort(log_lks)[::-1]
        
        # join the list
        out_list = []
        for k in range(best_k):
            out_list.append(hyps[sorted_idx[k]])
        # end k-for

        return out_list       
    # end sortHypothesis

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


    def __init__(self, n_layers=2, src_vocab_dict=None, tgt_vocab_dict=None, w_vec_dim=500, lstm_dim=500, encoder_type='rnn', dropout=0.1, gpu=0):
        """
        Instantiate and initialize the entire NMT network.

        :param n_layers: number of LSTM stacked layers, shared among encoder and decoder
        :param src_vocab_dict: dictionary of source vocabulary
        :param tgt_vocab_dict: dictionary of target vocabulary
        :param w_vec_dim: word embedding dimension. shared among encoder and decoder.
        :param lstm_dim: lstm hidden vector dimension. shared among encoder and decoder
        :param encoder_type: 'rnn' for uni-directional encoder, 'brnn' for bi-directional encoder
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
        self.encoder_type = encoder_type
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
            self.encoder = encoder.Encoder(n_layers, self.src_vocab_size, w_vec_dim, lstm_dim, encoder_type, dropout, gpu)
            #self.encoder = encoder.Encoder(n_layers, self.src_vocab_size, w_vec_dim, lstm_dim, encoder_type, DEBUG_BRNN, dropout, gpu)

            # ID sequences and encoder hidden vector sequences are fed into the decoder, hidden vector sequences are emitted
            self.decoder = decoder.Decoder(n_layers, self.tgt_vocab_size, w_vec_dim, lstm_dim, gpu)

            # encdoer/decoder hidden vector sequences are fed into the Attention network, augmented decoder hiddens state are emited
            self.global_attention = attention.GlobalAttention(lstm_dim, gpu)

            # augmented decoder hidden vector sequences are fed into the Generator network, log p(y_t|X, Y_t-1) is emitted.
            self.generator = generator.Generator(lstm_dim * 2, self.tgt_vocab_size, dropout)
        # end with

        print("nmt_model.SimpleAttentionNMT is initialized")
        if encoder_type=='rnn':
            print("Encoder=Stacked LSTM, Decoder=Stacked LSTM, Attention=GlobalAttention, Generator=Generator")
        elif encoder_type=='brnn':
            print("Encoder=Stacked BiDirectional LSTM, Decoder=Stacked LSTM, Attention=GlobalAttention, Generator=Generator")
        # end if
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

        ## padding the source sentences
        # padded_src = F.pad_sequence(src, None, -1)
        ## forward the encoder with the entire sequence
        #hs, cs, xs = self.encoder.forward(padded_src)       
        #xs_mat = F.stack(xs)

        # forward the encoder with the entire sequence
        hs, cs, xs = self.encoder.forward(src)       
        xs_mat = F.stack(xs)
        # generate a mask
        padded_src = xp.array(F.pad_sequence(src, None, -1).data)
        enable_src_mask = xp.where( padded_src != -1, 1, 0)[:, :, xp.newaxis]
        
        ### THIS IS A BAD WAY (super slow!!###
        #(B, src_len, ls_dim) = np.shape(xs)
        #xs_mat = xp.zeros( (B, src_len, ls_dim), dtype=xp.float32 )
        #for b in range(B):
        #    xs_mat[b, :, :] = xp.reshape(xs[b].data, (1, src_len, ls_dim))
        ## end b-for

        ### for debug ###
        #print("####### for DEBUG: Encoder forwarding done.######")
        #print("hs is: ")
        #print(type(hs))
        #print(np.shape(hs))
        #print(type(hs[0]))
        #print(np.shape(hs[0]))              
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
        #print(xp.sum(xs[0][0, :]))
        #print(xp.sum(xs_mat[0, 0, :] ) )
        #print("####################")

        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        #if self.encoder.encoder_type=='rnn':
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
            tgt_batch_size = len(tgt_tokens_at_t)

            if t==0:
                BOSID_array = xp.ones(tgt_batch_size) * self.getTgtID("<BOS>")
                input_tokens_at_t = chainer.Variable(xp.array(BOSID_array.astype(xp.int32)))
            else:
                #tgt_t1 = transposed_tgt[t-1][0:tgt_batch_size]
                #input_tokens_at_t = xp.array(tgt_t1.data).astype(xp.int32)
                input_tokens_at_t = transposed_tgt[t-1][0:tgt_batch_size]
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

            # attentino with padding mask
            augmented_vec = self.global_attention(xs_mat[0:tgt_batch_size], h, enable_src_mask[0:tgt_batch_size])
            
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
        # generate a mask
        padded_src = xp.array(F.pad_sequence(src, None, -1).data)
        enable_src_mask = xp.where( padded_src != -1, 1, 0)[:, :, xp.newaxis]

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
            augmented_vec = self.global_attention(xs_mat, h, enable_src_mask)
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
    # end decode_translate_greedy-def

    def decode_translate_beam(self, src, max_tgt_length, beam_size):
        """
        k-width beam search for decoding (translation) given a src ID sequence.

        :param src: an ID sequence of source inputs (int32 nump/cuda array)
        :param max_tgt_length: an maximum number of tokens for a translation
        :param beam_size: beam width for search
        :return: an ID sequences of the predicted target outputs, len(??)-dim numpy array
                  log likelihood of the sequence
        """

        # forward the encoder with the entire sequence
        #self.encoder.reset_state()
        hs, cs, xs = self.encoder.forward(src)
        # convert xs into a matrix (Variable)
        xs_mat = F.stack(xs)
        # generate a mask
        padded_src = xp.array(F.pad_sequence(src, None, -1).data)
        enable_src_mask = xp.where( padded_src != -1, 1, 0)[:, :, xp.newaxis]

        #(b, src_len, ls_dim) = np.shape(xs) # b should be 1
        #assert(b == 1)
        #xs_mat = F.reshape(xs[0].data, (1, src_len, ls_dim))
                
        # given the encoder states, initialize the decoder. each network memorizes (at most) B rnn histories.
        #self.decoder.reset_state()
        self.decoder.decoder_init(cs, hs)

        # set up beam hypotheses list. if the hyp emit EOS, i leaves this list
        beam_hyps = []
        # at the first iteration, only one hypothesis (all start from BOS)
        beam_b = DecoderHypothesis(self.decoder) 
        beam_hyps.append(beam_b)


        # if a beam hyp emit <EOS>, it joins this list. 
        # when all hyps moved, then the search is done. 
        finished_hyps = []

        # forward the decoder+attention+generator for each time(token step)
        # for loop-ing w.r.t. time step t.
        input_token_at_t = xp.asarray( [self.getTgtID("<BOS>")] ).astype(xp.int32)

        tgt_token_at_t = None
        pred_y = []
        log_lk = 0.0
        
        for t in range(max_tgt_length):
            
            candid_hyps = []
            for b, hyp in enumerate(beam_hyps):
                
                # input the previously emitted target word
                if t > 0:
                    temp = hyp.tgt_token_at_t
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
                h = hyp.decoder.onestep_forward(input_token_at_t)
                augmented_vec = self.global_attention(xs_mat, h, enable_src_mask)
                pY_t = self.generator(augmented_vec)                                
                
                #print("####### For DEBUG: check generator output at t=" + str(t) + " ######")
                #print("beam hypothesis No. " + str(b))
                #print("pY_t is;")
                #print(type(pY_t))
                #print(len(pY_t))
                #print(pY_t)
                #print(pY_t.data)
                #print("#############")
                
                # take the top-k candidates
                for k in range(beam_size):
                    tgt_token_at_t = xp.argmax(pY_t.data)
                    log_lk = hyp.log_lk + pY_t.data[0][tgt_token_at_t]
                    
                    # deepcopy a new hyp
                    new_hyp = hyp.copy()
                    
                    # extend the hypothesis with the emitted word and the new log score
                    new_hyp.extend(tgt_token_at_t, log_lk)
                    
                    # append the extended hypothesis
                    candid_hyps.append(new_hyp)
                    
                    # reduce the score for this choice
                    pY_t.data[0][tgt_token_at_t] = -10000000000
                    
                    #print("####### For DEBUG: check candid extension. at t=" + str(t) + " ######")
                    #print("beam hypothesis No. " + str(b) + " child No. " + str(k))
                    #print("log_lk is;")
                    #print(log_lk)
                    #print("tgt_token_at_t is:")
                    #print(tgt_token_at_t)
                    #print("#############")


                    ### simple 1-best greedy search            
                    #tgt_token_at_t = xp.argmax(pY_t.data)
                    #log_lk = log_lk + pY_t.data[0][tgt_token_at_t]
                    
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
                    
                # end b,hyp-enumerate       
            # end hyp-for
            
            # sort the candidate hyps by the score, retain bests.
            num_top = len(beam_hyps)
            if t == 0:
                num_top = beam_size
            top_hyps = self.sortHypotheses(candid_hyps, num_top)
                    
            # if the sorted hyp emits EOS, move to the finisehd list. 
            # otherwise, keep in the beam_hyps
            beam_hyps = [] # clear
            for hyp in top_hyps:
                if hyp.tgt_token_at_t == self.getTgtID("<EOS>"):
                    finished_hyps.append(hyp)
                else:
                    beam_hyps.append(hyp)
                # end EOS-ifelse
            # end top_hyps-for
            
            # if no hypthesis remains, out the t-loop
            if len(beam_hyps) < 1:
                break
            # end if
            
        # end t-for
        
        # if still remaining hyps, force return. 
        if len(beam_hyps) > 0:
            for hyp in beam_hyps:
                finished_hyps.append(hyp)
            # end hyp-for
        # end len-if

        # return the best hypothesis
        for i, hyp in enumerate(finished_hyps):
            print("beam result: hypothesis " + str(i)+ ": score=" + str(hyp.log_lk) + ", length=" + str(len(hyp.pred_y)-1) + ", normalized score=" + str(hyp.log_lk / (len(hyp.pred_y)-1)) )

        best_hyp = self.sortHypotheses(finished_hyps, beam_size)[0]
        pred_y = best_hyp.pred_y
        log_lk = best_hyp.log_lk
        
        return pred_y, log_lk
    # end decode_translate_baem-def

# end MLP-classs


