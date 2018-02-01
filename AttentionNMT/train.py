# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        train.py
# Purpose:     A Chainer implementation of Attent ion NMT: training script
#              based on the OpenNMT-py (pytorch) official implementations
#
#              required inputs:
#              serialized object, containing lists of tokenized training sequences (source, target) and vocabulary dictionaries (source, target)
#              output_prefix
#              Note all sequences are sorted by the TARGET token length in descending order (handled by preprocess.py)
#
#              ToDo: add options to control network structure
#
#              outputs:
#              serialized trained NMT model binary, including:
#                - trained NMT network
#                - src/tgt vocabulary dictionary
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     19/01/2018 (DD/MM/YY)
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

matplotlib.use('Agg')  # do it background
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

rcParams['font.family'] = 'IPAGothic'

# import pandas as pd
# from pandas import Series, DataFrame

# import boto3
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import cuda

from . import nmt_model

def getID(vocab_dict, token_str):
    """
    return the token ID of the input token_str, in the vocab_dict dictionary

    :param vocab_dict: dictionary to look up
    :param token_str: target key
    :return: inrteer, ID of the token str
    """

    return vocab_dict[token_str]

def make_datatuples(src_pckl, tgt_pckl):
    """
    load the src/tgt sequence list (pickled), then zip them, return as a list of tuples (src_seq, tgt_seq)
    :param src_pckl: file path to source sequence list pickle
    :param tgt_pckl: file path to source sequence list pickle
    :return: a list of tuples, each element is (src_seq, tgt_seq)
    """

    # load pickle
    src_seqs = pickle.load(src_pckl)
    tgt_seqs = pickle.load(tgt_pckl)

    assert(len(src_seqs) == len(tgt_seqs))

    # zip them
    return zip(src_seqs, tgt_seqs)
# end make_datatuples-def

def minibatchToListTuple(train_batch, gpuid):
    """
    Decompose the minibatch (list of (src, tgt) tuple) into two chainer Variable, which are lists of sequences (src, tgt).
    Transfer them to the designated device:
    if gpuid = 0: cpu (numpy)
    else cpu (cupy)

    :param train_batch:list of (src seq., tgt seq.) tuples.
    :param gpuid: 0 if CPU, else GPU
    :return:Chainer Variables which are lists of src seqeunces and tgt sequecnes
    """

    src_list, tgt_list = zip(*train_batch)

    src_list = chainer.Variable(xp.array(src_list, dtype=xp.float32))
    tgt_list = chainer.Variable(xp.array(src_list, dtype=xp.float32))

    return src_list, tgt_list

def main(args):
    """
    main training loop.

    :param args: argparse instance. having argument values as menber variable.
    :return: serialized binary of the trained model: <args.output_prefix>-ep(epoch).model, including:
               - trained NMT network
               - vocabulary dictionaries of source and target
    """


    ###
    # load the training data. target side sentences end with <EOS>.
    ###
    (train_src, train_tgt, valid_src, valid_tgt, src_vocab_dictionary, tgt_vocab_dictionary) = pickle.load(args.data)

    train_tuples = make_datatuples(train_src, train_tgt)
    valid_tuples = make_datatuples(valid_src, valid_tgt)

    # just a iterators, but is able to repeat and shuffle.
    train_iter = chainer.iterators.SerialIterator(train_tuples, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid_tuples, args.batchsize, repeat=False, shuffle=False)

    print("training data pickle " + str(args.data) + " loaded. ")
    print("vocabulary size= " + str(len(src_vocab_dictionary)) + "(src), " + str(len(tgt_vocab_dictionary)) + "(tgt)")
    print("sample size= " + str(len(train_tuples)) + "(training), " + str(len(valid_tuples)) + "(valid)")
    print("minibathces: size=" + str(args.batchsize) + " number=" + str(len(train_iter)) )

    ###
    # set up the network, optimizer, loss etc
    ###

    model = nmt_model.SimpleAttentionNMT(args.n_layers,len(src_vocab_dictionary), len(src_vocab_dictionary), args.w_vec_dim, args.lstm_dim, args.dropout, args.gpu)

    global xp
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id((args.gpu)).use()
        model.to_gpu(args.gpu) # copy the chain to the GPU
        xp = cuda.cupy
    else:
        xp = np
    # end args.gpu-if
    print("model set up. ")

    # set up the optimizer with gradient clipping
    optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    optimizer.setup(model)
    print("optimizer: SGD")

    ###
    # Training loop
    ###
    learning_rate = args.learning_rate
    former_valid_loss = -1234567890.0

    # for each epoch
    for epoch in range(args.epoch):
        # iterate training minibatches
        train_batch = train_iter.next()

        # reshape the data into (src list) and (tgt list), then transfer to gpu if necessary
        src_list, tgt_list = minibatchToListTuple(train_batch, args.gpu)

        # compute the loss
        loss = model.forward_train(src_list, tgt_list, getID(tgt_vocab_dictionary, 'BOS'))

        # back-prop by auto differential
        model.cleargrads()
        loss.backward()
        loss.unchain.backward()
        optimizer.update()

        # one-pass through of training data done.
        if train_iter.is_new_epoch:

            # display training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float( loss.data )), end='')

            # compute a score on validation set
            valid_losses = []
            while True:
                valid_batch = valid_iter.next()
                val_src_list, val_tgt_list = minibatchToListTuple(valid_batch, args.gpu)

                # forward: cross entropy as validation loss
                val_loss = model.forward_train(val_src_list, val_tgt_list, getID(tgt_vocab_dictionary, 'BOS'))
                valid_losses.append( val_loss.data )

            # end true-while
            valid_loss = np.mean(valid_losses)

            # if valid-score get worse, decrease the learning rate
            if former_valid_loss < valid_loss and epoch > args.learning_rate_decay_start:
                learning_rate = learning_rate * args.learning_rate_decay
                print('val_loss:{:.04f} learning rate(changed):{:.04f}'.format(valid_loss), learning_rate)
            else:
                print('val_loss:{:.04f} learning rate:{:.04f}'.format(valid_loss), learning_rate)

            former_valid_loss = valid_loss

            # end train_iter.is_new_epochif

        # dump the model and the dictionaries on this epoch
        dump_variable = (model, src_vocab_dictionary, tgt_vocab_dictionary)
        pickle.dump(dump_variable, args.out_prefix + "_ep" + str(epoch) + ".model.pckl")

    # end epoch-for


# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer Example: Attention NMT')

    # forced inputs: data sets and output prefix
    parser.add_argument('-data', type=str, required=True,
                        help='serialized training data, containing ID-sequences of train/valid corpus and the vocabulary dictionary (src/tgt)')
    parser.add_argument('-out_prefix', type=str, required=True,
                        help='Output prefix name')

    # training specs
    parser.add_argument('--batchsize', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='Initial valud of learning ratet')
    parser.add_argument('--learning_rate_decay', type=float, default=0.9,
                        help='Discount decaying ratio of learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=10,
                        help='Epoch of decay starting')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout ratio throughout the network')

    # translation specs
    parser.add_argument('--max_tgt_len', type=int, default=50,
                        help='Maximum sequence length of translation')

    # NMT network architecture
    parser.add_argument('--n_layers', '-n', type=int, default=2,
                        help='Number of LSTM stack layers (shared between encoder and decoder)')
    parser.add_argument('--w_vec_dim', '-w', type=int, default=500,
                        help='Dimension of word embeddings (shared between encoder and decoder)')
    parser.add_argument('--lstm_dim', '-l', type=int, default=500,
                        help='Dimension of LSTM hidden states (shared between encoder and decoder)')

    # computational resources, I/F, outputs
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--frequency', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')

    args = parser.parse_args()
    print(args)

    # secure output directory
    last_slash = args.out_prefix.rfind("/")
    if last_slash > -1:
        outdir = args.out_prefix[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
            # end if
    # end last_slash-ifr

    main(args)