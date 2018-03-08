# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        train.py
# Purpose:     A Chainer implementation of Attent ion NMT: training script
#              based on the OpenNMT-py (pytorch) official implementations
#
#              required inputs:
#              serialized object, containing lists of tokenized training sequences (source, target) and vocabulary dictionaries (source, target)
#              output_prefix
#              Note:
#                all target seqeucnes end with <EOS>. 
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
import time
import matplotlib

matplotlib.use('Agg')  # do it background
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

#rcParams['font.family'] = 'IPAGothic'

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

import nmt_model

def make_datatuples(src_list, tgt_list):
    """
    load the src/tgt sequence list, convert to xp.array, then zip them, return as a list of tuples (src_seq, tgt_seq)
    :param src_list: list of source sequences
    :param tgt_list: list of target sequences 
    :return: a list of tuples, each element is (src_seq, tgt_seq)
    """

    assert(len(src_list) == len(tgt_list))

    # pair them
    out_tuples = []
    for (s, t) in zip(src_list, tgt_list):
        out_tuples.append( (xp.array(s, dtype=xp.int32), xp.array(t, dtype=xp.int32)) )

    return out_tuples
# end make_datatuples-def

def minibatchToListTuple(train_batch, gpuid):
    """
    Decompose the minibatch (list of (array(src), array(tgt)) tuple) into two lists of ndarray(int), which are lists of wordID sequences (src, tgt).
    Sort the target sequences in descending order of sequence length
    Transfer them to the designated device:
    if gpuid = 0: cpu (numpy)
    else cpu (cupy)

    :param train_batch:list of (src seq., tgt seq.) tuples.
    :param gpuid: 0 if CPU, else GPU
    :return: src_list: lists of (len-seq) numpy array (type=int)
    """

    src_list, tgt_list = zip(*train_batch)   

   
    # sort target sentences
    tgt_lens = [ len(y) for y in tgt_list  ]
    tgt_permute_idx = sorted(range(len(tgt_lens)), key=lambda k: tgt_lens[k], reverse=True)

    src_list_of_array = [ src_list[i] for i in tgt_permute_idx  ]
    tgt_list_of_array = [ tgt_list[i] for i in tgt_permute_idx  ]
    #src_list_of_array = [ xp.array(src_list[i], dtype=xp.int32) for i in tgt_permute_idx  ]
    #tgt_list_of_array = [ xp.array(tgt_list[i], dtype=xp.int32) for i in tgt_permute_idx  ]

    return src_list_of_array, tgt_list_of_array

def main(args):
    """
    main training loop.

    :param args: argparse instance. having argument values as menber variable.
    :return: serialized binary of the trained model: <args.output_prefix>-ep(epoch).model, including:
               - trained NMT network
               - vocabulary dictionaries of source and target
    """

    print("################")
    print("setting up the data and model")
    print("################")

    ###
    # load the training data. target side sentences end with <EOS>.
    ###
    with open(args.data, 'rb') as fin:
        (train_src, train_tgt, valid_src, valid_tgt, src_vocab_dictionary, tgt_vocab_dictionary) = pickle.load(fin)
    # end open-with

    ###
    # set up the network, optimizer, loss etc
    ###
    print("setting the model up...")
    model = nmt_model.SimpleAttentionNMT(args.n_layers, src_vocab_dictionary, tgt_vocab_dictionary, args.w_vec_dim, args.lstm_dim, args.encoder_type, args.dropout, args.gpu)

    print("model set up complete. ")

    # set up the optimizer with gradient clipping
    optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    print("optimizer: SGD, leargning rate=" + str(optimizer.lr))

    global xp
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id((args.gpu)).use()
        model.to_gpu(args.gpu) # copy the chain to the GPU
        xp = cuda.cupy
    else:
        xp = np
    # end args.gpu-if

    # make a list of type(array, array)
    train_tuples = make_datatuples(train_src, train_tgt)
    valid_tuples = make_datatuples(valid_src, valid_tgt)
    print("training data pickle: " + str(args.data) + " loaded. ")
    print("vocabulary size= " + str(len(src_vocab_dictionary)) + "(src), " + str(len(tgt_vocab_dictionary)) + "(tgt)")
    print("sample size= " + str(len(train_tuples)) + "(training), " + str(len(valid_tuples)) + "(valid)")

    # just a iterators, but is able to repeat and shuffle.
    train_iter = chainer.iterators.SerialIterator(train_tuples, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid_tuples, args.batchsize, repeat=False, shuffle=False)

    print("minibathces: size=" + str(args.batchsize))
    print("data setup complete")
    

    ###
    # Training loop
    ###
    learning_rate = args.learning_rate
    former_valid_loss = -1234567890.0
    forward_time = 0
    backward_time = 0

    print("################")
    print("start training...")
    print("################")
    # for each epoch
    while train_iter.epoch < args.epoch:                

        # iterate training minibatches
        train_batch = train_iter.next()

        # reshape the data into (src list) and (tgt list)
        src_list, tgt_list = minibatchToListTuple(train_batch, args.gpu)

        # compute the loss
        tick = time.time()
        loss = model.forward_train(src_list, tgt_list)
        forward_time = forward_time + time.time() - tick
        
        # back-prop by auto differential
        tick2 = time.time()
        model.cleargrads()
        loss.backward()
        #loss.unchain()
        optimizer.update()
        backward_time = backward_time + time.time() - tick2

        # one-pass through of training data done.
        if train_iter.is_new_epoch:


            # display training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float( loss.data )), end='')
            print("forward time={:.02f}, backward time={:.02f}, ".format(forward_time, backward_time), end='')
            forward_time = 0
            backward_time = 0
            
            # compute a score on validation set
            valid_losses = []
            while True:
                valid_batch = valid_iter.next()
                val_src_list, val_tgt_list = minibatchToListTuple(valid_batch, args.gpu)
                
                # forward: cross entropy as validation loss
                val_loss = model.forward_train(val_src_list, val_tgt_list)
                val_loss.to_cpu()
                valid_losses.append(val_loss.data)
                            
                # sweeped all validation sentences
                if valid_iter.is_new_epoch:
                    valid_iter.epoch = 0
                    valid_iter.current_position = 0
                    valid_iter.is_new_epoch = False
                    valid_iter._pushed_position = None
                    valid_loss = np.mean(valid_losses)

                    # if valid-score get worse, decrease the learning rate
                    if former_valid_loss < valid_loss and train_iter.epoch > args.learning_rate_decay_start:
                        learning_rate = learning_rate * args.learning_rate_decay
                        optimizer.hyperparam.lr = learning_rate
                        print('val_loss:{:.04f} learning rate(changed):{:.04f}'.format(valid_loss, learning_rate))
                    else:
                        print('val_loss:{:.04f} learning rate:{:.04f}'.format(valid_loss, learning_rate))
                    # end valid_loss-decay-ifelse
                    
                    former_valid_loss = valid_loss

                    ##
                    # dump the model specification, and the model binary
                    ##

                    # parameters for model specificaton, and dictionary
                    dump_filename = args.out_prefix + "_ep" + str(train_iter.epoch) + ".model.spec"
                    dump_vars = (args.n_layers, src_vocab_dictionary, tgt_vocab_dictionary, args.w_vec_dim, args.lstm_dim, args.encoder_type, args.dropout, args.gpu)
                    with open(dump_filename, mode='wb') as fout:
                        pickle.dump(dump_vars, fout)
                    
                    # model binary
                    dump_filename = args.out_prefix + "_ep" + str(train_iter.epoch) + ".model.npz"
                    serializers.save_npz(dump_filename, model, compression=True)
                
                    # break the valid_iter-whlie. go toe next training iteration. 
                    break 
                # end valid_iter_is_new_epoch-if
            # end true-while for valid_iter

        # end train_iter.is_new_epochif

    # end true-while for train_iter

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
    parser.add_argument('--encoder_type', '-e', type=str, default="rnn",
                        help='Choose \'rnn\' for uni-directional LSTM encoder, \'brnn\' for bi-directional LSTM encoder')

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