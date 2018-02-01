# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        preprocess.py
# Purpose:     A Chainer implementation of Attent ion NMT: preprocessing the trainng dataset
#              based on the OpenNMT-py (pytorch) official implementations
#
#              required inputs:
#              a text file for src-training corpus, one line one sequence, tokenized by spaces.
#              a text file for tgt-training corpus, one line one sequence, tokenized by spaces.
#              a text file for src-validation corpus, one line one sequence, tokenized by spaces.
#              a text file for tgt-validation corpus, one line one sequence, tokenized by spaces.
#              out_prefix
#
#              Special tokens forced to included in the dictionary:
#              <BOS> - beginning of sentence
#              <EOS> - end of sentecne
#              <UNK> - out-of-vocabulary tokens
#
#              sequences: all tokens are replaced with the dictionary index. <EOS> is attached to all target sentences.
#                          Note all sequences are sorted by the TARGET token length in descending order
#
#              dictionary: sorted by the number of occurences. Last three entries are <BOS>, <EOS>, and <UNK>
#
#
#               outputs:
#               serialized file, including:
#                - ID-converted sequences of {src,tgt}-{train,target} corpus. All sentences are appended with <BOS>, <EOS>
#                - src/tgt vocabulary dictionaries
#               two vocabulary dictionary text files (src, tgt, for human reading)
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     29/01/2018 (DD/MM/YY)
# Last update: 01/02/2018 (DD/MM/YY)
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


def rearrangeLists(input_src_list, input_tgt_list, max_sentence_length):
    """
    rearrange the order of sentence pairs based on the token-length of target sentences

    :param input_src_list: list of token-Id lists, source side
    :param input_tgt_list: list of token-Id lists, source side
    :param max_sentence_length: maximum number of tokens allowed in a sentence
    :return: reordered lists
    """

    # we sort the sentences into 'buckets', then concat them in descending order

    bucket = [] # will be two-axis list
    for l in range(max_sentence_length):
        bucket.append([])

    for l in range( len(input_tgt_list) ):
        len_tgt = len(input_tgt_list[l])
        bucket[len_tgt].append(l)
    # end for

    # sanity check
    sum_bucket = 0
    for bucket_i in bucket:
        sum_bucket = sum_bucket + len(bucket_i)
    assert(sum_bucket == len(input_tgt_list))


    # reordered lists
    out_src_list = []
    out_tgt_list = []

    ls = range(max_sentence_length)[::-1]
    for l in ls:
        indices = bucket[l]
        for i in indices:
            out_src_list.append(input_src_list[i])
            out_tgt_list.append(input_tgt_list[i])
        # end i-for
    # end l-for

    return out_src_list, out_tgt_list
# emd filterList-def


def filterLists(input_src_list, input_tgt_list, max_sentence_length):
    """
    filtered out the empty lines and too-long lines from two input lists.

    :param input_src_list: list of token-Id lists, source side
    :param input_tgt_list: list of token-Id lists, source side
    :param max_sentence_length: maximum number of tokens allowed in a sentence
    :return: filtered list
    """

    assert(len(input_src_list) == len(input_tgt_list))

    out_src_list = []
    out_tgt_list = []
    for l in range( len(input_src_list) ):
        if len(input_src_list[l]) < 1:
            pass
        elif len(input_src_list[l]) > max_sentence_length - 1: #-1 for EOS
            pass
        if len(input_tgt_list[l]) < 1:
            pass
        elif len(input_tgt_list[l]) > max_sentence_length - 1: #-1 for EOS
            pass
        else:
            out_src_list.append(input_src_list[l])
            out_tgt_list.append(input_tgt_list[l])
        # end if-else
    # end for

    return out_src_list, out_tgt_list
# emd filterList-def

def convertToIDs(input_file, vocab_dict):
    """
    Convert the raw token sentences into vocabulary-ID sentences. return as a list.

    :param input_file: file path, many lines of space-tokenized sentences. could be very many lines
    :param vocab_dict: a dictionary object, <key>=token <val> = index
    :return: list of ID-seqeucnes, each sequence is a list of token IDs
    """

    out_list= []
    with open(input_file, 'r') as fin:
        line = fin.readline()
        while line:
            out_list_line = []
            #out_list_line = [vocab_dict["<BOS>"]]
            line = line.rstrip()
            tokens = line.split()
            for token in tokens:
                if token in vocab_dict:
                    out_list_line.append(vocab_dict[token])
                else:
                    out_list_line.append(vocab_dict["<UNK>"])
                # end token-in-ifelse
            # end token-for

            # end of line
            #out_list_line.append(vocab_dict["<EOS>"])
            
            out_list.append(out_list_line)
            # go next.
            line = fin.readline()
        # end line-while
    # end fin-with

    return out_list

# end convertToIDs

def make_vocabulary(input_file, max_vocab_size, output_file):
    """
    Read the input file, sort tokens appeared by the frequency, cut-off entries lowerer than max_vocab_size.

    :param input_file: file path, many lines of space-tokenized sentences. could be very many lines
    :param max_vocab_size:number of maximum vocabulary, excluding <BOS>, <EOS>, <UNK>
    :param: output_file: file path of the text file output
    :return: a text file of dictionary entries. in a descending order of token frequencies. Each line, token \t frequency
              a dictionary object, for each entry, <key>=token <val>=index (frequency descending order)
    """

    # count all the tokens
    freq_dict = {}
    with open(input_file, 'r') as fin:
        line = fin.readline()
        while line:
            line = line.rstrip()
            tokens = line.split()
            for token in tokens:
                if token in freq_dict:
                    freq_dict[token] = freq_dict[token] + 1
                else:
                    freq_dict[token] = 1
                # end token-in-ifelse
            # end token-for
            line = fin.readline()
        # end line-while
    # end fin-with

    # sort by frequency. write to a text file
    numElement = 0
    vocab_dict = {}
    with open(output_file, "w") as fout:
        for k, v in sorted(freq_dict.items(), key=lambda x: -x[1]):
            fout.write(str(k) + "\t" + str(v) + "\n")
            vocab_dict[k] = numElement

            numElement = numElement + 1

            if numElement >= max_vocab_size:
                break
            # end if
        # end sort-for

        # add special tokens
        fout.write('<BOS>" + "\t" + "0" + \n')
        fout.write('<EOS>" + "\t" + "0" + \n')
        fout.write('<UNK>" + "\t" + "0" + \n')

        vocab_dict["<BOS>"] = numElement
        vocab_dict["<EOS>"] = numElement + 1
        vocab_dict["<UNK>"] = numElement + 2

        print(output_file + " created, vocabulary size=" + str(numElement+2))

    # end opne-with

    return vocab_dict
# end make_vocabulary-def

def main(args):
    """
    main training loop.

    :param args: argparse instance. having argument values as member variable.
    :return: serialized file, including:
               - ID-converted sequences of {src,tgt}-{train,target} corpus. All target sentences are appended with <EOS>
               - src/tgt vocabulary dictionaries
    """

    ###
    # make vocabulary dictionaries
    ###
    print("making vocabulary dictionaries...")
    src_vocab_dictionary = make_vocabulary(args.train_src, args.src_vocab_size, args.out_prefix+".src.vocab")
    tgt_vocab_dictionary = make_vocabulary(args.train_tgt, args.src_vocab_size, args.out_prefix+".tgt.vocab")

    ###
    # read the training parallel corpus, replace with token ids, attach EOS.
    ###
    train_src_id = convertToIDs(args.train_src, src_vocab_dictionary)
    train_tgt_id = convertToIDs(args.train_tgt, tgt_vocab_dictionary)

    ###
    # filter out empty or too long sentences. otherwise,
    ###
    train_src_id, train_tgt_id = filterLists(train_src_id, train_tgt_id, args.max_sentence_length)

    ###
    # re-arrange the order of sentence list by the target sentences' token lengths
    ###
    train_src_id, train_tgt_id = rearrangeLists(train_src_id, train_tgt_id, args.max_sentence_length)

    ###
    # validation corpus (no re-arrange)
    ###
    val_src_id = convertToIDs(args.val_src, src_vocab_dictionary)
    val_tgt_id = convertToIDs(args.val_tgt, tgt_vocab_dictionary)
    val_src_id, val_tgt_id = filterLists(val_src_id, val_tgt_id, args.max_sentence_length)

    ###
    # append <EOS> for all target sentences
    ###
    for tgt1 in train_tgt_id:
        tgt1.append( tgt_vocab_dictionary["<EOS>"] )
    for tgt2 in val_tgt_id:
        tgt2.append( tgt_vocab_dictionary["<EOS>"] )

    ###
    # dump
    ###
    out_tuple = (train_src_id, train_tgt_id, val_src_id, val_tgt_id, src_vocab_dictionary, tgt_vocab_dictionary)
    with open(args.out_prefix + ".traindata.pckl", "w") as fout:
        pickle.dump(out_tuple, fout)
    # end with-open


# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer Example: Attention NMT')

    # forced inputs: train/valid corpuses and output prefix
    parser.add_argument('-train_src', type=str, required=True,
                        help='path to the src-training corpus')
    parser.add_argument('-train_tgt', type=str, required=True,
                        help='path to the tgt-training corpus')
    parser.add_argument('-val_src', type=str, required=True,
                        help='path to the src-validation corpus')
    parser.add_argument('-val_tgt', type=str, required=True,
                        help='path to the tgt-validation corpus')
    parser.add_argument('-out_prefix', type=str, required=True,
                        help='Output prefix name')

    # options
    parser.add_argument('--src_vocab_size', type=int, default=50000,
                        help='maximum number of source side vocabulary')
    parser.add_argument('--tgt_vocab_size', type=int, default=50000,
                        help='maximum number of target side vocabulary')

    parser.add_argument('--max_sentence_length', type=int, default=80,
                        help='maximum number of tokens in a sentence. too long sentences will be removed from corpus')


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