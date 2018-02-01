# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        translate.py
# Purpose:     A Chainer implementation of Attent ion NMT: translation script
#              based on the OpenNMT-py (pytorch) official implementations
#
#              required inputs:
#              serialized Chainer NMT model (w/ src and tgt vocabulary dictionary)
#              input text file, list of tokenized source sequences
#              output_file name
#
#              # TODO: re-write for beam decoding.
#
#              outputs:
#              text file of translated tokenized target sentences
#              print the translation and the log likelihood on stdout
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     25/01/2018 (DD/MM/YY)
# Last update: 25/01/2018 (DD/MM/YY)
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

def convertToWordSequence(vocab_dict, IDseq):
    """
    Convert an ID array into a tokenized string.

    :param vocab_dict: dictionary to look up
    :param IDseq: np.array, sequence of token IDs
    :return: a tokenized string
    """

    out_list = []
    for id in IDseq:
        out_list.append(vocab_dict.keys()[vocab_dict.values().index(id)])

    # remove EOS
    if out_list[-1] == "<EOS>":
        out_list.remove("<EOS>")

    return " ".join(out_list)

def convertToIDSequence(vocab_dict, line):
    """
    Convert a tokenized string into a ID array

    :param vocab_dict: dictionary to look up
    :param line: string, a tokenized sentence
    :return: numpy array of token IDs
    """

    tokens = line.rstrip().split()
    out_array = np.zeros( len(tokens) )
    for i, token in enumerate(tokens):
        if vocab_dict.has_key(token):
            out_array[i] = vocab_dict[token]
        else:
            out_array[i] = vocab_dict["<UNK>"]
        # end has_key-ifelse
    # end token-for

    return out_array
# end convertToIDSequence-def

def main(args):
    """
    main translation code

    :param args: argparse instance. having argument values as menber variable.
    :return: text file of translated tokenized target sentences
              print the translation and the log likelihood on stdout
    """


    ###
    # load the trained model
    ###

    (model, src_vocab_dictionary, tgt_vocab_dictionary) = pickle.load(args.model)
    global xp
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id((args.gpu)).use()
        model.to_gpu(args.gpu) # copy the cphain to the GPU
        xp = cuda.cupy
    else:
        xp = np
    # end args.gpu-if

    print("trained model pickle " + str(args.model) + " loaded. ")

    ###
    # read the input and perform translation
    ###

    snt_id = 0
    with open(args.src, 'r') as fin, open(args.out_name, 'w') as fout:
        snt_id = snt_id + 1
        src_lines = fin.readlines()
        for src_line in src_lines:
            # convert the line into srcID sequence
            src_IDseq = convertToIDSequence(src_vocab_dictionary, src_line)

            # translate
            # TODO: change to decode_translate_beam
            tgt_IDseq, log_lk = model.decode_translate_greedy(src_IDseq, args.max_tgt_length, getID(tgt_vocab_dictionary, '<BOS>'), getID(tgt_vocab_dictionary, '<EOS>'))

            # convert back to the target words
            tgt_line = convertToWordSequence(tgt_vocab_dictionary, tgt_IDseq)

            # write the translation into output file
            fout.write(tgt_line + "\n")

            # print to stdout unless suppressed
            if not args.quiet:
                print("input " + str(snt_id) + ": " + src_line.rstrip())
                print("translation " + str(snt_id) + ": " + tgt_line)
                print("log likelihood: " + log_lk)
                print(" ")
        # end src_lines-for
    # end with-open

# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer Example: Attention NMT')

    # forced inputs: trained model binary, input source, output target
    parser.add_argument('-model', type=str, required=True,
                        help='serialized chainer model, containing the trained NMT model, src/tgt vocabulary dictionary')
    parser.add_argument('-src', type=str, required=True,
                        help="a text file with source input sentences")
    parser.add_argument('-out_name', type=str, required=True,
                        help='Translation output file name')

    # output specs
    parser.add_argumnt('--detailed_output', action='store_true', default=False,
                      help='enable if log likelihood is needed in translation')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='enable if you prefer no outputs in stdout')

    # translation specs
    parser.add_argument('--max_tgt_len', type=int, default=50,
                        help='Maximum sequence length of translation')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='beam size. if 1 then use greedy search')

    # computational resources, I/F, outputs
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')

    args = parser.parse_args()
    print(args)

    # secure output directory
    last_slash = args.out_name.rfind("/")
    if last_slash > -1:
        outdir = args.out_name[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
            # end if
    # end last_slash-ifr

    main(args)