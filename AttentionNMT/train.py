# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        train.py
# Purpose:     A Chainer implementation of Attent ion NMT: training script.
#              based on the OpenNMT-py (pytorch) official implementations
#
#              inputs:
#              minibatches of tokenized training sequences: src/tgt.
#              the lenght of sequences in the mini-batch must be a DESCENDING ORDER.
#              vocabulray list: src/tgt
#              output_prefix
#
#              ToDo: add options to control network structure
#
#              outputs:
#              trained NMT model
#
# Author:      Katsuhiko Ishiguro <k.ishiguro.jp@ieee.org>
#
#
# License:     All rights reserved unless specified.
# Created:     08/01/2018 (DD/MM/YY)
# Last update: 08/01/2018 (DD/MM/YY)
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

import net

def main(batchsize, epoch, frequency, gpu, outdir, resume, unit, plot):
    """
    main training loop.

    :param batchsize:
    :param epoch:
    :param frequency:
    :param gpu:
    :param outdir:
    :param resume:
    :param unit:
    :param plot:
    :return:
    """

    # set up the NN
    # chainer.links.Classifier(predictor, lossfun=<function softmax_cross_entropy>, accfun=<function accuracy>, label_key=-1)
    model = L.Classifier(net.MLP(unit, 10))
    if gpu >= 0:
        chainer.cuda.get_device_from_id((gpu)).use()
        model.to_gpu() # copy the chain to the GPU

    # set up the optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load Mnist data
    train, test = chainer.datasets.get_mnist() #train and test are TupleDataset variable

    # just a iterators, but is able to repeat and shuffle.
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # set up a trainer and updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=outdir)

    ### trainer extentions ###
    # evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    # Dump a computationl graph from 'loss' variable at the first itertion
    # The "main" referes to the target link of the "main" optimizer
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snopshot for each specified epoch
    frequency = epoch if frequency == -1 else max(1, frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation stats. for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()

    # serizlize the final model
    serializers.save_npz('my_mnist.model', model)

# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer Example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()
    print(args)

    # secure output directory
    last_slash = args.outdir.rfind("/")
    if last_slash > -1:
        outdir = args.outdir[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
            # end if
    # end last_slash-ifr

    main(args.batchsize, args.epoch, args.frequency, args.gpu, args.outdir, args.resume, args.unit, args.plot)