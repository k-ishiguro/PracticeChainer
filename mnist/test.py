# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        test.py
# Purpose:     manually written test code for chaner/example/mnist
#
#              inputs: 
#
#              outputs: 
#
# Author:      Katsuhiko Ishiguro <ishiguro.katsuhiko@lab.ntt.co.jp>
# 
# 
# License:     All rights reserved unless specified. 
# Created:     19/12/2017 (DD/MM/YY)
# Last update: 19/12/2017 (DD/MM/YY)
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
from chainer.cuda import to_cpu


import net

def main(learned_model, gpu, outdir, plot):
    """
    main test process.

    :param learned_model:
    :param gpu:
    :param outdir:
    :param plot:
    :return:
    """

    # set up the NN
    # chainer.links.Classifier(predictor, lossfun=<function softmax_cross_entropy>, accfun=<function accuracy>, label_key=-1)
    infer_net = net.MLP(None,None)
    serializers.load_npz('result/snapshot_iter_12000', infer_net, path='updater/model:main/predictor/')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id((gpu)).use()
        infer_net.to_gpu() # copy the chain to the GPU

    # Load Mnist data, again. we only use test.
    train, test = chainer.datasets.get_mnist() #train and test are TupleDataset variable

    for i in range(10):
        x, t = test[i]
        #"plt.imshow(x.reshape(28, 28), cmap='gray')
        #plt.show()

        x = infer_net.xp.asarray(x[None, ...])
        y = infer_net(x)
        y = to_cpu(y.array)

        print('予測ラベル:', y.argmax(axis=1)[0])
    # end i-for

# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer Example: MNIST-test')
    parser.add_argument('--learned_model', '-m', type=str, default=None,
                        help='path to the learned Chainer MLP model for MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
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

    main(args.learned_model, args.gpu, args.outdir, args.plot)