# " -*- coding: utf-8 -*-"
# -------------------------------------------------------------------------------
# Name:        __init__.py
# Purpose:     
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

def main():
    pass

# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do something')
    parser.add_argument('data_file', help='data file name')
    parser.add_argument('output_prefix', help='output prefix')
    args = parser.parse_args()
    print(args)

    # secure output directory
    last_slash = args.output_prefix.rfind("/")
    if last_slash > -1:
        outdir = args.output_prefix[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
            # end if
    # end last_slash-ifr

    main(args.data_file, args.output_prefix)