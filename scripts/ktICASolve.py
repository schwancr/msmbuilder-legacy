#!/usr/bin/env python

import os
from msmbuilder import io
from msmbuilder import arglib
from msmbuilder import ktica
import scipy.spatial
import numpy as np
import scipy
import logging
import copy
logger = logging.getLogger('msmbuilder.arglib')
logger.setLevel(logging.INFO)

parser = arglib.ArgumentParser(get_kernel=True)
parser.add_argument('dist_mat', help='distance matrix file from ktICADistance.py')
parser.add_argument('reg_factor', type=float, help='regularization factor on identity matrix.')
parser.add_argument('output', help='output filename to save results. you can use ktica_svd.load to read it.')

args, kernel = parser.parse_args()

if os.path.exists(args.output):
    logger.error("path (%s) exists..." % args.output)
    exit()

dt = io.loadh(args.dist_mat, 'dt')[0]
d_flat = io.loadh(args.dist_mat, 'dist_mat')
ptraj = io.loadh(args.dist_mat, 'ptraj')

n = ptraj.shape[0]

if (n % 2) != 0:
    raise Exception("ptraj must be concatenation of two arrays of the same length.")

D = scipy.spatial.distance.squareform(d_flat)

kt = ktica.ktICA(kernel, dt, reg_factor=args.reg_factor)
kt.use_matrix(D, ptraj[:(n / 2)], ptraj[(n / 2):])
kt.solve()

print kt.acf_vals[:10]
print - kt.dt / np.log(kt.acf_vals[:10])

kt.save(args.output)
