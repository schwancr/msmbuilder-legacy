#!/usr/bin/env python
from __future__ import print_function, absolute_import, division


import logging
import os
import sys
import re
import numpy as np
import scipy
import mdtraj as md
from mdtraj import io
from mdtraj.utils.six.moves import xrange
from msmbuilder import arglib
from msmbuilder import Project
from msmbuilder.reduce.tICA import tICA
logger = logging.getLogger('msmbuilder.scripts.tICA_train')


parser = arglib.ArgumentParser(get_metric=True, description="""
Calculate the time-lag correlation and covariance matrices for use in the tICA
metric. This method attempts to find projection vectors such that they have a
maximal autocorrelation function.

For more details see:
Schwantes, CR and Pande, VS. J. Chem. Theory Comput., 2013, 9 (4),
pp 2000-2009. DOI: 10.1021/ct300878a""")
parser.add_argument('project')
parser.add_argument('stride', type=int, default=1,
                    help='stride to subsample input trajectories')
parser.add_argument('atom_indices', default='all',
                    help='atom indices to restrict trajectories to')
parser.add_argument('output', default='tICAData.h5',
                    help='output filename to save results to')
parser.add_argument('delta_time', type=int, help="""delta time to 
    use in calculating the time-lag correlation matrix""")
parser.add_argument('min_length', type=int, default=0,
                    help="""only train on trajectories greater than <min_length> 
    number of frames""")
parser.add_argument('traj_indices', default=None, help="""a filename with a line 
    for each trajectory in your project. Each line contains two indices corresponding
    to an interval of the frames to use in the calculation [a, b)""")


def run(prep_metric, project, delta_time, atom_indices=None,
        output='tICAData.h5', min_length=0, stride=1, traj_indices=None):

    # We will load the trajectories at the stride, so we need to find
    # what dt should be once we've strided by some amount
    lag = delta_time / stride

    if (float(delta_time) / stride) != lag:
        raise Exception("Stride must be a divisor of delta_time.")

    if lag > 0:  # Then we're doing tICA
        tica_obj = tICA(lag=lag, calc_cov_mat=True, prep_metric=prep_metric)
    else:  # If lag is zero, this is equivalent to regular PCA
        tica_obj = tICA(lag=lag, calc_cov_mat=False, prep_metric=prep_metric)

    for i in xrange(project.n_trajs):
        logger.info("Working on trajectory %d" % i)

        if project.traj_lengths[i] <= lag:
            logger.info("Trajectory is not long enough for this lag "
                        "(%d vs %d)", project.traj_lengths[i], lag)
            continue

        if project.traj_lengths[i] < min_length:
            logger.info("Trajectory is not longer than min_length "
                        "(%d vs %d)", project.traj_lengths[i], min_length)
            continue

        # it would be more memory efficient if we trained incrementally
        # at least for long trajectories
        traj_chunk = md.load(project.traj_filename(
            i), stride=stride, atom_indices=atom_indices)

        if not traj_indices is None:
            a, b = traj_indices[i] / stride 
            # integer arithmetic should truncate correctly when
            # the indices are not multiples of the stride
            traj_chunk = traj_chunk[a:b]

        tica_obj.train(trajectory=traj_chunk)

    tica_obj.solve()
    tica_obj.save(output)
    logger.info("Saved output to %s", output)

    return tica_obj


def entry_point():
    args, prep_metric = parser.parse_args()
    arglib.die_if_path_exists(args.output)

    if args.atom_indices.lower() == 'all':
        atom_indices = None
    else:
        atom_indices = np.loadtxt(args.atom_indices).astype(int)

    project = Project.load_from(args.project)
    min_length = int(float(args.min_length))
    # need to convert to float first because int can't
    # convert a string that is '1E3' for example...weird.

    if not args.traj_indices is None:
        traj_indices = np.loadtxt(args.traj_indices).astype(int)
    else:
        traj_indices = None

    tica_obj = run(
        prep_metric, project, args.delta_time, atom_indices=atom_indices,
        output=args.output, min_length=min_length, stride=args.stride,
        traj_indices=traj_indices)

if __name__ == "__main__":
    entry_point()
