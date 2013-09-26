#!/usr/bin/env python 

from msmbuilder import arglib
import numpy as np
from msmbuilder import Project, Trajectory, io
import tables
import logging
import pickle
logger = logging.getLogger('msmbuilder.arglib')
logger.setLevel(logging.INFO)


def load_traj_subset(project, ind, start, end, stride):

    trj_file = tables.openFile(project.traj_filename(ind))

    trj = project.empty_traj()

    trj['XYZList'] = trj_file.root.XYZList[start:end:stride].astype(float) / 1000.

    trj_file.close()
    
    return trj


def run(project, output, dt=1, stride=1, randomize=False, skip_every=1):

    if randomize:
        start_ind = lambda : np.random.randint(stride)
    else:
        start_ind = lambda : 0

    # estimate the size of the matrix.
    est_size = np.sum(project.traj_lengths[::skip_every][np.where(project.traj_lengths[::skip_every] > dt)] / stride + 1)
    logger.info('will have approximately %d pairs' % est_size)

    trjA = project.empty_traj()
    trjB = project.empty_traj()

    for i in xrange(0, project.n_trajs, skip_every):

        if project.traj_lengths[i] <= dt:
            logger.info('trajectory (%s) not long enough (%d)' % (project.traj_filename(i), project.traj_lengths[i]))
            continue

        i0 = start_ind() % (project.traj_lengths[i] - dt)
        indA = np.arange(i0, project.traj_lengths[i], stride)
        indB = np.arange(i0 + dt, project.traj_lengths[i], stride)

        indA = indA[: len(indB)]
        assert indA.shape == indB.shape

        trjA += load_traj_subset(project, i, indA[0], indA[-1] + stride, stride)
        trjB += load_traj_subset(project, i, indB[0], indB[-1] + stride, stride)

        logger.info('added trajectory %s (%d pairs)' % (project.traj_filename(i), len(trjA)))

    traj = trjA + trjB

    ptraj = metric.prepare_trajectory(traj)

    if isinstance(ptraj, np.ndarray):
        ptraj = ptraj.astype(np.double)

    dist_mat = metric.all_pairwise(ptraj)

    try:
        metric_str = pickle.dumps(metric)
    except:
        metric_str = 'unpickleable'
        print "metric is not pickleable."

    io.saveh(output, dist_mat=dist_mat, dt=np.array([dt]), ptraj=ptraj,
            metric_str=np.array([metric_str]))

    logger.info('saved output to %s' % output)


if __name__ == '__main__':
    
    parser = arglib.ArgumentParser(get_basic_metric=True)
    parser.add_argument('project')
    parser.add_argument('output')
    parser.add_argument('stride', type=int, default=1, help='stride to subsample data by')
    parser.add_argument('dt', default=1, type=int, help='correlation lag time in frames')
    parser.add_argument('randomize', default=False, action='store_true', help='randomize location of first pair in each trajectory. Otherwise start with frame 0')
    parser.add_argument('skip', type=int, default=1, help='only use every <n> trajectories')

    args, metric = parser.parse_args()

    project = Project.load_from(args.project)

    arglib.die_if_path_exists(args.output)

    run(project, args.output, dt=args.dt, stride=args.stride, randomize=args.randomize, 
        skip_every=args.skip)
