#!/usr/bin/env python
 
from msmbuilder import arglib, sshfs_tools, tICA
from msmbuilder.kernels.dotproduct import DotProduct
from msmbuilder import Project, Trajectory, io
import numpy as np
import scipy
import os, sys, re
import scipy
import logging
logger = logging.getLogger( __name__ )

def run(kernel, project, atom_indices, out_fn, min_length, lag, stride):

    # First load all of the data into memory:
    
    ptraj_0 = []
    ptraj_dt = []

    for i in xrange(project.n_trajs):
        logger.info("Loading trajectory %d", i)

        traj = project.load_traj(i, atom_indices=atom_indices, stride=stride)
        temp_ptraj = kernel.prepare_trajectory(traj)

        ptraj_0.extend(temp_ptraj[:-lag])
        ptraj_dt.extend(temp_ptraj[lag:])

    ptraj_0 = np.array(ptraj_0)
    ptraj_dt = np.array(ptraj_dt)

    gram_mat = tICA.GramMatrix(kernel)
    gram_mat.calc_gram_matrices(ptraj_0, ptraj_dt)

    vals, vecs = gram_mat.get_eigensolution()

    # Need to normalize each of the vectors

    norm_mat = np.sqrt(np.square(gram_mat.gram_matrix_0.dot(vecs)).sum(axis=0))

    vecs /= norm_mat
    
    io.saveh(out_fn, vals=vals, vecs=vecs, gram_mat=gram_mat.gram_matrix_0, gram_mat_dt=gram_mat.gram_matrix_dt,
             trained_ptraj=ptraj_0) 

    return

if __name__ == '__main__':
    parser = arglib.ArgumentParser(get_kernel=True)
    parser.add_argument('project')
    parser.add_argument('stride',help='stride to subsample input trajectories',type=int,default=1)
    parser.add_argument('atom_indices',help='atom indices to restrict trajectories to',default='all')
    parser.add_argument('out_fn',help='output filename to save results to',default='tICAData.h5')
    parser.add_argument('delta_time',help='delta time to use in calclating the time-lag correlation matrix',type=int)
    parser.add_argument('min_length',help='only train on trajectories greater than some number of frames',type=int,default=0)
    
    args, kernel = parser.parse_args()
    
    arglib.die_if_path_exists(args.out_fn)
    
    try: 
        atom_indices = np.loadtxt(args.atom_indices).astype(int)
    except: 
        atom_indices = None

    stride = int( args.stride )
    dt = int( args.delta_time )
    project = Project.load_from( args.project )
    min_length = int( float( args.min_length ) ) # need to convert to float first because int can't convert a string that is '1E3' for example...wierd.
    lag = int( dt / stride )

    if float(dt)/stride != lag:
        logger.error( "Stride must be a divisor of dt..." )
        sys.exit()

    run(kernel, project, atom_indices, args.out_fn, min_length, lag, stride)

