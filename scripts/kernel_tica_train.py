#!/usr/bin/env python
 
from msmbuilder import arglib, sshfs_tools, tICA
from msmbuilder.kernels.dotproduct import DotProduct
from msmbuilder import Project, Trajectory, io
from msmbuilder.Trajectory import _convert_from_lossy_integers, DEFAULT_PRECISION
import tables
import numpy as np
import scipy
import os, sys, re
import scipy
import logging
logger = logging.getLogger( __name__ )

def run(kernel, project, atom_indices, out_fn, dt, num_samples):

    # First sample the dataset into memory:
    
    traj_inds = np.concatenate([ [i]*project.traj_lengths[i] for i in xrange(project.n_trajs) ])

    frame_inds = np.concatenate([ np.arange(project.traj_lengths[i]) for i in xrange(project.n_trajs) ])

    which_inds = np.array(zip(traj_inds, frame_inds))

    sampled_inds = np.random.permutation(np.arange(len(which_inds)))
    sampled_inds_dt = sampled_inds + dt

    good_inds = np.where(sampled_inds_dt < len(which_inds))
    sampled_inds = sampled_inds[good_inds]
    sampled_inds_dt = sampled_inds_dt[good_inds]

    permuted_which = which_inds[sampled_inds]
    permuted_which_dt = which_inds[sampled_inds_dt]

    good_inds = np.where( permuted_which[:,0] == permuted_which_dt[:,0] )

    permuted_which = permuted_which[good_inds][:num_samples]
    permuted_which_dt = permuted_which_dt[good_inds][:num_samples]

    traj_0 = project.empty_traj()
    traj_dt = project.empty_traj()

    for traj_ind in np.unique(np.concatenate((permuted_which[:,0], permuted_which_dt[:,0]))):
        F = tables.openFile(project.traj_filename(traj_ind))
        frame_inds = permuted_which[:,1][ np.where(permuted_which[:,0] == traj_ind) ]
        logger.info("Loading %d frames from trajectory %s", len(frame_inds), project.traj_filename(traj_ind))
        for i in frame_inds:
            if traj_0['XYZList'] is None:
                traj_0['XYZList'] = np.array([ F.root.XYZList[i] ])
            else:
                traj_0['XYZList'] = np.vstack([ traj_0['XYZList'], np.array([ F.root.XYZList[i] ]) ])

        frame_inds = permuted_which_dt[:,1][ np.where(permuted_which_dt[:,0] == traj_ind) ]
        for i in frame_inds:
            if traj_dt['XYZList'] is None:
                traj_dt['XYZList'] = np.array([ F.root.XYZList[i] ])
            else:
                traj_dt['XYZList'] = np.vstack([ traj_dt['XYZList'], np.array([ F.root.XYZList[i] ]) ])
        
        F.close()

    traj_0['XYZList'] = _convert_from_lossy_integers( traj_0['XYZList'], DEFAULT_PRECISION )
    traj_dt['XYZList'] = _convert_from_lossy_integers( traj_dt['XYZList'], DEFAULT_PRECISION )
    ptraj_0 = kernel.prepare_trajectory(traj_0)
    ptraj_dt = kernel.prepare_trajectory(traj_dt)

    gram_mat = tICA.GramMatrix(kernel, symmetrize=True, diagonal_load=0.0)
    # Still need to figure out how to pick the diagonally loading parameter...
    gram_mat.calc_gram_matrices(ptraj_0, ptraj_dt)

    vals, vecs = gram_mat.get_eigensolution()

    print np.sort(vals)[::-1][:10]

    # Need to normalize each of the vectors

    K_0, K_dt = gram_mat.get_centered_matrices()

    norm_mat = np.sqrt(np.square(K_0.dot(vecs)).sum(axis=0))
    #norm_mat = np.sqrt(np.square(gram_mat.gram_matrix_0.dot(vecs)).sum(axis=0))

    vecs /= np.reshape(norm_mat, (1,-1))  # Be sure it is a row vector
    
    io.saveh(out_fn, vals=vals, vecs=vecs, gram_mat=gram_mat.gram_matrix_0, gram_mat_dt=gram_mat.gram_matrix_dt,
             trained_ptraj=ptraj_0) 

    return

if __name__ == '__main__':
    parser = arglib.ArgumentParser(get_kernel=True)
    parser.add_argument('project')
    parser.add_argument('num_samples',help='number of samples to grab from the data. Note that we '
                                           'need to diagonalize a square matrix of this size, so this'
                                           ' can\'t be too big...',type=int,default=1000)
    parser.add_argument('atom_indices',help='atom indices to restrict trajectories to',default='all')
    parser.add_argument('out_fn',help='output filename to save results to',default='ktICAData.h5')
    parser.add_argument('delta_time',help='delta time to use in calclating the time-lag correlation matrix',type=int)
    
    args, kernel = parser.parse_args()
    
    arglib.die_if_path_exists(args.out_fn)
    
    try: 
        atom_indices = np.loadtxt(args.atom_indices).astype(int)
    except: 
        atom_indices = None

    dt = int( args.delta_time )
    num_samples = int( args.num_samples )
    project = Project.load_from( args.project )

    run(kernel, project, atom_indices, args.out_fn, dt, num_samples)

