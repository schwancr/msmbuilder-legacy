#!/usr/bin/env python
# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import os, sys
import numpy as np
from msmbuilder.metrics import RMSD
from msmbuilder import Project
from msmbuilder import Trajectory
from msmbuilder import io
from msmbuilder import arglib
import warnings

import logging
logger = logging.getLogger('msmbuilder.scripts.CalculateProjectRMSD')

def run(project, pdb, atom_indices):    
    distances = -1 * np.ones((project.n_trajs, np.max(project.traj_lengths)))
    rmsd = RMSD(atom_indices)
    ppdb = rmsd.prepare_trajectory(pdb)
    
    for i in xrange(project.n_trajs):
        ptraj = rmsd.prepare_trajectory(project.load_traj(i))
        d = rmsd.one_to_all(ppdb, ptraj, 0)
        distances[i, 0:len(d)] = d
    
    return distances
    
    
if __name__ == '__main__':
    deprecationmessage = """
===============================================================================
This script is deprecated and will be removed in v2.7 
Please use CalculateProjectDistance.py
===============================================================================
"""
    parser = arglib.ArgumentParser(description="""
Calculate the RMSD between an input PDB and all conformations in your project.
Output as a HDF5 file (load using msmbuilder.io.loadh())
""" + deprecationmessage)
    warnings.warn(deprecationmessage, DeprecationWarning)
    
    parser.add_argument('pdb')
    parser.add_argument('atom_indices', help='Indices of atoms to compare',
        default='AtomIndices.dat')
    parser.add_argument('output', help='''Output file name. Output is an
        .h5 file with RMSD entries corresponding to the Assignments.h5 file.''',
        default='Data/RMSD.h5')
    parser.add_argument('project')
    args = parser.parse_args()

    arglib.die_if_path_exists(args.output)

    project = Project.load_from(args.project)
    pdb = Trajectory.load_trajectory_file( args.pdb )
    atom_indices = np.loadtxt( args.atom_indices ).astype(int)

    distances = run(project, pdb, atom_indices)
    
    io.saveh(args.output, distances)
    logger.info('Saved to %s', args.output)
