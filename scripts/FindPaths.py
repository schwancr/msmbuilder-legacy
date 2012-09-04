#!/usr/bin/python
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

#TJL 2011, PANDE GROUP

import sys
import numpy as np
import os
import scipy.io

from msmbuilder import transition_path_theory
from msmbuilder import Serializer
from msmbuilder import arglib

def run(NFlux, A, B, n):

    (Paths, Bottlenecks, Fluxes) = transition_path_theory.DijkstraTopPaths(A, B, NFlux, NumPaths=n, NodeWipe=False)

    # We have to pad the paths with -1s to make a square array
    maxi = 0 # the maximum path length
    for path in Paths:
        if len(path) > maxi: maxi = len(path)
    PaddedPaths = -1 * np.ones( (len(Paths), maxi ) )
    for i, path in enumerate(Paths):
        PaddedPaths[i,:len(path)] = np.array(path)
    
    
    return PaddedPaths, np.array(Bottlenecks), np.array(Fluxes)

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
"""Finds the highest flux paths through an MSM.
Returns: an HDF5 file (default: Paths.h5), which contains three items:
(1) The highest flux pathways (a list of ints)
(2) The bottlenecks in these pathways (a list of 2-tuples)
(3) The flux of each pathway

Paths.h5 can be read by RenderPaths.py which generates a .dot file capturing these paths.""")
    
    parser.add_argument('number', help='''Number of pathways you want
        to retreive''', type=int)
    parser.add_argument('flux_matrix', help='Net flux matrix from DoTPT.py',
        default='NFlux.mtx')
    parser.add_argument('starting', help='''Vector of states in the
        starting/reactants/unfolded ensemble.''', default='U_states.dat')
    parser.add_argument('ending', help='''Vector of states in the
        ending/products/folded ensemble.''', default='F_states.dat')
    parser.add_argument('output', default='Paths.h5')
    args = parser.parse_args()
    
    F = np.loadtxt( args.ending ).astype(int)
    U = np.loadtxt( args.starting ).astype(int)
    flux_matrix = scipy.io.mmread( args.flux_matrix )
    # deal with case where have single start or end state
    if F.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(F)
        F = tmp.copy()
    if U.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(U)
        U = tmp.copy()
    
    arglib.die_if_path_exists(args.output)
    paths, bottlenecks, fluxes = run(flux_matrix, U, F, args.number)
    
    Serializer({'Paths': paths,
                           'Bottlenecks': bottlenecks,
                           'fluxes': fluxes}).SaveToHDF(args.output)
    print '\nSaved to %s' % args.output
