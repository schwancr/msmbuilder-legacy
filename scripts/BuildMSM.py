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

import os
import numpy as np
import scipy.io
from msmbuilder import arglib
import msmbuilder.io
from msmbuilder import MSMLib
import logging
logger = logging.getLogger('msmbuilder.scripts.BuildMSM')


def run(lagtime, assignments, symmetrize='MLE', input_mapping="None", out_dir="./Data/"):

    # set the filenames for output
    FnTProb = os.path.join(out_dir, "tProb.mtx")
    FnTCounts = os.path.join(out_dir, "tCounts.mtx")
    FnMap = os.path.join(out_dir, "Mapping.dat")
    FnAss = os.path.join(out_dir, "Assignments.Fixed.h5")
    FnPops = os.path.join(out_dir, "Populations.dat")

    # make sure none are taken
    outputlist = [FnTProb, FnTCounts, FnMap, FnAss, FnPops]
    arglib.die_if_path_exists(outputlist)

    # if given, apply mapping to assignments
    if input_mapping != "None":
        MSMLib.apply_mapping_to_assignments(assignments, input_mapping)

    n_assigns_before_trim = len(np.where(assignments.flatten() != -1)[0])

    counts = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=lagtime, sliding_window=True)

    rev_counts, t_matrix, populations, mapping = MSMLib.build_msm(counts, symmetrize=symmetrize, ergodic_trimming=True)

    MSMLib.apply_mapping_to_assignments(assignments, mapping)
    n_assigns_after_trim = len(np.where(assignments.flatten() != -1)[0])

    # if had input mapping, then update it
    if input_mapping != "None":
        mapping = mapping[input_mapping]

    # Print a statement showing how much data was discarded in trimming
    percent = (1.0 - float(n_assigns_after_trim) / float(n_assigns_before_trim)) * 100.0
    logger.warning("Ergodic trimming discarded: %f percent of your data", percent)

    # Save all output
    np.savetxt(FnPops, populations)
    np.savetxt(FnMap, mapping, "%d")
    scipy.io.mmwrite(str(FnTProb), t_matrix)
    scipy.io.mmwrite(str(FnTCounts), rev_counts)
    msmbuilder.io.saveh(FnAss, assignments)

    for output in outputlist:
        logger.info("Wrote: %s", output)

    return

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
                                   """Estimates the counts and transition matrices from an
                                   Assignments.h5 file. Reversible models can be calculated either from naive
                                   symmetrization or estimation of the most likely reversible matrices (MLE,
                                   recommended). Also calculates the equilibrium populations for the model
                                   produced. Outputs will be saved in the directory of your input Assignments.h5
                                   file.
                                   \nOutput: tCounts.mtx, tProb.mtx, Populations.dat,  Mapping.dat,
                                   Assignments.Fixed.h5""")
    parser.add_argument('assignments')
    parser.add_argument('symmetrize', help="""Method by which to estimate a
                        symmetric counts matrix. Symmetrization ensures reversibility, but may skew
                        dynamics. We recommend maximum likelihood estimation (MLE) when tractable,
                        else try Transpose. It is strongly recommended you read the documentation
                        surrounding this choice.""", default='MLE',
                        choices=['MLE', 'Transpose', 'None'])
    parser.add_argument('lagtime', help='''Lag time to use in model (in
        number of snapshots. EG, if you have snapshots every 200ps, and set the
        lagtime=50, you'll get a model with a lagtime of 10ns)''', type=int)
    parser.add_argument('mapping', help='''Mapping, EG from microstates to macrostates. If given, this mapping will be applied to the specified assignments before creating an MSM.''', default="None")
    parser.add_argument('output_dir')
    args = parser.parse_args()

    try:
        assignments = msmbuilder.io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = msmbuilder.io.loadh(args.assignments, 'Data')

    if args.mapping != "None":
        args.mapping = np.array(np.loadtxt(args.mapping), dtype=int)

    run(args.lagtime, assignments, args.symmetrize, args.mapping, args.output_dir)
