import os
import numpy as np
from msmbuilder import io
from msmbuilder import arglib
from msmbuilder import Project
from msmbuilder import MSMLib
from msmbuilder.clustering import concatenate_trajectories
import logging
logger = logging.getLogger('msmbuilder.scripts.SaveStructures')
DEBUG = True


def run(project, assignments_list, states, n_per_state, random=None, replacement=True):
    """Extract random conformations from states

    Parameters
    ----------
    project : msmbuilder.project
    assignments_list : list
        list of assignments. Each assignments array should be an np.ndarray
    states : array_like
        The indices of the states to pull from
    n_per_state : int
        Number of conformations to extract per state
    random : np.random.RandomState, optional
        Source of randomness

    Returns
    -------
    confs : [msmbuilder.Trajectory]
        List of trajectories, each of length n_per_state. confs[i][j] is
        the `j`th conformation sampled from state `states[i]`.
    """

    if random is None:
        random = np.random

    results = []
    ind_inv_list = []
    # get a mapping from microstate -> trj/frame

    n_traj_list = [ a.shape[0] for a in assignments_list ]

    for assignments in assignments_list:
        ind_inv = MSMLib.invert_assignments(assignments)
        ind_inv_list.append(ind_inv)

    uniq_states = np.unique(np.concatenate([d.keys() for d in ind_inv_list]))

    for s in states:
        trajs, frames = ind_inv_list[0][s]
        for i in xrange(1, len(ind_inv_list)):
            temp_trajs, temp_frames = ind_inv_list[i][s]
            temp_trajs += np.sum(n_traj_list[:i])
        
            trajs = np.concatenate((trajs, temp_trajs))
            frames = np.concatenate((frames, temp_frames))
    
        if len(trajs) != len(frames):
            raise RuntimeError('inverse assignments corrupted?')

        if replacement:
            if len(trajs) < n_per_state:
                logger.error("Asked for %d confs per state, but state %d only has %d" % (n_per_state, s, len(trajs)))
            # indices of the confs to select
            r = random.randint(len(trajs), size=n_per_state)
        else:
            if len(trajs) < n_per_state:
                raise ValueError("Asked for %d confs per state, but state %d only has %d" % (n_per_state, s, len(trajs)))
            # draw n_per_state random numbers between `0` and `len(trajs)` without replacement
            r = random.permutation(len(trajs))[:n_per_state]


        results.append(project.load_frame(trajs[r], frames[r]))

    return results


def save(confs_by_state, states, style, format, outdir):
    "Save the results to disk"

    if style == 'sep':
        for i, trj in enumerate(confs_by_state):
            for j in xrange(len(trj)):

                fn = os.path.join(outdir, 'State%d-%d.%s' % (states[i], j,
                    format))
                arglib.die_if_path_exists(fn)

                logger.info("Saving file: %s" % fn)
                trj[j].save(fn)

    elif style == 'tps':
        for i, trj in enumerate(confs_by_state):
            fn = os.path.join(outdir, 'State%d.%s' % (states[i], format))
            arglib.die_if_path_exists(fn)

            logger.info("Saving file: %s" % fn)
            trj.save(fn)

    elif style == 'one':
        fn = os.path.join(outdir, 'Confs.%s' % format)
        arglib.die_if_path_exists(fn)

        logger.info("Saving file: %s" % fn)
        concatenate_trajectories(confs_by_state).save(fn)

    else:
        raise ValueError('Invalid style: %s' % style)


def main():
    """Parse command line inputs, load up files, then call run() and save() to do
    the real work"""

    parser = arglib.ArgumentParser(description="""
Yank a number of randomly selected conformations from each state in a model.

The conformations can either be saved in separate files (i.e. one PDB file per
conformations), or in the same file.
""")
    parser.add_argument('project')
    parser.add_argument('assignments', default=['Data/Assignments.Fixed.h5'], nargs='+')
    parser.add_argument('conformations_per_state', default=5, type=int,
        help='Number of conformations to sample from each state')
    parser.add_argument('states', nargs='+', type=int,
        help='''Which states to sample from. Pass a list of integers, separated
        by whitespace. To specify ALL of the states, include the integer -1.''',
        default=[-1])
    # nargs=+ means the values will come out as type list, but this isn't
    # naturally applied to the default, so we just put the default as [-1]
    parser.add_argument('format', choices=['pdb', 'xtc', 'lh5'],
        help='''Format for the outputted conformations. PDB is the standard
        plaintext protein databank format. XTC is the gromacs binary trajectory
        format, and lh5 is the MSMBuilder standard hdf5 based format''',
        default='pdb')
    parser.add_argument('style', choices=['sep', 'tps', 'one'], help='''Controls
        the number of conformations save per file. If "sep" (SEPARATE), all of
        the conformations will be saved in separate files, named in the format
        State{i}-{j}.{ext} for the `j`th conformation sampled from the `i`th
        state. If "tps" (Trajectory Per State), each of the conformations
        sampled from a given state `i` will be saved in a single file, named
        State{i}.{ext}. If "one", all of the conformations will be saved in a
        single file, such that the `j`th conformation from the `states[i]`-th
        microstate will be the `i+j*N`th frame in the trajectory file. The file
        will be namaed Confs.{ext}
        ''', default='sep')
    parser.add_argument('replacement', type=bool, help='''Draw random stuctures from
        those assigned to a state either with replacement or without. If you ask for k
        structures from a state with < k assignments and are drawing with --replacement False,
        you will get an error.''', default=True)

    parser.add_argument('output_dir', default='PDBs')
    args = parser.parse_args()

    # load...
    # project
    project = Project.load_from(args.project)

    # assignments
    assignments_list = []
    for fn in args.assignments:
        try:
            assignments = io.loadh(fn, 'arr_0')
        except KeyError:
            assignments = io.loadh(fn, 'Data')
        assignments_list.append(assignments)

    # states
    if -1 in args.states:
        n_states = len(np.unique(np.concatenate(
            [assignments[np.where(assignments != -1)] 
            for assignments in assignments_list])))
        logger.info('Yanking from all %d states', n_states)
        states = np.arange(n_states)
    else:
        # ensure that the states are sorted, and that they're unique -- you
        # can only request each state once
        states = np.unique(args.states)
        logger.info("Yanking from the following states: %s", states)


    # extract the conformations using np.random for the randomness
    confs_by_state = run(project=project, assignments_list=assignments_list,
        states=states, n_per_state=args.conformations_per_state,
        random=np.random, replacement=args.replacement)

    # save the conformations to disk, in the requested style
    save(confs_by_state=confs_by_state, states=states, style=args.style,
         format=args.format, outdir=args.output_dir)

if __name__ == '__main__':
    main()
