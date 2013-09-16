import os, sys
import numpy as np
from msmbuilder.testing import *
from msmbuilder import Trajectory

def test_traj_0():
    
    aind = np.unique( np.random.randint( 22, size=4) )
    stride = np.random.randint(1, 100 )
    
    r_traj = get('Trajectories/trj0.lh5')

    r_traj.restrict_atom_indices( aind )

    r_traj['XYZList'] = r_traj['XYZList'][ ::stride ]

    traj = Trajectory.load_from_lhdf(get('Trajectories/trj0.lh5', just_filename=True),
        Stride=stride, AtomIndices=aind)

    # make sure we loaded the right number of atoms
    assert traj['XYZList'].shape[1] == len(aind)

    for key in traj.keys():
        if key in ['SerializerFilename'] :
            continue
        
        if key in ['IndexList']:
            for row, r_row in zip( traj[key], r_traj[key] ):
                eq(row, r_row)
        elif key == 'XYZList':
            eq(traj[key], r_traj[key])
        else:
            eq(traj[key], r_traj[key])

def test_traj_1():
    for i in range(20):
        test_traj_0()
        
def test_xtc_dcd():
    pdb_filename = get("native.pdb", just_filename=True)
    xtc_filename = get('RUN00_frame0.xtc', just_filename=True)
    dcd_filename = get('RUN00_frame0.dcd', just_filename=True)
    r_xtc = Trajectory.load_from_xtc(xtc_filename, pdb_filename)
    r_dcd = Trajectory.load_from_dcd(dcd_filename, pdb_filename)

    x_xtc = r_xtc["XYZList"]
    x_dcd = r_dcd["XYZList"]

    eq(x_xtc, x_dcd, decimal=4)
