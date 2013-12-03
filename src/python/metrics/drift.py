"""
To USE this code, from a client perspective, all you want to do is

>> from metrics import RMSD, Dihedral, Contact

Nothing else in this modules' namespace will be useful to you as a client.

and then for example

>> rmsdtraj1 = RMSD.prepare_trajectory(traj, atomindices)
>> RMSD.one_to_all(rmsdtraj1, rmsdtraj1, 0)
>> dihedraltraj1 = Dihedral.prepare_trajectory(traj)
>> Dihedral.one_to_all(dihedraltraj1, dihedraltraj1, 0)

this would compute the distances from frame 0 to all other frames under both
the rmsd metric and dihedral metric. There are a lot more options and ways you can
calcuate distances (euclidean distance vs. cityblock vs pnorm, etc etc.) and select
the frames you care about (one_to_all(), one_to_many(), many_to_many(), all_to_all(), etc).

NOTE: Because the code leverages inheritance, if you just casually browse the code
for Dihedral for example, you ARE NOT going to see all methods that the class
actually implements. I would browsing the docstrings in ipython.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=

To DEVELOP on top of this code, just implement some classes that inherit from
either AbstractDistanceMetric or Vectorized. In particular, if you inherit
from Vectorized, you get a fully functional DistanceMetric class almost for
free. All you have to do is define a prepare_trajectory() fuction.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This should be an (almost) drop in replacement for
msmbuilder's DistanceMetic.py, but I think it is much
easier to extend.

Changes (combared to msmbuilder.DistanceMetric):
(1) Moved all the RMSD specific code in DistanceMetric.py
    into the RMSD class. (TheoData stuff). I also
    eliminated the TheoData getters and setters because this is python and
    everything is public / we're all consenting adults.
(2) All distance metrics now inherit from the abstract base
    class AbstractDistanceMetric, and define (at minimum) the
    methods prepare_trajectory(), one_to_many() and one_to_all()
(3) Dihedral and Contact implement the same interface
    as RMSD
(4) Dihedral and Contact share code by subclassing
    Vectorized, which lets you use any distance metric on vectors
    (euclidean, manhattan, chebychev, pnorm, etc) that you like. All
    that Dihedral and Contact have to actually implement
    themselves is their prepare_trajectory function which processes the
    trajectory into a bunch of vectors.
(5) So, anyone can really easily make a new Vectorized. All you have to
    do is put the code that prepares the vectors into a function called
    prepare_trajectory and put that into a class that subclasses Vectorized.
    Furthermore, if you would like to set the default metric used to not be
    euclidean (for instance in Contact since the prepared_trajectories are
    boolean vectors, you dont want euclidean to be the default), you set the
    the class variable 'default_scipy_metric' to be whatever you like.
    Thats all there is to it!
    
    
#=#=#=#+#+#+#+#+#+#+#+#+#+#+#+#+#
ALSO:
    This should be documented better somewhere, because it will cause cryptic
    errors if you don't do it. Whatever data structure you return from prepare_trajectory()
    needs to support slice sytax. If you return an array or something, then this is
    no problem, but if you create your own object to hold the data that prepare_trajectory()
    returns, you need to add a __getitem__(), __setitem__() and __len__() methods. See the
    RMSD.TheoData object below. Also, if you're not familiar with this side of python,
    these docs (http://docs.python.org/reference/datamodel.html#emulating-container-types)
    are pretty good. Only __getitem__, __setitem__ and __len__ are necessary.
#=#=#=#+#+#+#+#+#+#+#+#+#+#+#+#+#
"""


import abc
import re
import copy
import numpy as np
import warnings
from msmbuilder.metrics import AbstractDistanceMetric

def get_epsilon_neighborhoods(metric,ptraj,tau):

    output = []
    N = len(ptraj)
    for i in xrange(N):
        if i < tau:
            output.append( metric.one_to_many(ptraj,ptraj,i,[i+tau])[0] )
        elif i >= N-tau:
            output.append( metric.one_to_many(ptraj,ptraj,i,[i-tau])[0] )
        else:
            output.append( metric.one_to_many(ptraj,ptraj,i,[i-tau,i+tau]).max() )

    return np.array( output )

class DriftMetric(AbstractDistanceMetric):
    """This class is used for normalizing another metric by its drift at some time. NOTE: THIS IS
    NOT MATHEMATICALLY A METRIC BECAUSE IT DOES NOT NECESSARILY SATISFY THE TRIANGLE INEQUALITY. It
    is however, a similarity criterion, and this problem will likely not cause any issues in the 
    clustering process."""

    class DriftTrajectory(dict):

        def __init__( self, trajectory, DriftMetricInst, lengths=None, epsilons=None):
            """
            Inputs:
            1) msmbuilder.Trajectory.Trajectory object
            2) schwancrtools.metrics_Drift.DriftMetric instance
            3) lengths [ None ] - an array of lengths if the trajectory is actually a concatenation of many trajectories
            4) epsilons [ None ] - If you already have the epsilons, or cannot calculate them from the given trajectory, then pass them here
            
            Outputs:
            1) Get a dictionary with keys: 'base_ptraj' and 'epsilons' corresponding to the prepared trajectory and its epsilons
            """
            def chop_object( obj, lengths ):
                if len(obj) != np.sum(lengths): raise Exception('Lengths do not correspond to this data.')
                sum = 0
                out = []
                for i in xrange( len(lengths) ):
                    out.append( obj[ sum : sum + lengths[i] ] )
                return out
            tempDict = {}
            tempDict['base_ptraj'] = copy.copy( DriftMetricInst.base_metric.prepare_trajectory(trajectory) )
            if epsilons==None:
                if lengths==None:
                    tempDict['epsilons'] = copy.copy( get_epsilon_neighborhoods( DriftMetricInst.base_metric, tempDict['base_ptraj'], DriftMetricInst.tau ) )
                else:
                    if isinstance( trajectory, np.ndarray ):
                        tempDict['epsilons'] = np.concatenate([ get_epsilon_neighborhoods( DriftMetricInst.base_metric, segment, DriftMetricInst.tau )
                                                         for segment in chop_object( tempDict['base_ptraj'], lengths ) ])
                    else:
                        tempDict['epsilons'] = np.concatenate([ get_epsilon_neighborhoods( DriftMetricInst.base_metric,
                                                         self.base_metric.prepare_trajectory( segment ), DriftMetricInst.tau ) for segment
                                                         in chop_object( trajectory, lengths ) ])
            else:
                tempDict['epsilons'] = copy.copy( epsilons )
                if len(epsilons) != len( trajectory ):
                    raise Exception( "Input epsilons (%d) are not the same length as the trajectory (%d)" % ( len(epsilons), len(trajectory)) )
            super( DriftMetric.DriftTrajectory, self).__init__(tempDict)
            #self = copy.copy( tempDict )

        def __len__( self ):
            return len( self['base_ptraj'] )

        def __getitem__(self, key):
            if isinstance(key, int) or isinstance(key, slice) or isinstance(key,np.ndarray):
                if isinstance(key, int):
                    key = [key]
                newTraj = copy.copy(self)
                newTraj['base_ptraj'] = newTraj['base_ptraj'][key]
                newTraj['epsilons'] = newTraj['epsilons'][key]
                return newTraj
            else:
                return super(DriftMetric.DriftTrajectory,self).__getitem__(key)

    def __init__(self, base_metric, tau):
        """Initialize the metric with its base metric and tau to use to calculate the drifts."""
        self.tau = int(tau)
        self.base_metric = base_metric

    def prepare_trajectory(self, trajectory, lengths=None):
        if isinstance( trajectory, tuple ):
            return self.DriftTrajectory( trajectory[0], self, lengths=lengths, epsilons=trajectory[1] )
        else:
            return self.DriftTrajectory( trajectory, self, lengths=lengths )

    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Compute the distance from prepared_traj1[index1] to each of the indices2
        frames of prepared_traj2"""
        
        distances = self.base_metric.one_to_many( prepared_traj1['base_ptraj'], prepared_traj2['base_ptraj'], index1, indices2 )
 
        eps1 = prepared_traj1['epsilons'][ index1 ]
        eps2 = prepared_traj2['epsilons'][ indices2 ]
        if eps2.shape == ():
            eps2 = [ eps2 ]
        min_eps = np.array([ min(eps1,eps2[i]) for i in xrange( len(eps2) ) ])

        return distances / min_eps

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Compute the distance from prepared_traj1[index1] to each of the frames in
        prepared_traj2"""
        
        distances = self.base_metric.one_to_all( prepared_traj1['base_ptraj'], prepared_traj2['base_ptraj'], index1 )

        eps1 = prepared_traj1['epsilons'][ index1 ]
        eps2 = prepared_traj2['epsilons']

        min_eps = np.array([ min(eps1,eps2[i]) for i in xrange( len(eps2) ) ])
        
        return distances / min_eps

    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances"""
        
        traj_length = len(prepared_traj['base_ptraj'])
        output = -1 * np.ones(traj_length * (traj_length - 1) / 2)
        p = 0
        for i in range(traj_length):
            cmp_indices = np.arange(i + 1, traj_length)
            output[p: p + len(cmp_indices)] = self.one_to_many(prepared_traj, prepared_traj, i, cmp_indices)
            p += len(cmp_indices)
        return output
