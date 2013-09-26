import abc
import re
import numpy as np
import warnings
from msmbuilder import io
try: from msmbuilder import metric_LPRMSD as lprmsd # This is down here because metric_LPRMSD imports this file, and so it is a bad recursion issue. This should be fixed by combining LP's metric into this file...
except: lprmsd = None
from msmbuilder.metrics.baseclasses import AbstractDistanceMetric, Vectorized

class ktICAPNorm(Vectorized, AbstractDistanceMetric):
    """
    This is a class for using a ktICA representation for the trajectory data. 
    """

    def __init__(self, kt_obj, num_vecs, metric='euclidean', p=2):
        """

        Parameters:
        kt_obj : msmbuilder.ktICA.ktICA instance
            this should correspond to the solution of the ktICA problem
        num_vecs : int
            number of vectors (eigenfunctions) to use in the projection
        metric : string, optional
            Should be a valid entry for the Vectorized class (see metrics.Vectorized)
        p : int, optional
             Exponent for the p-norm (if using p-norm)

        Returns:
        --------
        """

        self.kt_obj = kt_obj
        self.num_vecs = num_vecs
        self.which_vecs = np.arange(num_vecs)

        super(ktICAPNorm, self).__init__(metric=metric, p=p)

        
    def prepare_trajectory(self, trajectory):

        prep_traj = self.kt_obj.kernel.prepare_trajectory(trajectory)

        proj_traj = self.kt_obj.project(prep_traj, which=self.which_vecs)

        return proj_traj

