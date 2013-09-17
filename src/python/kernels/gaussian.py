
from msmbuilder.kernels.baseclasses import AbstractKernel
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np

class Gaussian(AbstractKernel):
    """
    This kernel is a gaussian kernel, whose induced vector space is 
    infinite dimensional:

    k(x, y) = exp[ - d(x, y)^2 / 2. / sigma^2 ]

    The only parameter is sigma which is the width of the gaussian.
    Any distance metric can be used, but if you use RMSD, this kernel
    may not be a reproducing kernel, which would mean the ktICA solutions
    could get screwy...
    """

    def __init__(self, metric, std_dev=1):

        self.metric = metric
        self.std_dev = std_dev
        self.denom = - 2. * std_dev * std_dev

    def __repr__(self):
        return "Gaussian kernel with norm defined by %s" % str(self.metric)

    def __call__(self, distance):
        return np.exp(np.square(distance) / self.denom)

    def prepare_trajectory(self, trajectory):
        return np.double(self.metric.prepare_trajectory(trajectory))

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return self.__call__(self.metric.one_to_all(prepared_traj1, prepared_traj2, index1))

