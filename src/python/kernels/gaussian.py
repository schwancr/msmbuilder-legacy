
from msmbuilder.kernels.baseclasses import AbstractKernel
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np

class Gaussian(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass. Note that only Vectorized metrics will work here.

    Also, if you plan on using a kernel trick, this really isn't very 
    useful, as it is the same as working in the vector space defined by
    your metric.
    """

    def __init__(self, metric, std_dev=1):

        self.metric = metric
        self.std_dev = std_dev
        self.denom = - 2. * std_dev * std_dev

    def __repr__(self):
        return "Gaussian kernel with norm defined by %s" % str(self.metric)

    def prepare_trajectory(self, trajectory):
        return self.metric.prepare_trajectory(trajectory)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return np.exp(np.square(self.metric.one_to_all(prepared_traj1, prepared_traj2, index1)) / self.denom)

