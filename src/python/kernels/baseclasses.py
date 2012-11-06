import abs
import numpy as np
import warnings

class AbstractKernel(object):
    """Abstract class for all new kernel functions"""

    @abc.abstractmethod
    def prepare_trajectory(self, trajectory):
        return

    @abc.abstractmethod
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return

    @abc.abstractmethod
    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        return

    @abc.abstractmethod
    def all_pairwise(self, prepared_traj):
        traj_length = len(prepared_traj)
        output = -1 * np.ones(traj_length * (traj_length - 1) / 2)
        p = 0
        for i in xrange(traj_length):
            cmp_indices = np.arange(i + 1, traj_length)
            output[p: p + len(cmp_indices)] = self.one_to_many(prepared_traj, prepared_traj, i, cmp_indices)
            p += len(cmp_indices)

        return output
