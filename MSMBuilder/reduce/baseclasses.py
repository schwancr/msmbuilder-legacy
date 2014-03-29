import abc
from mdtraj.utils.six import with_metaclass


class AbstractDimReduction(with_metaclass(abc.ABCMeta, object)):

    """
    abstract class for defining dimensionality reduction
    techniques.

    Any subclass of this class can be used with the RedDimPNorm
    distance metric
    """

    @abc.abstractmethod
    def project(self, trajectory=None, prep_trajectory=None, which=None):
        """
        this method should take at least three kwargs:

        Parameters:
        -----------
        trajectory: mdtraj.Trajectory instance, optional
        prep_trajectory: prepared msmbuilder.Trajectory instance, optional
            prepared trajectory
        which: np.ndarray
            which vectors to use to project onto

        Returns:
        --------
        proj_trajectory: np.ndarray
            projected trajectory (n_frames, len(which))
        """

        return
