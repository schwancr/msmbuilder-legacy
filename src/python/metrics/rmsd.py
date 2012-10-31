import warnings
import numpy as np
from collections import namedtuple
from baseclasses import AbstractDistanceMetric
from msmbuilder import _rmsdcalc


class RMSD(AbstractDistanceMetric):
    """
    Compute distance between frames using the Room Mean Square Deviation
    over a specifiable set of atoms using the Theobald QCP algorithm

    References
    ----------
    .. [1] Theobald, D. L. Acta. Crystallogr., Sect. A 2005, 61, 478-480.

    """

    class TheoData(object):
        """Stores temporary data required during Theobald RMSD calculation.

        Notes:
        Storing temporary data allows us to avoid re-calculating the G-Values
        repeatedly. Also avoids re-centering the coordinates."""

        theo_slice = namedtuple('theo_slice', ('xyz', 'G'))
        # I'm not sure what "G" is yet so I don't know how to rename it
        def __init__(self, xyz_data, num_atoms=None, G=None):
            """Create a container for intermediate values during 
            RMSD Calculation.

            Notes:
            1.  We remove center of mass.
            2.  We pre-calculate matrix magnitudes (ConfG)"""

            if num_atoms is None or G is None:
                num_confs = len(xyz_data)
                num_atoms = xyz_data.shape[1]

                self.center_conformations(xyz_data)

                num_atoms_with_padding = 4 + num_atoms - num_atoms % 4

                # Load data and generators into aligned arrays
                xyz_data2 = np.zeros((num_confs, 3, num_atoms_with_padding), 
                                    dtype=np.float32)
                for i in range(num_confs):
                    xyz_data2[i, 0:3, 0:num_atoms] = xyz_data[i].transpose()

                #Precalculate matrix magnitudes
                conf_G = np.zeros((num_confs,), dtype=np.float32)
                for i in xrange(num_confs):
                    conf_G[i] = self.calc_G_value(xyz_data[i, :, :])

                self.xyz_data = xyz_data2
                self.G = conf_G
                self.num_atoms = num_atoms
                self.num_atoms_with_padding = num_atoms_with_padding
                self.check_centered()
            else:
                self.xyz_data = xyz_data
                self.G = G
                self.num_atoms = num_atoms
                self.num_atoms_with_padding = xyz_data.shape[2]

        def __getitem__(self, key):
            # to keep the dimensions right, we make everything a slice
            if isinstance(key, int):
                key = slice(key, key+1)
            return RMSD.TheoData(self.xyz_data[key], num_atoms=self.num_atoms,
                                 G=self.G[key])

        def __setitem__(self, key, value):
            self.xyz_data[key] = value.xyz_data
            self.G[key] = value.G

        def check_centered(self, epsilon=1E-5):
            """Raise an exception if XYZAtomMajor has nonnzero center of 
            mass(CM)."""

            xyz = self.xyz_data.transpose(0, 2, 1)
            x = np.array([ max(abs(xyz[i].mean(0))) for i in 
                          xrange(len(xyz)) ]).max()
            if x > epsilon:
                raise Exception("The coordinate data does not appear to have "
                                "been centered correctly.")

        @staticmethod
        def center_conformations(xyz_list):
            """Remove the center of mass from conformations.  Inplace to minimize mem. use."""

            for ci in xrange(xyz_list.shape[0]):
                x = xyz_list[ci].astype('float64')  # To improve the accuracy of RMSD, it can help to do certain calculations in double precision.
                x -= x.mean(0)
                xyz_list[ci] = x.astype('float32')
            return

        @staticmethod
        def calc_G_value(xyz):
            """Calculate the sum of squares of the key matrix G.  
            A necessary component of Theobold RMSD algorithm."""

            conf = xyz.astype('float64')  # Doing this operation in double significantly improves numerical precision of RMSD
            G = 0
            G += np.dot(conf[:, 0], conf[:, 0])
            G += np.dot(conf[:, 1], conf[:, 1])
            G += np.dot(conf[:, 2], conf[:, 2])
            return G

        def __len__(self):
            return len(self.xyz_data)

    def __init__(self, atom_indices=None, omp_parallel=True):
        """Initalize an RMSD calculator

        Parameters
        ----------
        atomindices : array_like, optional
            List of the indices of the atoms that you want to use for the RMSD
            calculation. For example, if your trajectory contains the coordinates
            of all the atoms, but you only want to compute the RMSD on the C-alpha
            atoms, then you can supply a reduced set of atom_indices. If unsupplied,
            all of the atoms will be used.
        omp_parallel : bool, optional
            Use OpenMP parallelized C code under the hood to take advantage of
            multicore architectures. If you're using another parallelization scheme
            (e.g. MPI), you might consider turning off this flag.

        Notes
        -----
        You can also control the degree of parallelism with the OMP_NUM_THREADS
        envirnoment variable


        """
        self.atom_indices = atom_indices
        self.omp_parallel = omp_parallel

    def __repr__(self):
        try:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % \
                   (repr(list(self.atom_indices)), self.omp_parallel)
        except:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % \
                   (self.atom_indices, self.omp_parallel)
        return val

    def prepare_trajectory(self, trajectory):
        """Prepare the trajectory for RMSD calculation.

        Preprocessing includes extracting the relevant atoms, centering the
        frames, and computing the G matrix.


        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            Molecular dynamics trajectory

        Returns
        -------
        theodata : array_like
            A msmbuilder.metrics.TheoData object, which contains some preprocessed
            calculations for the RMSD calculation
        """

        if self.atom_indices is not None:
            return self.TheoData(trajectory['XYZList'][:, self.atom_indices])

        return self.TheoData(trajectory['XYZList'])

    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`

        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
        Vector of distances of length len(indices2)

        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster. Using omp_parallel = False is
        advised if indices2 is a short list and you are paralellizing your
        algorithm (say via mpi) at a different
        level.
        """

        if isinstance(indices2, list):
            indices2 = np.array(indices2)
        if not isinstance(prepared_traj1, RMSD.TheoData):
            raise TypeError('Theodata required')
        if not isinstance(prepared_traj2, RMSD.TheoData):
            raise TypeError('Theodata required')

        if self.omp_parallel:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices(
                      prepared_traj1.num_atoms, prepared_traj1.num_atoms_with_padding,
                      prepared_traj1.num_atoms_with_padding, prepared_traj2.xyz_data,
                      prepared_traj1.xyz_data[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)
        else:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices_serial(
                      prepared_traj1.num_atoms, prepared_traj1.num_atoms_with_padding,
                      prepared_traj1.num_atoms_with_padding, prepared_traj2.xyz_data,
                      prepared_traj1.xyz_data[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate a vector of distances from one frame of the first trajectory
        to all of the frames in the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2`

        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`

        Returns
        -------
        Vector of distances of length len(prepared_traj2)

        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster.
        """

        if self.omp_parallel:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g(
                prepared_traj1.num_atoms, prepared_traj1.num_atoms_with_padding,
                prepared_traj1.num_atoms_with_padding, prepared_traj2.xyz_data,
                prepared_traj1.xyz_data[index1], prepared_traj2.G,
                prepared_traj1.G[index1])
        else:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_serial(
                    prepared_traj1.num_atoms, prepared_traj1.num_atoms_with_padding,
                    prepared_traj1.num_atoms_with_padding, prepared_traj2.xyz_data,
                    prepared_traj1.xyz_data[index1], prepared_traj2.G,
                    prepared_traj1.G[index1])

    def _square_all_pairwise(self, prepared_traj):
        """Reference implementation of all_pairwise"""
        warnings.warn('This is HORRIBLY inefficient. This operation really '
                      'needs to be done directly in C')
        traj_length = prepared_traj.xyz_data.shape[0]
        output = np.empty((traj_length, traj_length))
        for i in xrange(traj_length):
            output[i] = self.one_to_all(prepared_traj, prepared_traj, i)
        return output
