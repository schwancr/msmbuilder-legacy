
import numpy as np
import scipy.linalg
from msmbuilder import io
import pickle
from msmbuilder.kernels import AbstractKernel

class ktICA(object):
    """ 
    class for calculating tICs in a high dimensional feature space
    """

    def __init__(self, kernel, dt, reg_factor=1E-10, n_components=100):

        """
        Initialize an instance of the ktICA solver

        Paramaters
        ----------
        kernel : subclass of kernels.AbstractKernel
            instance of a subclass of kernels.AbstractKernel that defines
            the kernel function for kernel-tICA

        dt : int
            correlation lagtime to compute the tICs for

        reg_factor : float, optional
            regularization parameter. This class will use a ridge regression
            when solving the generalized eigenvalue problem

        n_components : int, optional
            number of components to calculate. Since the number of 
            eigenfunctions scales with the amount of data, it makes more
            sense to only calculating the <n_components> that are the 
            slowest
        """

        if not isinstance(kernel, AbstractKernel):
            raise Exception("kernel must be an instance of subclass of AbstractKernel")
        self.kernel = kernel

        self._Xa = None
        self._Xb = None
        
        self.K = None
        self.K_uncentered = None

        self.reg_factor = float(reg_factor)

        self.dt = int(dt)

        self.n_components = int(n_components)

        self._normalized = False
        self.acf_vals = None
        self.vec_vars = None


    def _center_K(self):

        if self.K is None:
            self.calculate_matrices()

        if self.K_uncentered is None:
            self.K_uncentered = np.array(K)

        N = self.K.shape[0]
        # now normalize the matrices.
        one_N = np.ones((N, N)) / float(N)

        self.K = self.K - one_N.dot(self.K) - self.K.dot(one_N) + one_N.dot(self.K).dot(one_N)

        self.K = (self.K + self.K.T) * 0.5 # for numerical issues, just make it symmetric


    def add_traj(self, trajectory, trajectory_dt=None, prepped=True):
        """
        append a trajectory to the calculation. Right now this just appends 
        the trajectory to the concatenated trajectories

        Parameters
        ----------
        trajectory : np.ndarray (2D)
            two dimensional np.ndarray with time in the first axis and 
            features in the second axis
        trajectory_dt : np.ndarray (2D), optional
            for each point we need a corresponding point that is separated
            in a trajectory by dt. If trajectory_dt is not None, then it should
            be the same length as trajectory such that trajectory[i] comes 
            exactly dt before trajectory_dt[i]. If trajectory_dt is None, then
            we will get all possible pairs from trajectory (trajectory[:-dt, 
            trajectory[dt:]).
        prepped : 
        
        """

        if trajectory_dt is None:
            A = trajectory[:-self.dt]
            B = trajectory[self.dt:]

        else:
            if trajectory_dt.shape != trajectory.shape:
                raise Exception("trajectory and trajectory_dt should be same shape!")

            A = trajectory
            B = trajectory_dt

        if self._Xa is None:
            self._Xa = A
            self._Xb = B
        else:
            self._Xa = np.concatenate((self._Xa, A)) 
            self._Xb = np.concatenate((self._Xb, B))  


    def use_matrix(self, dist_mat, ptrajA, ptrajB):
        """
        Add a single distance matrix which has been prepared correctly.
        You should only use this if you actually know what you are doing.
        
        Parameters
        ----------
        dist_mat : np.ndarray (2 N, 2 N)
            distance matrix corresponding to the distance between conformation
            i and conformation j in the concatenation of ptrajA and ptrajB
        ptrajA : np.ndarray
            prepared trajectory
        ptrajB : np.ndarray
            prepared trajectory such that ptrajB[i] is found <dt> frames after
            ptrajA[i] in a trajectory.

        """

        N = len(ptrajA)

        if N != len(ptrajB):
            raise Exception("ptrajA and ptrajB must be the same length.")

        if len(dist_mat.shape) != 2:
            raise Exception("dist_mat must be two-dimensional")

        if ((2 * N) != dist_mat.shape[0]) or ((2 * N) != dist_mat.shape[1]):
            raise Exception("dist_mat must be square and the same size as the ptraj's")

        self.K = self.kernel(dist_mat)
        self.K_uncentered = np.array(self.K)

        self._Xall = np.concatenate([ptrajA, ptrajB])
        self._Xa = ptrajA
        self._Xb = ptrajB

        self._center_K()


    def calculate_matrices(self):
        """
        calculate the two matrices we need, K and Khat and then normalize them
        """

        N = len(self._Xa) * 2

        self.K = np.zeros((N, N))

        self._Xall = np.concatenate((self._Xa, self._Xb))

        for i in xrange(N):
            self.K[i] = self.kernel.one_to_all(self._Xall, self._Xall, i)
        
        self.K = K

        self._center_K()


    def solve(self):
        """
        solve the generalized eigenvalue problem for kernel-tICA:
    
        K R K b = w K K b

        Returns
        -------
        eigenvalues : np.ndarray
            eigenvalues from eigensolution
        eigenvectors : np.ndarray
            eigenvectors are stored in the columns
        """

        if self.K is None:
            self.calculate_matrices()

        N = self.K.shape[0] / 2

        rot_mat = np.zeros((2 * N, 2 * N))
        rot_mat[:N, N:] = np.eye(N)
        rot_mat += rot_mat.T

        lhs = self.K.dot(rot_mat).dot(self.K)
        rhs = self.K.dot(self.K) + np.eye(self.K.shape[0]) * self.reg_factor

#        self.eigen_sol = scipy.linalg.eig(lhs, b=rhs)
        self.eigen_sol = scipy.sparse.linalg.eigsh(lhs, M=rhs, k=self.n_components, which='LR')
        # use the sparse implementation which uses ARPACK. Ideally this will
        # be replaced by a better package, but this is still faster (by a lot)
        # than getting all eigenvectors with scipy.linalg.eigs

        self._normalize()

        self._sort()

        return self.eigen_sol
    

    def _sort(self):
        """
        sort eigenvectors / eigenvalues so they are decreasing
        """
        if self.eigen_sol is None:
            logger.warn("have not calculated eigenvectors yet...")
            return

        if not self._normalized:
            self._normalize()

        ind = np.argsort(self.eigen_sol[0])[::-1]

        self.eigen_sol = (self.eigen_sol[0][ind], self.eigen_sol[1][:, ind])

        vecs = self.eigen_sol[1]
        term2 = self.reg_factor * np.square(vecs).sum(axis=0) / vecs.shape[0]

        self.acf_vals = self.eigen_sol[0].real * (1 + term2.real)


    def _normalize(self):
        """
        normalize the eigenvectors to unit variance.

        Note: This is the same normalization as the right eigenvectors
            of the transfer operator under the assumption that self._Xall 
            is distributed according to the true equlibrium populations.
        """

        KK = self.K.dot(self.K)

        M = float(self.K.shape[0])

        vKK = self.eigen_sol[1].T.dot(KK)

        self.vec_vars = np.sum(vKK * self.eigen_sol[1].T, axis=1) / M
        # dividing by M instead of M - 1. Shouldn't really matter...

        norm_vecs = self.eigen_sol[1] / np.sqrt(self.vec_vars)

        self.eigen_sol = (self.eigen_sol[0], norm_vecs)

        self._normalized = True


    def project(self, trajectory, which):
        """
        project a point onto an eigenvector

        Parameters
        ----------
        trajectory : np.ndarray
            trajectory to project onto eigenvector
        which : list or int
            which eigenvector(s) (0-indexed) to project onto
        
        Returns
        -------
        proj_trajectory : np.ndarray
            projected value of each point in the trajectory
        """

        if isinstance(which, int):
            which = [which]

        which = np.array(which)

        if which.max() >= self.n_components:
            raise RuntimeError("cannot project onto more components than we've calculated")

        if not self._normalized:
            self._normalize()

        comp_to_all = []

        for i in xrange(len(self._Xall)):
            comp_to_all.append(self.kernel.one_to_all(self._Xall, trajectory, i))
        
        comp_to_all = np.array(comp_to_all).T
        # rows are points from trajectory
        # cols are comparisons to the library points

        M = self.K_uncentered.shape[0]

        comp_to_all = comp_to_all - np.reshape(comp_to_all.sum(axis=1), (-1, 1)) / float(M) \
                        - self.K_uncentered.sum(axis=0) / float(M) \
                        + self.K_uncentered.sum() / float(M) / float(M)

        vecs = self.eigen_sol[1][:, which]

        proj_trajectory = comp_to_all.dot(vecs)

        return proj_trajectory


    def log_likelihood(self, equilA, equilB, trajA=None, trajB=None, projA=None,
        projB=None, num_vecs=10, timestep=1):

        if not trajA is None and not trajB is None:
            if np.unique([len(ary) for ary in [trajA, trajB, equilA, equilB]]).shape[0] != 1:
                raise Exception("trajA, trajB, equilA, and equilB should all be the same length.")
        else:
            if np.unique([len(ary) for ary in [projA, projB, equilA, equilB]]).shape[0] != 1:
                raise Exception("trajA, trajB, equilA, and equilB should all be the same length.")

        if timestep < self.dt:
            raise Exception("can't model dynamics less than original dt.")

        elif timestep == self.dt:
            exponent = 1

        else:
            if (timestep % self.dt):
                raise Exception("for timestep > dt, timestep must be a multiple of dt.")

            exponent = timestep / self.dt

        if projA is None:
            projA = self.project(trajA, which=np.arange(num_vecs))
    
        if projB is None:
            projB = self.project(trajB, which=np.arange(num_vecs))

        N = projA.shape[0]
        projA = np.hstack([np.ones((N, 1)), projA])
        projB = np.hstack([np.ones((N, 1)), projB])
        vals = np.concatenate([[1], self.eigen_sol[0][:num_vecs]]).real
        vals = np.power(vals, exponent)
        vals = np.reshape(vals, (-1, 1))
        if len(equilA.shape) == 1:
            equilA = np.reshape(equilA, (-1, 1))
        if len(equilB.shape) == 1:
            equilB = np.reshape(equilB, (-1, 1))
        temp_array = projA * projB * equilB
        # don't multiply by muA because that is the normalization
        # constraint on the output PDF
        temp_array = temp_array.dot(vals)
        # NOTE: The above likelihood is merely proportional to the actual likelihood
        # we would really need to multiply by a volume of phase space, since this
        # is the likelihood PDF...
        log_like = np.log(temp_array).sum()
        print log_like
        return log_like


    def save(self, output_fn):
        """
        save results to a .h5 file
        """
    
        try: 
            kernel_str = pickle.dumps(self.kernel)
        except:
            kernel_str = ""
            print "cannot pickle this kernel :("   

        io.saveh(output_fn, ktica_vals=self.eigen_sol[0],
            ktica_vecs=self.eigen_sol[1], K=self.K, 
            K_uncentered=self.K_uncentered, 
            reg_factor=np.array([self.reg_factor]),
            traj=self._Xall, dt=np.array([self.dt]),
            normalized=np.array([self._normalized]), 
            kernel_str=np.array([kernel_str]),
            acf_vals=self.acf_vals)


def load(input_fn, kernel=None):
    """
    load a ktica object saved via the .save method. 

    Parameters
    ----------
    input_fn : str
        input filename
    kernel : kernel instance, optional
        kernel to use when calculating inner products. If None,
        then we will look in the file. If it's not there, then an 
        exception will be raised

    Returns
    -------
    kt : ktica instance
    """

    f = io.loadh(input_fn)

    if not kernel is None:
        kernel = kernel
    elif 'kernel_str' in f.keys():
        kernel = pickle.loads(f['kernel_str'][0])
    else:
        raise Exception("kernel_str not found in %s. Need to pass a kernel object")

    kt = ktICA(kernel, f['dt'][0], reg_factor=f['reg_factor'][0])  
    # dt and reg_factor were saved as arrays with one element

    kt.K_uncentered = f['K_uncentered']
    kt.K = f['K']

    kt._Xall = f['traj'].astype(np.double)
    kt._Xa = kt._Xall[:len(kt._Xall) / 2]
    kt._Xb = kt._Xall[len(kt._Xall) / 2:]


    kt.eigen_sol = (f['ktica_vals'], f['ktica_vecs'])

    kt._normalized = False
    kt._sort()
    # ^^^ sorting also normalizes 
 
    return kt

