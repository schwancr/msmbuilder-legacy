
import numpy as np
import re, sys, os
from time import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CovarianceMatrix(object):
    """
    CovarianceMatrix is a class for calculating covariance matrices. It can be
    used to calculate both the time-lag correlation matrix and covariance
    matrix. The advantage it has is that you can calculate the matrix for a 
    large dataset by "training" pieces of the dataset at a time. 

    Notes
    -----

    It can be shown that the time-lag correlation matrix is the same as:

    C = E[Outer(X[t], X[t+lag])] - Outer(E[X[t]], E[X[t+lag]])

    Because of this it is possible to calculate running sums corresponding 
    to variables A, B, D:

    A = E[X[t]]
    B = E[X[t+lag]]
    D = E[Outer(X[t], X[t+lag])]

    Then at the end we can calculate C:

    C = D - Outer(A, B)

    Finally we can get a symmetrized C' from our estimate of C, for
    example by adding the transpose:

    C' = (C + C^T) / 2
     
    """
    def __init__(self, lag, procs=1, calc_cov_mat=True, size=None, tProb=None,
                 populations=None):
        """
        Create an empty CovarianceMatrix object.

        To add data to the object, use the train method.

        Parameters
        ----------
        lag: int
            The lag to use in calculating the time-lag correlation
            matrix. If zero, then only the covariance matrix is
            calculated
        procs: int, optional
            number of processors to use when training.
            CURRENTLY NOT IMPLEMENTED
        calc_cov_mat: bool, optional
            if lag > 0, then will also calculate the covariance matrix
        size: int, optional
            the size is the number of coordinates for the vector
            representation of the protein. If None, then the first
            trained vector will be used to initialize it.
        
        """

        self.tProb = tProb
        self.populations = populations
        if self.tProb is None:
            self.populations = None
        else:
            self.tProb = self.tProb.tolil()

        self.corrs = None
        self.sum_t = None
        self.sum_t_dt = None
        # The above containers hold a running sum that is used to 
        # calculate the time-lag correlation matrix as well as the
        # covariance matrix

        self.coors_lag0 = None  # needed for calculating the covariance
                                # matrix       
        self.sum_all = None

        self.trained_frames = 0
        self.trained_frames_t = 0
        self.trained_frames_dt = 0
        self.total_frames = 0
        # Track how many frames we've trained

        self.lag=int(lag)
        if self.lag < 0:
            raise Exception("lag must be non-negative.")
        elif self.lag == 0:  # If we have lag=0 then we don't need to
                             # calculate the covariance matrix twice
            self.calc_cov_mat = False
        else:
            self.calc_cov_mat = calc_cov_mat

        self.size = size
        if not self.size is None:
            self.set_size(size)
            
        self.procs=procs
        # Currently not-used...

    def set_size(self, N):
        """
        Set the size of the matrix.

        Parameters
        ----------
        N : int
            The size of the square matrix will be (N, N)

        """

        self.size = N

        self.corrs = np.zeros((N,N), dtype=float)
        self.sum_t = np.zeros(N, dtype=float)
        self.sum_t_dt = np.zeros(N, dtype=float)
        self.sum_all = np.zeros(N, dtype=float)

        if self.calc_cov_mat:
            self.corrs_lag0 = np.zeros((N, N), dtype=float)

    def train(self, data_vector, assignments=None):
        a=time()  # For debugging we are tracking the time each step takes

        if not self.populations is None:
            if assignments is None:
                raise Exception("Need to input assignments array for this trajectory")
    
        if self.size is None:  
        # then we haven't started yet, so set up the containers
            self.set_size(data_vector.shape[1])


        if data_vector.shape[1] != self.size:
            raise Exception("Input vector is not the right size. axis=1 should "
                            "be length %d. Vector has shape %s" %
                            (self.size, str(data_vector.shape)))

        if data_vector.shape[0] <= self.lag:
            logger.warn("Data vector is too short (%d) "
                        "for this lag (%d)", data_vector.shape[0],self.lag)
            return

        b=time()
        if not self.populations is None:
            good_ass_ind = np.where((assignments[:-self.lag] != -1) & (assignments[self.lag:] != -1))[0]
            # need to account for trimmed states. We will do this by having them be probability zero.

            pops_t = np.zeros((len(data_vector) - self.lag, 1))
            pops_dt = np.zeros((len(data_vector) - self.lag, 1))
            trans_t_dt = np.zeros((len(data_vector) - self.lag, 1))
            pops_all = np.zeros((len(data_vector), 1))

            pops_t[good_ass_ind] = self.populations[assignments[good_ass_ind]].reshape((-1,1))
            pops_dt[good_ass_ind] = self.populations[assignments[good_ass_ind + self.lag]].reshape((-1,1))
            trans_t_dt[good_ass_ind] = self.tProb[assignments[good_ass_ind], assignments[good_ass_ind + self.lag]].toarray().reshape((-1,1))
            pops_all[np.where(assignments != -1)] = self.populations[assignments[np.where(assignments != -1)]].reshape((-1,1))
        else:
            pops_t = np.ones((len(data_vector[:-self.lag]), 1))
            pops_dt = pops_t
            trans_t_dt = pops_t
            pops_all = np.ones((len(data_vector), 1))

        if self.lag != 0:
            self.corrs += (pops_t * trans_t_dt * data_vector[:-self.lag]).T.dot(data_vector[self.lag:])
            self.sum_t += np.sum(data_vector[:-self.lag] * pops_t, axis=0)
            self.sum_t_dt += np.sum(data_vector[self.lag:] * pops_dt, axis=0)
        else:
            self.corrs += (data_vector * pops_all).T.dot(data_vector) 
            self.sum_t += np.sum(data_vector * pops_all, axis=0)
            self.sum_t_dt += np.sum(data_vector * pops_all, axis=0)

        if self.calc_cov_mat:
            self.corrs_lag0 += (pops_all * data_vector).T.dot(data_vector)
            self.sum_all += np.sum(data_vector * pops_all, axis=0)

            self.total_frames += pops_all.sum()

        self.trained_frames += (pops_t * trans_t_dt).sum()
        self.trained_frames_t += pops_t.sum()
        self.trained_frames_dt += pops_dt.sum()
        # this accounts for us having finite trajectories, so we really are 
        #  only calculating expectation values over N - \Delta t total samples

        c=time()

        logger.debug("Setup: %f, Corrs: %f" %(b-a, c-b))
        # Probably should just get rid of this..

    def get_current_estimate(self):
        """Calculate the current estimate of the time-lag correlation
        matrix and the covariance matrix (if asked for).

        Currently, this is done by symmetrizing the sample time-lag
        correlation matrix, which can cause problems!
        
        """
        
        time_lag_corr = (self.corrs) / float(self.trained_frames)

        outer_expectations = np.outer(self.sum_t, self.sum_t_dt) / float(self.trained_frames_t) / float(self.trained_frames_dt)

        current_estimate = time_lag_corr - outer_expectations
        #current_estimate += current_estimate.T  # symmetrize the matrix
        #above suffers from a bug in numpy.ndarray.__iadd__
        current_estimate = current_estimate + current_estimate.T
        current_estimate /= 2.

        if self.calc_cov_mat:
            cov_mat = self.corrs_lag0 / float(self.total_frames) -  \
                np.outer(self.sum_all, self.sum_all) / float(self.total_frames) ** 2

            return current_estimate, cov_mat

        return current_estimate
