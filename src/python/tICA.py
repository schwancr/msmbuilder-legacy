
from scipy import signal
import scipy
import numpy as np
import re, sys, os
import multiprocessing as mp
from time import time
import gc
from scipy.weave import inline
import logging

logger = logging.getLogger( __name__ )
logger.setLevel( logging.DEBUG )
def correlate_C( lag ):

    N0, N = data_vector.shape
    correlate_mat = np.zeros( ( N, N ) )
    code = \
"""
int i,j,k; 
float sum=0.;

lag = (int) lag;

#pragma omp parallel for private(sum,i,j)
for ( i = 0; i < N; i++ )
{
    for ( j = 0; j < N; j++ )
    {
        sum=0.;
        for ( k = 0; k < N0 - lag; k++ )
        {   
            // correlate_mat[ i * N + j ] += data_vector[ k * N + i ] * data_vector[ ( k + lag ) * N + j];
            sum += data_vector[ k * N + i ] * data_vector[ ( k + lag ) * N + j];
        }
        correlate_mat[ i * N + j ] = sum;
    }
}
"""
    inline( code, ['N', 'N0', 'lag', 'correlate_mat', 'data_vector'], headers=['"omp.h"'], extra_compile_args=[ '-fopenmp' ], libraries=['gomp'], verbose=2 )

    return correlate_mat

def np_dot_row( args ):
    row_ind = args[0]
    lag = args[1]
    sol = []

    if lag == 0:
        a = data_vector[:,row_ind].reshape( (-1,1) )
        sol = (a * data_vector ).sum(axis=0)
    else:
        a = data_vector[:-lag,row_ind].reshape( (-1,1) )
        sol = (a * data_vector[lag:]).sum(axis=0)

    del a
    return sol

def np_correlate_row( args ):
    row_ind = args[0]
    lag = args[1]
    sol = []

    if lag == 0:
        a = data_vector[:,row_ind]
        for j in xrange( data_vector.shape[1] ):
            sol.append( np.correlate( a, data_vector[:,j],mode='valid' )[0] )
    else:
        a = data_vector[:-lag,row_ind]
        for j in xrange( data_vector.shape[1] ):
            sol.append( np.correlate( a, data_vector[lag:,j], mode='valid' )[0] ) 
    del a
    return sol

class CovarianceMatrix:

    def __init__( self, lag=0, procs=1, normalize=False ):

        self.corrs = None
        self.left_sum = None
        self.right_sum = None
        self.tot_sum = None

        self.coors_lag0 = None        
        self.normalize = normalize

        self.trained_frames=0
        self.total_frames = 0

        self.lag=int(lag)

        self.size=None

        self.procs=procs

    def set_size( self, N ):
        if self.corrs !=None:
            logger.warn( "There is still a matrix stored! Not overwriting, use method start_over() to delete the old matrix." )
            return
        self.size = N

        self.corrs = np.zeros( (N,N) )
        self.left_sum = np.zeros( N )
        self.right_sum = np.zeros( N )
        self.tot_sum = np.zeros( N )

        if self.normalize:
            self.corrs_lag0 = np.zeros( (N, N) )

    def start_over( self ):
        self.trained_frames = 0
        self.total_frames = 0

        self.corrs = None
        self.left_sum = None
        self.right_sum = None
        self.tot_sum = None
#        self.sq_tot_sum = None

        self.corrs_lag0 = None

    def train( self, data_vector_orig ):
        global data_vector
        a=time()
        data_vector = data_vector_orig.copy()
        #data_vector /= data_vector.std(axis=0)
        if data_vector.shape[1] != self.size:
            raise Exception("Input vector is not the right size. axis=1 should be length %d. Vector has shape %s" %(self.size, str(data_vector.shape)) )

        if data_vector.shape[0] <= self.lag:
            logger.warn( "Data vector is too short (%d) for this lag (%d)" % (data_vector.shape[0],self.lag) )
            return

        temp_mat = np.zeros( (self.size,self.size) )

        num_frames = data_vector.shape[0] - self.lag
        b=time()
#        Pool = mp.Pool( self.procs )
        #sol = [ np_dot_row( (i,self.lag) ) for i in xrange( self.size ) ]
        #debug for memory leak ^^^

#        result = Pool.map_async( np_dot_row, zip( range( self.size ), [self.lag]*self.size ) )
#        result.wait()
#        sol=result.get()

#        Pool.close()
#        Pool.join()
#        temp_mat = np.vstack( sol )
        temp_mat = correlate_C( self.lag )
        if self.normalize:
  
        #    Pool = mp.Pool( self.procs )

        #    result_lag0 = Pool.map_async( np_dot_row, zip( range( self.size ), [0]*self.size ) )
        #    result_lag0.wait()
        #    sol=result_lag0.get()

        #    Pool.close()
        #    Pool.join()
        #    temp_mat_lag0 = np.vstack( sol )
            temp_mat_lag0 = correlate_C( 0 )
            self.corrs_lag0 += temp_mat_lag0

        c=time()

        self.corrs += temp_mat
        self.left_sum += data_vector[: -self.lag].sum(axis=0)
        self.right_sum += data_vector[ self.lag :].sum(axis=0)
        self.tot_sum += data_vector.sum(axis=0)
        # self.sq_tot_sum += (data_vector * data_vector).sum(axis=0)
        
        self.trained_frames += num_frames
        self.total_frames += data_vector.shape[0]
        f=time()

        #print np.abs(self.corrs_lag0 - self.corrs_lag0.T).max()
        logger.debug( "Setup: %f, Corrs: %f, Finish: %f" %( b-a, c-b, f-c) )

    def get_current_estimate(self):

        tot_means = ( self.tot_sum / float( self.total_frames ) ).reshape( (-1,1) )
 
        tot_means_mat = np.dot( tot_means, tot_means.T ) * self.trained_frames

        left_sum = self.left_sum.reshape( (-1,1) )
        right_sum = self.right_sum.reshape( (-1,1) )

        tot_mean_left_sum = np.dot( tot_means, left_sum.T )
        tot_mean_right_sum = np.dot( tot_means, right_sum.T )



        temp_mat = self.corrs - tot_mean_left_sum - tot_mean_right_sum + tot_means_mat
        temp_mat /= self.total_frames

        temp_mat = (temp_mat+temp_mat.T)/2.

        if self.normalize:

            logger.debug( 'Error in symmetry of covariance matrix is %f' % np.abs(self.corrs_lag0 - self.corrs_lag0.T).max() )
            temp_mat_lag0 = self.corrs_lag0 / float( self.total_frames ) - np.dot( tot_means, tot_means.T )
            return temp_mat, temp_mat_lag0

        return temp_mat

class GramMatrix:
    """ This class is similar to the covariance matrix, but it stores a Gram matrix using a 
    particular kernel function (one of msmbuilder.kernels)"""

    def __init__(self, kernel, store_in_memory=True):
        self.kernel = kernel
        self.store_in_memory = store_in_memory

    def calc_gram_matrices(self, prepared_traj_0, prepared_traj_dt):

        self.M = len(prepared_traj_0)
        self.gram_matrix_0 = self.kernel.all_to_all(prepared_traj_0, prepared_traj_0)
        self.gram_matrix_dt = self.kernel.all_to_all(prepared_traj_0, prepared_traj_dt)
        #self.gram_matrix_dt = np.concatenate([ self.kernel.one_to_all(prepared_traj_0, prepared_traj_dt, i) for i in xrange(self.M) ], axis=0)

        return 

    def get_centered_matrix(self, mat):

        M = float(self.M)
        print mat.shape
        return mat - mat.sum(axis=0) / M - np.reshape(mat.sum(axis=1), (self.M, 1)) / M - \
               mat.sum() / M / M

    def get_eigensolution(self):
        
        K_dt = self.get_centered_matrix(self.gram_matrix_dt)
        K_0 = self.get_centered_matrix(self.gram_matrix_0)

        left_mat = K_dt.dot(K_0)
        right_mat = K_0.dot(K_0)

        eigensolution = scipy.linalg.eig(left_mat, b=right_mat)
        
        return eigensolution
