import numpy as np
import numpy.testing as npt
from scipy.spatial.distance import squareform
from msmbuilder.clustering import BaseFlatClusterer
import scipy.weave
import itertools
import logging
    
logger = logging.getLogger(__name__)


################################################################################
#          A simple and fast algorithm for K-medoids clustering
#          Hae-Sang Park, Chi-Hyuck Jun 
#          Expert Systems with Applications 36 (2009) 3336-3341
################################################################################

#DEBUG = True


class ParkKMedoids(BaseFlatClusterer):
    def __init__(self, metric, trajectories, k=None, max_iters=None):
        """Run kcenters clustering algorithm.

        Terminates either when `k` clusters have been identified, or when every data
        is clustered better than `distance_cutoff`.

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectory : Trajectory or list of msmbuilder.Trajectory
            data to cluster
        k : int
            number of desired clusters
        max_iters : {int, None}
            Maximum number of iterations
 
        """
        super(ParkKMedoids, self).__init__(metric, trajectories)
        gi, asgn, dl = _park_medoids(metric, self.ptraj, k, max_iters)
        
        # note that the assignments here are with respect to the numbering
        # in the trajectory -- they are not contiguous. Using the get_assignments()
        # method defined on the superclass (BaseFlatClusterer) will convert them
        # back into the contiguous numbering scheme (with respect to position in the
        # self._generator_indices).
        self._generator_indices = gi
        self._assignments = asgn
        self._distances = dl

def _park_assign(d_a2a, medoids, n_frames, k):
    membership = -1 * np.ones(n_frames, dtype=np.int)
    distances_to_medoid = np.inf * np.ones(n_frames, dtype=np.float32)
    
    scipy.weave.inline(r"""
    int i, j, ptr;
    
    for (i = 0; i < k; i++) {
        for (j = 0; j < n_frames; j++ ) {
            ptr = n_frames * medoids[i] + j;
            if (d_a2a[ptr] < distances_to_medoid[j]) {
                membership[j] = i;
                distances_to_medoid[j] = d_a2a[ptr];
            }
        }
    }
    """, ['membership', 'distances_to_medoid', 'n_frames',
          'k', 'd_a2a', 'medoids'],
    extra_compile_args = ["-O3"], compiler='gcc')
    
    return membership, distances_to_medoid


def _park_medoids(metric, ptraj, k, max_iterations=None):
    """
    Parameters
    ----------
    
    
    max_iterations : {int, None}
        max # iterations, or until convergence if you supply None
    """
    n_frames = len(ptraj)
    logger.info('getting all-to-all')
    d_a2a = squareform(metric.all_pairwise(ptraj)).astype(np.float32)
    logger.info('got all-to-all')
    # compute v_j = sum_i^{n_frames} (d_{ij} / sum_l^{n_frames} d_{il})
    # we have to use some transpose tricks to get aound
    # numpy broadcasting rules
    v = np.sum((d_a2a.T / np.sum(d_a2a, axis=1)), axis=1)
    
    # if DEBUG:
    #     j = 10
    #     v_j = 0
    #     for i in range(n_frames):
    #         v_j += d_a2a[i,j] / np.sum(d_a2a[i,:])
    #     assert v_j == v[j]
    
    # indices of the k smallest
    medoids = np.argsort(v)[:k]
    
    
    membership, distances_to_medoid = _park_assign(d_a2a, medoids, n_frames, k)
    new_sd = np.mean(distances_to_medoid)
    logger.info('Initial Clustering Score %f', new_sd)
    
    if max_iterations is None:
        iterable = itertools.count(0)
    else:
        iterable = xrange(max_iterations)
    
    for i in iterable:
        medoids = _park_update(d_a2a, medoids, membership, n_frames, k)
        membership, distances_to_medoid = _park_assign(d_a2a, medoids, n_frames, k)
        old_sd, new_sd = new_sd, np.mean(distances_to_medoid)
        logger.info("Iteration %5d clustering score %f", i, new_sd)

        if new_sd >= old_sd:
            logger.info('Converged. You win!')
            break

    # the assignments that we return (middle return value) are going to be read
    # like assignments[i] = j indicates that frame i is assigned to frame j
    # (NOT frame i is assigned to medoids[j])
    return medoids, medoids[membership], distances_to_medoid
    

def _park_update(d_a2a, medoids, membership, n_frames, k):
    #Find a new medoid of each cluster, which
    #is the object minimizing the total distance
    #to other objects in its cluster. Update the
    # current medoid in each cluster by replacing
    # with the new medoid.
    best_medoids = -1 * np.ones(k, dtype=np.int)
    best_medoids_score = np.inf * np.ones(k, dtype=np.float32)
    
    # RELEASE THE GIL, BITCHES!!!!!
    scipy.weave.inline(r"""
    Py_BEGIN_ALLOW_THREADS
    int i, j, membership_i;
    float score_i;
    
    #pragma omp parallel for private(j, score_i, membership_i) shared(best_medoids_score, best_medoids, d_a2a, membership, n_frames)
    for (i = 0; i < n_frames; i++) {
        score_i = 0.0;
        membership_i = membership[i];
        for (j = 0; j < n_frames; j++) {
            if (membership_i == membership[j]) {
                score_i += d_a2a[i*n_frames + j];
            }
        }
        
        // be careful here, if two threads tried to update
        // this at the same time, it could get ugly
        #pragma omp critical(best_update)
        {
            if (score_i < best_medoids_score[membership_i]) {
                best_medoids_score[membership_i] = score_i;
                best_medoids[membership_i] = i;
            }
        }
    }
    Py_END_ALLOW_THREADS
    """, ['best_medoids_score', 'best_medoids', 'd_a2a', 'membership',
           'n_frames'],
         extra_link_args = ['-lgomp'],
         extra_compile_args = ["-O3", "-fopenmp"],
         headers = ['<omp.h>'],
         compiler='gcc')
    
    return best_medoids





##################################################
#    REFERENCE PYTHON IMPLEMENTATION OF UPDATE   #
#                    AND ASSIGN                  #
##################################################


def _park_update_reference(d_a2a, medoids, membership, n_frames, k):
    #Find a new medoid of each cluster, which
    #is the object minimizing the total distance
    #to other objects in its cluster. Update the
    # current medoid in each cluster by replacing
    # with the new medoid.
    best_medoids = -1 * np.ones(k)
    best_medoids_score = np.inf * np.ones(k)
    
    for i in xrange(n_frames):
        score_i = 0
        membership_i = membership[i]
        for j in xrange(n_frames):
            if membership_i == membership[j]:
                score_i += d_a2a[i,j]
        
        if score_i < best_medoids_score[membership_i]:
            best_medoids_score[membership_i] = score_i
            best_medoids[membership_i] = i

    #print best_medoids
    return best_medoids


def _park_assign_reference(d_a2a, medoids, n_frames, k):
    membership = -1 * np.ones(n_frames, dtype=np.int)
    distances_to_medoid = np.inf * np.ones(n_frames)
    for i in xrange(k):
        where = np.where(d_a2a[medoids[i],:] < distances_to_medoid)[0]
        membership[where] = i
        distances_to_medoid[where] = d_a2a[medoids[i],where]
        
    return membership, distances_to_medoid