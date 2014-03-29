from __future__ import print_function, absolute_import, division

import numpy as np
from numpy import dot, diag
inv = np.linalg.inv
norm = np.linalg.norm
tr = np.trace

import scipy.optimize

from .lumper import EigenvectorLumper

import logging
logger = logging.getLogger(__name__)


def metastability(alpha, T, right_eigenvectors, square_map, pi):
    """Return the metastability PCCA+ objective function.

    Parameters
    ----------
    alpha : ndarray
        Parameters of objective function (e.g. flattened A)
    T : csr sparse matrix
        Transition matrix
    right_eigenvectors : ndarray
        The right eigenvectors.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    pi : ndarray
        Equilibrium Populations of transition matrix.

    Returns
    -------
    obj : float
        The objective function

    Notes
    -------
    metastability: try to make metastable fuzzy state decomposition.
    Defined in ref. [2].
    """

    num_micro, num_eigen = right_eigenvectors.shape

    A, chi, mapping = calculate_fuzzy_chi(alpha, square_map, right_eigenvectors)

    # If current point is infeasible or leads to degenerate lumping.
    if len(np.unique(mapping)) != right_eigenvectors.shape[1] or has_constraint_violation(A, right_eigenvectors):
        return -1.0 * np.inf

    obj = 0.0
    for i in range(num_eigen):  # Calculate  metastabilty of the lumped model.  Eqn 4.20 in LAA.
        obj += np.dot(T.dot(chi[:, i]), pi * chi[:, i]) / np.dot(chi[:, i], pi)

    return obj


def crisp_metastability(alpha, T, right_eigenvectors, square_map, pi):
    """Return the crisp_metastability PCCA+ objective function.

    Parameters
    ----------
    alpha : ndarray
        Parameters of objective function (e.g. flattened A)
    T : csr sparse matrix
        Transition matrix
    right_eigenvectors : ndarray
        The right eigenvectors.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    pi : ndarray
        Equilibrium Populations of transition matrix.

    Returns
    -------
    obj : float
        The objective function

    Notes
    -------
    crisp_metastability: try to make the resulting crisp msm metastable.
    This is the recommended choice.  This is the metastability (trace)
    of a transition matrix computed by forcing a crisp (non-fuzzy)
    microstate mapping.  Defined in ref. [2].
    """

    num_micro, num_eigen = right_eigenvectors.shape

    A, chi_fuzzy, mapping = calculate_fuzzy_chi(alpha, square_map, right_eigenvectors)

    chi = 0.0 * chi_fuzzy  # Make the membership matrix "crisp"
    chi[np.arange(num_micro), mapping] = 1.

    # If current point is infeasible or leads to degenerate lumping.
    if len(np.unique(mapping)) != right_eigenvectors.shape[1] or has_constraint_violation(A, right_eigenvectors):
        return -1.0 * np.inf

    obj = 0.0
    for i in range(num_eigen):  # Calculate  metastabilty of the lumped model.  Eqn 4.20 in LAA.
        obj += np.dot(T.dot(chi[:, i]), pi * chi[:, i]) / np.dot(chi[:, i], pi)

    return obj


def crispness(alpha, T, right_eigenvectors, square_map, pi):
    """Return the crispness PCCA+ objective function.

    Parameters
    ----------
    alpha : ndarray
        Parameters of objective function (e.g. flattened A)
    T : csr sparse matrix
        Transition matrix
    right_eigenvectors : ndarray
        The right eigenvectors.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    pi : ndarray
        Equilibrium Populations of transition matrix.

    Returns
    -------
    obj : float
        The objective function

    Notes
    -------
    Tries to make crisp state decompostion.  This function is
    defined in [3].
    """

    A, chi, mapping = calculate_fuzzy_chi(alpha, square_map, right_eigenvectors)

    # If current point is infeasible or leads to degenerate lumping.
    if len(np.unique(mapping)) != right_eigenvectors.shape[1] or has_constraint_violation(A, right_eigenvectors):
        return -1.0 * np.inf

    obj = tr(dot(diag(1. / A[0]), dot(A.transpose(), A)))

    return obj


class PCCAPlus(EigenvectorLumper):

    def __init__(self, T, num_macrostates, objective_function="crisp_metastability", flux_cutoff=None, do_minimization=True):
        """Perform PCCA+ lumping and return PCCA+ object.

        Parameters
        ----------
        T : csr sparse matrix
            Transition matrix
        num_macrostates : int
            desired (maximum) number of macrostates
        flux_cutoff : float, optional
            If desired, discard eigenvectors with flux below this value.
        do_minimization : bool, optional
            If False, skip the optimization of the transformation matrix.
            In general, minimization is recommended.
        objective_function: {'crisp_metastablility', 'metastability', 'metastability'}
            Possible objective functions.  See objective for details.

        Notes
        -----
        PCCA+ is used to construct a "lumped" state decomposition.  First,
        The eigenvalues and eigenvectors are computed for a transition matrix.
        An optimization problem is then used to estimate a mapping from
        microstates to macrostates.

        For each microstate i, microstate_mapping[i] is chosen as the
        macrostate with the largest membership (chi) value.

        The membership matrix chi is given by chi = dot(vr,A).

        The transformation matrix A is the output of a constrained
        optimization problem.

        You have three choices for objective function: crispness, metastability,
        or crisp_metastability

        Our implementation of PCCA+ REQUIRES DETAILED BALANCE.

        Here are some key quantities that will be saved as member variables:

        A : ndarray
            The transformation matrix.
        chi : ndarray
            The membership matrix
        vr : ndarray
            The right eigenvectors.
        microstate_mapping : ndarray
            Mapping from microstates to macrostates.

        References
        ----------

        .. [1]  Deuflhard P, Weber, M.,  "Robust perron cluster analysis in
         conformation dynamics,"
        Linear Algebra Appl., vol 398 pp 161-184 2005.

        .. [2]  Kube S, Weber M.  "A coarse graining method for the
        identification of transition rates between molecular conformations,"
        J. Chem. Phys., vol 126 pp 24103-024113, 2007.

        .. [3]  Kube S.,  "Statistical Error Estimation and Grid-free
        Hierarchical Refinement in Conformation Dynamics," Doctoral Thesis.
        2008

        .. [4]  Deuflhard P, et al.  "Identification of almost invariant
        aggregates in reversible nearly uncoupled markov chains,"
        Linear Algebra Appl., vol 315 pp 39-59, 2000.

        See Also
        --------
        PCCA
        """

        if objective_function == "crispness":
            self.objective_function = crispness
        elif objective_function == "metastability":
            self.objective_function = metastability
        elif objective_function == "crisp_metastability":
            self.objective_function = crisp_metastability
        else:
            possible_objective_functions = ["crispness", "crisp_metastability", "metastability"]
            raise Exception("objective_function must be one of ", possible_objective_functions)

        EigenvectorLumper.__init__(self, T, num_macrostates, flux_cutoff=flux_cutoff)
        self.lump(do_minimization=do_minimization)

    def lump(self, do_minimization=True):
        """Perform PCCA+ algorithm by optimizing transformation matrix A.

        Creates the following member variables:
        -------
        A : ndarray
            The transformation matrix.
        chi : ndarray
            The membership matrix
        microstate_mapping : ndarray
            Mapping from microstates to macrostates.

        """

        index = index_search(self.right_eigenvectors)

        # compute transformation matrix A as initial guess for local optimization
        # (maybe not feasible)
        A = self.right_eigenvectors[index, :]

        A = inv(A)
        A = fill_A(A, self.right_eigenvectors)

        if do_minimization == True:
            logger.warning("Minimizing PCCA+ objective function.")
            A = self.optimize_A(A)
        else:
            logger.warning("Skipping Minimization")

        A = fill_A(A, self.right_eigenvectors)

        chi = dot(self.right_eigenvectors, A)

        microstate_mapping = np.argmax(chi, 1)

        self.A, self.chi, self.microstate_mapping = A, chi, microstate_mapping

    def optimize_A(self, A):
        """Find optimal transformation matrix A by minimization.

        Parameters
        ----------
        A : ndarray
        The transformation matrix A.

        Returns
        -------
        A : ndarray
            The transformation matrix.
        """
        flat_map, square_map = get_maps(A)
        alpha = to_flat(1.0 * A, flat_map)

        obj = lambda x: -1 * \
            self.objective_function(
                x, self.T, self.right_eigenvectors, square_map, self.populations)
        self.obj = obj
        self.alpha = alpha.copy()

        logger.info("Initial value of objective function: f = %f", obj(alpha))

        alpha = scipy.optimize.anneal(
            obj, alpha, lower=0.0, maxiter=1, schedule="boltzmann", dwell=1000, feps=1E-3, boltzmann=2.0, T0=1.0)[0]

        alpha = scipy.optimize.fmin(
            obj, alpha, full_output=True, xtol=1E-4, ftol=1E-4, maxfun=5000, maxiter=100000)[0]

        logger.info("Final value: f = %f" % (obj(alpha)))

        if np.isneginf(obj(alpha)):
            raise ValueError("Error: minimization has not located a feasible point.")

        A = to_square(alpha, square_map)

        return A


def get_maps(A):
    """Get mappings from the square array A to the flat vector of parameters alpha.

    Helper function for PCCA+ optimization.

    Parameters
    ----------
    A : ndarray
        The transformation matrix A.

    Returns
    -------
    flat_map : ndarray
        Mapping from flat indices (k) to square (i,j) indices.
    square map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    """

    N = A.shape[0]
    flat_map = []
    for i in range(1, N):
        for j in range(1, N):
            flat_map.append([i, j])

    flat_map = np.array(flat_map)

    square_map = np.zeros(A.shape, 'int')

    for k in range((N - 1) ** 2):
        i, j = flat_map[k]
        square_map[i, j] = k

    return flat_map, square_map


def to_flat(A, flat_map):
    """Convert a square matrix A to a flat array alpha.

    Parameters
    ----------
    A : ndarray
        The transformation matrix A
    flat_map : ndarray
        Mapping from flat indices (k) to square (i,j) indices.

    Returns
    -------
    FlatenedA : ndarray
        flattened version of A
    """
    return A[flat_map[:, 0], flat_map[:, 1]]


def to_square(alpha, square_map):
    """Convert a flat array alpha to a square array A.

    Parameters
    ----------
    alpha : ndarray
        An array of (n-1)^2 parameters used as optimization parameters.
        alpha is a minimal, flat representation of A.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    Returns
    -------
    SquareA : ndarray
        Square version of alpha
    """
    return alpha[square_map]


def has_constraint_violation(A, right_eigenvectors, epsilon=1E-8):
    """Check for constraint violations in transformation matrix.

    Parameters
    ----------
    A : ndarray
        The transformation matrix.
    right_eigenvectors : ndarray
        The right eigenvectors.
    epsilon : float, optional
        Tolerance of constraint violation.

    Returns
    -------
    truth : bool
        Whether or not the violation exists


    Notes
    -------
    Checks constraints using Eqn 4.25 in [1].


    References
    ----------
    .. [1]  Deuflhard P, Weber, M.,  "Robust perron cluster analysis in
     conformation dynamics,"
    Linear Algebra Appl., vol 398 pp 161-184 2005.
    """

    lhs = 1 - A[0, 1:].sum()
    rhs = dot(right_eigenvectors[:, 1:], A[1:, 0])
    rhs = -1 * rhs.min()

    if abs(lhs - rhs) > epsilon:
        return True
    else:
        return False


def index_search(right_eigenvectors):
    """Find simplex structure in eigenvectors to begin PCCA+.


    Parameters
    ----------
    right_eigenvectors :  ndarray
        Right eigenvectors of transition matrix

    Returns
    -------
    index : ndarray
        Indices of simplex
    """

    num_micro, num_eigen = right_eigenvectors.shape

    index = np.zeros(num_eigen, 'int')

    # first vertex: row with largest norm
    index[0] = np.argmax([norm(right_eigenvectors[i]) for i in range(num_micro)])

    ortho_sys = right_eigenvectors - np.outer(np.ones(num_micro), right_eigenvectors[index[0]])

    for j in range(1, num_eigen):
        temp = ortho_sys[index[j - 1]].copy()
        for l in range(num_micro):
            ortho_sys[l] -= temp * dot(ortho_sys[l], temp)

        dist_list = np.array([norm(ortho_sys[l]) for l in range(num_micro)])

        index[j] = np.argmax(dist_list)

        ortho_sys /= dist_list.max()

    return index


def fill_A(A, right_eigenvectors):
    """Construct feasible initial guess for transformation matrix A.


    Parameters
    ----------
    A : ndarray
        Possibly non-feasible transformation matrix.
    right_eigenvectors :  ndarray
        Right eigenvectors of transition matrix

    Returns
    -------
    A : ndarray
        Feasible transformation matrix.
    """
    num_micro, num_eigen = right_eigenvectors.shape

    A = A.copy()

    # compute 1st column of A by row sum condition
    A[1:, 0] = -1 * A[1:, 1:].sum(1)

    # compute 1st row of A by maximum condition
    A[0] = -1 * dot(right_eigenvectors[:, 1:].real, A[1:]).min(0)

    # rescale A to be in the feasible set
    A /= A[0].sum()

    return A


def calculate_fuzzy_chi(alpha, square_map, right_eigenvectors):
    """Calculate the membership matrix (chi) from parameters alpha.

    Parameters
    ----------
    alpha : ndarray
        Parameters of objective function (e.g. flattened A)
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    right_eigenvectors : ndarray
        The right eigenvectors.

    Returns
    -------
    A : ndarray
        The transformation matrix A
    chi_fuzzy : ndarray
        The (fuzzy) membership matrix.
    mapping: ndarray
        The mapping from microstates to macrostates.
    """
    A = to_square(alpha, square_map)  # Convert parameter vector into matrix A
    A = fill_A(A, right_eigenvectors)  # Make A feasible.
    chi_fuzzy = np.dot(right_eigenvectors, A)  # Calculate the fuzzy membership matrix.
    mapping = np.argmax(chi_fuzzy, 1)  # Calculate the microstate mapping.
    return A, chi_fuzzy, mapping
