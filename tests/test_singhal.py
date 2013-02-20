
from msmbuilder.error import SinghalError
from msmbuilder.testing import eq, get
from msmbuilder import msm_analysis, MSMLib
from scipy.io import mmread
import numpy as np
import IPython




class test_SinghalError():

    def test(self):

        c = get('singhal_reference/tCounts.mtx').toarray()

        s = SinghalError(c, force_dense=True)

        variances = s.get_eigenvalue_variances(which_eigenvalues=range(1,10))
        qi_list = s.get_eigenvalue_variances(which_eigenvalues=range(1,10), return_list=True)

        ref_vars = get('singhal_reference/reference_variances.dat')
        ref_qis = get('singhal_reference/reference_qis.dat')

        eq(ref_vars, variances)
        eq(ref_qis, qi_list)

