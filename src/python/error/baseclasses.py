
class MSMError(object):
    """
    Abstract class for calculating errors in MSMs.
    """

    def get_eigenvalue_variances(self):
        """
        This method returns the variance of a particular eigenvalue or set of
        eigenvalues.
        """
        raise Exception("Not Implemented")

    
    def get_eigenvector_variances(self):
        """
        This method returns the variance of a particular eigenvector or set
        of eigenvectors.
        """
        raise Exception("Not Implemented")

    
