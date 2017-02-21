"""
Insert module description here.
"""

from __future__ import print_function, division

import numpy as np

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2015-11-30"
__version__ = "0.1"


class PrincipalComponentAnalysis(object):
    """
    Perform a principal component analysis of some input data.

    Properties:
        data              Array of all input vectors
        mean              The mean of the input vectors
        scatter_matrix    S = \sum_i (x_i -mean) * (x_i - mean)^T
                          for all input vectors x_i
        eigenvectors      Eigenvectors of S; principal components (PCs)
        eigenvalues       Eigenvalues of S; weights of the PCs
        transformed_data  Input data in the PC basis
    """

    def __init__(self, data):
        """
        Arguments:
            data     2d array with input data.  Each row is taken to be one
                     data point with its coordinates in the columns.
        """

        self.data = np.array(data)
        self.mean = np.mean(data, axis=0)

        # scatter matrix and its eigenvalues/-vectors
        # (estimate of the covariance matrix)
        self.scatter_matrix = np.zeros((self.dimension, self.dimension))
        for x in data:
            self.scatter_matrix += np.outer(x - self.mean, x - self.mean)
        self.scatter_matrix /= self.dimension
        (self.eigenvalues, self.eigenvectors
         ) = np.linalg.eig(self.scatter_matrix)

        if np.any(abs(self.eigenvectors.imag) > 1.0e-4):
            print(" Warning: imaginary eigenvectors")
        self.eigenvectors = self.eigenvectors.real

        # principal components
        self.transformed_data = self.input2pc(self.data)

        np.savetxt('eigenval-pca', self.eigenvalues)
        np.savetxt('eigenvec-pca', self.eigenvectors)
        np.savetxt('corr.in-pca', self.transformed_data)

    @classmethod
    def from_file(cls, fname):
        """
        Use numpy.loadtxt with all default arguments to load the input data
        from a file.

        fname can be a file object or a string with the filesystem path.
        """

        data = np.loadtxt(fname)
        return cls.__init__(data)

    def __str__(self):
        return

    @property
    def dimension(self):
        return len(self.mean)

    def input2pc(self, vectors):
        """
        Transform vector(s) from the input basis into the principal component
        basis.

        """

        return np.dot(vectors, self.eigenvectors.T)
