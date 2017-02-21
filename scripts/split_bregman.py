#!/usr/bin/env python

"""
Insert script description here.
"""

from __future__ import print_function, division

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__   = "2014-08-14"
__version__ = "0.1"

import argparse
import numpy as np
from scipy.io.mmio import mmread

from bregman import split_bregman

#----------------------------------------------------------------------#

def solve_bregman(A_file, E_file, weights_file, mu, lmbda, maxiter=100,
                  tol=0.001, eci_cutoff=1.0e-8, outfile='ECIs.dat'):

    A = np.array(mmread(A_file).todense())
    En = np.loadtxt(E_file)
    (Nstruc, Ncorr) = A.shape
    w = np.ones(Nstruc)
    if weights_file is not None:
        with open(weights_file, 'r') as fp:
            for line in fp:
                (struc, weight) = line.split()[:2]
                struc = int(struc)
                weight = float(weight)
                w[struc-1] = weight
        A[:] = (A.T * w).T
        En[:] = En * w

    ecis = np.zeros(Ncorr)

    print(" Input Parameters:  "
          "mu = {},  lambda = {},  eci_cutoff = {}".format(
              mu, lmbda, eci_cutoff))

    mu = 1.0/mu
    ecis[:] = split_bregman(maxiter, tol, mu, lmbda, A, En, ecis)

    nonzero = 0
    with open(outfile, 'w') as fp:
        print(" Saving ECIs to file `{}'".format(outfile))
        for i in range(Ncorr):
            if (abs(ecis[i]) > eci_cutoff):
                fp.write("{:6d}   {:14.10f}\n".format(i+1,ecis[i]))
                nonzero += 1
            else:
                fp.write("{:6d}   {:14.10f}\n".format(i+1,0.0))
    print(" {} non-zero ECIs found.".format(nonzero))

    E = A.dot(ecis)
    rmse = np.sqrt(np.sum((E - En)**2)/Nstruc)
    print(" RMSE = {}".format(rmse))

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "A",
        help    = "Path to correlation matrix in sparse matrix market format.")

    parser.add_argument(
        "E",
        help    = "Path to file with structural energies.")

    parser.add_argument(
        "--weights", "-w",
        help    = "Path to file with structure weights.",
        default = None)

    parser.add_argument(
        "--mu",
        help    = "Estimated amplitude of noise.",
        type    = float,
        default = 0.01)

    parser.add_argument(
        "--lmbda",
        help    = "Algorithm parameter (lambda)",
        type    = float,
        default = 100)

    parser.add_argument(
        "--maxiter",
        help    = "Maximum number of iteration.",
        type    = int,
        default = 100)

    parser.add_argument(
        "--tol",
        help    = "Convergence criterium.",
        type    = float,
        default = 1.0e-3)

    parser.add_argument(
        "--eci-cutoff",
        help    = "ECIs below this cutoff are considered equal to zero.",
        type    = float,
        default = 1.0e-8)

    args = parser.parse_args()

    solve_bregman(args.A, args.E, args.weights, args.mu, args.lmbda,
                  args.maxiter, args.tol, args.eci_cutoff)
