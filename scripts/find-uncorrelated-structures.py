#!/usr/bin/env python

"""
Find structures that are most dissimilar to a reference by computing
the scalar products of their correlations.

"""

from __future__ import print_function, division

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2015-01-13"
__version__ = "0.1"

import argparse
import numpy as np


def score_structure_sum(A, ref, i):
    nref = len(ref)
    score = 0.0
    for j in ref:
        score += abs(np.dot(A[i], A[j]))
    score /= float(nref)
    return score

def score_structure_max(A, ref, i):
    score = 0.0
    for j in ref:
        score = max(abs(np.dot(A[i], A[j])), score)
    return score

def print_correlation_matrix(A, ref):
    nref = len(ref)
    C = np.identity(nref)
    for i in range(nref):
        for j in range(i+1, nref):
            C[i, j] = C[j, i] = np.dot(A[ref[i]], A[ref[j]])
    print(("#   " + nref*"{:6d} ").format(*ref))
    frmt = nref*"{:6.3f} "
    for i in range(nref):
        print("{:3d} ".format(ref[i]) + frmt.format(*C[i, :]))

def find_structures(reference_structure, corr_in_file, clusters, N,
                    scorefunc):

    if scorefunc == "max":
        score_structure = score_structure_max
    elif scorefunc == "sum":
        score_structure = score_structure_sum
    else:
        print("Error: unrecognized scoring function: {}".format(scorefunc))
        return

    if clusters is not None:
        cols = np.array(clusters)
        A = np.loadtxt(corr_in_file, skiprows=3, usecols=cols)
    else:
        A = np.loadtxt(corr_in_file, skiprows=3)

    nstruc = len(A)
    ref = [reference_structure]

    # normalize all vectors and ignore structures that are not
    # normalizable
    norm = np.linalg.norm(A, axis=1)
    ignore = list(np.arange(nstruc)[norm < 1.0e-6])
    norm[ignore] = 1.0
    A = A/np.reshape(norm,(nstruc,1))
    if len(ignore) > 0:
        print("Warning: {}".format(len(ignore))
              + " structure(s) could not be normalized: "
              + (len(ignore)*"{} ").format(*ignore))

    for i in range(N):
        score_min = 2.0
        for istruc in range(nstruc):
            if istruc in (ref + ignore):
                continue
            score = score_structure(A, ref, istruc)
            if score < score_min:
                istruc_min = istruc
                score_min  = score
        ref.append(istruc_min)

    print_correlation_matrix(A, ref)

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "reference_structure",
        help="ID of the reference structure (starting with 0)",
        type=int)

    parser.add_argument(
        "--correlations",
        help="Path to CASM's 'corr.in' file (default: corr.in).",
        type=str,
        default="corr.in")

    parser.add_argument(
        "--clusters", "-c",
        help="List of clusters to be used starting with 0 (default: all)",
        type=int,
        default=None,
        nargs="+")

    parser.add_argument(
        "--num-structures", "-n",
        help="Number of structures to be returned (default: 1).",
        type=int,
        default=1)

    parser.add_argument(
        "--score",
        help="Scoring function to be used. "
             "Options are: sum, max (default: sum)",
        type=str,
        default="sum")

    args = parser.parse_args()

    find_structures(args.reference_structure,
                    args.correlations,
                    args.clusters,
                    args.num_structures,
                    args.score)
