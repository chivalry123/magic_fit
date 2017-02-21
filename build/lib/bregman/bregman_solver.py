"""
This module implements the Split-Bregman algorithm for compressive
sensing.

Translated from Fortran 90 to Python by Alexander Urban (MIT DMSE).
Original Fortran code written by Vivuds Ozolins (UCLA Materials Science
& Eng).

REFERENCES:

The Bregman algorithm is described in W. Yin, S. Osher, D. Goldfarb, and
J. Darbon, "Bregman Iterative Algorithms for L1-Minimization with
Applications to Compressed Sensing," SIAM J. Img. Sci. 1, 143 (2008).

The split Bregman algorithm is described in T. Goldstein and S. Osher,
"The split Bregman method for L1 regularized problems", SIAM Journal on
Imaging Sciences Volume 2 Issue 2, Pages 323-343 (2009).

The conjugae gradient algorithm is described in S. Boyd and
L. Vandenberghe, "Convex Optimization" (Cambridge University Press,
2004).

Coded by Vidvuds Ozolins (UCLA Materials Science & Eng).
Last modified: June 19, 2012

"""

from __future__ import print_function, division

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2014-08-14"
__version__ = "0.1"

import numpy as np
import sys

CGtol = 0.1


def split_bregman(MaxIt, tol, mu, lmbda, A, f, u, verbose=False, silent=False):
    """Performs split Bregman iterations using conjugate gradients (CG)
    for the L2 minimizations and shrinkage for L1 minimizations.

       u = arg min_u { ||u||_1 + mu/2 ||Au-f||_2^2 }

    The algorithm is described in T. Goldstein and S. Osher, "The split
    Bregman method for L1 regularized problems", SIAM Journal on Imaging
    Sciences Volume 2 Issue 2, Pages 323-343 (2009).

    Arguments:
      MaxIt   Number of outer split Bregman loops
      tol     Required tolerance for the residual. The algorithm stops when
              tol > ||Au-f||_2 / ||f||_2
      mu      weight for the L1 norm of the solution
      lambda  weight for the split constraint (affects speed of convergence,
              not the result)
      A       sensing matrix A
      f       array with measured signal values (of length M)
      u       solution array (of length N)
      verbose if True, print additional information to stdout
      silent  if True, suppress warning even when not converged

    Returns:
      u       solution
    """

    # number of unknown expansion coefficients
    N = len(u)

    MaxCGit = max(10, int(N/2))
    crit1 = 1.0
    crit2 = 1.0

    uprev = np.empty(N)
    b = np.zeros(N)
    d = np.zeros(N)
    bp = np.empty(N)

    for k in range(1, MaxIt+1):
        uprev[:] = u[:]
        bp[:] = d[:] - b[:]
        u[:] = cg_min(A, f, bp, mu, lmbda, MaxCGit, CGtol*crit1, u,
                      verbose, silent)
        d[:] = b[:] + u[:]
        d = shrink(d, 1.0/lmbda)
        u_dot_u = np.dot(u, u)
        if abs(u_dot_u) < 1.0e-12:
            if not silent:
                sys.stderr.write(" Warning: Split Bregman failed.\n")
            break
        crit1 = np.sqrt(np.dot(u-uprev, u-uprev)/u_dot_u)
        crit2 = np.sqrt(np.dot(u-d, u-d)/np.dot(u, u))
        if verbose:
            print(" SplitBregman: it = {}, ".format(k)
                  + "||deltaU||/||U|| = {}, ".format(crit1)
                  + "||d-U||/||U|| = {}".format(crit2))
        if (crit1 <= tol):
            break
        b[:] += u[:] - d[:]

    if (crit1 > tol) and not silent:
        sys.stderr.write(
            " Warning: Did not reach prescribed accuracy in SplitBregman:\n")
        sys.stderr.write(
            ("          ||deltaU||/||U|| = {}, \n"
             "             ||d-U||/||U|| = {}\n").format(crit1, crit2))

    return u


def cg_min(A, f, b, mu, lmbda, MaxIt, gtol, u, verbose=False, silent=False):
    """
    Conjugate gradient routine to perform L2-based minimization of

        min_u { mu/2 ||Au-f||_2^2 + lambda/2 ||b-u||_2^2 }

    Algorithm is described in S. Boyd and L. Vandenberghe,
    "Convex Optimization" (Cambridge University Press, 2004).

    Inut parameters:
       A    - sensing matrix of dimensions (M,N)
       f    - values of measurements
       b    - vector enforcing the split-off L1 constraint
       mu   - weight of the L2 constraint on Au=f
       lambda - weight of the split-off constraint
       MaxIt  - max. number of CG iterations
       gtol - tolerance for the gradient; exit if gtol > ||grad||_2
       u    - starting guess for the solution

    Output parameter:
       u    - converged solution
    """

    # M - number of measurements
    # N - number of expansion coefficients
    N = len(u)
    M = len(f)

    p = np.empty(N)
    r = np.empty(N)
    rp = np.empty(N)
    x = np.empty(M)

    x[:] = np.dot(A, u) - f
    r[:] = -(mu*np.dot(np.asarray(A).T, x) - lmbda*(b - u))

    p[:] = r[:]
    delta = np.dot(r, r)
    if verbose:
        print(" Initial norm of gradient = {}".format(np.sqrt(delta)))

    for k in range(1, MaxIt+1):
        x = np.dot(A, p)
        rp = mu*np.dot(np.asarray(A).T, x) + lmbda*p
        p_dot_rp = np.dot(p, rp)
        if abs(p_dot_rp) < 1.0e-12:
            if not silent:
                sys.stderr.write(
                    " Warning: Conjugate Gradient optimization failed.\n")
            break
        alpha = delta/p_dot_rp
        u += alpha*p
        r -= alpha*rp
        deltaprev = delta
        delta = np.dot(r, r)

        if (np.sqrt(delta) < gtol):
            break

        beta = delta/deltaprev
        p = beta*p + r

    if (np.sqrt(delta) > gtol) and not silent:
        sys.stderr.write(" Warning: Unconverged gradient after CGmin = "
                         "{}\n".format(np.sqrt(delta)))

    return u


def shrink(u, alpha):
    """
    L1-based shrinkage
    """
    return np.sign(u)*np.maximum(abs(u)-alpha, 0.0)
