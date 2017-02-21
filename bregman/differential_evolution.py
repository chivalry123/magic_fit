"""
Global function optimization using differential evolution.
"""

from __future__ import print_function, division

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2015-09-21"
__version__ = "0.1"

import numpy as np
import sys


class Evolution(object):
    """
    Arguments:
      func                   function func(x) to be minimized
      x0 (array)             initial guess for optimal x
      bounds (list)          tuple (min, max) for each component of x
      population (int)       GA population
      amplitude (float)      magnitude of variations to create population
      crossing_rate (float)  average fraction of components of x that
                             undergo crossing upon every iteration
      diff_weight (float)    weight of the difference occuring in the
                             differential evolution crossing

    """

    def __init__(self, func, x0, bounds=None, population=10,
                 amplitude=0.1, crossing_rate=0.9, diff_weight=0.8):
        self.func = func
        self.dim = len(x0)
        self.x0 = np.asarray(x0)
        self.f0 = self.func(self.x0)
        self.bounds = bounds
        self.population = population
        self.amplitude = amplitude

        self.crossing_rate = crossing_rate
        self.diff_weight = diff_weight

        self._generate_trials()

        self.best_x = self.x0
        self.best_f = self.f0

    def __str__(self):
        return

    def _generate_trials(self):
        """
        Generate initial population by randomixing the initial guess vector.

        """
        self.trial = [self.x0]
        self.fitness = [self.f0] + (self.population - 1)*[None]
        shift = np.zeros(self.dim)
        amp = np.ones(self.dim)*self.amplitude
        if self.bounds is not None:
            for i in range(self.dim):
                lower = max(self.bounds[i][0], self.x0[i] - self.amplitude)
                upper = min(self.bounds[i][1], self.x0[i] + self.amplitude)
                amp[i] = (upper - lower)*0.5
                shift[i] = (lower + upper)*0.5 - self.x0[i]
        for i in range(self.population - 1):
            r = 2.0*(np.random.random(self.dim) - 0.5)
            x = self.x0.copy() + shift + r*amp
            self.trial.append(x)
        self._eval_fitness()

    def _eval_fitness(self):
        """
        Make sure that the fitness of all current trials is known.
        """
        for i in range(self.population):
            if self.fitness[i] is None:
                self.fitness[i] = self.func(self.trial[i])

    def _mate(self):
        """
        Generate new generation of trials.
        """
        for a in range(self.population):
            # select 3 trials from current generation
            b, c, d = np.argsort(np.random.random(self.population - 1))[:3]
            if b >= a:
                b += 1
            if c >= a:
                c += 1
            if d >= a:
                d += 1
            t0 = self.trial[a]
            t1 = self.trial[b]
            t2 = self.trial[c]
            t3 = self.trial[d]
            new_trial = t0.copy()
            for i in np.argsort(np.random.random(self.dim)):
                r = np.random.random()
                if r < self.crossing_rate:
                    new_trial[i] = t1[i] + self.diff_weight*(t2[i] - t3[i])
            if self.bounds is not None:
                for i in range(self.dim):
                    new_trial[i] = max(self.bounds[i][0], new_trial[i])
                    new_trial[i] = min(self.bounds[i][1], new_trial[i])
            new_fitness = self.func(new_trial)
            if new_fitness <= self.fitness[a]:
                self.trial[a] = new_trial
                self.fitness[a] = new_fitness
                if new_fitness <= self.best_f:
                    self.best_f = new_fitness
                    self.best_x = new_trial

    def optimize(self, maxiter=100, maxfeval=None, verbose=False,
                 output_file=sys.stdout):
        """
        Run optimization.

        Argument:
          maxiter (int)         max. number of iterations
          maxfeval (int)        max. number of function evaluations
                                = maxiter/population size
          verbose (bool)        if True, print message upon completion
          output_file (filep)   file pointer or string with path to output
                                file (defaults to stdout)

        Returns:
          Tuple (best_x, best_f) where best_x is the best-so-far argument
          vector and best_f is f(best_x).

        """

        if isinstance(output_file, file):
            close_file = False
            fp = output_file
        elif output_file is None:
            close_file = False
            fp = None
        else:
            close_file = True
            fp = open(output_file, 'w')

        if maxfeval is not None:
            maxiter = min(maxiter, np.floor(maxfeval/self.population))

        current_best = self.best_f
        if fp is not None:
            fp.write("# iteration, current best, difference\n")
        for it in range(maxiter):
            self._mate()
            if fp is not None:
                fp.write(" {} {} {}\n".format(
                    it, self.best_f, self.best_f - current_best))
                fp.flush()
                current_best = self.best_f

        if verbose:
            print(" Function value after {} iterations: {}".format(
                maxiter, self.best_f))

        if close_file:
            fp.close()

        return (self.best_x, self.best_f)
