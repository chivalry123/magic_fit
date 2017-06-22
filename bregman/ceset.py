"""
Interface class that provides access to cluster expansion data and
operations on it.

"""

from __future__ import print_function, division

import random
import os
import re
import sys
import numpy as np
from  pprint import pprint
from scipy.spatial import ConvexHull
from sets import Set
from bregman import split_bregman
from bregman.differential_evolution import Evolution
from bregman.pca import PrincipalComponentAnalysis

from cvxopt import matrix
from cvxopt import solvers
from cvxopt.modeling import variable
from cvxopt.modeling import op

from fractions import Fraction

from datetime import datetime

# from cvxopt import normal, uniform
# from cvxopt.modeling import variable, dot, op, sum
import cvxopt

__author__ = "Alexander Urban,Wenxuan Huang"
__email__ = "alexurba@mit.edu, key01027@mit.edu"
__date__ = "2015-12-09"
__version__ = "0.1"


EPS = np.finfo(float).eps




def tuple_all_strict_left_larger_than_right(tuple_a,tuple_b):
    tuple_a_np=np.array(tuple_a)
    tuple_b_np=np.array(tuple_b)
    assert len(tuple_a_np)==len(tuple_b_np)
    logic=(tuple_a_np>tuple_b_np).all()
    return logic



class CESet(object):

    def __init__(self, energy, energy_shift, correlations, directories,
                 concentrations, pca=False):
        """
        Arguments:
          energy (array)          energies of all reference configurations
          energy_shift (float)    energies have been shifted by this value
          correlations (ndarray)  cluster correlations of all configurations
          directories (list)      paths to the directories with the data
                                  of all configurations
          concentrations (list)   each item is a list of concentrations of
                                  all species in the CE
          pca (bool)              perform principal component analysis of
                                  the cluster correlations
        """
        random.seed(42)
        self.already_compute_special_decomposition_data=False
        solvers.options['show_progress'] = False
        # solvers.options['abstol'] = 1e-12
        # solvers.options['reltol'] = 1e-10
        # solvers.options['feastol'] = 1e-8
        self.already_compute_decomposition_data=False
        self.energy_in = energy
        self.energy_shift = energy_shift
        if pca:
            self.pca = PrincipalComponentAnalysis(correlations)
            self.correlations_in = self.pca.transformed_data
        else:
            self.pca = None
            self.correlations_in = correlations
        self.structure_directory = directories
        self.concentrations = concentrations
        self.dimension = len(self.concentrations[0])

        self.rms_distance = self.read_rms_distance(self.structure_directory)

        self.structure_weight = np.ones(self.N_struc)
        self.ignored_structure = np.array([])

        self.ecis = np.zeros(self.N_corr)

        self.simplified = False
        self.all_ecis = None
        self.ce_energies_all_ecis = None
        self.ce_errors_all_ecis = None

        # compute formation energy convex hull and energy above hull of
        # all configurations based on reference energies
        # print("self.energy_in")
        # print(self.energy_in)
        # print("self.energy_shift")
        # print(self.energy_shift)

        E = self.energy_in + self.energy_shift
        (self.formation_energies_in, self.formation_energy_basis_in,
         self.formation_energy_levels_in
         ) = self.compute_formation_energies(
             self.concentrations, E, return_reference=True)
        if self.formation_energy_basis_in is not None:
            (self.hull_in, self.energy_above_hull_in
             ) = self.compute_convex_hull(self.concentrations,
                                          self.formation_energies_in)
        # Wenxuan just to make the hull forgetting structs with positive formation E

            # print ("first time done")
            (self.hull_in, self.energy_above_hull_in
             ) = self.compute_convex_hull(self.concentrations,
                                          self.formation_energies_in,Eah_to_remove_positive=self.energy_above_hull_in,output_ref=False)

        else:
            self.hull_in = None
            self.energy_above_hull_in = self.formation_energies_in[:]

        self.formation_energies_ce = None
        self.hull_ce = None
        self.energy_above_hull_ce = None

        self.hull_ce_wrt_input_GS = None
        self.energy_above_hull_ce_wrt_input_GS = None
        self.eci_cutoff=None   ## Wenxuan added
        self.compute_hull_idx()

        # print("special debug mode 2017-02-01:6:33pm")
        # debug_now = True
        # if debug_now:
        #     index = range(len(self.concentrations))
        #     total_zip = zip(self.concentrations,index,self.energy_in,self.formation_energies_in,self.energy_above_hull_in)
        #     total_zip.sort()
        #     print("total_zip is")
        #     pprint(total_zip,width=130)
        #
        #     hull_idx_another_approach=self.compute_hull_idx()
        #     hull_idx=hull_idx_another_approach
        #     hull_conc=[self.concentrations[i] for i in hull_idx]
        #     hull_form_e=[self.formation_energies_in[i] for i in hull_idx]
        #     hull_e_above_hull_in=[self.energy_above_hull_in[i] for i in hull_idx]
        #     hull_zip=zip(hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)
        #     hull_zip.sort()
        #     (hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)=zip(*hull_zip)
        #
        #     print("hull_zip is")
        #     pprint(hull_zip)



    def __str__(self):
        return

    @property
    def N_struc(self):
        """ Number of reference structures """
        return np.asarray(self.correlations_in).shape[0]

    @property
    def N_corr(self):
        """ Number of clusters/correlations """
        return np.asarray(self.correlations_in).shape[1]

    @property
    def N_ignored(self):
        """ Number of ignored reference structures """
        return len(self.ignored_structure)

    @property
    def N_nonzero(self):
        """ Number of non-zero ECIs """
        return len(self.nonzero_ECIs())

    def read_rms_distance(self, structure_directory):
        """
        Read root mean squared displaceements of relaxed structures from
        ideal structures if present in structure directory.

        Arguments:
          structure_directory[i] (str)    Path to the directory of the
                                          i-th structure

        Returns:
          rms_dist[i]  (float or None)    RMSD of the i-th structure or
                                          None if no RMSD file was found

        """
        rms_dist = []
        for d in structure_directory:
            try:
                with open(d + os.sep + 'RMSD', 'r') as fp:
                    try:
                        rmsd = map(float, fp.readline(). split())
                        if rmsd == []:
                            rmsd = None
                    except ValueError:
                        rmsd = [100.0, None]
                    rms_dist.append(rmsd)
            except IOError:
                rms_dist.append(None)
        return rms_dist

    def compute_structure_weights(self, max_rmsd, bias_low_energy=None,
                                  bias_high_energy=None,
                                  bias_stable=False,
                                  low_energy_bias=2.0,
                                  high_energy_bias=0.1,
                                  ground_state_bias=4.0,
                                  give_hypothetical_structure_trivial_weights=True,
                                  concentrationmin=None,concentrationmax=None):

        """
        Compute a weight for each input structure depending on its RMSD and
        energy.

        Arguments:
          max_rmsd (float)    Ignore structures with RMSD values greater
                              than this number.
          bias_low_energy (float) Increase the weight for structures whose
                              energies are less than this value above
                              the hull.
          bias_high_energy (float) Decrease the weight for structures whose
                              energies are more than this value above
                              the hull.
          low_energy_bias (float)  Weight multiple to be applied to structures
                              within the energy range defined with
                              'bias_energy'.
          bias_stable (bool)  If true, bias all ground states (i.e.,
                              configurations that are on the convex hull)
          ground_state_bias (float)  Weight multiple to be applied to
                              structures selected with 'bias_stable'.

        Sets:
          self.structure_weight[i] (float)
                              weight of the i-th structure
          self.ignored_structure[j] (int)
                              index of the j-th structure that will be ignored
                              due to the height of its RMSD
        """

        Eah = np.array(self.energy_above_hull_in)

        w = np.ones(self.N_struc)
        ignore = []
        for (i, rmsd) in enumerate(self.rms_distance):
            if rmsd is not None:
                if rmsd[0] > max_rmsd:
                    w[i] = 0.0
                    ignore.append(i)
        ignore = np.array(ignore)

        if bias_stable:
            w[:] = np.where(abs(Eah) <= 1.0e-6, w*ground_state_bias, w)

        if bias_low_energy is not None:
            w[:] = np.where(Eah <= bias_low_energy, w*low_energy_bias, w)

        if bias_high_energy is not None:
            w[:] = np.where(Eah > bias_high_energy, w*high_energy_bias, w)

        if self.LinearScalingToUnweightHighEahStructs is not None:
            for i in range(len(self.structure_directory)):
                if Eah[i] > self.LinearScalingToUnweightHighEahStructs:
                    w[i] = w[i]*((self.LinearScalingToUnweightHighEahStructs/Eah[i])**2)

        if give_hypothetical_structure_trivial_weights:
            for i in range(len(self.structure_directory)):
                if "enum-hypo-" in self.structure_directory[i]:
                    w[i]=w[i]*1e-15


        if concentrationmin is not None :
            for i in range(len(self.structure_directory)):
                if tuple_all_strict_left_larger_than_right(concentrationmin-1e-3,self.concentrations[i]):
                    w[i]=w[i]*1e-15

        if concentrationmax is not None:
            for i in range(len(self.structure_directory)):
                if tuple_all_strict_left_larger_than_right(self.concentrations[i],concentrationmax+1e-3):
                    w[i]=w[i]*1e-15


            # for i in range(len(w)):
            #     if self.structure_directory

        self.structure_weight = w
        self.ignored_structure = ignore

        self._apply_structure_weights()




    def read_structure_weights(self, weights_file):
        """
        Read structure weights for the ECI fit from a file.

        Arguments:
          weights_file (str)  Path to the file containing the
                              structure weights.

        Sets:
          self.structure_weight[i] (float)
                              weight of the i-th structure
          self.ignored_structure[j] (int)
                              index of the j-th structure that will be
                              ignored due to the height of its RMSD
        """
        w = np.zeros(self.N_struc)
        ignore = []
        istruc = 0
        comment = re.compile(r"^ *#")
        with open(weights_file, 'r') as fp:
            for line in fp:
                if not comment.match(line):
                    i = int(line.split()[0])
                    w[i] = float(line.split()[1])
                    if w[i] == 0.0:
                        ignore.append(i)
                    istruc += 1
        if istruc != self.N_struc:
            raise IOError("Incompatible weights file: "
                          "{}".format(weights_file))

        self.structure_weight = w
        self.ignored_structure = ignore

        self._apply_structure_weights()

    def save_structure_weights(self, weights_file):
        """
        Write structure weights for the ECI fit to an output file.

        Arguments:
          weights_file (str)  Path to the output file.
        """
        w = self.structure_weight
        En = self.energy_in
        Eah = self.energy_above_hull_in
        rms_dist = self.rms_distance
        dirs = self.structure_directory
        with open(weights_file, 'w') as fp:
            fp.write("#     wght      energy       above hull "
                     " rms distance  directory\n")
            for i in range(self.N_struc):
                if rms_dist[i] is not None:
                    fp.write((" {:3d}  {:4.1f}  {:12.6f}  {:12.6f}  "
                              + "{:12.6f}  {}\n").format(
                                  i, w[i], En[i], Eah[i], rms_dist[i][0],
                                  dirs[i]))
                else:
                    fp.write((" {:3d}  {:4.1f}  {:12.6f}  {:12.6f}      "
                              "N/A       {}\n").format(
                                  i, w[i], En[i], Eah[i], dirs[i]))

    def _apply_structure_weights(self, selected_clusters=None):
        """
        Transform structure energies and correlation matrix according to the
        structure weights.

        Arguments:
          selected_clusters (list)   List of indices of selected clusters;
                                     correlations belonging to other clusters
                                     will be set to 0

        Sets:
          self._energy_w
          self._correlations_w
        """

        self.correlations_w = (self.correlations_in.T*self.structure_weight).T
        self.energy_w = self.energy_in * self.structure_weight
        if selected_clusters is not None:
            idx_del = [i for i in range(self.N_corr) if i not in
                       selected_clusters]
            self.correlations_w[:, idx_del] = 0.0

    def optimize_structure_weights(self, maxiter_opt=100, verbose=False,
                                   maxiter=100, tol=1.0e-3, mu=0.005,
                                   lmbda=1000, simplify=None, refit=False):
        """
        Optimize the structure weights by minimizing an ad-hoc error
        function based on the energy above hull.

        """

        class ErrorFunction(object):

            progress_sym = ">"
            progress_width = 70

            def __init__(self, ce, max_calls):
                """
                Arguments:
                  ce         the parent cluster expansion object
                  max_calls  max. number of calls (for progress bar)
                """
                self.iteration = 0
                self.ce = ce
                self.progess_calls = max_calls
                self.progress_update = self.progress_width/max_calls
                self.progress_status = 0.0
                print(" " + self.progress_width*"-")
                print("  {:3d}%".format(0), end="")

            def _progress(self):
                self.iteration += 1
                self.progress_status += self.progress_update
                print("\b\b\b\b\b", end="")
                while self.progress_status >= 1.0:
                    print(self.progress_sym, end="")
                    self.progress_status -= 1.0
                percentage = (100*self.iteration) // self.progess_calls
                print(" {:3d}%".format(percentage), end="")
                sys.stdout.flush()

            def close(self):
                print("")

            def __call__(self, weights):
                self._progress()
                initial_weights = self.ce.structure_weight[:]
                self.ce.structure_weight[:] = weights[:]
                self.ce._apply_structure_weights()
                if simplify is not None:
                    self.ce.simplify(target_rmse=simplify,
                                     maxiter=maxiter, tol=tol, mu=mu,
                                     lmbda=lmbda, verbose=False,
                                     refit=refit)
                else:
                    self.ce.compute_ECIs(
                        maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda)
                self.ce._update_ce_hull()
                E_ce = self.ce.formation_energies_ce
                E_in = self.ce.formation_energies_in
                error = 0.0
                for i, Eah_in in enumerate(self.ce.energy_above_hull_in):
                    # energy difference scaled to 1/2 of the E above hull
                    E_diff = (E_in[i] - E_ce[i])/max(0.5*Eah_in, 5.0e-3)
                    n = 6
                    a = (1.0/(1.0 + (4.0/9.0)**(n-1.0)))**(1.0/(2.0*n))
                    b = (1.0 - a**(2.0*n))/a**2
                    error += b*(a*E_diff)**2
                    error += (a*E_diff)**(2.0*n)
                    E_diff = (E_in[i] - E_ce[i])/max(0.5*Eah_in, 1.0e-3)
                    if (E_diff>= 1):
                        error+=(E_diff-1)*1e6

                self.ce.structure_weight[:] = initial_weights[:]
                return error

        popsize = 10
        errfunc = ErrorFunction(self, maxiter_opt*popsize + popsize)
        w0 = self.structure_weight[:]
        E0 = errfunc(w0)
        bounds = len(w0)*[(0.1, 100)]
        de = Evolution(errfunc, w0, bounds=bounds, amplitude=0.5,
                       population=popsize)
        (w_opt, E_opt) = de.optimize(
            maxiter=maxiter_opt, output_file="opt.dat")
        errfunc.close()
        if verbose:
            print(" Initial and final error: {} {} ({})".format(
                E0, E_opt, E_opt - E0))

        self.structure_weight = w_opt.copy()
        self._apply_structure_weights()
        if simplify is not None:
            self.simplify(target_rmse=simplify,
                          maxiter=maxiter, tol=tol, mu=mu,
                          lmbda=lmbda, verbose=False, refit=refit)
        else:
            self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda)
        self._update_ce_hull()

    def nonzero_ECIs(self, eci_cutoff=EPS, ecis=None):
        """
        List of indices of all clusters with non-zero ECIs.

        Arguments:
          eci_cutoff (float)     ECIs with absolute values below this
                                 threshold will be set to zero.

        Returns:
          An np.ndarray with cluster indices

        """
        if ecis is None:
            ecis = self.ecis
        nonzero = []
        for i, j in enumerate(ecis):
            if abs(j) >= eci_cutoff:
                nonzero.append(i)
        return np.array(nonzero)

    def save_ECIs(self, eci_file, eci_cutoff=EPS):
        """
        Writes effective cluster interactions (ECIs) to an output file.

        Arguments:
          eci_file (str)     Path to the output file.
          eci_cutoff (float) ECIs smaller than this value will be set to zero

        """
        ecis_local = self.ecis.copy()
        if self.pca is not None:
            ecis_local = np.dot(self.pca.eigenvectors.T, ecis_local)
        ecis_local[0] += self.energy_shift
        with open(eci_file, 'w') as fp:
            for i in range(self.N_corr):
                if (abs(self.ecis[i]) >= eci_cutoff):
                    fp.write(
                        "{:6d}   {:14.10f}\n".format(i+1, ecis_local[i]))
                else:
                    fp.write("{:6d}   {:14.10f}\n".format(i+1, 0.0))

    def read_ECIs(self, eci_file):
        """
        Read effective cluster interactions (ECIs) from a file.

        Arguments:
          eci_file (str)      Path to the file containing the ECIs

        Sets.
          self.ecis
        """
        ecis = np.loadtxt(eci_file, usecols=(1,))
        if len(ecis) != self.N_corr:
            raise IOError("Incompatible ECI file.")
        if self.pca is not None:
            self.ecis[:] = np.dot(self.pca.eigenvectors_inv, ecis)
        else:
            self.ecis[:] = ecis[:]

    def compute_ECIs(self, maxiter=100, tol=1.0e-3, mu=0.005, lmbda=1000,
                     eci_cutoff=EPS, silent=False):
        """
        Use compressive sensing to compute the effective cluster
        interactions.

        Arguments:
          maxiter (int)    Maximum number of iterations
          tol (float)      Convergence threshold
          mu (float)       Data noise parameter in energy units
          lmbda (float)    Split Bregman lambda parameter
          eci_cutoff (float)  All ECIs with absolute values below this
                           threshold will be set to zero.
          silent           if True, suppress warnings

          'mu' weights the sparsity of the solution vector against the
          error of the solution.  The larger 'mu' the sparser the solution
          vector.

          'lambda' does not affect the solution, but may have an impact on
          the convergence of the algorithm.  Typically, values between 100
          and 1000 work well.

        Sets:
          self.ecis

        """
        self.ecis = np.zeros(self.N_corr)
        self.ecis[:] = split_bregman(maxiter, tol, mu, lmbda,
                                     self.correlations_w, self.energy_w,
                                     self.ecis, silent=silent)
        idx_del = [i for i in range(self.N_corr)
                   if i not in self.nonzero_ECIs(eci_cutoff=eci_cutoff)]
        self.ecis[idx_del] = 0.0

    def compute_ce_energies(self):
        """
        Return the cluster expansion energy based of all structures based on
        the current ECIs.

        Returns:
          ce_energy[i] (float)   CE energy of the i-th reference structure
        """
        return self.correlations_in.dot(self.ecis)

    def save_ce_energies(self, ce_energy_file, ce_ignored_file=None):
        """
        Write the CE energies for the current ECIs to an output file.

        Arguments:
          ce_energy_file (str)   Path to a new CE energy file.
          ce_ignored_file (str)  Path to the corresponding file for all
                                 structures that were not used for the
                                 ECI fit.

        """

        def print_selected(selected_structures, outfile):
            if len(selected_structures) == 0:
                return
            with open(outfile, 'w') as f:
                header = ("#       concentrations "
                          + (self.dimension - 1)*15*" "
                          + "      input           "
                          + "full CE         reduced CE     "
                          + "Error (full CE)  Error (red. CE) "
                          + "E above hull (input)\n")
                f.write(header)
                frmt = "{:3d}  " + (self.dimension + 6)*"{:15.8f}  " + "\n"
                for i in range(self.N_struc):
                    if i in selected_structures:
                        f.write(frmt.format(
                            *([i] + conc[i] + [E_ref[i]+E_shift]
                              + [E_all[i]+E_shift] + [E_simp[i]+E_shift]
                              + [E_all[i]-E_ref[i]] + [E_simp[i]-E_ref[i]]
                              + [self.energy_above_hull_in[i]])))

        E_ref = self.energy_in
        E_shift = self.energy_shift
        if self.simplified:
            E_simp = self.compute_ce_energies()
            E_all = self.ce_energies_all_ecis
        else:
            E_all = self.compute_ce_energies()
            E_simp = E_all

        conc = self.concentrations

        selected = [i for i in range(self.N_struc)
                    if i not in self.ignored_structure]
        print_selected(selected, ce_energy_file)

        if ce_ignored_file is not None:
            print_selected(self.ignored_structure, ce_ignored_file)

    def compute_ce_errors(self, ce_energies=None, target_energies=None,concentrationmin=None,concentrationmax=None,weighted_rmse_no_hypo=False):
        """
        Return cluster expansion errors for the current set of ECIs.
        The structure weights are considered.

        Arguments:
          ce_energies[i] (float)      CE energy of the i-th structure;
                                      when not specified, the CE energies
                                      will be computed
          target_energies[i] (float)  Reference energy of the i-th structure;
                                      when not specified, the input
                                      energies will be used

        Returns:
          Tuple (rmse, mue, mse, rmse_noW) with
            rmse (float)      Root mean squared error
            mue (float)       Mean unsigned error
            mse (float)       Mean signed error
            rmse_noW (float)  RMSE evaluated without structure weights

        """
        if ce_energies is None:
            ce_energies = self.compute_ce_energies()
        if target_energies is None:
            target_energies = self.energy_in
            # target_energies = self.formation_energies_in

        norm = np.sum(self.structure_weight)
        errors = ce_energies - target_energies
        rmse = np.sqrt(np.sum((errors)**2*self.structure_weight)/norm)
        rmse_noW = np.sqrt(np.sum((errors)**2)/len(errors))
        mue = np.sum(np.abs(errors)*self.structure_weight)/norm
        mse = sum(errors*self.structure_weight)/norm

        valid_index= range(len(errors))
        # print ("valid_index is",valid_index)
        valid_index= [i for i in valid_index if "enum-hypo-" not in self.structure_directory[i]]
        # print ("valid_index is",valid_index)

        if concentrationmin is not None:
            valid_index=[i for i in valid_index if not tuple_all_strict_left_larger_than_right(concentrationmin-1e-3,self.concentrations[i])]
        # print ("valid_index is",valid_index)

        if concentrationmax is not None:
        # print ("self.concentrations is",self.concentrations)
            valid_index=[i for i in valid_index if not tuple_all_strict_left_larger_than_right(self.concentrations[i],concentrationmax+1e-3) ]

            valid_index=[i for i in valid_index if not  self.concentrations[i][0]>concentrationmax+1e-3]
        # print ("valid_index is",valid_index)


        errors_no_hypo=[errors[i] for i in valid_index ]

        # print ("errors_no_hypo is ",errors_no_hypo)

        errors_no_hypo_sq=[errors[i]**2 for i in valid_index ]

        rmse_no_hypo_no_weight=np.sqrt(np.sum((errors_no_hypo_sq))/len(errors_no_hypo_sq))

        if weighted_rmse_no_hypo:
            weight_no_hypo=[self.structure_weight[i] for i in valid_index]
            # weight_no_hypo=np.ones(len(weight_no_hypo))
            # print ("weight_no_hypo is, ",weight_no_hypo)
            #
            # print ("[target_energies[i] for i in valid_index] is, ",[target_energies[i] for i in valid_index])
            errors_no_hypo=np.array(errors_no_hypo);
            norm_no_hypo = np.sum(weight_no_hypo)
            rmse_no_hypo_weighted = np.sqrt(np.sum((errors_no_hypo)**2*weight_no_hypo)/norm_no_hypo)
            return (rmse, mue, mse, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted)
            1;



        return (rmse, mue, mse, rmse_noW,rmse_no_hypo_no_weight)







    def compute_cv_score(self, selected_clusters=None, maxiter=100,
                         tol=1.0e-3, mu=0.005, lmbda=1000,
                         cv_energy_file=None):
        """
        Compute the leave-one-out cross-validation score for the current
        cluster expansion Hamiltonian.

        Arguments:
          selected_clusters[i] (int)   Index of the i-th selected cluster;
                                       when not specified, use all clusters
                                       with non-zero ECI
          maxiter, tol, mu, lmbda --> see computeECIs()
          cv_energy_file (str)  Path to an output file to store the CE
                                energies computed without including the
                                structure

        Returns:
          Tuple (cv_ref, cv_all) with
              cv_ref (float)   cross-validation RMSE with respect to
                               reference energies
              cv_all (float)   cross-validation RMSE with respect to
                               cluster expansion fit using all structures

        """
        if selected_clusters is None:
            selected_clusters = self.nonzero_ECIs()
        # remove ignored structures (structures with weight==0)
        # to speed up the computation of the CV score
        A = np.delete(self.correlations_in, self.ignored_structure, axis=0)
        E = np.delete(self.energy_in, self.ignored_structure)
        w = np.delete(self.structure_weight, self.ignored_structure)
        idx_all = np.delete(np.arange(self.N_struc, dtype=int),
                            self.ignored_structure)
        (Nstruc, Ncorr) = A.shape
        idx_del = [i for i in range(Ncorr) if i not in selected_clusters]
        A[:, idx_del] = 0.0
        # fit ECIs using all structures
        Aw_all = (A.T * w).T
        Ew_all = E * w
        ecis_all = np.zeros(Ncorr)
        ecis_all[:] = 0.0
        ecis_all[:] = split_bregman(
            maxiter, tol, mu, lmbda, Aw_all, Ew_all, ecis_all)
        # CE energy from fit with all structures
        E_all = np.dot(A, ecis_all)
        # output file header (if output requested)
        if cv_energy_file is not None:
            fp = open(cv_energy_file, 'w')
            fp.write("#         reference        "
                     "included       not included\n")
        # compute CV score
        ecis = np.zeros(Ncorr)
        rmse_cv_ref = 0.0
        rmse_cv_all = 0.0
        norm = 0.0
        for i in range(Nstruc):
            A_cv = np.delete(A, [i], axis=0)
            E_cv = np.delete(E, [i])
            w_cv = np.delete(w, [i])
            Aw = (A_cv.T * w_cv).T
            Ew = E_cv * w_cv
            ecis[:] = 0.0
            ecis[:] = split_bregman(maxiter, tol, mu, lmbda, Aw, Ew, ecis)
            E_i = np.dot(A[i, :], ecis)
            # CV error with respect to reference energies
            rmse_cv_ref += w[i]*(E_i - E[i])**2
            # CV error with respect to fit using all structures
            rmse_cv_all += w[i]*(E_i - E_all[i])**2
            norm += w[i]
            if cv_energy_file is not None:
                fp.write("{:3d}  {:15.8f}  {:15.8f}  {:15.8f}\n".format(
                    idx_all[i], E[i]+self.energy_shift,
                    E_all[i]+self.energy_shift, E_i+self.energy_shift))
        rmse_cv_ref = np.sqrt(rmse_cv_ref/norm)
        rmse_cv_all = np.sqrt(rmse_cv_ref/norm)
        if cv_energy_file is not None:
            fp.close()
        return (rmse_cv_ref, rmse_cv_all)

    def compute_formation_energies(self, concentrations, energies,
                                   references=None,
                                   reference_energies=None,
                                   return_reference=False):
        """
        Compute formation energies relative to some reference
        concentrations.

        Arguments:
          concentrations (list) List of concentrations in the format of
                                self.concentrations.
          energies (list)       Energies of all configurations.
          references (list)     List of reference concentrations for the
                                calculation of the formation energy.
          reference_energies    Energies of all reference concentrations
          return_reference      if true, return final references and
                                reference energies

        Example:
          Given the binary system Li-CoO2---Vac-CoO2, reasonable reference
          concentrations would be [[0.0], [1.0]].
          For a ternary CE, say an A-B-C alloy with compositions A_x B_y C_z,
          the reference concentrations are vectors.  For example:
          references = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

        Returns:
          is (return_reference == False) just the array of formations
          energies is returned; otherwise, the tuple
            (formation_energies, references, reference_energies)
          is returned

        """
        if references is None:
            references = np.identity(self.dimension + 1)[:, :-1].tolist()

        if reference_energies is not None:
            E_basis = np.array(reference_energies)
        else:
            E_basis = []
            id_basis = []
            for k, c in enumerate(references):
                if c in concentrations:
                    conc = c
                else:
                    i = np.linalg.norm(np.array(concentrations)
                                       - np.array(c), axis=1).argmin()
                    conc = concentrations[i]
                    print(" No configuration found with "
                          "concentrations: " + (
                              self.dimension*"{} ").format(*c))
                    print(" Using instead reference "
                          "concentrations: " + (
                              self.dimension*"{} ").format(*conc))
                    references[k] = conc
                idx = np.array([j for j in range(len(concentrations))
                                if concentrations[j] == conc])
                i = np.asarray(energies)[idx].argmin()
                id_basis.append(i)
                E_basis.append(np.asarray(energies)[idx][i])
            E_basis = np.array(E_basis)
        basis = []
        for c in references:
            b = list(c[:])
            b.append(1.0 - np.sum(b))
            basis.append(b)
        basis = np.array(basis)
        try:
            basis_inv = np.linalg.inv(basis.T)
        except np.linalg.linalg.LinAlgError:
            basis_inv = None
            print(" Formation energies could not be computed.")
            print(" Do all configurations have the same concentrations?")
            formation_energies = energies[:]
            references = None
            E_basis = None
        if basis_inv is not None:
            formation_energies = []
            for i, conc in enumerate(concentrations):
                c = list(conc[:]) + [1.0 - np.sum(conc)]
                coeff = np.dot(basis_inv, c)
                Ef = energies[i] - np.sum(coeff*E_basis)
                formation_energies.append(Ef)

        if return_reference:
            return (formation_energies, references, E_basis)
        else:
            return formation_energies

    def compute_convex_hull_OLD(self, concentrations, energies):
        """
        Compute the convex hull of the formation energies to determine
        stable phases.

        Arguments:
          concentrations (list)   list of concentrations in the format of
                                  self.concentrations
          energies (list)         list of total energies (floats) for each
                                  of the concentrations

        Returns:
          (hull, E_above_hull)
          hull is a `hull' object created by scipy's ConvexHull() method
          E_above_hull is a list that contains the energy above the convex
          hull for all input concentrations

        """
        def hull_energy(simplexes, simplex_energies, conc):
            check_energies = []
            for i, S in enumerate(simplexes):
                E_S = simplex_energies[i]
                # check whether point (conc) is within simplex
                S0 = S - S[0]
                P0 = conc - S[0]
                try:
                    c = np.linalg.solve(S0[1:].T, P0)
                except np.linalg.linalg.LinAlgError:
                    # Some simplices may contain vertical edges/facets
                    # (edges in the direction of the energy axis).  When
                    # projected into concentration space, those
                    # simplices contain redundant vertices and are
                    # therefore not valid --> ignore singular matrices
                    continue
                if np.all(c >= 0.0) and np.sum(c) <= 1.0:
                    E0 = E_S - E_S[0]
                    E = E_S[0] + E0[1:].T.dot(c)
                    check_energies.append(E)
            return np.min(check_energies)
        points = np.concatenate((concentrations,
                                 np.array([energies]).T), axis=1)
        hull = ConvexHull(points)
        # projection to concentration space (discarding the energies)
        simplexes = [hull.points[s][:, :-1] for s in hull.simplices]
        # projection to energy space (discarding concentrations)
        simplex_energies = [hull.points[s][:, -1] for s in hull.simplices]
        print(np.array(simplexes))
        print(np.array(simplex_energies))
        exit()
        E_above_hull = []
        for i, conc in enumerate(concentrations):
            E = energies[i]
            E_hull = hull_energy(simplexes, simplex_energies, conc)
            E_above_hull.append(E - E_hull)
        return (hull, E_above_hull)

    def compute_convex_hull(self, concentrations, energies,Eah_to_remove_positive=None,output_ref=True):
        """
        Compute the convex hull of the formation energies to determine
        stable phases.

        Arguments:
          concentrations (list)   list of concentrations in the format of
                                  self.concentrations
          energies (list)         list of total energies (floats) for each
                                  of the concentrations

        Returns:
          (hull, E_above_hull)
          hull is a `hull' object created by scipy's ConvexHull() method
          E_above_hull is a list that contains the energy above the convex
          hull for all input concentrations

        """
        def hull_energy(simplices, conc):
            check_energies = []
            for i, S in enumerate(simplices):
                # check whether conc is a vertex of the simplex
                for v in S:
                    if np.linalg.norm(v[:-1] - conc) <= 1.0e-6:
                        check_energies.append(v[-1])
                # check whether point (conc) is within simplex
                A = np.zeros(S.shape)
                A[:, :-1] = (S[1:] - S[0]).T
                A[-1, -1] = -1.0
                b = -S[0]
                b[:-1] += conc
                try:
                    x = np.linalg.solve(A, b)
                except np.linalg.linalg.LinAlgError:
                    # If there is no solution, the simplex is irrelevant
                    continue
                if np.all(x[:-1] >= -1.0e-6) and np.sum(x[:-1]) - 1 <= 1.0e-6:
                    check_energies.append(x[-1])
            return np.min(check_energies)

        # Wenxuan just to make the hull forgetting structs with positive formation E
        # if hasattr(self,'energy_above_hull_in') and range(len(self.structure_directory))==range(len(self.energy_above_hull_in)):
        #     print ("I am at if hasattr(self,'energy_above_hull_in') and range(len(self.structure_directory))==range(len(self.energy_above_hull_in))")
        #
        #     temp_engr_input=[]
        #     for i in range(len(self.structure_directory)) :
        #         if  self.energy_above_hull_in>1e-4:
        #             temp_engr_input.append(self.formation_energies_in[i]-self.energy_above_hull_in[i]/2)
        #         else:
        #             temp_engr_input.append(self.formation_energies_in[i])
        #
        #
        #     points = np.concatenate((concentrations,
        #                              np.array([temp_engr_input]).T), axis=1)
        #
        # else:

        if Eah_to_remove_positive is not None:

            temp_engr_input=[]
            for i in range(len(Eah_to_remove_positive)) :
                if  Eah_to_remove_positive>1e-4:
                    temp_engr_input.append(energies[i]-Eah_to_remove_positive[i]/10*9)
                else:
                    temp_engr_input.append(energies[i])

            points = np.concatenate((concentrations,
                                     np.array([temp_engr_input]).T), axis=1)
        else:
            # print ("I am at else ansianw1241")
            points = np.concatenate((concentrations,
                                     np.array([energies]).T), axis=1)

        try:
            hull = ConvexHull(points)
        except:
            print("error in convex hull of points")
            print("let's see what is concentration and energy")
            zipped_conc_energ = zip(list(concentrations),list(energies))
            zipped_conc_energ.sort()
            pprint(zipped_conc_energ)
            self.convexhull_error = True
            # exit()
        E_above_hull = []
        for i, conc in enumerate(concentrations):
            E = energies[i]
            E_hull = hull_energy(hull.points[hull.simplices], conc)
            Eah = E - E_hull
            if Eah < 0.0:
                if Eah >= -1.0e-6:
                    Eah = 0.0
                else:
                    print("Error: negative energy above hull encountered:")
                    print(conc)
                    print(E, E_hull, Eah)
                    print(hull.vertices)
                    exit()
            E_above_hull.append(Eah)

        if hasattr(self,'energy_above_hull_in') and output_ref:

            # print ("I am printing concentrations_wrt_input_GS, energies_wrt_input_GS")
            concentrations_wrt_input_GS=[concentrations[i] for i in range(len(concentrations)) if self.energy_above_hull_in[i]<1e-5]
            energies_wrt_input_GS=[energies[i] for i in range(len(energies)) if self.energy_above_hull_in[i]<1e-5]
            # print (repr(concentrations_wrt_input_GS))
            # print (repr(energies_wrt_input_GS))


            points_wrt_input_GS = np.concatenate((concentrations_wrt_input_GS,
                                 np.array([energies_wrt_input_GS]).T), axis=1)
            hull_wrt_input_GS = ConvexHull(points_wrt_input_GS)
            E_above_hull_wrt_input_GS = []
            for i, conc in enumerate(concentrations):
                E = energies[i]
                E_hull_wrt_input_GS = hull_energy(hull_wrt_input_GS.points[hull_wrt_input_GS.simplices], conc)
                Eah_wrt_input_GS = E - E_hull_wrt_input_GS

                E_above_hull_wrt_input_GS.append(Eah_wrt_input_GS)

            return (hull, E_above_hull,hull_wrt_input_GS,E_above_hull_wrt_input_GS)
        else:
            return (hull, E_above_hull)



    def _update_ce_hull(self):
        """
        Sets

          self.formation_energies_ce
          self.hull_ce
          self.energy_above_hull_ce

        for present CE Hamiltonian.

        """
        conc = self.concentrations
        E_tot_ce = self.compute_ce_energies() + self.energy_shift
        E_ce = self.compute_formation_energies(
            self.concentrations, E_tot_ce,
            references=self.formation_energy_basis_in,
            reference_energies=self.formation_energy_levels_in)
        hull_ce, Eah_ce, hull_ce_wrt_input_GS, Eah_ce_wrt_input_GS = self.compute_convex_hull(conc, E_ce)

        self.formation_energies_ce = E_ce
        self.hull_ce = hull_ce
        self.energy_above_hull_ce = Eah_ce
        self.hull_ce_wrt_input_GS=hull_ce_wrt_input_GS
        self.energy_above_hull_ce_wrt_input_GS=Eah_ce_wrt_input_GS


    def save_convex_hull(self, hull_file):
        """
        Save formation energy convex hull to an output file.

        Arguments:
          hull_file (str)         Path to the output file

        """

        E_ref = self.formation_energies_in
        Eah_ref = self.energy_above_hull_in

        self._update_ce_hull()
        conc = self.concentrations
        E_ce = self.formation_energies_ce
        hull_ce = self.hull_ce
        Eah_ce = self.energy_above_hull_ce

        selected = np.array([i for i in range(self.N_struc)
                             if i not in self.ignored_structure])

        # sort entries by first concentration
        idx = np.argsort(np.array([c[0] for c in conc])[selected])
        with open(hull_file, 'w') as fp:
            fp.write("#ID       " + "concentrations "
                     + (17*(self.dimension - 1))*" "
                     + "  form. E (ref)    " + "form. E (CE)     "
                     + "above hull (ref) " + "above hull (CE)  "
                     + "hull (ref)       " + "hull (CE)" + "\n")
            frmt = ("{:3d}  " + (self.dimension + 4)*"{:15.8f}  "
                    + 2*"{:15s}  " + "\n")
            for i in selected[idx]:
                c = conc[i]
                values = [i] + c + [E_ref[i], E_ce[i], Eah_ref[i], Eah_ce[i]]
                if Eah_ref[i] > 1.0e-6:
                    values.append("     *")
                else:
                    values.append("{:15.8f}".format(E_ref[i]))
                if Eah_ce[i] > 1.0e-6:
                    values.append("     *")
                else:
                    values.append("{:15.8f}".format(E_ce[i]))
                fp.write(frmt.format(*values))

        # save tesselation for 3d plots for ternary cluster expansions
        if self.dimension == 2:
            def save_tesselation(fp, points, Eah):
                stable = points[np.array(Eah) <= 1.0e-6]
                hull = ConvexHull(stable)
                for S in hull.points[hull.simplices]:
                    for p in S:
                        fp.write((3*"{:10.6f} ").format(*p) + "\n")
                    fp.write((3*"{:10.6f} ").format(*S[0]) + "\n")
                    fp.write("\n\n")
            with open(hull_file + "-ce-hull", "w") as fp:
                save_tesselation(fp, hull_ce.points, Eah_ce)
            with open(hull_file + "-ref-hull", "w") as fp:
                save_tesselation(fp, self.hull_in.points, Eah_ref)



    def simplify(self, target_rmse, refit=True, maxiter=100, tol=1.0e-3,
                 mu=0.005, lmbda=1000, verbose=True):
        """
        Reduce the number of clusters in the cluster expansion by increasing
        the ECI cutoff until a target RMSE (with respect to the full CE)
        has been reached.  This is implemented simply by bracketing.

        The original ECIs will be saved as self.all_ecis .

        Arguments:
          target_rmse (float)  Target RMSE for the reduced cluster expansion
          refit (bool)         If True, refit ECIs with reduced set of clusters
          maxiter, tol, mu, lmbda --> see computeECIs()
          verbose              Print message when target RMSE was not achieved

        Sets:
          self.ecis
          self.all_ecis
          self.ce_energies_all_ecis
          self.ce_errors_all_ecis
          self.simplified
          self.cluster_index_simp
          self.eci_cutoff_simp

        """

        if not self.simplified:
            self.all_ecis = self.ecis.copy()
            self.ce_energies_all_ecis = self.compute_ce_energies()
            self.ce_errors_all_ecis = self.compute_ce_errors()
            self.simplified = True

        def reduce_CE(eci_cutoff):
            ecis = self.all_ecis
            sel = abs(ecis) >= eci_cutoff
            idx = np.arange(self.N_corr)[sel]
            ecis2 = np.zeros(self.N_corr)
            ecis2[idx] = ecis[idx]
            self.ecis[:] = ecis2[:]
            return idx

        t1 = 1.0
        t0 = 0.0
        (rmse_sel, mue_sel, mse_sel, rmse_noW) = self.compute_ce_errors(
            target_energies=self.ce_energies_all_ecis)
        rmse0 = rmse_sel
        idx = reduce_CE(t1)
        (rmse_sel, mue_sel, mse_sel, rmse_noW) = self.compute_ce_errors(
            target_energies=self.ce_energies_all_ecis)
        rmse1 = rmse_sel
        correlations_w_orig = self.correlations_w.copy()
        while True:
            t = 0.5*(t0+t1)
            idx = reduce_CE(t)
            if refit:
                idx_del = [i for i in range(self.N_corr) if i not in idx]
                self.correlations_w = correlations_w_orig.copy()
                self.correlations_w[:, idx_del] = 0.0
                self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu,
                                  lmbda=lmbda, silent=True)
            (rmse_sel, mue_sel, mse_sel, rmse_noW) = self.compute_ce_errors()
            if ((abs(rmse_sel - target_rmse) <= 1.0e-4)
                    or (rmse0 <= rmse1 <= target_rmse)):
                eci_cutoff = t
                break
            elif rmse_sel <= target_rmse:
                t0 = t
                rmse0 = rmse_sel
            else:
                t1 = t
                rmse1 = rmse_sel
            if (rmse0 == rmse1) or ((t1-t0) < 1.0e-6):
                if verbose:
                    print(" Warning: Unable to achieve target RMSE of "
                          "{}".format(target_rmse))
                eci_cutoff = t
                break

        if not refit:
            idx_del = [i for i in range(self.N_corr) if i not in idx]
            self.correlations_w = correlations_w_orig.copy()
            self.correlations_w[:, idx_del] = 0.0
            self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda)

        self.cluster_index_simp = idx
        self.eci_cutoff_simp = eci_cutoff



    def preserve_GS(self, maxiter_opt=100, verbose=False,
                                   maxiter=100, tol=1.0e-3, mu=0.005,
                                   lmbda=1000, simplify=None, refit=False,eci_cutoff=1e-8):
        """
        this GS preservation algorithm preservs GS in this way. At each iteration,
        fits are performed accroding to the weights, then we obtain energy_above_hull_ce and
        energy_above_hull_in. If for structure i, (energy_above_hull_in!=0 and energy_above_hull_ce=0)
        or (energy_above_hull_in=0 and energy_above_hull_ce!=0) and its fit is incorrect more than 10 meV, weight for
        structure i doubles....

        """

        print("Ground state preservation algorithm activated")
        if simplify is not None:
            self.simplify(target_rmse=simplify,
                          maxiter=maxiter, tol=tol, mu=mu,
                          lmbda=lmbda, verbose=False, refit=refit)
        else:
            self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda,eci_cutoff=eci_cutoff)
        self.compute_ce_energies()
        self._update_ce_hull()

        for iteration in range(maxiter_opt):
            # print(self.structure_weight);
            # print(self.structure_weight[0]);

            # print(self.energy_above_hull_ce)
            # print(self.energy_above_hull_in)
            break_out=True

            form_E_diff_max=0;
            for i in range(len(self.structure_weight)):
                if ((((self.energy_above_hull_in[i]<1e-5 and self.energy_above_hull_ce[i]>1e-3) or
                    (self.energy_above_hull_in[i]>1e-3 and self.energy_above_hull_ce[i]<1e-5)) ) ):
                    if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max:
                        form_E_diff_max=abs(self.formation_energies_ce[i]-self.formation_energies_in[i])
                    break_out=False

            for i in range(len(self.structure_weight)):
                if ((((self.energy_above_hull_in[i]<1e-5 and self.energy_above_hull_ce[i]>1e-3) or
                    (self.energy_above_hull_in[i]>1e-3 and self.energy_above_hull_ce[i]<1e-5))) ):
                    if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max*0.5:
                        print ("in iteration ",iteration," Updating weight of structure ",i)
                        self.structure_weight[i]=self.structure_weight[i]*5
                    break_out=False


            #print(self.structure_weight);

            self._apply_structure_weights();
            if simplify is not None:
                self.simplify(target_rmse=simplify,
                              maxiter=maxiter, tol=tol, mu=mu,
                              lmbda=lmbda, verbose=False, refit=refit)
            else:
                self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda,eci_cutoff=eci_cutoff)
            self.compute_ce_energies()
            self._update_ce_hull()


            if break_out:
                break



    def preserve_GS_range(self, verbose=False,
                                   maxiter=100, tol=1.0e-3, mu=0.005,
                                   lmbda=1000, simplify=None, refit=False,eci_cutoff=1e-8,
                                            preserve_ground_state_range=None,
                                            preserve_ground_state_range_iterations=None,
                                            give_hypothetical_structure_trivial_weights=None):
        """
        this GS preservation algorithm preservs GS in this way. At each iteration,
        fits are performed accroding to the weights, then we obtain energy_above_hull_ce and
        energy_above_hull_in. If for structure i, (energy_above_hull_in!=0 and energy_above_hull_ce=0)
        or (energy_above_hull_in=0 and energy_above_hull_ce!=0) and its fit is incorrect more than 10 meV, weight for
        structure i doubles....

        """
        maxiter_opt=preserve_ground_state_range_iterations
        print("Ground state range algorithm activated")
        if simplify is not None:
            self.simplify(target_rmse=simplify,
                          maxiter=maxiter, tol=tol, mu=mu,
                          lmbda=lmbda, verbose=False, refit=refit)
        else:
            self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda,eci_cutoff=eci_cutoff)
        self.compute_ce_energies()
        self._update_ce_hull()


        for iteration in range(maxiter_opt):
            # print(self.structure_weight);
            # print(self.structure_weight[0]);

            # print(self.energy_above_hull_ce)
            # print(self.energy_above_hull_in)
            break_out=True

            form_E_diff_max=0;
            weight_min=1e100;

            valid_list=[]

            for i in range(len(self.structure_weight)):
                if ((((self.energy_above_hull_in[i]<1e-5 and self.energy_above_hull_ce[i]>preserve_ground_state_range) or
                    (self.energy_above_hull_in[i]>1e-3 and self.energy_above_hull_ce_wrt_input_GS[i]<-preserve_ground_state_range) or (self.energy_above_hull_in[i]<1e-5 and abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>1) ) ) ):
                    if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max:
                        form_E_diff_max=abs(self.formation_energies_ce[i]-self.formation_energies_in[i])
                    if self.structure_weight[i]<weight_min:
                        weight_min=self.structure_weight[i]
                    valid_list.append(i)
                    break_out=False

            print ('the valid list is',repr(valid_list), 'the weight min now is ',repr(weight_min))

            for i in range(len(self.structure_weight)):
                if ((((self.energy_above_hull_in[i]<1e-5 and self.energy_above_hull_ce[i]>preserve_ground_state_range) or
                    (self.energy_above_hull_in[i]>1e-3 and self.energy_above_hull_ce_wrt_input_GS[i]<-preserve_ground_state_range) or (self.energy_above_hull_in[i]<1e-5 and abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>1))) ):
                    # if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max*0.01:
                    if self.structure_weight[i]<=weight_min*10:
                        print ("in iteration ",iteration," Updating weight of structure ",i)
                        self.structure_weight[i]=self.structure_weight[i]*4
                    break_out=False


            #print(self.structure_weight);

            self._apply_structure_weights();
            if simplify is not None:
                self.simplify(target_rmse=simplify,
                              maxiter=maxiter, tol=tol, mu=mu,
                              lmbda=lmbda, verbose=False, refit=refit)
            else:
                self.compute_ECIs(maxiter=maxiter, tol=tol, mu=mu, lmbda=lmbda,eci_cutoff=eci_cutoff)
            self.compute_ce_energies()
            self._update_ce_hull()


            if break_out:
                break



    def perform_weight_adjusting_algo_incremental(self,mu=10,concentrationmin=None,
                           concentrationmax=None):
        print("weight_adjusting algorithm QP activated")

        #FIXME:need to check if this works with hypothetical structure also!
        #FIXME: check if this works with out of concentration configuration also

        maxiter_opt=2000

        self.perform_QP(mu,concentrationmin,concentrationmax,activate_GS_preservation=False)
        self.compute_ce_energies()
        self._update_ce_hull()

        self.weight_algo_succeed=1
        for i in range(len(self.structure_weight)):
            if ((((self.energy_above_hull_in[i]<1e-6 and self.energy_above_hull_ce[i]>1e-6) or
                (self.energy_above_hull_in[i]>1e-6 and self.energy_above_hull_ce[i]<1e-6)) ) ):

                self.weight_algo_succeed=0


        for iteration in range(maxiter_opt):
            # print(self.structure_weight);
            # print(self.structure_weight[0]);
            # print(self.energy_above_hull_ce)
            # print(self.energy_above_hull_in)
            break_out=True

            form_E_diff_max=0;
            weight_min=1e100;

            self.add_concentration_min_max(concentrationmin,concentrationmax)
            self.decide_valid_lists()

            valid_list=[]


            print ("in iteration ",iteration)
            print("np.where(self.energy_above_hull_in<1e-6)[0] ")
            print(np.where(np.array( self.energy_above_hull_in )<1e-4)[0])

            print("np.where(self.energy_above_hull_ce<1e-6)[0]")
            print(np.where(np.array( self.energy_above_hull_ce)<1e-6)[0])



            problem_set=Set(np.where(np.array( self.energy_above_hull_in )<1e-4)[0])
            form_E_diff_max=0;
            for i in range(len(self.structure_weight)):
                if ((((self.energy_above_hull_in[i]<1e-6 and self.energy_above_hull_ce[i]>1e-6) or
                    (self.energy_above_hull_in[i]>1e-6 and self.energy_above_hull_ce[i]<1e-6)) ) ):
                    problem_set.add(i)

                    # if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max:
                    #     form_E_diff_max=abs(self.formation_energies_ce[i]-self.formation_energies_in[i])
                    break_out=False



            if break_out:
                self.weight_algo_succeed=1
                break

            form_E_diff_max=np.max([abs(self.formation_energies_ce[i]-self.formation_energies_in[i]) for i in problem_set])
            to_update_list=[]
            for i in problem_set:
                if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max*0.5:
                    to_update_list.append(i)
            print("to_update_list")
            print(to_update_list)

            for i in to_update_list:
                if self.structure_weight[i]<1:
                    self.structure_weight[i]=(self.structure_weight[i])*3
                else:
                    self.structure_weight[i]=(self.structure_weight[i])*1.4



            # for i in range(len(self.structure_weight)):
            #     if ((((self.energy_above_hull_in[i]<1e-6 and self.energy_above_hull_ce[i]>1e-6) or
            #         (self.energy_above_hull_in[i]>1e-6 and self.energy_above_hull_ce[i]<1e-6)) ) ):
            #         if abs(self.formation_energies_ce[i]-self.formation_energies_in[i])>form_E_diff_max*0.5:
            #             print ("in iteration ",iteration," Updating weight of structure ",i, " energy_above_hull_in[i]: ",
            #                    self.energy_above_hull_in[i]," self.energy_above_hull_ce[i] ",
            #                    self.energy_above_hull_ce[i]," self.formation_energies_ce[i] ",
            #                    self.formation_energies_ce[i]," self.formation_energies_in[i] ",
            #                    self.formation_energies_in[i])
            #             self.structure_weight[i]=(self.structure_weight[i])*1.4
            #         break_out=False


            #print(self.structure_weight);
            self._apply_structure_weights();
            self.perform_QP(mu,concentrationmin,concentrationmax,activate_GS_preservation=False)
            self.compute_ce_energies()
            self._update_ce_hull()

            break_out_due_to_unreasonable_weight=False
            print("max(self.structure_weight) now is ",max(self.structure_weight))
            if max(self.structure_weight)>1e10:
                break_out_due_to_unreasonable_weight=True

            if break_out_due_to_unreasonable_weight:
                self.weight_algo_succeed=0
                break






    def perform_QP(self,mu=10,concentrationmin=None,
                           concentrationmax=None,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=None):

        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b
        self.generate_QP_MIQP_matrix(mu,concentrationmin,
                   concentrationmax,activate_GS_preservation,AbsoluteErrorConstraintOnHull,MIQP=False)

        P = self.P
        q = self.q
        G_3 = self.G_3
        h_3 = self.h_3
        G3_without_preserve_GS = self.G3_without_preserve_GS
        h_3_without_preserve_GS = self.h_3_without_preserve_GS



        P_matrix=matrix(P)
        q_matrix=matrix(q)
        G_3_matrix=matrix(G_3)
        h_3_matrix=matrix(h_3)

        # print (repr(hull_zip))

        sol = solvers.qp(P_matrix,q_matrix,G_3_matrix,h_3_matrix)
        if sol["status"]!="optimal":
            print("*"*1000)
            print("this subsystem's GS cannot be conserved (mathematically impossible)")
            print("*"*1000)
            G3_without_preserve_GS_matrix=matrix(G3_without_preserve_GS)
            h_3_without_preserve_GS_matrix = matrix(h_3_without_preserve_GS)
            sol = solvers.qp(P_matrix,q_matrix,G3_without_preserve_GS_matrix,h_3_without_preserve_GS_matrix)


        self.ecis = np.zeros(self.N_corr)
        self.ecis[0:self.N_corr]=sol['x'][0:self.N_corr].T

        if self.eci_cutoff is not None:
            idx_del = [i for i in range(self.N_corr)
               if i not in self.nonzero_ECIs(eci_cutoff=self.eci_cutoff)]
            self.ecis[idx_del] = 0.0

        self.compute_ce_energies()
        self._update_ce_hull()



    def perform_MIQP(self,mu=10,concentrationmin=None,
                           concentrationmax=None,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=None):
        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b
        import gurobi

        self.generate_QP_MIQP_matrix(mu,concentrationmin,
                           concentrationmax,activate_GS_preservation,AbsoluteErrorConstraintOnHull,MIQP=True)
        P = self.P
        q = self.q
        G_3 = self.G_3
        h_3 = self.h_3
        G3_without_preserve_GS = self.G3_without_preserve_GS
        h_3_without_preserve_GS = self.h_3_without_preserve_GS
        binary_list = self.binary_list

        # print("in performing MIQP ")
        # print("P.shape")
        # print(P.shape)

        # print (repr(hull_zip))

        # sol = solvers.qp(P_matrix,q_matrix,G_3_matrix,h_3_matrix)

        # sol = solvers.qp(P,q,G_3,h_3)
        sol = self.solve_MIQP_matrix(P,q,G_3,h_3,binary_list)

        if sol["status"]!="optimal" and sol["status"]!="timelimit":
            print("*"*1000)
            print("this subsystem's GS cannot be conserved (mathematically impossible)")
            print("*"*1000)
            # sol = solvers.qp(P,q,G3_without_preserve_GS,h_3_without_preserve_GS)
            sol = self.solve_MIQP_matrix(P,q,G3_without_preserve_GS,h_3_without_preserve_GS,binary_list)

        self.ecis = np.zeros(self.N_corr)
        self.ecis[0:self.N_corr]=sol['x'][0:self.N_corr]

        if self.eci_cutoff is not None:
            idx_del = [i for i in range(self.N_corr)
               if i not in self.nonzero_ECIs(eci_cutoff=self.eci_cutoff)]
            self.ecis[idx_del] = 0.0

        self.convexhull_error = False
        self.compute_ce_energies()
        self._update_ce_hull()


        if self.convexhull_error:
            print("ECIs is")
            print(self.ecis)


    def solve_MIQP_matrix(self,P_matrix,q_matrix,G_matrix,h_3_matrix,binary_list):
        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b

        # if len(binary_list)>10:
        #     import pickle
        #     filename='tmp.pkl'
        #     if os.path.exists(filename):
        #         os.remove(filename)
        #
        #     output = open(filename, 'wb')
        #     pickle.dump(P_matrix, output)
        #     pickle.dump(q_matrix, output)
        #     pickle.dump(G_matrix, output)
        #     pickle.dump(h_3_matrix, output)
        #     pickle.dump(binary_list, output)
        #     output.close()

        # try:
        #     time_limit = float(self.time_limit_miqp)
        # except:
        #     if G_matrix.shape[0]>G_matrix.shape[1]*2:
        #         time_limit = 300
        #     else:
        #         time_limit = 100


        if "prGS" in self.matrix_type:
            time_limit = self.MIQPGSPrsSolvingTime
        else:
            time_limit = self.MIQPNonGSPrsSolvingTime


        try:
            heuristics = self.heuristics
        except:
            heuristics = 1


        binary_list = set(binary_list)
        if len(q_matrix.shape)==2:
            q_matrix = q_matrix.T[0]
        if len(h_3_matrix.shape)==2:
            h_3_matrix = h_3_matrix.T[0]


        import gurobi
        model = gurobi.Model()

        # print("using Gurobi for MIQP matrix")

        vars = []
        cols = P_matrix.shape[1]

        for i in range(cols):
            if i not in binary_list:
               vars.append(model.addVar(lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY, vtype=gurobi.GRB.CONTINUOUS))
            else:
               vars.append(model.addVar(lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY, vtype=gurobi.GRB.BINARY))

        for i in range(G_matrix.shape[0]):
            expr = gurobi.LinExpr()
            for j in range(G_matrix.shape[1]):
                if G_matrix[i][j]!= 0:
                    pass
                    expr += G_matrix[i][j]*vars[j]
            model.addConstr(expr, gurobi.GRB.LESS_EQUAL, h_3_matrix[i])

        obj = gurobi.QuadExpr()
        for i in range(cols):
            for j in range(cols):
                if P_matrix[i][j] != 0:
                    obj += 0.5*P_matrix[i][j]*vars[i]*vars[j]
        for j in range(cols):
            if q_matrix[j] != 0:
                obj += q_matrix[j]*vars[j]
        obj += self.constant_term
        model.setObjective(obj)

        model.setParam(gurobi.GRB.Param.TuneTimeLimit, 3000)
        model.setParam(gurobi.GRB.Param.TuneCriterion, 2)
        model.setParam(gurobi.GRB.Param.TuneTrials, 2)


        model.setParam(gurobi.GRB.Param.OutputFlag, True)
        model.setParam(gurobi.GRB.Param.Heuristics, heuristics)
        model.setParam(gurobi.GRB.Param.MIPFocus, 1)
        model.setParam(gurobi.GRB.Param.PSDTol, 1e-2)

        # model.tune()

        model.setParam(gurobi.GRB.Param.TimeLimit, time_limit)


        model.optimize()

        sol = {}
        if model.status == gurobi.GRB.Status.OPTIMAL or model.status == gurobi.GRB.Status.TIME_LIMIT:
            if model.status == gurobi.GRB.Status.OPTIMAL:
                sol["status"] = "optimal"
            elif model.status == gurobi.GRB.Status.TIME_LIMIT:
                sol["status"] = "timelimit"
            x = model.getAttr('x', vars)
            sol['x'] = np.zeros(cols)
            for i in range(cols):
                sol['x'][i] = x[i]
        else:
            sol["status"] = model.status

        return sol


    def generate_QP_MIQP_matrix(self,mu=10,concentrationmin=None,
                           concentrationmax=None,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=None,MIQP=False,big_M=20.0):

        if MIQP==False:
            big_M = 1.0

        L0L1 = self.L0L1
        L0mu = self.L0mu
        # print("mu is ",mu)

        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b
        corr_in=np.array(self.correlations_in)
        engr_in=np.array(self.energy_in)
        engr_in.shape=(len(engr_in),1)

        weight_vec=np.array(self.structure_weight)

        weight_matrix= np.diag(weight_vec.transpose())

        P_corr_part=2*((weight_matrix.dot(corr_in)).transpose()).dot(corr_in)


        if len(self.diff_foscus_lists_of_lists)>0:

            P_diff_focus_part = 0
            q_diff_focus_part = 0
            for index,list_now in enumerate(self.diff_foscus_lists_of_lists):
                for i in range(1,len(list_now)):
                    corr_diff_focus_now = np.array([corr_in[list_now[i]]-corr_in[list_now[0]]]) #explicit row vector
                    assert corr_diff_focus_now.shape[0] == 1
                    energy_diff_now = engr_in[list_now[i]]-engr_in[list_now[0]]
                    P_diff_focus_part += 2*self.diff_foscus_weights_lists[index]*corr_diff_focus_now.transpose().dot(corr_diff_focus_now)
                    q_diff_focus_part += -2*self.diff_foscus_weights_lists[index]*corr_diff_focus_now.transpose()*energy_diff_now
            P_corr_part += P_diff_focus_part



        P=np.lib.pad(P_corr_part,((0,self.N_corr),(0,self.N_corr)),mode='constant', constant_values=0)
        # P_z_part=np.zeros(self.N_corr)

        q_corr_part=-2*((weight_matrix.dot(corr_in)).transpose()).dot(engr_in)

        # print("q_diff_focus_part.shape")
        # print(q_diff_focus_part.shape)

        if len(self.diff_foscus_lists_of_lists)>0:
            q_corr_part+=q_diff_focus_part

        q_z_part=np.ones((self.N_corr,1))*mu

        Dim = len(self.concentrations[0])
        count_of_two = 0
        min_clust_length = 0

        if not self.CompressAllTerms:
            for i in range(len(corr_in[0])):
                if self.cluster_size[i] < 2:
                   q_z_part[i]=0

                if not self.CompressFirstPair:
                    if self.cluster_size[i] == 2:
                       count_of_two += 1
                       if count_of_two == 1:
                           min_clust_length=self.cluster_length[i]
                       if self.cluster_length[i] < min_clust_length*(1+1e-4):
                           q_z_part[i]=0

                if self.UnCompressPairUptoDist is not None:
                    if self.cluster_size[i] == 2 and self.cluster_length[i]<=self.UnCompressPairUptoDist:
                       q_z_part[i]=0

        q=np.concatenate((q_corr_part,q_z_part),axis=0)
        # G_1=np.concatenate((np.identity(self.N_corr),-np.identity(self.N_corr)*big_M),axis=1)
        # G_2=np.concatenate((-np.identity(self.N_corr),-np.identity(self.N_corr)*big_M),axis=1)
        G_1=np.concatenate((np.identity(self.N_corr),-np.identity(self.N_corr)),axis=1)
        G_2=np.concatenate((-np.identity(self.N_corr),-np.identity(self.N_corr)),axis=1)
        G_3=np.concatenate((G_1,G_2),axis=0)
        G3_without_preserve_GS = G_3[:]
        h_3=np.zeros((2*self.N_corr,1))
        h_3_without_preserve_GS = h_3[:]

        small_error_global=self.SmallErrorOnInequality
        self.add_concentration_min_max(concentrationmin,concentrationmax)
        self.decide_valid_lists()


        if AbsoluteErrorConstraintOnHull is not None:
            hull_idx_another_approach=self.compute_hull_idx()
            hull_idx=hull_idx_another_approach
            for i in hull_idx:
                if i in self.valid_index:
                    # G_3_new_line=np.concatenate((self.correlations_in[global_index]-self.correlations_in[i],np.zeros((self.N_corr))))
                    G_3_new_line=np.concatenate((self.correlations_in[i],np.zeros(self.N_corr)))
                    G_3_new_line.shape=(1,2*self.N_corr)
                    G_3=np.concatenate((G_3,G_3_new_line),axis=0)
                    G3_without_preserve_GS=np.concatenate((G3_without_preserve_GS,G_3_new_line),axis=0)
                    form_E_Upper=np.array(self.formation_energies_in[i]+AbsoluteErrorConstraintOnHull)
                    form_E_Upper.shape=(1,1)
                    h_3=np.concatenate((h_3,form_E_Upper),axis=0)
                    h_3_without_preserve_GS=np.concatenate((h_3_without_preserve_GS,form_E_Upper),axis=0)


                    G_3_new_line=np.concatenate((-self.correlations_in[i],np.zeros(self.N_corr)))
                    G_3_new_line.shape=(1,2*self.N_corr)
                    G_3=np.concatenate((G_3,G_3_new_line),axis=0)
                    G3_without_preserve_GS=np.concatenate((G3_without_preserve_GS,G_3_new_line),axis=0)
                    form_E_Upper=np.array(-self.formation_energies_in[i]+AbsoluteErrorConstraintOnHull)
                    form_E_Upper.shape=(1,1)
                    h_3=np.concatenate((h_3,form_E_Upper),axis=0)
                    h_3_without_preserve_GS=np.concatenate((h_3_without_preserve_GS,form_E_Upper),axis=0)



        G_3_GS_preservation_part = np.array([])
        h_3_GS_preservation_part = np.array([])

        if activate_GS_preservation:

            hull_idx_another_approach=self.compute_hull_idx()
            hull_idx=hull_idx_another_approach
            hull_conc=[self.concentrations[i] for i in hull_idx]
            hull_form_e=[self.formation_energies_in[i] for i in hull_idx]
            hull_e_above_hull_in=[self.energy_above_hull_in[i] for i in hull_idx]
            hull_zip=zip(hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)
            hull_zip.sort()
            (hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)=zip(*hull_zip)
            # print("hull_zip is ")
            # pprint(hull_zip,width=200)

            self.decide_valid_lists()
            valid_index=self.valid_index
            valid_index_only_consider_conc=self.valid_index_only_consider_conc
            invalid_index_due_to_conc_min=self.invalid_index_due_to_conc_min
            invalid_index_due_to_conc_max=self.invalid_index_due_to_conc_max


            decomposition_data=self.compute_decomposition_data()
            special_decomposition_data=self.compute_special_decomposition_data()




            for i in valid_index_only_consider_conc:
                if i not in hull_idx:
                    # print (i,"is not in hull_idx")
                    decomposition_now=decomposition_data[i]
                    array_now = 0
                    for target_index,decomposition_value in decomposition_now.iteritems():
                        array_now += decomposition_value*self.correlations_in[target_index]
                    array_now -= self.correlations_in[i]
                    G_3_new_line=np.concatenate((array_now,np.zeros(self.N_corr)))
                    G_3_new_line.shape=(1,2*self.N_corr)
                    G_3=np.concatenate((G_3,G_3_new_line),axis=0)
                    if G_3_GS_preservation_part.size:
                        G_3_GS_preservation_part = np.concatenate((G_3_GS_preservation_part,G_3_new_line),axis=0)
                    else:
                        G_3_GS_preservation_part = G_3_new_line


                    small_error=np.array(-small_error_global) # from -4 to -3
                    small_error.shape=(1,1)
                    h_3=np.concatenate((h_3,small_error),axis=0)
                    if h_3_GS_preservation_part.size:
                        h_3_GS_preservation_part = np.concatenate((h_3_GS_preservation_part,small_error),axis=0)
                    else:
                        h_3_GS_preservation_part = small_error

            ## now ,we would only consider the situation where data out side the desired region is not
            ## considered at all in this GS preserving step

            for i in hull_idx:
                if i not in self.undecomposable_hull_idx:
                    special_decomposition_now=special_decomposition_data[i]
                    array_now = 0
                    for target_index,decomposition_value in special_decomposition_now.iteritems():
                        array_now -= decomposition_value*self.correlations_in[target_index]
                    array_now += self.correlations_in[i]
                    G_3_new_line=np.concatenate((array_now,np.zeros(self.N_corr)))
                    G_3_new_line.shape=(1,2*self.N_corr)
                    G_3=np.concatenate((G_3,G_3_new_line),axis=0)
                    G_3_GS_preservation_part = np.concatenate((G_3_GS_preservation_part,G_3_new_line),axis=0)

                    small_error=np.array(-small_error_global) # from -4 to -3
                    small_error.shape=(1,1)
                    h_3=np.concatenate((h_3,small_error),axis=0)
                    h_3_GS_preservation_part = np.concatenate((h_3_GS_preservation_part,small_error),axis=0)


        W_dot_E = weight_matrix.dot(engr_in)
        self.constant_term = W_dot_E.T.dot(engr_in)
        # print("self.constant_term")
        # print(self.constant_term)
        if MIQP:

            P = np.lib.pad(P,((0,self.N_corr),(0,self.N_corr)),mode='constant', constant_values=0)
            q_integer_part = np.ones((self.N_corr,1))*L0mu
            q = np.concatenate((q,q_integer_part),axis=0)

            # np.identity(self.N_corr)
            # np.zeros(self.N_corr,self.N_corr)
            # -np.identity(self.N_corr)*big_M

            G_1_integer=np.concatenate((np.identity(self.N_corr),np.zeros((self.N_corr,self.N_corr)),-np.identity(self.N_corr)*big_M),axis=1)
            G_2_integer=np.concatenate((-np.identity(self.N_corr),np.zeros((self.N_corr,self.N_corr)),-np.identity(self.N_corr)*big_M),axis=1)
            G_3_integer=np.concatenate((G_1_integer,G_2_integer),axis=0)
            h_3_integer=np.zeros((2*self.N_corr,1))


            if self.MaxNumClusts is not None:
                G_3_integer_new_line = np.concatenate((np.zeros((1,self.N_corr*2)),np.ones((1,self.N_corr))),axis=1)
                h_3_integer_new_line = np.ones((1,1))*self.MaxNumClusts
                G_3_integer = np.concatenate((G_3_integer,G_3_integer_new_line),axis=0)
                h_3_integer = np.concatenate((h_3_integer,h_3_integer_new_line),axis=0)

            if self.L0Hierarchy is not None:
                print("adding L0Hierarchy now")
                for child_idx, list_tmp in enumerate(self.cluster_hierarchy):
                    for parent_idx in list_tmp:
                        G_3_integer_new_line = np.zeros((1,self.N_corr))
                        G_3_integer_new_line[0][parent_idx] = -1
                        G_3_integer_new_line[0][child_idx] = 1
                        G_3_integer_new_line = np.concatenate((np.zeros((1,self.N_corr*2)),G_3_integer_new_line),axis=1)
                        h_3_integer_new_line = np.zeros((1,1))
                        G_3_integer = np.concatenate((G_3_integer,G_3_integer_new_line),axis=0)
                        h_3_integer = np.concatenate((h_3_integer,h_3_integer_new_line),axis=0)

            G3_without_preserve_GS = np.lib.pad(G3_without_preserve_GS,((0,0),(0,self.N_corr)),mode='constant', constant_values=0)
            G3_without_preserve_GS = np.concatenate((G3_without_preserve_GS,G_3_integer),axis=0)

            h_3_without_preserve_GS = np.concatenate((h_3_without_preserve_GS,h_3_integer),axis=0)

            G_3 = np.lib.pad(G_3,((0,0),(0,self.N_corr)),mode='constant', constant_values=0)

            # print(G_3.shape)
            # print(G_3_integer.shape)

            G_3 = np.concatenate((G_3,G_3_integer),axis=0)
            h_3 = np.concatenate((h_3,h_3_integer),axis=0)

            binary_list = range(2*self.N_corr,3*self.N_corr)
            self.binary_list = binary_list



        if MIQP and activate_GS_preservation:
            self.matrix_type = "L0L1prGS"
        elif MIQP and not activate_GS_preservation:
            self.matrix_type = "L0L1nonGS"
        elif not MIQP and  activate_GS_preservation:
            self.matrix_type = "L1prGS"
        elif not MIQP and not activate_GS_preservation:
            self.matrix_type = "L1nonGS"

        # # import numpy
        # # numpy.set_printoptions(threshold='nan')
        # def is_pos_semi_def(x):
        #     eigen_values = np.linalg.eigvals(x)
        #     # print ("eigen_values is ",eigen_values)
        #     print ("violating eigenvalues are eigen_values[np.where(eigen_values<0)]")
        #     # print ("is ",eigen_values[0]," >=0 ",eigen_values[0]>=0)
        #     # print ("is ",eigen_values[-1]," >=0 ",eigen_values[-1]>=0)
        #     return np.all(eigen_values >= 0)
        #
        # is_P_psd = is_pos_semi_def(P)
        # print ("is_P_psd ",is_P_psd)
        #
        # # if is_P_psd == False:
        # #     print ("P is")
        # #     print (P)

        self.P = P
        self.q = q
        self.G_3 = G_3
        self.h_3 = h_3
        self.G3_without_preserve_GS = G3_without_preserve_GS
        self.h_3_without_preserve_GS = h_3_without_preserve_GS


    def add_self_eci_cutoff(self,eci_cutoff):
        self.eci_cutoff= eci_cutoff;


    def add_concentration_min_max(self,concentrationmin,concentrationmax):
        self.concentrationmin=concentrationmin
        self.concentrationmax=concentrationmax



    def decide_valid_lists(self):

        valid_index= range(len(self.correlations_in))
        valid_index= [i for i in valid_index if "enum-hypo-" not in self.structure_directory[i]]
        if self.concentrationmin is not None:
            valid_index=[i for i in valid_index if not tuple_all_strict_left_larger_than_right(self.concentrationmin-1e-3,self.concentrations[i])]
        if self.concentrationmax is not None:
            valid_index=[i for i in valid_index if not  tuple_all_strict_left_larger_than_right(self.concentrations[i],self.concentrationmax+1e-3) ]

        valid_index_only_consider_conc= range(len(self.correlations_in))
        if self.concentrationmin is not None:
            valid_index_only_consider_conc=[i for i in valid_index_only_consider_conc if not tuple_all_strict_left_larger_than_right(self.concentrationmin-1e-3,self.concentrations[i])]
        if self.concentrationmax is not None:
            valid_index_only_consider_conc=[i for i in valid_index_only_consider_conc if not  tuple_all_strict_left_larger_than_right(self.concentrations[i],self.concentrationmax+1e-3)]

        invalid_index_due_to_conc_min=[];
        if self.concentrationmin is not None:
            invalid_index_due_to_conc_min=[i for i in range(len(self.correlations_in)) if  tuple_all_strict_left_larger_than_right(self.concentrationmin-1e-3,self.concentrations[i])]

        # print ("concentrationmax is",concentrationmax)
        invalid_index_due_to_conc_max=[];
        if self.concentrationmax is not None:
            invalid_index_due_to_conc_max=[i for i in range(len(self.correlations_in)) if  tuple_all_strict_left_larger_than_right(self.concentrations[i],self.concentrationmax+1e-3)]
        # print ("invalid_index_due_to_conc_max is",invalid_index_due_to_conc_max)

        self.non_hypothetical_list= [i for i in range(len(self.correlations_in)) if "enum-hypo-" not in self.structure_directory[i]]
        self.hypothetical_list =[i for i in range(len(self.correlations_in)) if "enum-hypo-"  in self.structure_directory[i]]
        self.valid_index=valid_index
        self.valid_index_only_consider_conc=valid_index_only_consider_conc
        self.invalid_index_due_to_conc_min=invalid_index_due_to_conc_min
        self.invalid_index_due_to_conc_max=invalid_index_due_to_conc_max


    def ReDefineFormE(self):

        self.decide_valid_lists()
        for i in self.hypothetical_list:
            # print(self.concentrations[i],self.formation_energies_in[i],self.energy_above_hull_in[i])
            self.formation_energies_in[i]=self.formation_energies_in[i]-self.energy_above_hull_in[i]+random.random()*0.001
            self.energy_in[i]=self.energy_in[i]-self.energy_above_hull_in[i]+random.random()*0.001
        print("Formation Energy redefined for hypothetical structure!")



    def compute_decomposition_data(self,repeat_calculation=False):

        if (self.already_compute_decomposition_data==False) or repeat_calculation:
            valid_index=self.valid_index
            valid_index_only_consider_conc=self.valid_index_only_consider_conc
            invalid_index_due_to_conc_min=self.invalid_index_due_to_conc_min
            invalid_index_due_to_conc_max=self.invalid_index_due_to_conc_max

            hull_idx_another_approach=self.compute_hull_idx()
            hull_idx=hull_idx_another_approach
            hull_conc=[self.concentrations[i] for i in hull_idx]
            hull_form_e=[self.formation_energies_in[i] for i in hull_idx]
            hull_e_above_hull_in=[self.energy_above_hull_in[i] for i in hull_idx]
            hull_zip=zip(hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)
            hull_zip.sort()
            (hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)=zip(*hull_zip)

            # print("hull_zip is")
            # pprint(hull_zip)

            decomposition_data = {}
            self.dimension

            for i in valid_index_only_consider_conc:
                if i not in hull_idx:
                    decomposition_now = {}
                    conc_i = self.concentrations[i]
                    # print("conc_i is")
                    # print(conc_i)
                    coeff = []
                    constr_list = []

                    for j in range(len(hull_conc)):
                        coeff.append(variable())

                    sum_coeff = 0
                    for j in range(len(hull_conc)):
                        sum_coeff = sum_coeff + coeff[j]

                    constr_1 = (sum_coeff == 1)
                    constr_list.append(constr_1)

                    for j in range(len(hull_conc)):
                        constr_list.append(coeff[j]>=0)



                    for d in range(self.dimension):
                        sum_conc_d_now = 0
                        for j in range(len(hull_conc)):
                            sum_conc_d_now += hull_conc[j][d]*coeff[j]
                        constr_d_now = (sum_conc_d_now==conc_i[d])
                        constr_list.append(constr_d_now)

                    obj = 0
                    for j in range(len(hull_conc)):
                        obj += float(hull_form_e[j])*coeff[j]

                    lp = op(obj,constr_list)
                    lp.solve()

                    for j in range(len(hull_conc)):
                        coeff_value_j = float(Fraction(coeff[j].value[0]).limit_denominator(2000))
                        if coeff_value_j>0:
                            decomposition_now[hull_idx[j]]=float(coeff_value_j)

                    decomposition_data[i]=decomposition_now

            self.already_compute_decomposition_data=True
            self.decomposition_data = decomposition_data

        return self.decomposition_data





    def compute_special_decomposition_data(self,repeat_special_calculation=False):

        if (self.already_compute_special_decomposition_data==False) or repeat_special_calculation:
            valid_index=self.valid_index
            valid_index_only_consider_conc=self.valid_index_only_consider_conc
            invalid_index_due_to_conc_min=self.invalid_index_due_to_conc_min
            invalid_index_due_to_conc_max=self.invalid_index_due_to_conc_max

            hull_idx=np.where(np.array( self.energy_above_hull_in )<1e-4)[0]
            hull_conc=[self.concentrations[i] for i in hull_idx]
            hull_form_e=[self.formation_energies_in[i] for i in hull_idx]
            hull_e_above_hull_in=[self.energy_above_hull_in[i] for i in hull_idx]
            hull_zip=zip(hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)
            hull_zip.sort()
            (hull_conc,hull_idx,hull_form_e,hull_e_above_hull_in)=zip(*hull_zip)

            special_decomposition_data = {}
            self.dimension
            self.undecomposable_hull_idx = []

            for i in hull_idx:
                decomposition_now = {}
                conc_i = self.concentrations[i]
                coeff = []
                constr_list = []

                for j in range(len(hull_conc)):
                    coeff.append(variable())

                sum_coeff = 0
                for j in range(len(hull_conc)):
                    sum_coeff = sum_coeff + coeff[j]

                constr_1 = (sum_coeff == 1)
                constr_list.append(constr_1)


                i_in_hull_idx_index = hull_idx.index(i)
                special_constr = (coeff[i_in_hull_idx_index] == 0)
                constr_list.append(special_constr)


                for j in range(len(hull_conc)):
                    constr_list.append(coeff[j]>=0)



                for d in range(self.dimension):
                    sum_conc_d_now = 0
                    for j in range(len(hull_conc)):
                        sum_conc_d_now += float (hull_conc[j][d])*coeff[j]

                    constr_d_now = (sum_conc_d_now==float(conc_i[d]))
                    constr_list.append(constr_d_now)

                obj = 0
                for j in range(len(hull_conc)):
                    obj += float(hull_form_e[j])*coeff[j]

                lp = op(obj,constr_list)
                lp.solve()

                if lp.status == "optimal":
                    for j in range(len(hull_conc)):
                        coeff_value_j = float(Fraction(coeff[j].value[0]).limit_denominator(2000))
                        if coeff_value_j>0:
                            decomposition_now[hull_idx[j]]=float(coeff_value_j)

                    special_decomposition_data[i]=decomposition_now
                else:
                    self.undecomposable_hull_idx.append(i)

            # print("special_decomposition_data")
            # pprint(special_decomposition_data)
            self.already_compute_special_decomposition_data=True
            self.special_decomposition_data = special_decomposition_data

        return self.special_decomposition_data

    def compute_hull_idx(self):
        hull_idx_another_approach=np.where(np.array( self.energy_above_hull_in )<1e-8)[0]

        hull_idx_dirs = [self.structure_directory[i] for i in hull_idx_another_approach]
        # print ("in compute_hull_idx, the identified hull is ")
        # print (hull_idx_dirs)

        self.hull_idx = hull_idx_another_approach

        return hull_idx_another_approach
