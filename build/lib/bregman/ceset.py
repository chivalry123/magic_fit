"""
Interface class that provides access to cluster expansion data and
operations on it.

"""

from __future__ import print_function, division

import os
import re
import sys
import numpy as np
from scipy.spatial import ConvexHull

from bregman import split_bregman
from bregman.differential_evolution import Evolution
from bregman.pca import PrincipalComponentAnalysis

from cvxopt import matrix
from cvxopt import solvers

# from cvxopt import normal, uniform
# from cvxopt.modeling import variable, dot, op, sum
import cvxopt

__author__ = "Alexander Urban,Wenxuan Huang"
__email__ = "alexurba@mit.edu, key01027@mit.edu"
__date__ = "2015-12-09"
__version__ = "0.1"


EPS = np.finfo(float).eps


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

        solvers.options['show_progress'] = False

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

            print ("first time done")
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

        if give_hypothetical_structure_trivial_weights:
            for i in range(len(self.structure_directory)):
                if "enum-hypo-" in self.structure_directory[i]:
                    w[i]=w[i]*1e-7


        if concentrationmin is not None :
            for i in range(len(self.structure_directory)):
                if self.concentrations[i][0]<concentrationmin-1e-3 :
                    w[i]=w[i]*1e-7

        if concentrationmax is not None:
            for i in range(len(self.structure_directory)):
                if self.concentrations[i][0]>concentrationmax+1e-3:
                    w[i]=w[i]*1e-7


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
            valid_index=[i for i in valid_index if not self.concentrations[i][0]<concentrationmin-1e-3]
        # print ("valid_index is",valid_index)

        if concentrationmax is not None:
            # print ("self.concentrations is",self.concentrations)
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

        hull = ConvexHull(points)
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



    def perform_QP(self,mu=10,concentrationmin=None,
                           concentrationmax=None,activate_GS_preservation=True):
        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b

        # print ("ready for QP")
        # print (repr(self.correlations_in))
        # print (repr(self.energy_in))
        # print (repr(self.concentrations))
        corr_in=np.array(self.correlations_in)
        engr_in=np.array(self.energy_in)
        engr_in.shape=(len(engr_in),1)
        concentration_in=np.array(self.concentrations)
        concentration_in.shape=(len(concentration_in),1)
        weight_vec=np.array(self.structure_weight)
        # weight_vec.shape=(len(weight_vec),1)
        weight_matrix= np.diag(weight_vec.transpose())

        P_corr_part=2*((weight_matrix.dot(corr_in)).transpose()).dot(corr_in)
        P=np.lib.pad(P_corr_part,((0,self.N_corr),(0,self.N_corr)),mode='constant', constant_values=0)
        # P_z_part=np.zeros(self.N_corr)
        q_corr_part=-2*((weight_matrix.dot(corr_in)).transpose()).dot(engr_in)
        q_z_part=np.ones((self.N_corr,1))*mu
        # q_z_part[0]=0
        q=np.concatenate((q_corr_part,q_z_part),axis=0)

        G_1=np.concatenate((np.identity(self.N_corr),-np.identity(self.N_corr)),axis=1)
        G_2=np.concatenate((-np.identity(self.N_corr),-np.identity(self.N_corr)),axis=1)
        G_3=np.concatenate((G_1,G_2),axis=0)

        h_3=np.zeros((2*self.N_corr,1))

        if activate_GS_preservation:
            hull_points=self.hull_in.points[self.hull_in.vertices]
            hull_idx=self.hull_in.vertices
            hull_conc=[hull_points[i][0] for i in range(len(hull_points))]
            hull_engr=[hull_points[i][1] for i in range(len(hull_points))]
            hull_zip=zip(hull_conc,hull_engr,hull_idx)
            hull_zip.sort()
            (hull_conc,hull_engr,hull_idx)=zip(*hull_zip)

            delete_idx=[]
            if len(hull_conc)>2:
                for i in range(2,len(hull_conc)-1):
                    if hull_engr[i]> (hull_conc[-1]-hull_conc[i])/(hull_conc[-1]-hull_conc[0])*hull_engr[0]+\
                        (hull_conc[i]-hull_conc[0])/(hull_conc[-1]-hull_conc[0])*hull_engr[-1]:
                        delete_idx.append([i])

            # print (repr(hull_zip[0]))

            hull_zip_new=[v for i,v in enumerate(hull_zip) if i not in delete_idx]
            hull_zip=hull_zip_new
            (hull_conc,hull_engr,hull_idx)=zip(*hull_zip)

            # print (hull_conc)



            valid_index= range(len(self.correlations_in))
            valid_index= [i for i in valid_index if "enum-hypo-" not in self.structure_directory[i]]
            if concentrationmin is not None:
                valid_index=[i for i in valid_index if not self.concentrations[i][0]<concentrationmin-1e-3]
            if concentrationmax is not None:
                valid_index=[i for i in valid_index if not  self.concentrations[i][0]>concentrationmax+1e-3]


            valid_index_only_consider_conc= range(len(self.correlations_in))
            if concentrationmin is not None:
                valid_index_only_consider_conc=[i for i in valid_index_only_consider_conc if not self.concentrations[i][0]<concentrationmin-1e-3]
            if concentrationmax is not None:
                valid_index_only_consider_conc=[i for i in valid_index_only_consider_conc if not  self.concentrations[i][0]>concentrationmax+1e-3]

            invalid_index_due_to_conc_min=[];
            if concentrationmin is not None:
                invalid_index_due_to_conc_min=[i for i in range(len(self.correlations_in)) if  self.concentrations[i][0]<concentrationmin-1e-3]

            # print ("concentrationmax is",concentrationmax)
            invalid_index_due_to_conc_max=[];
            if concentrationmax is not None:
                invalid_index_due_to_conc_max=[i for i in range(len(self.correlations_in)) if  self.concentrations[i][0]>concentrationmax+1e-3]
            # print ("invalid_index_due_to_conc_max is",invalid_index_due_to_conc_max)



            for i in valid_index_only_consider_conc:
                # print (i,self.concentrations[i])

                if i not in hull_idx:
                    # print (i,"is not in hull_idx")

                    upper_conc=1e10
                    upper_idx=0
                    lower_conc=-1e10
                    lower_idx=0
                    for j in range(len(hull_conc)):
                        # print ("j is ",j, "self.concentrations[i][0] is ",self.concentrations[i][0],"hull_conc[j] is ",hull_conc[j],"lower_conc is",lower_conc,"upper_conc is ",upper_conc )

                        if self.concentrations[i][0]>=hull_conc[j]-1e-7 and hull_conc[j]>=lower_conc-1e-7:
                            # print ("update lower concentration")
                            lower_conc=hull_conc[j]
                            lower_idx=j
                        if self.concentrations[i][0]<=hull_conc[j]+1e-7 and hull_conc[j]<=upper_conc+1e-7:
                            upper_conc=hull_conc[j]
                            upper_idx=j

                    # print ("upper conc is ",upper_conc)
                    # print ("lower conc is ",lower_conc)

                    if abs(lower_conc-upper_conc)<1e-7:

                        # index=lower_idx
                        # engr_now=hull_engr[index]
                        global_index=hull_idx[lower_idx]
                        # print ("global_index is ",global_index)
                        # print (repr(self.correlations_in[global_index]-self.correlations_in[i]))
                        # print (repr(np.zeros((self.N_corr))))
                        # G_3_new_line=np.concatenate((self.correlations_in[global_index]-self.correlations_in[i],np.zeros((self.N_corr))),axis=1)

                        G_3_new_line=np.concatenate((self.correlations_in[global_index]-self.correlations_in[i],np.zeros((self.N_corr))))
                        # print (G_3_new_line)
                        G_3_new_line.shape=(1,2*self.N_corr)

                        # print (G_3)
                        # print (G_3.shape)
                        # print (G_3_new_line.shape)
                        G_3=np.concatenate((G_3,G_3_new_line),axis=0)
                        # print (repr(h_3))
                        small_error=np.array(-1e-3)
                        small_error.shape=(1,1)
                        # print (repr(small_error))
                        h_3=np.concatenate((h_3,small_error),axis=0)
                    else:
                        conc_now=self.concentrations[i][0]
                        # G_3_new_line=np.concatenate(((upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]+
                        #                              (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                        #                              -self.correlations_in[i]
                        #                              ,np.zeros((self.N_corr))),axis=1)
                        G_3_new_line=np.concatenate(((upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]+
                             (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                             -self.correlations_in[i]
                             ,np.zeros((self.N_corr))))
                        G_3_new_line.shape=(1,2*self.N_corr)
                        G_3=np.concatenate((G_3,G_3_new_line),axis=0)

                        small_error=np.array(-1e-3) # from -4 to -3
                        small_error.shape=(1,1)
                        h_3=np.concatenate((h_3,small_error),axis=0)


            #     valid_index= range(len(self.correlations_in))
            #     valid_index= [i for i in valid_index if "enum-hypo-" not in self.structure_directory[i]]
            #     if concentrationmin is not None:
            #         valid_index=[i for i in valid_index if not self.concentrations[i][0]<concentrationmin-1e-3]
            #     if concentrationmax is not None:
            #         valid_index=[i for i in valid_index if not  self.concentrations[i][0]>concentrationmax+1e-3]


            if len(hull_idx)>3: ## not yet complete
                for ind in range(1,len(hull_idx)-1):
                    if hull_idx[ind] in valid_index_only_consider_conc:
                        conc_now=hull_conc[ind];
                        # print ("looking into the hull inequalities, conc_now is",conc_now)
                        upper_conc=hull_conc[ind+1]
                        lower_conc=hull_conc[ind-1]
                        i=hull_idx[ind]
                        lower_idx=ind-1
                        upper_idx=ind+1
                        if hull_idx[upper_idx] in invalid_index_due_to_conc_max:
                            # print ("I am in upper_idx in invalid_index_due_to_conc_max")
                            for j in invalid_index_due_to_conc_max:
                                upper_conc=self.concentrations[j]
                                upper_idx_global=j
                                # G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]-
                                #      (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[upper_idx_global]+
                                #      +self.correlations_in[i]
                                #      ,np.zeros((self.N_corr))),axis=1)

                                G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]-
                                     (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[upper_idx_global]+
                                     +self.correlations_in[i]
                                     ,np.zeros((self.N_corr))))

                                G_3_new_line.shape=(1,2*self.N_corr)
                                G_3=np.concatenate((G_3,G_3_new_line),axis=0)

                                small_error=np.array(-1e-3) ## this is for the case where no hull shape preserving is used
                                hull_shape_preservation=True;
                                if hull_shape_preservation:
                                    gap=self.energy_in[i]-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.energy_in[hull_idx[lower_idx]]\
                                            -(conc_now-lower_conc)/(upper_conc-lower_conc)*self.energy_in[upper_idx_global]


                                    # print ("energy_in[i] is ",self.energy_in[i]," energy_in[hull_idx[lower_idx]] is ",self.energy_in[hull_idx[lower_idx]]," energy_in[hull_idx[upper_idx]] is ", self.energy_in[hull_idx[upper_idx]])
                                    # print ("gap value is", gap)
                                    gap=min(gap/10,-1e-3)
                                    # print ("gap value is", gap)
                                    small_error=np.array(gap) ## this is for the case where hull shape preserving is used

                                small_error.shape=(1,1)
                                h_3=np.concatenate((h_3,small_error),axis=0)
                        elif hull_idx[lower_idx] in invalid_index_due_to_conc_min:
                            for j in invalid_index_due_to_conc_max:
                                lower_conc=self.concentrations[j]
                                lower_idx_global=j


                                # G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[lower_idx_global]-
                                #      (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                                #      +self.correlations_in[i]
                                #      ,np.zeros((self.N_corr))),axis=1)

                                G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[lower_idx_global]-
                                     (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                                     +self.correlations_in[i]
                                     ,np.zeros((self.N_corr))))

                                G_3_new_line.shape=(1,2*self.N_corr)
                                G_3=np.concatenate((G_3,G_3_new_line),axis=0)

                                small_error=np.array(-1e-3) ## this is for the case where no hull shape preserving is used
                                # hull_shape_preservation=False;
                                # if hull_shape_preservation:
                                #     gap=self.energy_in[i]-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.energy_in[hull_idx[lower_idx]]\
                                #             -(conc_now-lower_conc)/(upper_conc-lower_conc)*self.energy_in[hull_idx[upper_idx]]
                                #
                                #
                                #     print ("energy_in[i] is ",self.energy_in[i]," energy_in[hull_idx[lower_idx]] is ",self.energy_in[hull_idx[lower_idx]]," energy_in[hull_idx[upper_idx]] is ", self.energy_in[hull_idx[upper_idx]])
                                #     print ("gap value is", gap)
                                #     gap=min(gap/10,-1e-3)
                                #     print ("gap value is", gap)
                                #     small_error=np.array(gap) ## this is for the case where hull shape preserving is used

                                small_error.shape=(1,1)
                                h_3=np.concatenate((h_3,small_error),axis=0)
                        else:
                            # G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]-
                            #              (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                            #              +self.correlations_in[i]
                            #              ,np.zeros((self.N_corr))),axis=1)

                            G_3_new_line=np.concatenate((-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[lower_idx]]-
                                         (conc_now-lower_conc)/(upper_conc-lower_conc)*self.correlations_in[hull_idx[upper_idx]]+
                                         +self.correlations_in[i]
                                         ,np.zeros((self.N_corr))))


                            G_3_new_line.shape=(1,2*self.N_corr)
                            G_3=np.concatenate((G_3,G_3_new_line),axis=0)

                            small_error=np.array(-1e-3) ## this is for the case where no hull shape preserving is used
                            hull_shape_preservation=True;
                            if hull_shape_preservation:
                                gap=self.energy_in[i]-(upper_conc-conc_now)/(upper_conc-lower_conc)*self.energy_in[hull_idx[lower_idx]]\
                                        -(conc_now-lower_conc)/(upper_conc-lower_conc)*self.energy_in[hull_idx[upper_idx]]


                                # print ("energy_in[i] is ",self.energy_in[i]," energy_in[hull_idx[lower_idx]] is ",self.energy_in[hull_idx[lower_idx]]," energy_in[hull_idx[upper_idx]] is ", self.energy_in[hull_idx[upper_idx]])
                                # print ("gap value is", gap)
                                gap=min(gap/10,-1e-3)
                                # print ("gap value is", gap)
                                small_error=np.array(gap) ## this is for the case where hull shape preserving is used

                            small_error.shape=(1,1)
                            h_3=np.concatenate((h_3,small_error),axis=0)



            # relative_range_preserve=True
            # if relative_range_preserve:
            #     valid_index= range(len(self.correlations_in))
            #     valid_index= [i for i in valid_index if "enum-hypo-" not in self.structure_directory[i]]
            #     if concentrationmin is not None:
            #         valid_index=[i for i in valid_index if not self.concentrations[i][0]<concentrationmin-1e-3]
            #     if concentrationmax is not None:
            #         valid_index=[i for i in valid_index if not  self.concentrations[i][0]>concentrationmax+1e-3]
            #     for ind in range(len(hull_idx)):
            #         relative_range=2e-1
            #
            #         conc_now=hull_conc[ind]
            #         i=hull_idx[ind]
            #
            #         if i in valid_index:
            # #             G_3_new_line=np.concatenate((
            # #                          +self.correlations_in[i]
            #                          ,np.zeros((self.N_corr))),axis=1)
            #             G_3_new_line=np.concatenate((
            #                          +self.correlations_in[i]
            #                          ,np.zeros((self.N_corr))))
            #
            #             G_3_new_line.shape=(1,2*self.N_corr)
            #             G_3=np.concatenate((G_3,G_3_new_line),axis=0)
            #
            #             Upper=np.array(self.energy_in[i]+relative_range) ## this is for the case where no hull shape preserving is used
            #             Upper.shape=(1,1)
            #             h_3=np.concatenate((h_3,Upper),axis=0)
            #
            #
            #            # G_3_new_line=np.concatenate((
            #            #              -self.correlations_in[i]
            #            #              ,np.zeros((self.N_corr))),axis=1)
            #             G_3_new_line=np.concatenate((
            #                          -self.correlations_in[i]
            #                          ,np.zeros((self.N_corr))))
            #
            #             G_3_new_line.shape=(1,2*self.N_corr)
            #             G_3=np.concatenate((G_3,G_3_new_line),axis=0)
            #
            #             Lower=np.array(-self.energy_in[i]+relative_range) ## this is for the case where no hull shape preserving is used
            #             Lower.shape=(1,1)
            #             h_3=np.concatenate((h_3,Lower),axis=0)




            # print (G_3.shape,h_3.shape)
            # print (G_3,h_3)

        P_matrix=matrix(P)
        q_matrix=matrix(q)
        G_3_matrix=matrix(G_3)
        h_3_matrix=matrix(h_3)



        # print (repr(hull_zip))

        sol = solvers.qp(P_matrix,q_matrix,G_3_matrix,h_3_matrix)



        # print (repr( self.hull_in))
        # print (repr(self.hull_in.simplices))
        # print (repr(self.hull_in.points))
        # print (repr(self.hull_in.points[self.hull_in.simplices]))
        # print (repr(self.hull_in.vertices))
        # hull.points

        # print (sol['x'][1])

        self.ecis = np.zeros(self.N_corr)
        self.ecis[0:self.N_corr]=sol['x'][0:self.N_corr].T

        self.compute_ce_energies()
        self._update_ce_hull()

        # zip(self.concentrations,self.structure_weight)
        # print ("zip(self.concentrations,self.structure_weight) is",zip(self.concentrations,self.structure_weight))


        # idx_del = [i for i in range(self.N_corr)
        #            if i not in self.nonzero_ECIs(eci_cutoff=eci_cutoff)]
        # self.ecis[idx_del] = 0.0


        # print (repr( weight_vec))
        # weight_vec.shape=(len(weight_vec),1)
        # print (repr( weight_vec))
        # print (repr( weight_vec.transpose()))
        # print (repr(( weight_vec.transpose()).transpose()))








