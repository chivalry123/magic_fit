"""
This module provides a class, CASMSet, to contain cluster expansion
sets for use with the CASM cluster expansion code.

"""

from __future__ import print_function, division

import os
import numpy as np
from pprint import pprint
from ceset import CESet, EPS

__author__ = "Alexander Urban, Wenxuan Huang"
__email__ = "alexurba@mit.edu, key01027@mit.edu"
__date__ = "2014-12-01"
__version__ = "0.1"


class CASMSet(CESet):
    """
    Extends CESet by I/O routines specific to the CASM cluster expansion
    package.

    Arguments:
      corr_in_file (str)     Path to CASM's 'corr.in' file.
      energy_file (str)      Path to CASM's 'energy' file.
      shift_energies (bool)  If True, shift lowest structural energy to
                             zero before the ECI fit (default: False).
      detect_redundant_clusters (bool)  If True, clusters that are redundant
                             for the specific set of input structures will
                             we removed (default: True).

    """

    def __init__(self, corr_in_file, energy_file, shift_energies=False,
                 detect_redundant_clusters=True, pca=False,DiffFocus=None,DiffFocusWeight=None,DiffFocusName=None,
                 SmallErrorOnInequality=None,OnlyKeppEcis=None):
        self.SmallErrorOnInequality=SmallErrorOnInequality
        self.corr_in_file = corr_in_file
        self.energy_file = energy_file

        # read energies of the input structures
        (energy, dimension, concentrations, directories, energy_shift
         ) = self._read_energies(energy_file, shift_energies)


        self.only_kept_ecis_list=None

        if OnlyKeppEcis is not None:
            self.only_kept_ecis_list=[]
            self.readOnlyKeppEcis(OnlyKeppEcis)

        # read cluster correlations for all input structures
        correlations = self._read_cluster_correlations(
            corr_in_file, detect_linear_dependencies=detect_redundant_clusters)
        # print("energy")
        # print(energy)
        # print("energy_shift")
        # print(energy_shift)

        self.cluster_multiplicity=[]
        self.cluster_size=[]

        self.read_eci_in_to_determine_multiplicity()
        self.diff_foscus_lists_of_lists=[]

        self.DiffFocusWeight=DiffFocusWeight

        if DiffFocus is not None:
            raise AssertionError("DiffFocus is disabled, please use DiffFocusName instead")
            self.read_diff_focused_txt(DiffFocus)
            if DiffFocusName is not None:
                raise AssertionError("only DiffFocus or DiffFocusName")
        if DiffFocusName is not None:
            self.diff_foscus_lists_of_lists=[]
            self.diff_foscus_names_lists_of_lists=[]
            self.diff_foscus_weights_lists = []
            self.read_diff_focused_name_txt(DiffFocusName)
            for list_now in self.diff_foscus_names_lists_of_lists:
                list_tmp = []
                for file_name_now in list_now:
                    match_found=False
                    for i in range(len(directories)):
                        if file_name_now in directories[i]:
                            list_tmp.append(i)
                            match_found=True
                            break
                    if match_found==False:
                        raise AssertionError(file_name_now+" does not correspond to any directories ")
                self.diff_foscus_lists_of_lists.append(list_tmp)


        print("diff focued list is")
        pprint(self.diff_foscus_lists_of_lists)

        super(CASMSet, self).__init__(
            energy, energy_shift, correlations, directories,
            concentrations=concentrations, pca=pca)

    def __str__(self):
        return

    def readOnlyKeppEcis(self,OnlyKeppEcis):
        with open(OnlyKeppEcis, 'r') as f:
            # next(f)
            for line in f:
                # line_now = line.split()
                list_now = map(int, line.split())
                self.only_kept_ecis_list+=list_now


    def read_diff_focused_name_txt(self,DiffFocusName):
        # print("reading diff focused txt")
        with open(DiffFocusName, 'r') as f:
            # next(f)
            for line in f:
                line_now = line.split()
                list_now = map(str, line.split())
                list_now_0 = list_now[0]
                list_now_0_split= list_now_0.split(":")
                if list_now_0_split[0]=="weight":
                    weight_now = float(list_now_0_split[1])
                    list_now = list_now[1:]
                else:
                    weight_now=self.DiffFocusWeight
                self.diff_foscus_names_lists_of_lists.append(list_now)
                self.diff_foscus_weights_lists.append(weight_now)


    def read_diff_focused_txt(self,DiffFocus):
        # print("reading diff focused txt")
        with open(DiffFocus, 'r') as f:
            # next(f)
            for line in f:
                line_now = line.split()
                # print("line now is ")
                # print(line_now)
                # print("len(line_now)")
                # print(len(line_now))
                list_now = map(int, line.split())
                self.diff_foscus_lists_of_lists.append(list_now)

    def _read_energies(self, energy_file, shift_energies):
        """
        Read structural energies from CASM 'energy' file.

        Arguments:
          energy_file (str)      path to the energy file
          shift_energies (bool)  if True, the energies will be shifted
                                 such that the lowest energy is 0
        Returns:
          Tuple (En, dim, conc, dirs, E_min) with
            En[i] (float)   Energy of the i-th structure
            dim (int)       Dimension of the CE; 1 = binary, 2 = ternary, ...
            conc[i] (float) List of concentration of the i-th structure
            dirs[i] (str)   Path to the diretory of the i-th structure
            E_min (float)   Energy shift imposed to all energies.

        """

        contents = np.loadtxt(energy_file, dtype=str, comments="#")
        dim = len(contents[0]) - 4
        En = np.asarray(contents[:, 0], dtype=float)
        dirs = list(contents[:, -1])
        conc = [map(float, c) for c in contents[:, 2:2+dim]]
        E_min = 0.0
        if shift_energies:
            E_min = np.min(En)
            En -= E_min

        return (En, dim, conc, dirs, E_min)

    def _read_cluster_correlations(
            self, corr_in_file, detect_linear_dependencies=True):
        """
        Read the cluster correlations from a file in CASM's corr.in format
        and optionally check the clusters for linear dependencies.

        Arguments:
          corr_in_file (str)  Path to CASM's 'corr.in' file
          detect_linear_dependencies (bool)
                              If True, attempt to remove redundant clusters
                              by setting the corresponding column of the
                              correlation matrix to 0.

        Returns:
          An np.ndarray containing the correlation matrix.
        """

        A = np.loadtxt(corr_in_file, skiprows=3)
        if detect_linear_dependencies:
            n_redundant = 0
            for i in range(A.shape[1]-1, 0, -1):
                x = np.linalg.lstsq(A[:, :i], A[:, i])
                error = A[:, :i].dot(x[0]) - A[:, i]
                if np.linalg.norm(error) < 1.0e-10:
                    n_redundant += 1
                    A[:, i] = 0.0
                elif self.only_kept_ecis_list is not None:
                    if i not in self.only_kept_ecis_list:
                        n_redundant += 1
                        A[:, i] = 0.0

            if n_redundant > 0:
                corr_in_file_red = corr_in_file + "-red"
                print(" {} redundant clusters removed.".format(n_redundant))
                header = "{} # number of clusters\n".format(A.shape[1])
                header += "{} # number of configurations\n".format(
                    A.shape[0])
                header += "clusters"
                np.savetxt(corr_in_file_red, A, fmt='%9.7f',
                           delimiter='  ', header=header, comments='')

        return A

    def save_casm_eci_file(self, eci_in_file='eci.in',
                           eci_out_file='eci.out', eci_cutoff=EPS):
        """
        Save non-zero ECIs to CASM's eci.out format.  The cluster
        multiplicities are read from CASM's eci.in file.

        Arguments:
          eci_in_file (str)   Path to the existing 'eci.in' file
          eci_out_file (str)  Path to the new 'eci.out' file; if the file
                              already exists, the operation will be aborted
          eci_cutoff (float)  All ECIs with absolute values below this
                              threshold will be ignored

        """
        if not os.path.exists(eci_in_file):
            raise IOError("File not found: {}".format(eci_in_file))
        ecis = self.ecis.copy()
        if self.pca is not None:
            ecis = np.dot(self.pca.eigenvectors, ecis)
        ecis[0] += self.energy_shift
        if os.path.exists(eci_out_file):
            print(" Warning: file '{}' already exists. Aborting.".format(
                eci_out_file))
        else:
            nonzero = self.nonzero_ECIs(eci_cutoff=eci_cutoff, ecis=ecis)
            with open(eci_out_file, 'w') as fout:
                for i in range(6):
                    fout.write("*** the header is irrelevant ***\n")
                # fout.write("           ECIs       ECI/mult    Cluster#\n")
                # Wenxuan's edit to improve output accuracy
                fout.write("                     ECIs                 ECI/mult    Cluster#\n")

                with open(eci_in_file, 'r') as f:
                    next(f)
                    for line in f:
                        (i, w, mult) = map(int, line.split()[0:3])
                        if i in nonzero:
                            # fout.write(" % 24.19f % 24.19f  %10d\n"

                            fout.write(" % 24.19f % 24.19f  %10d\n"
                                       % (ecis[i], ecis[i]/mult, i))

    def read_eci_in_to_determine_multiplicity(self, eci_in_file='eci.in'):
        old_i = -1
        with open(eci_in_file, 'r') as f:
            next(f)
            for line in f:
                (i, w, mult,size) = map(int, line.split()[0:4])
                assert old_i + 1 == i
                self.cluster_multiplicity.append(mult)
                self.cluster_size.append(size)
                old_i = i







class CASMSet_WX_create_sub(CESet):
    def __init__(self, grand_casm, indexes_to_use ,shift_energies=False,
                 detect_redundant_clusters=True, pca=False):
        self.only_kept_ecis_list = grand_casm.only_kept_ecis_list
        self.diff_foscus_weights_lists=grand_casm.diff_foscus_weights_lists

        self.SmallErrorOnInequality=grand_casm.SmallErrorOnInequality
        energy=np.asarray([grand_casm.energy_in[i] for i in indexes_to_use])
        correlations=np.asarray([grand_casm.correlations_in[i]  for i in indexes_to_use])
        energy_shift=grand_casm.energy_shift
        # print("energy")
        # print(energy)
        # print("grand_casm.energy_shift")
        # print(grand_casm.energy_shift)
        directories=[grand_casm.structure_directory[i] for i in indexes_to_use]
        concentrations=[grand_casm.concentrations[i] for i in indexes_to_use]

        self.cluster_multiplicity=grand_casm.cluster_multiplicity
        self.cluster_size=grand_casm.cluster_size
        diff_foscus_lists_of_lists_tmp = grand_casm.diff_foscus_lists_of_lists
        self.diff_foscus_lists_of_lists=self.map_lists_of_lists_to_new_list_of_list(indexes_to_use,diff_foscus_lists_of_lists_tmp)
        self.DiffFocusWeight=grand_casm.DiffFocusWeight
        # print("in this subcasm  indexes_to_use is ")
        # print(indexes_to_use)
        # print("initial diff_foscus_lists_of_lists is")
        # print(diff_foscus_lists_of_lists_tmp)
        # print("converted diff_foscus_lists_of_lists is")
        # print(self.diff_foscus_lists_of_lists)


        super(CASMSet_WX_create_sub, self).__init__(
            energy, energy_shift, correlations, directories,
            concentrations=concentrations, pca=pca)


    def map_lists_of_lists_to_new_list_of_list(self,indexes_to_use,list_of_list_now):
        new_list_of_list_tmp = []
        for list_now in list_of_list_now:
            new_list = self.map_list_to_new_lists(indexes_to_use,list_now)
            if new_list is not None:
                new_list_of_list_tmp.append(new_list)
        if len(new_list_of_list_tmp) > 0:
            return new_list_of_list_tmp
        else:
            return []



    def map_list_to_new_lists(self,indexes_to_use,list_now):
        new_list_tmp = []
        for i in list_now:
            new_num = self.map_to_new_index(indexes_to_use,i)
            if new_num is not None:
                new_list_tmp.append(new_num)
        if len(new_list_tmp) > 0:
            return new_list_tmp
        else:
            return None

    def map_to_new_index(self,indexes_to_use,number_now):
        if number_now  not in indexes_to_use:
            return None
        if number_now in indexes_to_use:
            list_tmp_1=range(number_now)
            list_tmp_2=[i for i in list_tmp_1 if i not in indexes_to_use]
            new_num= number_now - len(list_tmp_2)
            return new_num


    def save_casm_eci_file(self, eci_in_file='eci.in',
                           eci_out_file='eci.out', eci_cutoff=EPS):
        """
        Save non-zero ECIs to CASM's eci.out format.  The cluster
        multiplicities are read from CASM's eci.in file.

        Arguments:
          eci_in_file (str)   Path to the existing 'eci.in' file
          eci_out_file (str)  Path to the new 'eci.out' file; if the file
                              already exists, the operation will be aborted
          eci_cutoff (float)  All ECIs with absolute values below this
                              threshold will be ignored

        """
        if not os.path.exists(eci_in_file):
            raise IOError("File not found: {}".format(eci_in_file))
        ecis = self.ecis.copy()
        if self.pca is not None:
            ecis = np.dot(self.pca.eigenvectors, ecis)
        ecis[0] += self.energy_shift
        if os.path.exists(eci_out_file):
            print(" Warning: file '{}' already exists. Aborting.".format(
                eci_out_file))
        else:
            nonzero = self.nonzero_ECIs(eci_cutoff=eci_cutoff, ecis=ecis)
            with open(eci_out_file, 'w') as fout:
                for i in range(6):
                    fout.write("*** the header is irrelevant ***\n")
                # fout.write("           ECIs       ECI/mult    Cluster#\n")
                # Wenxuan's edit to improve output accuracy
                fout.write("                     ECIs                 ECI/mult    Cluster#\n")

                with open(eci_in_file, 'r') as f:
                    next(f)
                    for line in f:
                        (i, w, mult) = map(int, line.split()[0:3])
                        if i in nonzero:
                            # fout.write(" % 24.19f % 24.19f  %10d\n"

                            fout.write(" % 24.19f % 24.19f  %10d\n"
                                       % (ecis[i], ecis[i]/mult, i))

    def read_eci_in_to_determine_multiplicity(self, eci_in_file='eci.in'):
        old_i = -1
        with open(eci_in_file, 'r') as f:
            next(f)
            for line in f:
                (i, w, mult) = map(int, line.split()[0:3])
                assert old_i + 1 == i
                self.cluster_multiplicity.append(mult)
                old_i = i
