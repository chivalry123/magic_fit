"""
This module provides a class, CASMSet, to contain cluster expansion
sets for use with the CASM cluster expansion code.

"""

from __future__ import print_function, division

import os
import numpy as np

from ceset import CESet, EPS

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
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
                 detect_redundant_clusters=True, pca=False):
        self.corr_in_file = corr_in_file
        self.energy_file = energy_file

        # read energies of the input structures
        (energy, dimension, concentrations, directories, energy_shift
         ) = self._read_energies(energy_file, shift_energies)
        # read cluster correlations for all input structures
        correlations = self._read_cluster_correlations(
            corr_in_file, detect_linear_dependencies=detect_redundant_clusters)
        # print("energy")
        # print(energy)
        # print("energy_shift")
        # print(energy_shift)

        super(CASMSet, self).__init__(
            energy, energy_shift, correlations, directories,
            concentrations=concentrations, pca=pca)

    def __str__(self):
        return

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


class CASMSet_WX_create_sub(CESet):
    def __init__(self, grand_casm, indexes_to_use ,shift_energies=False,
                 detect_redundant_clusters=True, pca=False):
        energy=np.asarray([grand_casm.energy_in[i] for i in indexes_to_use])
        correlations=np.asarray([grand_casm.correlations_in[i]  for i in indexes_to_use])
        energy_shift=grand_casm.energy_shift

        # print("energy")
        # print(energy)
        # print("grand_casm.energy_shift")
        # print(grand_casm.energy_shift)
        directories=[grand_casm.structure_directory[i] for i in indexes_to_use]
        concentrations=[grand_casm.concentrations[i] for i in indexes_to_use]

        super(CASMSet_WX_create_sub, self).__init__(
            energy, energy_shift, correlations, directories,
            concentrations=concentrations, pca=pca)




