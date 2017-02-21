#!/usr/local/opt/python/bin/python2.7

"""
Insert script description here.
"""

from __future__ import print_function, division

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2015-07-17"
__version__ = "0.1"

import argparse
import numpy as np
import os
import re
import shutil as sh
import pymatgen as mg


def select_structures(infile, outdir, nstruc, conc, nsites, E_max):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    candidates = []
    with open(infile, 'r') as fp:
        for line in fp:
            if re.search(r"^ *[0-9]", line):
                (x, E, N, idx) = line.split()
                if ((conc[0] <= float(x) <= conc[1])
                        and (int(N) <= nsites) and (float(E) <= E_max)):
                    candidates.append((float(x), float(E), int(N), int(idx)))

    if len(candidates) <= nstruc:
        selected = candidates[:]
    else:
        idx = np.argsort(np.random.random(len(candidates)))[:nstruc]
        selected = [c for i, c in enumerate(candidates) if i in idx]

    with open(os.path.join(outdir, "selected.dat"), 'w') as log:
        for i, (x, E, N, idx) in enumerate(selected):
            src = os.path.join("index{}".format(idx), "POSCAR")
            dest = os.path.join(outdir, "POSCAR_{:04d}".format(i))
            sh.copyfile(src, dest)
            log.write("{:8.6f}  {:15.8f}  {:3d}  {:6d}\n".format(x, E, N, idx))


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "hull_states_info",
        help="File with hull states info from GS solver")

    parser.add_argument(
        "--num-structures", "-n",
        help="Maximum number of structures to be selected (default: 20).",
        type=int,
        default=20)

    parser.add_argument(
        "--concentration", "-x",
        help="Specify concentration range (default: all).",
        type=float,
        default=[0.0, 1.0],
        nargs=2)

    parser.add_argument(
        "--max-sites", "-s",
        help="Maximum number of sites (default: no maximum).",
        type=int,
        default=float("inf"))

    parser.add_argument(
        "--max-energy", "-E",
        help="Maximum formation energy (default: no limit).",
        type=float,
        default=float("inf"))

    parser.add_argument(
        "--output-dir", "-o",
        help="Path to an output directory (will be created if necessary).",
        default="selected_structures",
        nargs="?")

    args = parser.parse_args()

    select_structures(args.hull_states_info,
                      outdir=args.output_dir,
                      nstruc=args.num_structures,
                      conc=args.concentration,
                      nsites=args.max_sites,
                      E_max=args.max_energy)
