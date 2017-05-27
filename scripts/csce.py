#!/usr/bin/env python

"""
Compressive sensing cluster expansion using the correlation matrix and
energy files generated with CASM.
"""

from __future__ import print_function, division

import argparse
import sys

import numpy as np
from math import log
import  random
import time

from bregman.casm import CASMSet, CASMSet_WX_create_sub


__author__ = "Alexander Urban, Wenxuan Huang"
__email__ = "alexurba@mit.edu, key01027@mit.edu"
__date__ = "2015-12-09"
__version__ = "0.1"


def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


def compressive_sensing_CE(args,energy_file, corr_in_file, eci_in_file,
                           max_rmsd, bias_stable, bias_low_energy,
                           bias_high_energy, optimize_weights, mu,
                           lmbda, maxiter, tol, shift_energies,
                           eci_cutoff, eci_file, casm_eci_file,
                           simplify, simplify_strict, ce_energies_file,
                           save_weights_file, read_weights_file,
                           ce_hull_file, no_cv, eci_input_file,
                           no_detect_redundant, do_pca,preserve_GS,give_hypothetical_structure_trivial_weights,
                           preserve_ground_state_range,preserve_ground_state_range_iterations,QP,concentrationmin,
                           concentrationmax,researchonfittingmode,partitionsnumberforcvscore,DeactivateGSPreservation,
                           researchonfittingmodeWeightadjusting,RemoveOutsideConcentration,MuPartition,weight_update_scheme,
                            MuStart,MuEnd,ReDefineFormE,AbsoluteErrorConstraintOnHull,DiffFocus,
                           DiffFocusWeight,SpecializedCvPengHaoGS,DiffFocusName,SmallErrorOnInequality,OnlyKeppEcis,CompressFirstPair,MIQP,
                           UnCompressPairUptoDist,CompressAllTerms,MIQPGSPrsSolvingTime,MIQPNonGSPrsSolvingTime,L0L1,L0Hierarchy,L1Hierarchy,MaxNumClusts,L0mu,
                           LinearScalingToUnweightHighEahStructs,ExpScalingToUnweightHighEahStructs):

    time_start = time.clock()

    if eci_input_file is not None:
        shift_energies = False

    casm = CASMSet(corr_in_file, energy_file, shift_energies=shift_energies,
                   detect_redundant_clusters=(not no_detect_redundant),
                   pca=do_pca,DiffFocus=DiffFocus,DiffFocusWeight=DiffFocusWeight,DiffFocusName=DiffFocusName,
                   SmallErrorOnInequality=SmallErrorOnInequality,OnlyKeppEcis=OnlyKeppEcis,
                   CompressFirstPair=CompressFirstPair,UnCompressPairUptoDist=UnCompressPairUptoDist,CompressAllTerms=CompressAllTerms,
                   MIQPGSPrsSolvingTime=MIQPGSPrsSolvingTime,MIQPNonGSPrsSolvingTime=MIQPNonGSPrsSolvingTime,L0L1=L0L1,
                   L0Hierarchy=L0Hierarchy,L1Hierarchy=L1Hierarchy,MaxNumClusts=MaxNumClusts,L0mu=L0mu,LinearScalingToUnweightHighEahStructs=LinearScalingToUnweightHighEahStructs,
                   ExpScalingToUnweightHighEahStructs=ExpScalingToUnweightHighEahStructs)
    casm.add_concentration_min_max(concentrationmin,concentrationmax)
    casm.decide_valid_lists()

    if ReDefineFormE:
        casm.ReDefineFormE()

    if RemoveOutsideConcentration:
        print("old casm valid list ",casm.valid_index_only_consider_conc)
        casm=CASMSet_WX_create_sub(casm,casm.valid_index_only_consider_conc)
        casm.add_concentration_min_max(concentrationmin,concentrationmax)
        casm.decide_valid_lists()
        print("new casm valid list ",casm.valid_index_only_consider_conc)
        print("new casm concentrations length ",len(casm.concentrations) )
        print(casm.concentrations)




    if read_weights_file is not None:
        print(" Reading structural weights from file "
              "`{}'.".format(read_weights_file))
        casm.read_structure_weights(read_weights_file)
    else:
        # compute weights based on RMSDs and structural energies
        if bias_stable:
            print(" Biasing stable configurations ")
        if bias_low_energy is not None:
            print(" Favoring structures with energies up to " +
                  "{} ".format(bias_low_energy) +
                  "energy units above the hull")
        if bias_high_energy is not None:
            print(" De-prioritizing structures with energies above "
                  "{} ".format(bias_high_energy) + "energy units above "
                  "the hull")
        casm.compute_structure_weights(max_rmsd,
                                       bias_high_energy=bias_high_energy,
                                       bias_low_energy=bias_low_energy,
                                       bias_stable=bias_stable,give_hypothetical_structure_trivial_weights=give_hypothetical_structure_trivial_weights,concentrationmin=concentrationmin,concentrationmax=concentrationmax)

    if casm.N_ignored > 0:
        print(" {} structures will be ignored ".format(casm.N_ignored) +
              "due to their high RMS distance.")

    if eci_input_file is not None:
        print(" Reading ECIs from file `{}'\n".format(eci_input_file))
        casm.read_ECIs(eci_input_file)

    if QP or MIQP or researchonfittingmode:
        1;
    else:
        # compressive-sensing CE
        print(" Split-Bregman Input Parameters:  "
              "mu = {},  lambda = {},  eci_cutoff = {}".format(
                  mu, lmbda, eci_cutoff))
        casm.compute_ECIs(maxiter=maxiter, tol=tol, mu=(1.0/mu),
                          lmbda=lmbda, eci_cutoff=eci_cutoff)
        print(" Writing ECIs to file `{}'.".format(eci_file))
        casm.save_ECIs(eci_file, eci_cutoff=eci_cutoff)
        print(" {} non-zero ECIs found.\n".format(len(casm.nonzero_ECIs())))

    # CE energies and errors using all non-zero ECIs

    if not (QP or MIQP or researchonfittingmode):
        E_all = casm.compute_ce_energies()
        (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight
         ) = casm.compute_ce_errors(ce_energies=E_all)
        idx_all = casm.nonzero_ECIs()




    # simplify CE, if requested
    if simplify is not None:
        print(" Attempting to simplify the CE using a target RMSE of "
              "{}".format(simplify))
        casm.simplify(simplify, maxiter=maxiter, tol=tol, mu=(1.0/mu),
                      lmbda=lmbda, refit=simplify_strict)
        (rmse_simp, mue_simp, mse_simp, rmse_noW_simp,rmse_no_hypo_no_weight
         ) = casm.compute_ce_errors()
        nsimp = len(casm.cluster_index_simp)
        print(" ECI cutoff        :  {}".format(casm.eci_cutoff_simp))
        print(" Size of reduced CE:  %d" % (nsimp, ))
        print(" Writing ECIs of simplified CE to file "
              "`{}'.".format(eci_file + "-simp"))
        casm.save_ECIs(eci_file + "-simp")
        print("")

    # automatic weight optimization
    if optimize_weights is not None:
        print(" Attempting automatic weight optimization ")
        if simplify is not None:
            casm.optimize_structure_weights(maxiter_opt=optimize_weights,
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True,
                                            simplify=simplify,
                                            refit=simplify_strict)
            nsimp = len(casm.cluster_index_simp)
        else:
            casm.optimize_structure_weights(maxiter_opt=optimize_weights,
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True)
        (rmse_simp, mue_simp, mse_simp, rmse_noW_simp,rmse_no_hypo_no_weight
         ) = casm.compute_ce_errors()
        print(" Writing ECIs of optimized CE to file "
              "`{}'.".format(eci_file + "-opt"))
        casm.save_ECIs(eci_file + "-opt")
        print("")




    if preserve_GS is not None:
        print(" Attempting Ground state preservation, currently only support simplify ")
        if simplify is not None:
            casm.preserve_GS(maxiter_opt=preserve_GS,
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True,
                                            simplify=simplify,
                                            refit=simplify_strict,eci_cutoff=eci_cutoff)
            nsimp = len(casm.cluster_index_simp)
            (rmse_simp, mue_simp, mse_simp, rmse_noW_simp,rmse_no_hypo_no_weight
             ) = casm.compute_ce_errors()

        else:
            casm.preserve_GS(maxiter_opt=preserve_GS,
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True,eci_cutoff=eci_cutoff)
            E_all = casm.compute_ce_energies()
            (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight
             ) = casm.compute_ce_errors(ce_energies=E_all)
            idx_all = casm.nonzero_ECIs()



        print(" Writing ECIs of optimized CE to file "
              "`{}'.".format(eci_file + "-opt"))
        casm.save_ECIs(eci_file + "-opt")
        print("")



    if preserve_ground_state_range is not None and preserve_ground_state_range_iterations is not None:
        print(" Attempting Ground state preservation ranges")
        if simplify is not None:
            casm.preserve_GS_range(
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True,
                                            simplify=simplify,
                                            refit=simplify_strict,eci_cutoff=eci_cutoff,
                                            preserve_ground_state_range=preserve_ground_state_range,
                                            preserve_ground_state_range_iterations=preserve_ground_state_range_iterations,
                                            give_hypothetical_structure_trivial_weights=give_hypothetical_structure_trivial_weights)
            nsimp = len(casm.cluster_index_simp)
            (rmse_simp, mue_simp, mse_simp, rmse_noW_simp,rmse_no_hypo_no_weight
             ) = casm.compute_ce_errors()

        else:
            casm.preserve_GS_range(
                                            maxiter=maxiter, tol=tol,
                                            mu=(1.0/mu), lmbda=lmbda,
                                            verbose=True,eci_cutoff=eci_cutoff,
                                            preserve_ground_state_range=preserve_ground_state_range,
                                            preserve_ground_state_range_iterations=preserve_ground_state_range_iterations,
                                            give_hypothetical_structure_trivial_weights=give_hypothetical_structure_trivial_weights)
            E_all = casm.compute_ce_energies()
            (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight
             ) = casm.compute_ce_errors(ce_energies=E_all)
            idx_all = casm.nonzero_ECIs()



        print(" Writing ECIs of optimized CE to file "
              "`{}'.".format(eci_file + "-opt"))
        casm.save_ECIs(eci_file + "-opt")
        print("")



    if (QP or MIQP) and not researchonfittingmode:
        min_mu=MuStart
        max_mu=MuEnd
        mus = list(np.logspace(log(min_mu,10), log(max_mu,10), MuPartition))
        non_weighted_rmse_list=[]
        weighted_rmse_list=[]
        num_non_zero_eci_list=[]

        if not researchonfittingmodeWeightadjusting:

            print ("without GS preservation")

            for i in range(len(mus)):
                mu_now=mus[i]
                if MIQP :
                    casm.perform_MIQP(mu=mu_now,concentrationmin=concentrationmin,
                                   concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
                else:
                    casm.perform_QP(mu=mu_now,concentrationmin=concentrationmin,
                                   concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)


                idx_del = [i for i in range(casm.N_corr)
                           if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
                casm.ecis[idx_del] = 0.0
                E_all = casm.compute_ce_energies()
                (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
                 ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
                idx_all = casm.nonzero_ECIs()
                # print ("without GS preservation")
                # print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
                #   "RMSE (no wght)    MAE               ME")
                # print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
                #       % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
                #          mue_all, mse_all))
                non_weighted_rmse_list.append(rmse_no_hypo_no_weight)
                weighted_rmse_list.append(rmse_no_hypo_weighted)
                num_non_zero_eci_now=len(idx_all)
                num_non_zero_eci_list.append(num_non_zero_eci_now)
                print ("mu_now, weighted_rmse_now, none-weighted_rmse_now , num_non_zero_eci_now is",mu_now, rmse_no_hypo_weighted, rmse_no_hypo_no_weight, num_non_zero_eci_now)

            # if MIQP :
            #     casm.perform_MIQP(mu=mu,concentrationmin=concentrationmin,
            #                concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            # else:
            #     casm.perform_QP(mu=mu,concentrationmin=concentrationmin,
            #                concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            #
            #
            # # casm.perform_QP(mu=mu,concentrationmin=concentrationmin,
            # #                    concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            # idx_del = [i for i in range(casm.N_corr)
            #            if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
            # casm.ecis[idx_del] = 0.0
            # E_all = casm.compute_ce_energies()
            # (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
            #  ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
            # idx_all = casm.nonzero_ECIs()
            # print ("without GS preservation")
            # print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
            #   "RMSE (no wght)    MAE               ME")
            # print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
            #       % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
            #          mue_all, mse_all))
            #
            #
            #
            # print ("with GS preservation")


            for i in range(len(mus)):
                mu_now=mus[i]
                if MIQP :
                    casm.perform_MIQP(mu=mu_now,concentrationmin=concentrationmin,
                                   concentrationmax=concentrationmax,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
                else:
                    casm.perform_QP(mu=mu_now,concentrationmin=concentrationmin,
                                   concentrationmax=concentrationmax,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)


                idx_del = [i for i in range(casm.N_corr)
                           if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
                casm.ecis[idx_del] = 0.0
                E_all = casm.compute_ce_energies()
                (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
                 ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
                idx_all = casm.nonzero_ECIs()
                # print ("without GS preservation")
                # print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
                #   "RMSE (no wght)    MAE               ME")
                # print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
                #       % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
                #          mue_all, mse_all))
                non_weighted_rmse_list.append(rmse_no_hypo_no_weight)
                weighted_rmse_list.append(rmse_no_hypo_weighted)
                num_non_zero_eci_now=len(idx_all)
                num_non_zero_eci_list.append(num_non_zero_eci_now)
                print ("mu_now, weighted_rmse_now, none-weighted_rmse_now , num_non_zero_eci_now is",mu_now, rmse_no_hypo_weighted, rmse_no_hypo_no_weight, num_non_zero_eci_now)


        # if researchonfittingmodeWeightadjusting:
        #     casm.add_self_eci_cutoff(eci_cutoff=eci_cutoff)
        #     casm.perform_weight_adjusting_algo_incremental(mu=mu,concentrationmin=concentrationmin,
        #            concentrationmax=concentrationmax)
        #
        # else:
        #     if MIQP:
        #         casm.perform_MIQP(mu=mu,concentrationmin=concentrationmin,
        #                            concentrationmax=concentrationmax,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
        #     elif QP:
        #         casm.perform_QP(mu=mu,concentrationmin=concentrationmin,
        #                            concentrationmax=concentrationmax,activate_GS_preservation=True,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
        #
        #
        #
        #
        # idx_del = [i for i in range(casm.N_corr)
        #            if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
        # casm.ecis[idx_del] = 0.0
        # E_all = casm.compute_ce_energies()
        # (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
        #  ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
        # idx_all = casm.nonzero_ECIs()
        # print ("with GS preservation")
        # print ("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
        #   "RMSE (no wght)    MAE               ME")
        # print (" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
        #       % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
        #          mue_all, mse_all))



        if (DeactivateGSPreservation==True):
            if MIQP:
                casm.perform_MIQP(mu=mu,concentrationmin=concentrationmin,
                           concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            else:
                casm.perform_QP(mu=mu,concentrationmin=concentrationmin,
                               concentrationmax=concentrationmax,activate_GS_preservation=False,AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)


            idx_del = [i for i in range(casm.N_corr)
                       if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
            casm.ecis[idx_del] = 0.0
            E_all = casm.compute_ce_energies()
            (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
             ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
            idx_all = casm.nonzero_ECIs()
            print ("without GS preservation")
            print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
              "RMSE (no wght)    MAE               ME")
            print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
                  % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
                     mue_all, mse_all))



    if researchonfittingmode:

        print("hello, I am in research mode")
        grand_list=range(len(casm.energy_in))
        shuffled_grandlist=grand_list[:]
        random.shuffle(shuffled_grandlist)

        partitioned_shuffled_grandlist= partition(shuffled_grandlist,partitionsnumberforcvscore)

        min_mu=MuStart
        max_mu=MuEnd
        # mus = list(np.logspace(log(min_mu,10), log(max_mu,10), 20))
        mus = list(np.logspace(log(min_mu,10), log(max_mu,10), MuPartition))
        avg_out_of_sample_rmse_list=[]
        rms_out_of_sample_rmse_list=[]
        weighted_rms_out_of_sample_rmse_list=[]
        avg_num_non_zero_eci_list=[]
        preservation_succeed=[]

        debug_now=False
        if (debug_now):
            for i in range(len(partitioned_shuffled_grandlist)):
                print("test creating subcasm now when excluding partition",i)
                partion_now= partitioned_shuffled_grandlist[i]
                rest_list_now=[x for x in grand_list if x not in partion_now]
                sub_casm_now=CASMSet_WX_create_sub(casm,rest_list_now)

                print("test creating subcasm for when only including partition",i)
                complement_casm=CASMSet_WX_create_sub(casm,partion_now)


        # mu_now=3

        for mu_now in mus:
            non_weighted_rmse_list=[]
            weighted_rmse_list=[]
            num_non_zero_eci_list=[]

            if (researchonfittingmodeWeightadjusting):
                test_casm_now=CASMSet_WX_create_sub(casm,grand_list)
                test_casm_now.add_self_eci_cutoff(eci_cutoff=eci_cutoff)
                test_casm_now.perform_weight_adjusting_algo_incremental(mu=mu_now,concentrationmin=concentrationmin,
                           concentrationmax=concentrationmax)
                preservation_succeed.append(test_casm_now.weight_algo_succeed)

            for i in range(len(partitioned_shuffled_grandlist)):

                partion_now= partitioned_shuffled_grandlist[i]
                rest_list_now=[x for x in grand_list if x not in partion_now]


                if SpecializedCvPengHaoGS:
                    rest_list_now=rest_list_now+list(casm.hull_idx)
                    rest_list_now=list(set(rest_list_now))
                    rest_list_now.sort()

                    paestion_now=[x for x in grand_list if x not in rest_list_now]

                sub_casm_now=CASMSet_WX_create_sub(casm,rest_list_now)

                deactivate=False
                if (DeactivateGSPreservation==True):
                    deactivate=True

                print("activate_GS_preservation=(not deactivate) is ", (not deactivate))


                if (researchonfittingmodeWeightadjusting and preservation_succeed[-1]==1):
                    print("i am performing cv score calculation with weight adjusting scheme with\
                     QP help but not with GS preservation constraint")
                    sub_casm_now.add_self_eci_cutoff(eci_cutoff=eci_cutoff)
                    sub_casm_now.perform_weight_adjusting_algo_incremental(mu=mu_now,concentrationmin=concentrationmin,
                           concentrationmax=concentrationmax)
                    # preservation_succeed.append(sub_casm_now.weight_algo_succeed)

                elif (researchonfittingmodeWeightadjusting and preservation_succeed[-1]==0):
                    sub_casm_now.ecis==np.zeros(sub_casm_now.N_corr)
                else:
                    if MIQP:
                        sub_casm_now.perform_MIQP(mu=mu_now,concentrationmin=concentrationmin,
                                       concentrationmax=concentrationmax,activate_GS_preservation=(not deactivate),AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)

                    else:
                        sub_casm_now.perform_QP(mu=mu_now,concentrationmin=concentrationmin,
                                       concentrationmax=concentrationmax,activate_GS_preservation=(not deactivate),AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)



                complement_casm=CASMSet_WX_create_sub(casm,partion_now)

                complement_casm.ecis=sub_casm_now.ecis

                idx_del = [i for i in range(complement_casm.N_corr)
                           if i not in complement_casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
                complement_casm.ecis[idx_del] = 0.0
                E_all = complement_casm.compute_ce_energies()
                (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
                 ) = complement_casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
                idx_all = complement_casm.nonzero_ECIs()

                # print("I am debugging now !")
                # print("complement_casm.ecis.shape is",complement_casm.ecis.shape)
                # print("complement_casm.correlations_in.shape",complement_casm.correlations_in.shape)
                # print("complement_casm.ecis ",complement_casm.ecis)
                # print("E_all is ",E_all)
                print("rmse_no_hypo_no_weight for this sub-partition i",i," is ",rmse_no_hypo_no_weight)
                print("rmse_no_hypo_weighted for this sub-partition i",i," is ",rmse_no_hypo_weighted)

                # print("finish debugging message")

                # print ("without GS preservation")
                # print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
                #   "RMSE (no wght)    MAE               ME")
                # print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
                #       % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
                #          mue_all, mse_all))

                non_weighted_rmse_list.append(rmse_no_hypo_no_weight)
                weighted_rmse_list.append(rmse_no_hypo_weighted)
                num_non_zero_eci_now=len(idx_all)
                num_non_zero_eci_list.append(num_non_zero_eci_now)
                # print ("mu_now, weighted_rmse_now, none-weighted_rmse_now , num_non_zero_eci_now is",mu_now, rmse_no_hypo_weighted, rmse_no_hypo_no_weight, num_non_zero_eci_now)

            avg_out_of_sample_rmse=np.mean(non_weighted_rmse_list)
            avg_out_of_sample_rmse_list.append(avg_out_of_sample_rmse)
            rms_out_of_sample_rmse = np.linalg.norm(np.array(non_weighted_rmse_list),ord=2)/np.sqrt(len(non_weighted_rmse_list))
            rms_out_of_sample_rmse_list.append(rms_out_of_sample_rmse)
            weighted_rms_out_of_sample_rmse = np.linalg.norm(np.array(weighted_rmse_list),ord=2)/np.sqrt(len(weighted_rmse_list))
            weighted_rms_out_of_sample_rmse_list.append(weighted_rms_out_of_sample_rmse)

            avg_num_non_zero_eci = np.mean(num_non_zero_eci_list)
            avg_num_non_zero_eci_list.append(avg_num_non_zero_eci)

            print("mu_now, avg_out_of_sample_rmse, rms_out_of_sample_rmse, weighted_rms_out_of_sample_rmse,avg_num_non_zero_eci is,",
                  mu_now, avg_out_of_sample_rmse,rms_out_of_sample_rmse, weighted_rms_out_of_sample_rmse,avg_num_non_zero_eci)

            if (debug_now):
                print("in case it fails in the future, let's print temporary results")
                mus_temp=[mus[i] for i in range(len(avg_out_of_sample_rmse_list))]
                for mu_print,avg_print,preserve_succeed_print in zip(mus_temp,avg_out_of_sample_rmse_list,preservation_succeed):
                    print(mu_print," ",avg_print," ",preserve_succeed_print)


        print("after all the iterations, let's print mu and avg_out_of_sample_rmse")


        print("mu     cv_score(avg_oos_rmse)    another_cv_score(rms_oos_rmse)    avg_num_non_zero_eci ")
        for mu_print,avg_print,rms_print,weighted_rms_print,avg_num_non_zero_eci_print in zip(mus,avg_out_of_sample_rmse_list,rms_out_of_sample_rmse_list,weighted_rms_out_of_sample_rmse_list,avg_num_non_zero_eci_list):
            print(mu_print," ",avg_print," ",rms_print," ",weighted_rms_print," ",avg_num_non_zero_eci_print)

        if (researchonfittingmodeWeightadjusting):
            print("additionally as we are doing  weight adjusting, let's check the following")
            print("mu     cv_score      succeed_in_preservation")
            for mu_print,avg_print,preserve_succeed_print in zip(mus,avg_out_of_sample_rmse_list,preservation_succeed):
                print(mu_print," ",avg_print," ",preserve_succeed_print)



        time_elapsed = (time.clock() - time_start)
        print("\ntotal time spent is ",time_elapsed)

        do_not_want_error=True
        if do_not_want_error:
            if MIQP:
                casm.perform_MIQP(mu=mu,concentrationmin=concentrationmin,
                   concentrationmax=concentrationmax,activate_GS_preservation=(not deactivate),AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            else:
                casm.perform_QP(mu=mu,concentrationmin=concentrationmin,
                       concentrationmax=concentrationmax,activate_GS_preservation=(not deactivate),AbsoluteErrorConstraintOnHull=AbsoluteErrorConstraintOnHull)
            idx_del = [i for i in range(casm.N_corr)
                       if i not in casm.nonzero_ECIs(eci_cutoff=eci_cutoff)]
            casm.ecis[idx_del] = 0.0
            E_all = casm.compute_ce_energies()
            (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
             ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
            idx_all = casm.nonzero_ECIs()
            if (not deactivate):
                print ("with GS preservation")
            elif deactivate:
                print("without GS preservation")
            print ("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
              "RMSE (no wght)    MAE               ME")
            print (" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
                  % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
                     mue_all, mse_all))


    E_all = casm.compute_ce_energies()
    (rmse_all, mue_all, mse_all, rmse_noW,rmse_no_hypo_no_weight,rmse_no_hypo_weighted
     ) = casm.compute_ce_errors(ce_energies=E_all,concentrationmin=concentrationmin,concentrationmax=concentrationmax,weighted_rmse_no_hypo=True)
    idx_all = casm.nonzero_ECIs()

    # CE energies and errors using all non-zero ECIs

    print("             RMSE_no_hypo_weighted        RMSE_no_hypo_noW       RMSE              "
          "RMSE (no wght)    MAE               ME")
    print(" %15s    %15.8f    %15.8f        %15.8f   %15.8f   %15.8f   %15.8f"
          % ("{} clusters".format(len(idx_all)),rmse_no_hypo_weighted,rmse_no_hypo_no_weight, rmse_all, rmse_noW,
             mue_all, mse_all))
    if simplify is not None:
        print(" %15s %15.8f   %15.8f   %15.8f   %15.8f"
              % ("{} clusters".format(nsimp), rmse_simp, rmse_noW_simp,
                 mue_simp, mse_simp))
    print("")



    # optionally, save structural weights
    if save_weights_file is not None:
        print(" Writing structure weights to file "
              "`{}'.".format(save_weights_file))
        casm.save_structure_weights(save_weights_file)

    # optionally, save CE energies
    if ce_energies_file is not None:
        print(" Writing CE energies to file `{}'.".format(ce_energies_file))
        casm.save_ce_energies(ce_energies_file, ce_energies_file + "-ignored")

    # optionally, save convex hull
    if ce_hull_file is not None:
        print(" Writing CE formation energy hull to "
              "file `{}'.".format(ce_hull_file))
        casm.save_convex_hull(ce_hull_file)

    # save ECIs in CASM format, if requested
    if casm_eci_file is not None:
        print(" Writing ECIs in CASM format to file "
              "`{}'.".format(casm_eci_file[1]))
        casm.save_casm_eci_file(eci_in_file=casm_eci_file[0],
                                eci_out_file=casm_eci_file[1])

    # compute leave-one-out CV score
    if not no_cv:
        print("\n Error estimate by cross validation for the CE with "
              "{} clusters:".format(casm.N_nonzero))
        (cv_score_ref, cv_score_all) = casm.compute_cv_score(
            maxiter=maxiter, tol=tol, mu=(1.0/mu), lmbda=lmbda,
            cv_energy_file='cv-energies.dat')
        print("   Leave-one-out CV score wrt. input   : "
              "{:.4f}".format(cv_score_ref))
        print("   Leave-one-out CV score wrt. full fit: " +
              "{:.4f}".format(cv_score_all))
        print("   Writing CV energies to file `cv-energies.dat'.")
        print("")





if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--energies", "-e",
        help="CASM 'energy' file",
        default="energy",
        dest="energies")

    parser.add_argument(
        "--correlations", "-c",
        help="CASM 'corr.in' file",
        default="corr.in",
        dest="correlations")

    parser.add_argument(
        "--ECIs",
        help="CASM 'eci.in' file",
        default="eci.in",
        dest="eci_in")

    parser.add_argument(
        "--max-rmsd",
        help="Maximum RMS distance between ideal and relaxed "
             "structures (default: 0.2).",
        type=float,
        default=0.2)

    parser.add_argument(
        "--favor-low-energy",
        help="Positively bias structures with energies below this value "
             "above the formation energy hull (default: None).",
        type=float,
        default=None)

    parser.add_argument(
        "--prevent-high-energy",
        help="Negatively bias structures with energies above this value "
             "above the formation energy hull (default: None).",
        type=float,
        default=None)

    parser.add_argument(
        "--bias-stable",
        help="Bias stable structures (default: 1).",
        action="store_true")

    parser.add_argument(
        "--optimize-weights",
        help="Run automatic structure weight optimization.",
        type=int,
        default=None)

    parser.add_argument(
        "--mu",
        help="Estimated amplitude of noise (default: 0.01).",
        type=float,
        default=0.01)

    parser.add_argument(
        "--lmbda",
        help="Algorithm parameter (lambda, default: 100)",
        type=float,
        default=100)

    parser.add_argument(
        "--maxiter",
        help="Maximum number of iteration (default: 100).",
        type=int,
        default=100)

    parser.add_argument(
        "--tol",
        help="Convergence criterium (default: 1.0e-3).",
        type=float,
        default=1.0e-3)

    parser.add_argument(
        "--shift-energies",
        help="Shift all input energies such that the lowest energy is zero.",
        action="store_true")

    parser.add_argument(
        "--eci-cutoff",
        help="ECIs below this cutoff are considered equal to zero "
             "(default: 1.0e-8).",
        type=float,
        default=1.0e-8)

    parser.add_argument(
        "--eci-file",
        help="File to store final ECIs (default: ecis.dat).",
        default="ecis.dat")

    parser.add_argument(
        "--casm-eci-file",
        help="Save ECIs in CASM format. "
             "First argument is the path to the 'eci.in' file, and "
             "the second argument is the path to the (new) output file.",
        type=str,
        default=None,
        nargs=2)

    parser.add_argument(
        "--simplify",
        help="Simplify CE by reducing the number of clusters while "
             "staying below the specified RMS error.",
        type=float,
        default=None)

    parser.add_argument(
        "--simplify-strict",
        help="Make '--simplify' try real hard to achieve the"
             "target RMS (slower).",
        action="store_true")

    parser.add_argument(
        "--save-energies",
        help="Store CE energies in specified file.",
        default=None)

    parser.add_argument(
        "--save-weights",
        help="Store structural weights in specified file.",
        default=None)

    parser.add_argument(
        "--save-hull",
        help="Store structural weights in specified file.",
        default=None)

    parser.add_argument(
        "--read-weights",
        help="Read structural weights from specified file.",
        type=str,
        default=None)

    parser.add_argument(
        "--no-cv-score",
        help="Do not compute CV score.",
        action="store_true")

    parser.add_argument(
        "--load-ecis",
        help="Load ECIs from file.  Do not perform CS-CE, "
             "just evaluate energies.",
        type=str,
        default=None)

    parser.add_argument(
        "--no-detect-redundant",
        help="Do not detect and delete redundant clusters",
        action="store_true")

    parser.add_argument(
        "--pca",
        help="Perform prinipcal component analysis to orthogonalize basis.",
        action="store_true")


    parser.add_argument(
        "--preserve-ground-state",
        help="Implementing weight correction to preserve Ground state. the integer value corresponds to the number of iterations where weight correction is applied",
        type=int,
        default=None)

    parser.add_argument(
        "--give-hypothetical-structure-trivial-weights",
        help="give very small weights to structure with directory named enum-hypo-",
        action="store_true")


    parser.add_argument(
        "--preserve-ground-state-range",
        help="Implementing weight correction to preserve Ground state energy within a certain range with respect to the previous gruond state",
        type=float,
        default=None)


    parser.add_argument(
        "--preserve-ground-state-range-iterations",
        help="Implementing weight correction to preserve Ground state. the integer value corresponds to the number of iterations where weight correction is applied",
        type=int,
        default=None)

    parser.add_argument(
        "--QP",
        help="use quadratic programming solver and hull constraints to solve the problem, we use mu with mu|x|+||Ax-b||^2 as the objective function",
        action="store_true",
        default=False)

    parser.add_argument(
        "--concentrationmin",
        help="relevant min concentration ",
        type=str,
        default=None)

    parser.add_argument(
        "--concentrationmax",
        help="relevant max concentration ",
        type=str,
        default=None)

    parser.add_argument(
        "--researchonfittingmode",
        help="calculate cv score ",
        action="store_true")


    parser.add_argument(
        "--partitionsnumberforcvscore",
        type=int,
        default=None)


    parser.add_argument(
        "--DeactivateGSPreservation",
        help="for QP fit, do not use GS preservation",
        action="store_true")

    parser.add_argument(
        "--researchonfittingmodeWeightadjusting",
        help="calculate cv score from weight adjusting for purpose of writing paper",
        default=False,
        action="store_true")

    parser.add_argument(
        "--RemoveOutsideConcentration",
        help="remove concentration outside ",
        default=False,
        action="store_true")

    parser.add_argument(
        "--weight_update_scheme",
        help="when we perform weight adjustment, we want to use different schemes to preserve GS, e.g. AddOne or Double",
        type=str,
        default="AddOne")

    parser.add_argument(
        "--MuPartition",
        help="how many mus do you perform your fit/crossvalidation",
        type=int,
        default=20)

    parser.add_argument(
        "--MuStart",
        help="starting mu",
        type=float,
        default=1e-4)

    parser.add_argument(
        "--MuEnd",
        help="ending mu",
        type=float,
        default=1e2)

    parser.add_argument(
        "--ReDefineFormE",
        help="redefining formation energy for hypothetical energy",
        default=False,
        action="store_true")


    parser.add_argument(
        "--AbsoluteErrorConstraintOnHull",
        help="Apply ",
        type=float,
        default=None)


    parser.add_argument(
        "--DiffFocus",
        help="Apply difference focused based on file diff_focused.txt please "
             "set this directed to  diff_focused.txt if you want to use it, it contains the index"
             "of structures",
        default=None)

    parser.add_argument(
        "--DiffFocusName",
        help="Apply difference focused based on file diff_focused_name.txt please "
             "set this directed to  diff_focused.txt if you want to use it, it contains the prtial"
             " dir names of structures",
        default=None)

    parser.add_argument(
        "--DiffFocusWeight",
        help="Apply ",
        type=float,
        default=100)

    parser.add_argument(
        "--SpecializedCvPengHaoGS",
        help="specialized procedure for computing cv score ground state set is always in ",
        default=False,
        action="store_true")

    parser.add_argument(
        "--SmallErrorOnInequality",
        help=" The small error on the inequality   ",
        type=float,
        default=1e-4)

    parser.add_argument(
        "--OnlyKeppEcis",
        help="point to only_keep_ecis.txt to only keep these ECIs",
        default=None)

    parser.add_argument(
        "--CompressFirstPair",
        help="do we want to  compress first pair? ",
        default=False,
        action="store_true")

    parser.add_argument(
        "--MIQP",
        help="use l0 norm compressed sensing ",
        default=False,
        action="store_true")

    parser.add_argument(
        "--UnCompressPairUptoDist",
        help="how long do we want to uncompress up to ",
        default=None,
        type=float)

    parser.add_argument(
        "--CompressAllTerms",
        help="CompressAllTerms ",
        default=False,
        action="store_true")

    parser.add_argument(
        "--MIQPGSPrsSolvingTime",
        help=" MIQP GS Preservation SolvingTime   ",
        type=float,
        default=300)

    parser.add_argument(
        "--MIQPNonGSPrsSolvingTime",
        help=" MIQP non GS Preservation SolvingTime   ",
        type=float,
        default=300)

    parser.add_argument(
        "--L0L1",
        help="Implement both l0 and l1 norm ",
        default=False,
        action="store_true")


    parser.add_argument(
        "--L0Hierarchy",
        help="L0 Hierarchy parent larger than  L0Hierarchy*child, L0 should just be 1",
        default=None,
        type=float)

    parser.add_argument(
        "--L1Hierarchy",
        help="L1 Hierarchy parent larger than  L1Hierarchy*child ",
        default=None,
        type=float)

    parser.add_argument(
        "--MaxNumClusts",
        help="maximum number of clusters",
        default=None,
        type=int)

    parser.add_argument(
        "--L0mu",
        help="L0 mu ",
        default=None,
        type=float)

    parser.add_argument(
        "--LinearScalingToUnweightHighEahStructs",
        help="use linear scaling to give structures with higher Eah lower weights w_i = w_i * (Eah/the_const)^2"
             "although you see that it seems to be quadratic, but the definition of this code is actually linear ",
        default=None,
        type=float)

    parser.add_argument(
        "--ExpScalingToUnweightHighEahStructs",
        help="Note! this is a place holder only, I have not implemented anything with this feature"
            "use exponential scaling to give structures with higher Eah lower weights w_i = w_i * Exp(-Eah/the_const)"
             "If assumed room temperature, the_const should be 0.01  ",
        default=None,
        type=float)




    args = parser.parse_args()
    print("\n " + " ".join(sys.argv) + "\n")

    if args.concentrationmin is not None:
        args.concentrationmax =np.array(tuple((eval(args.concentrationmax),)))

    if args.concentrationmax is not None:
        args.concentrationmax = np.array(tuple((eval(args.concentrationmax),)))


    print("args.concentrationmin is")
    print(args.concentrationmin)
    print("args.concentrationmax is")
    print(repr(args.concentrationmax))

    if args.L0L1:
        print("setting MIQP to be true since you set L0L1 is true")

        args.MIQP = True



    compressive_sensing_CE(args=args,
                           energy_file=args.energies,
                           corr_in_file=args.correlations,
                           eci_in_file=args.eci_in,
                           max_rmsd=args.max_rmsd,
                           bias_stable=args.bias_stable,
                           bias_low_energy=args.favor_low_energy,
                           bias_high_energy=args.prevent_high_energy,
                           optimize_weights=args.optimize_weights,
                           mu=args.mu,
                           lmbda=args.lmbda,
                           maxiter=args.maxiter,
                           tol=args.tol,
                           shift_energies=args.shift_energies,
                           eci_cutoff=args.eci_cutoff,
                           eci_file=args.eci_file,
                           casm_eci_file=args.casm_eci_file,
                           simplify=args.simplify,
                           simplify_strict=args.simplify_strict,
                           ce_energies_file=args.save_energies,
                           save_weights_file=args.save_weights,
                           read_weights_file=args.read_weights,
                           ce_hull_file=args.save_hull,
                           no_cv=args.no_cv_score,
                           eci_input_file=args.load_ecis,
                           no_detect_redundant=args.no_detect_redundant,
                           do_pca=args.pca,
                           preserve_GS=args.preserve_ground_state,
                           give_hypothetical_structure_trivial_weights=args.give_hypothetical_structure_trivial_weights,
                           preserve_ground_state_range=args.preserve_ground_state_range,
                           preserve_ground_state_range_iterations=args.preserve_ground_state_range_iterations,
                           QP=args.QP,
                           concentrationmin=args.concentrationmin,
                           concentrationmax=args.concentrationmax,
                           researchonfittingmode=args.researchonfittingmode,
                           partitionsnumberforcvscore=args.partitionsnumberforcvscore,
                           DeactivateGSPreservation=args.DeactivateGSPreservation,
                           researchonfittingmodeWeightadjusting=args.researchonfittingmodeWeightadjusting,
                           RemoveOutsideConcentration=args.RemoveOutsideConcentration,
                           MuPartition=args.MuPartition,
                           weight_update_scheme=args.weight_update_scheme,
                           MuStart=args.MuStart,
                           MuEnd=args.MuEnd,
                           ReDefineFormE=args.ReDefineFormE,
                           AbsoluteErrorConstraintOnHull=args.AbsoluteErrorConstraintOnHull,
                           DiffFocus=args.DiffFocus,
                           DiffFocusWeight=args.DiffFocusWeight,
                           SpecializedCvPengHaoGS=args.SpecializedCvPengHaoGS,
                           DiffFocusName=args.DiffFocusName,
                           SmallErrorOnInequality=args.SmallErrorOnInequality,
                           OnlyKeppEcis=args.OnlyKeppEcis,
                           CompressFirstPair=args.CompressFirstPair,
                           MIQP=args.MIQP,
                           UnCompressPairUptoDist=args.UnCompressPairUptoDist,
                           CompressAllTerms=args.CompressAllTerms,
                           MIQPGSPrsSolvingTime=args.MIQPGSPrsSolvingTime,
                           MIQPNonGSPrsSolvingTime=args.MIQPNonGSPrsSolvingTime,
                           L0L1=args.L0L1,
                           L0Hierarchy=args.L0Hierarchy,
                           L1Hierarchy=args.L1Hierarchy,
                           MaxNumClusts=args.MaxNumClusts,
                           L0mu=args.L0mu,
                           LinearScalingToUnweightHighEahStructs=args.LinearScalingToUnweightHighEahStructs,
                           ExpScalingToUnweightHighEahStructs=args.ExpScalingToUnweightHighEahStructs)

