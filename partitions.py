#!/usr/bin/env python3

##################################################################
#
# This file contains all the scripts necessary to reproduce 
# the results presented in the paper.
#
# The TESTS section at the end of this file provides functions  
# to recalculate the numerical results and generate the graphs 
# featured in the paper.
#
##################################################################

import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from fractions import Fraction
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from sympy import symbols, simplify
from math import log

def precompute_binomial_coefficients(max_n):
    binom_coeffs = {}
    for n in range(max_n + 1):
        for k in range(n + 1):
            binom_coeffs[(n, k)] = comb(n, k, exact=True)
    return binom_coeffs

## -------------------------------------------------------------
##              Gadget RP Refresh
## -------------------------------------------------------------

def generate_cardinal_set(n):
    stack = [(n, 1, [])]  # (remaining_n, start_value, current_comp)
    cardinal_sets = set()
    
    while stack:
        remaining_n, start_value, current_comp = stack.pop()

        for i in range(start_value, remaining_n + 1):
            new_comp = current_comp + [i]

            if remaining_n - i == 0:
                cardinal_sets.add(tuple(sorted(new_comp)))
            else:
                stack.append((remaining_n - i, i, new_comp))
    
    return sorted(list(cardinal_sets))

def initial_probability(cardinal_set):
    # Initial probability for each partition

    if max(cardinal_set)==1:
        return np.float64(1.0)
    else:
        return np.float64(0.0)

def new_probabilities(cardinal_sets_list, probabilities, binom_coeffs, p, n):
    # New probabilities for each partition after one iteration
    
    initial_probabilities = list(probabilities)
    new_probabilities = [np.float64(0.0)] * len(probabilities)

    cardinal_sets_dict = {tuple(cs): idx for idx, cs in enumerate(cardinal_sets_list)}

    z1z2_leak = p*p
    z1z2_dont_leak = 1-2*p+p*p
    r_doesnot_leak = 1 - (3*p - 3*p*p + p*p*p)
    r_leaks = 3*p - 3*p*p + p*p*p
    one_zi_leaks = p-p*p


    for ind_cs, cs in enumerate(cardinal_sets_list):
        
        if initial_probabilities[ind_cs] == 0.0:
            continue

        cs_count = Counter(cs)

        """
        Both drawn indices belong to the same set
        """           
        for cardinal_i, freq in cs_count.items():

            if cardinal_i < 2:
                continue
     
            proba_same_set_i = np.float64(cardinal_i)*np.float64(cardinal_i-1)/(np.float64(n)*np.float64(n-1))
            proba_for_update = initial_probabilities[ind_cs]*proba_same_set_i*freq

            i = cs.index(cardinal_i)

            # 1. i1 and i2 are in the same set, zi1, zi2 and r leak 
            #   => [1,1,c1,...,c{i-1},c{i}-2,c{i+1},...,cn]
            updated_set_1 = list(cs)
            if updated_set_1[i]==2:
                del updated_set_1[i]
            else:
                updated_set_1[i] -= 2
            updated_set_1.extend([1, 1])
            updated_proba_1 = z1z2_leak*r_leaks
            index_1 = find_cardinal_set_index(cardinal_sets_dict,updated_set_1)
            new_probabilities[index_1] += proba_for_update*updated_proba_1

            # 2. i1 and i2 are in the same set, zi1, zi2 leak, r does not leak
            #   => [2,c1,...,c{i-1},c{i}-2,c{i+1},...,cn]
            updated_set_2 = list(cs)
            if updated_set_2[i]>2:
                updated_set_2[i] -= 2
                updated_set_2.append(2)
                index_2 = find_cardinal_set_index(cardinal_sets_dict,updated_set_2)
            else:
                index_2 = ind_cs
            updated_proba_2 = z1z2_leak*r_doesnot_leak
            new_probabilities[index_2] += proba_for_update*updated_proba_2

            # 3/5. i1 and i2 are in the same set, exactly 1 of zi1/zi1 leaks, r leaks
            #   => [1,c1,...,c{i-1},c{i}-1,c{i+1},...,cn]
            updated_set_3 = list(cs)
            updated_set_3[i] -= 1
            updated_set_3.append(1)
            updated_proba_3 = 2*one_zi_leaks*r_leaks
            index_3 = find_cardinal_set_index(cardinal_sets_dict,updated_set_3)
            new_probabilities[index_3] += proba_for_update*updated_proba_3

            # 4/6. i1 and i2 are in the same set, exactly 1 of zi1/zi1 leaks, r does not leak
            #   => initial partition
            updated_proba_4 = 2*one_zi_leaks*r_doesnot_leak

            # 7/8. i1 and i2 are in the same set, zi1 does not leak, z2 does not leak
            #   => initial partition
            updated_proba_7 = z1z2_dont_leak
            new_probabilities[ind_cs] += proba_for_update*(updated_proba_4+updated_proba_7)

        """
        The two drawn indices belong to different sets.
        """ 
        for i1 in range(len(cs)):
            for i2 in range(i1+1,len(cs)):

                proba_sets_i1_i2 = np.float64(2.0)*np.float64(cs[i1])*np.float64(cs[i2])/(np.float64(n)*np.float64(n-1))
                proba_for_update = initial_probabilities[ind_cs]*proba_sets_i1_i2

                # If n_{i2} = 1, then n_{i1}=1 because they are ordered,
                if cs[i2]==1:
                    # 9/11/13/15. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                    #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                    updated_proba_9 = z1z2_leak*r_leaks
                    new_probabilities[ind_cs] += proba_for_update*r_leaks

                    # 10/12/14/16. i1 and i2 are not in the same set, r does not leak
                    #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                    updated_set_10 = list(cs) 
                    updated_set_10[i2] +=1
                    del updated_set_10[i1]
                    index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)
                    new_probabilities[index_10] += proba_for_update*r_doesnot_leak

                else:
                    # cs[i2]>1

                    if cs[i1]==1:
                        # 9. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                        #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_9 = list(cs)
                        updated_set_9[i2] -= 1
                        updated_set_9.append(1)
                        updated_proba_9 = z1z2_leak*r_leaks
                        index_9 = find_cardinal_set_index(cardinal_sets_dict,updated_set_9)

                        # 10. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r does not leak
                        #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_10 = list(cs) 
                        updated_set_10[i2] -= 1
                        updated_set_10[i1] += 1
                        updated_proba_10 = z1z2_leak*r_doesnot_leak
                        index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)

                        # 11. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2},c{i2+1},...,cn]
                        updated_proba_11 = one_zi_leaks*r_leaks

                        # 12. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}+1,c{i2+1},...,cn]
                        updated_set_12 = list(cs) 
                        updated_set_12[i2] +=1
                        del updated_set_12[i1]
                        updated_proba_12 = one_zi_leaks*r_doesnot_leak
                        index_12 = find_cardinal_set_index(cardinal_sets_dict,updated_set_12)

                        # 13. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1},c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_proba_13 = one_zi_leaks*r_leaks

                        # 14. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}+1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_proba_14 = one_zi_leaks*r_doesnot_leak

                        # 15. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r leaks
                        #   => ensemble initial
                        updated_proba_15 = z1z2_dont_leak*r_leaks

                        # 16. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1+1},...,c{i2-1},c{i2+1},...,cn,c{i1}+c{i2}]
                        updated_proba_16 = z1z2_dont_leak*r_doesnot_leak

                        new_probabilities[ind_cs] += proba_for_update*(updated_proba_11+updated_proba_15)
                        new_probabilities[index_9] += proba_for_update*(updated_proba_9+updated_proba_13)
                        new_probabilities[index_10] += proba_for_update*(updated_proba_10+updated_proba_14)
                        new_probabilities[index_12] += proba_for_update*(updated_proba_12+updated_proba_16)

                    else:   
                        # cs[i1]>1

                        # 9. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                        #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_9 = list(cs)
                        updated_set_9[i1] -= 1
                        updated_set_9.append(1)
                        updated_set_9[i2] -= 1
                        updated_set_9.append(1)
                        updated_proba_9 = z1z2_leak*r_leaks
                        index_9 = find_cardinal_set_index(cardinal_sets_dict,updated_set_9)
                        new_probabilities[index_9] += proba_for_update*updated_proba_9

                        # 10. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r does not leak
                        #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_10 = list(cs) 
                        updated_set_10[i2] -= 1
                        updated_set_10[i1] -= 1
                        updated_set_10.append(2)
                        updated_proba_10 = z1z2_leak*r_doesnot_leak
                        index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)
                        new_probabilities[index_10] += proba_for_update*updated_proba_10

                        # 11. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2},c{i2+1},...,cn]
                        updated_set_11 = list(cs)
                        updated_set_11[i1] -= 1
                        updated_set_11.append(1)
                        updated_proba_11 = one_zi_leaks*r_leaks
                        index_11 = find_cardinal_set_index(cardinal_sets_dict,updated_set_11)
                        new_probabilities[index_11] += proba_for_update*updated_proba_11

                        # 12. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}+1,c{i2+1},...,cn]
                        updated_set_12 = list(cs) 
                        updated_set_12[i2] +=1
                        updated_set_12[i1] -=1
                        updated_proba_12 = one_zi_leaks*r_doesnot_leak
                        index_12 = find_cardinal_set_index(cardinal_sets_dict,updated_set_12)
                        new_probabilities[index_12] += proba_for_update*updated_proba_12

                        # 13. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1},c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_13 = list(cs)
                        updated_set_13[i2] -= 1
                        updated_set_13.append(1)
                        updated_proba_13 = one_zi_leaks*r_leaks
                        index_13 = find_cardinal_set_index(cardinal_sets_dict,updated_set_13)
                        new_probabilities[index_13] += proba_for_update*updated_proba_13

                        # 14. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}+1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_14 = list(cs)
                        updated_set_14[i2] -=1
                        updated_set_14[i1] +=1
                        updated_proba_14 = one_zi_leaks*r_doesnot_leak
                        index_14 = find_cardinal_set_index(cardinal_sets_dict,updated_set_14)
                        new_probabilities[index_14] += proba_for_update*updated_proba_14

                        # 15. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r leaks
                        #   => initial set
                        updated_proba_15 = z1z2_dont_leak*r_leaks
                        new_probabilities[ind_cs] += proba_for_update*updated_proba_15

                        # 16. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1+1},...,c{i2-1},c{i2+1},...,cn,c{i1}+c{i2}]
                        updated_set_16 = list(cs)
                        updated_set_16[i2] += updated_set_16[i1]
                        del updated_set_16[i1]
                        updated_proba_16 = z1z2_dont_leak*r_doesnot_leak
                        index_16 = find_cardinal_set_index(cardinal_sets_dict,updated_set_16)
                        new_probabilities[index_16] += proba_for_update*updated_proba_16

    return new_probabilities

def final_probabilities(cardinal_sets_list, probas, binom_coeffs, p, n):
    # New probabilities for each partition with the possible leakage of (zi) while they are added to the inputs
    initial_probabilities = list(probas)
    new_probabilities = [0] * len(probas)

    # Convert cardinal_sets_list to a dictionary for O(1) lookups
    cardinal_sets_dict = {tuple(cs): idx for idx, cs in enumerate(cardinal_sets_list)}

    # Precompute the powers of p and (1-p) for efficiency
    p_powers = [p**i for i in range(n+1)]
    one_minus_p_powers = [(1-p)**i for i in range(n+1)]

    for ind_cs, cs in enumerate(cardinal_sets_list): 
        # (k1, k2, ..., kl) with ki the number of elements leaking in the set of cardinal ci
        all_combinations = itertools.product(*[range(c + 1) for c in cs])
        for combo in all_combinations:
            # we build the corresponding set cs_2
            s = sum(combo)
            # we remove all the leaking elements from the cardinals
            # we remove all the elements equal to zero
            # we add ones for all the leaking elements
            updated_set = [ci - ki for ci, ki in zip(cs, combo) if ci - ki > 0]
            updated_set.extend([1] * s)
            index_updated_set = find_cardinal_set_index(cardinal_sets_dict,updated_set)
            # we compute the probability of the leakage
            pleak = 1
            for ci, ki in zip(cs, combo):
                pleak *= binom_coeffs[(ci, ki)]*p_powers[ki]*one_minus_p_powers[ci-ki]
            new_probabilities[index_updated_set] += initial_probabilities[ind_cs]*pleak

    return new_probabilities

def find_cardinal_set_index(cardinal_sets_dict, target_set):
    # Find the index of target_set in cardinal_sets_list
    target_set = tuple(sorted(target_set))
    return cardinal_sets_dict.get(target_set)

def sum_probabilities(probabilities):
    s = 0
    for proba in probabilities:
        s += proba
    return simplify(s)

def process_partitions(n,nb_iter,p, binom_coeffs):
    cardinal_sets_list = generate_cardinal_set(n)
    initial_probabilities = [initial_probability(cs) for cs in cardinal_sets_list]
    probas = initial_probabilities
    for i in range(nb_iter):
        new_probas = new_probabilities(cardinal_sets_list, probas, binom_coeffs, p, n)
        probas = new_probas

    # leakage on the shares (z_i) during the addition with the inputs
    final_probas_new = final_probabilities(cardinal_sets_list, probas, binom_coeffs, p, n)
    return cardinal_sets_list, final_probas_new, p

def compute_probas_x_outputs(n,t_out,cardinal_sets_list,probas,binom_coeffs):
    # probabilities that the internal leakage and the given outputs
    # make it possible to recover x input shares
    probas_x_outputs = [0] * (t_out+1)
    binom_n_t_out = binom_coeffs[(n, t_out)]

    for ind_cs, cs in enumerate(cardinal_sets_list):    
        if min(cs)>t_out:
            probas_x_outputs[0] += probas[ind_cs]
        else:
            # Multihypergeometric law
            # 1. We enumerate all the (k1,k2,k_ell) with ell the number of cardinals cs
                    # ki = number of known outputs in the set of cardinal ci
            valid_combinations = [
                combo for combo in itertools.product(*[range(c + 1) for c in cs])
                if sum(combo) == t_out
            ]
            for combo in valid_combinations:
                # 2. for each of them, we compute s_k = sum(k_i, tq k_i = c_i)
                s_k = sum(k for k, c in zip(combo, cs) if k == c)
                # 3. we update probas_x_outputs[s_k] with probas[ind_cs]*(see multihypergeometric law)
                proba_this_combo = 1
                for i, k_i in enumerate(combo):    
                    proba_this_combo *= binom_coeffs[(cs[i],k_i)]
                proba_this_combo = proba_this_combo/binom_n_t_out
                probas_x_outputs[s_k] += probas[ind_cs]*proba_this_combo

    log2_probas_x_outputs = [math.log2(pr) if pr > 0 else float('-inf') for pr in probas_x_outputs]
    return probas_x_outputs

def compute_proba_leaking_input_shares(n,p,binom_coeffs):
    proba_leaking_input_shares = np.zeros((n+1, n+1))
    p_powers = [p**i for i in range(n+1)]
    one_minus_p_powers = [(1-p)**i for i in range(n+1)]
    for nt in range(0,n+1):
        for it in range(0,nt+1):
            proba_leaking_input_shares[nt,it] = binom_coeffs[(nt, it)]*p_powers[it]*one_minus_p_powers[nt-it]
    return proba_leaking_input_shares

def cardinal_rpc_refresh_precomp(n,t_in,t_out,p,nb_iter,cardinal_sets_list,probas,binom_coeffs,probas_x_outputs,proba_input_shares):
    p=p             

    proba_rpc = 0
    for tt in range(0,min(t_in+1,t_out+1)):
        # tt input shares are recovered from the internal leakage and the outputs
        # j input shares are still missing among the remaining ones
        j= t_in -tt
        proba_input_shares= binom_coeffs[(n-tt, j)]*(p**j)*(1-p)**(n-tt-j)
        proba_rpc += probas_x_outputs[tt]*proba_input_shares
    return proba_rpc

def cardinal_rpc_refresh(n,tin,tout,p,nb_iter):
    binom_coeffs = precompute_binomial_coefficients(n)
    cardinal_sets_list, probas, p = process_partitions(n, nb_iter, p, binom_coeffs)
    proba_leaking_input_shares = compute_proba_leaking_input_shares(n,p,binom_coeffs)
    probas_x_outputs = compute_probas_x_outputs(n,tout,cardinal_sets_list,probas,binom_coeffs)
    rpc_gref = cardinal_rpc_refresh_precomp(n,tin,tout,p,nb_iter,cardinal_sets_list,probas,binom_coeffs,probas_x_outputs,proba_leaking_input_shares)
    return rpc_gref

def cardinal_rpc_refresh_envelope(n,p,nb_iter):
    binom_coeffs = precompute_binomial_coefficients(n)
    cardinal_sets_list, probas, p = process_partitions(n, nb_iter, p, binom_coeffs)
    proba_leaking_input_shares = compute_proba_leaking_input_shares(n,p,binom_coeffs)
    pgref = np.zeros((n+1, n+1))
    for j in range(n+1):
        probas_x_outputs = compute_probas_x_outputs(n,j,cardinal_sets_list,probas,binom_coeffs)
        for i in range(n+1):
            pgref[i, j] = cardinal_rpc_refresh_precomp(n,i,j,p,nb_iter,cardinal_sets_list,probas,binom_coeffs,probas_x_outputs,proba_leaking_input_shares)
    return pgref

def cardinal_rpc_refresh_envelope_constraints(n,p,nb_iter,low_bound_tin,low_bound_tout):
    binom_coeffs = precompute_binomial_coefficients(n)
    print("process_partitions")
    cardinal_sets_list, probas, p = process_partitions(n, nb_iter, p, binom_coeffs)
    print("compute_proba_leaking_input_shares")
    proba_leaking_input_shares = compute_proba_leaking_input_shares(n,p,binom_coeffs)
    pgref = np.zeros((n+1, n+1))
    print("compute_probas_x_outputs")
    for j in range(low_bound_tout,n+1):
        probas_x_outputs = compute_probas_x_outputs(n,j,cardinal_sets_list,probas,binom_coeffs)
        for i in range(low_bound_tin,n+1):
            pgref[i, j] = cardinal_rpc_refresh_precomp(n,i,j,p,nb_iter,cardinal_sets_list,probas,binom_coeffs,probas_x_outputs,proba_leaking_input_shares)
    return pgref

## -------------------------------------------------------------
##              Gadget Gadd
## -------------------------------------------------------------
def cardinal_rpc_add_pgref(n,tin1,tin2,tout,p,nb_iter,pgref,binom_coeffs,p_powers,one_minus_p_powers):
    proba_rpc = 0
    for l1 in range(0,n-tout+1):
        pl1 = binom_coeffs[(n-tout,l1)]*pgref[tin1,tout+l1]
        for l2 in range(0,n-tout+1):
            pl2 = binom_coeffs[(n-tout,l2)]*p_powers[l1+l2] * one_minus_p_powers[2*n-2*tout-l1-l2] * pgref[tin2,tout+l2]
            proba_rpc += pl1*pl2
    return proba_rpc

def rpc_add(n,tin1,tin2,tout,p,nb_iter):
    binom_coeffs = precompute_binomial_coefficients(n)
    p_powers = [p**i for i in range(n+1)]
    one_minus_p_powers = [(1-p)**i for i in range(n+1)]
    pgref = cardinal_rpc_refresh_envelope_constraints(n,p,nb_iter,0,tout)
    filtered_tin_add = [
        tin for tin in itertools.product(range(n + 1), repeat=2) 
        if tin[0] > tin1 or tin[1] > tin2
    ]
    rpc_add=0
    for tin in filtered_tin_add:
        rpc_add += cardinal_rpc_add_pgref(n,tin[0],tin[1],tout,p,nb_iter,pgref,binom_coeffs,p_powers,one_minus_p_powers)
    return rpc_add

def rpc_add_pgadd(n,tin1,tin2,tout,p,nb_iter,pgadd):
    all_tin = list(itertools.product(range(n+1), repeat=2))
    filtered_tin_add = [tin for tin in all_tin if (tin[0]>tin1 or tin[1]>tin2)]
    rpc_add=0
    for tin in filtered_tin_add:
        rpc_add += pgadd[tin[0],tin[1],t]
    return rpc_add

def cardinal_rpc_add_envelope(n,p,pgref):
    pgadd = np.zeros((n+1, n+1, n+1))
    for i_1 in range(n+1):
        for i_2 in range(n+1):
            for j in range(n+1):
                pgadd[i_1, i_2, j] = 0
                for l1 in range(0,n-j+1):
                    for l2 in range(0,n-j+1):
                        pgadd[i_1, i_2, j] += comb(n-j, l1) * comb(n-j, l2) * (p**(l1+l2)) * (1-p)**(2*n-2*j-l1-l2) * pgref[i_1, j+l1] * pgref[i_2, j+l2]
    return pgadd

## -------------------------------------------------------------
##              Gadget Gsum with chain structure
## -------------------------------------------------------------
def cardinal_rpc_gaddpchain_pgadd(n,ell,tin,tout,pgadd):
    def recursive_sum(index, i_prev):
        if index == ell-1:  # Last sum (index ell-1)
            return pgadd[i_prev, tin[index], tout]
        else:
            total_sum = 0.0
            for i_curr in range(n+1):
                prob = pgadd[i_prev, tin[index], i_curr] * recursive_sum(index + 1, i_curr)
                total_sum += prob
            return total_sum

    total_probability = 0.0
    for i_3 in range(n+1):
        total_probability += pgadd[tin[0], tin[1], i_3] * recursive_sum(2, i_3)  # Commence à l'indice 2 pour la récursion
    return total_probability

def cardinal_rpc_gaddpchain_envelope_pgadd_constraints_sum(n,ell,pgadd,tin_a):
    pgaddpchain = np.zeros([n+1] * (ell+1))
    all_tin = list(itertools.product(range(n+1), repeat=ell))
    filtered_tin = [tin for tin in all_tin if sum(tin) > tin_a]
    for tin in filtered_tin:
        for j in range(n+1):
            pgaddpchain[(*tin, j)] = cardinal_rpc_gaddpchain_pgadd(n,ell,tin,j,pgadd)
    return pgaddpchain

def cardinal_rpc_gaddpchain_envelope_pgadd(n,ell,pgadd):
    pgaddpchain = np.zeros([n+1] * (ell+1))
    all_tin = list(itertools.product(range(n+1), repeat=ell))
    for tin in all_tin:
        for j in range(n+1):
            pgaddpchain[(*tin, j)] = cardinal_rpc_gaddpchain_pgadd(n,ell,tin,j,pgadd)
    return pgaddpchain

## -------------------------------------------------------------
##              Gadget Gsum with binary tree structure
## -------------------------------------------------------------
def cardinal_rpc_gaddptree_pgadd(n,ell,tin,tout,pgadd, memo=None):
    if memo is None:
        memo = {}

    key = (ell, tuple(tin), tout)
    if key in memo:
        return memo[key]

    if ell==2:
        result = pgadd[tin[0], tin[1], tout]
        memo[key] = result 
        return result

    if ell==3:
        p = 0
        for i in range(n + 1):
            p += pgadd[tin[0], tin[1], i] * pgadd[i, tin[2], tout]
        memo[key] = p 
        return p

    p = 0
    half_ell = ell // 2
    for i1 in range(n+1):
        for i2 in range(n+1):
            part1 = cardinal_rpc_gaddptree_pgadd(n, half_ell, tin[:half_ell], i1, pgadd, memo)
            part2 = cardinal_rpc_gaddptree_pgadd(n, ell - half_ell, tin[half_ell:], i2, pgadd, memo)
            p += part1 * part2 * pgadd[i1, i2, tout]

    memo[key] = p
    return p

def cardinal_rpc_gaddptree_envelope_pgadd(n,ell,pgadd):
    pgaddptree = np.zeros([n+1] * ell+ [n+1])
    all_tin = list(itertools.product(range(n+1), repeat=ell))
    for tin in all_tin:
        for j in range(n+1):
            pgaddptree[(*tin, j)] = cardinal_rpc_gaddptree_pgadd(n,ell,tin,j,pgadd)

    return pgaddptree

def rpc_gsumtree(n,t,ell,pgadd):
    #all_tin = list(itertools.product(range(n+1), repeat=ell))
    #filtered_tin = [tin for tin in all_tin if any(x > t for x in tin)]
    rpc_gaddp=0
    for tin in itertools.product(range(n+1), repeat=ell):
        if any(x > t for x in tin):
            rpc_gaddp+= cardinal_rpc_gaddptree_pgadd(n,ell,tin,t,pgadd)
    return rpc_gaddp

## -------------------------------------------------------------
##              Gadget AddNoiseTo
## -------------------------------------------------------------
def cardinal_rpc_addnoiseto_pgaddp_pgadd(n,ell,tin_x,tin_a,tout,pgaddp,pgadd):
    p = 0
    for i in range(n+1):
        p = p + pgaddp[(*tin_a, i)] * pgadd[tin_x,i,tout]
    return p

def cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,ell,pgaddp,pgadd):
    paddnoiseto = np.zeros([n+1]+[n+1]*ell+[n+1])
    all_tin_a = list(itertools.product(range(n+1), repeat=ell))
    for tin_a in all_tin_a:
        for tin_x in range(0,n+1):
            for tout in range(0,n+1):
                for i in range(n+1):
                    paddnoiseto[(tin_x,*tin_a,tout)] += pgaddp[(*tin_a, i)] * pgadd[tin_x,i,tout]
    return paddnoiseto

def rpc_addnoiseto_pgaddp_pgadd(n,ell,tin_x,tin_a,tout,pgaddp,pgadd):
    p = 0
    all_tin_a = list(itertools.product(range(n+1), repeat=ell))
    filtered_tin_a = [tin for tin in all_tin_a if sum(tin) > tin_a]
    for s_tin_x in range(tin_x+1,n):
        for s_tin_a in filtered_tin_a:
            for i in range(n+1):
                p = p + pgaddp[(*s_tin_a, i)] * pgadd[s_tin_x,i,tout]
    return p

## -------------------------------------------------------------
##              Gadget Gcopy
## -------------------------------------------------------------
def cardinal_rpc_gcopy_envelope_pgref_prec(n,pgref):
    binom_coeffs = precompute_binomial_coefficients(n)
    pgcopy = np.zeros((n+1, n+1, n+1))
    for tin in range(0,n+1):
        for i in range(0,tin+1):
            for j in range(tin-i,tin+1):
                ptemp = binom_coeffs[(i, i+j-tin)]*binom_coeffs[(n-i, tin-i)]/binom_coeffs[(n, j)]
                for tout_1 in range(0,n+1):
                    for tout_2 in range(0,n+1):
                        pgcopy[tin,tout_1,tout_2] += pgref[i,tout_1]*pgref[j,tout_2]*ptemp
    return pgcopy


#Added by Victor
def cardinal_rpc_gcopy_envelope_pgref(n, p, pgref):
    prec = cardinal_rpc_gcopy_envelope_pgref_prec(n,pgref)
    pgcopy = np.zeros((n + 1, n + 1, n + 1))
    for tin in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (tin + 1) :
            pgcopy[tin, tout1, tout2] += (comb(n - l, tin - l) * p**(tin - l) * 
                                    (1 - p)**(n - tin) * prec[l, tout1, tout2]) 
          pgcopy[tin, tout1, tout2] = min(1, pgcopy[tin, tout1, tout2])
    return pgcopy
  
              




## -------------------------------------------------------------
##              Gadget Gcmult
## -------------------------------------------------------------
def cardinal_rpc_gcmult_envelope_pgref(n,p,pgref):
    binom_coeffs = precompute_binomial_coefficients(n)
    pgcmult = np.zeros((n+1, n+1))
    for tin in range(0,n+1):
        for tout in range(0,n+1):
            for i in range(0,n-tout+1):
                pgcmult[tin,tout] += binom_coeffs[(n-tout, i)]*(p**i)*((1-p)**(n-tout-i))*pgref[tin,tout+i]
    return pgcmult

def rpc_gcmult_pgref(n,t,p,pgref):
    binom_coeffs = precompute_binomial_coefficients(n)
    p_rpc_cmult = 0
    for tin in range(t+1,n+1):
        for i in range(0,n-t+1):
            p_rpc_cmult += binom_coeffs[(n-t, i)]*(p**i)*((1-p)**(n-t-i))*pgref[tin,t+i]
    return p_rpc_cmult

## -------------------------------------------------------------
##              Gadget Gmmult
## -------------------------------------------------------------
def cardinal_rpc_gcopytree_pgcopy(n,ell,tin,tout,pgcopy, memo=None):
    if memo is None:
        memo = {}
    key = (ell, tin)
    if key in memo:
        return memo[key]

    if ell==2:
        result = pgcopy[tin, tout, tout]
        memo[key] = result
        return result
    if ell==3: 
        p = 0
        for i in range(0,n+1):
            p += pgcopy[tin,i,tout]*pgcopy[i,tout,tout]
        result = p
        memo[key] = result
        return result
    p = 0
    half_ell = ell // 2
    for o1 in range(0,n+1):
        for o2 in range(0,n+1):
            part1 = cardinal_rpc_gcopytree_pgcopy(n, half_ell, o1, tout, pgcopy, memo)
            part2 = cardinal_rpc_gcopytree_pgcopy(n, ell - half_ell, o2, tout, pgcopy, memo)
            p += part1 * part2 * pgcopy[tin, o1, o2]

    memo[key] = p
    return p

def rpc_gcopytree(n,t,ell,pgcopy):
    rpc_gcopytree=0
    for tin in range(t+1,n+1):
        rpc_gcopytree+= cardinal_rpc_gcopytree_pgcopy(n,ell,tin,t,pgcopy)
    return rpc_gcopytree

def rpc_gpolymult(n,t,p,raccoon_n,pgcopy,pgadd,pgref):
    # First copy step
    p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_n,pgcopy)*raccoon_n
    # Linear multiplication step
    p_rpc_gpolymult = rpc_gcmult_pgref(n,t,p,pgref)*raccoon_n*raccoon_n
    # Addition step
    p_rpc_gsumtree_4 = rpc_gsumtree(n,t,4,pgadd)
    p_rpc_gsumtree = p_rpc_gsumtree_4*(128+32+8+4+1)*raccoon_n
    return p_rpc_gcopytree+p_rpc_gpolymult+p_rpc_gsumtree

## -------------------------------------------------------------
##              Gadget Decode
## -------------------------------------------------------------
def nb_subtrees(n,x):
    if x>n:
        return 0
    if x==n:
        return 1
    if x==0:
        return 1
    return (nb_subtrees(n//2,x)+nb_subtrees(n-(n//2),x))

# Not optimal, with the number of leaking trees
def cardinal_rpc_gdecode_envelope_pgref(n,p,pgref):
    pgdecode= np.zeros(n+1)
    for i in range(0,n+1):
        if i==0:
            pshareadd = (1-p)**(2*n-1)
        elif i>=n-1:
            pshareadd = (1-(1-p)**3)**(n//2)
        else:
            nb_leaves = 2**math.ceil(math.log2(n-i+1))
            pshareadd = (1-(1-p)**(2*nb_leaves-1))**nb_subtrees(n,nb_leaves)            
        for tin in range(0,n+1): 
            pgdecode[tin] += pgref[tin,i]*pshareadd
    for i in range(0,n+1):
        pgdecode[i] = min(pgdecode[i],1)
    return pgdecode

# tight cRPC-O security

# Function to recursively compute the PMF of D^{(n)}
def compute_pmf_Dn(n, p, memo):
    # Base case for n=1
    if n == 1:
        return {(1, 1): p, (0, 0): 1 - p}
    
    # Check if already computed
    if n in memo:
        return memo[n]

    # Recursive case: compute PMF for smaller n
    pmf_D_half_floor = compute_pmf_Dn(math.floor(n / 2), p, memo)
    pmf_D_half_ceil = compute_pmf_Dn(math.ceil(n / 2), p, memo)

    # Dictionary to hold PMF for current n
    pmf_Dn = defaultdict(float)

    # Combine PMFs of D^{(floor(n/2))} and D^{(ceil(n/2))}
    for (r1, s1), prob1 in pmf_D_half_floor.items():
        for (r2, s2), prob2 in pmf_D_half_ceil.items():
            prob_combined = prob1 * prob2

            s = min(r1 + r2, s1 + math.ceil(n / 2), s2 + math.floor(n / 2))

            # First case: probability p
            r = n
            pmf_Dn[(r, s)] += p * prob_combined

            # Second case: probability (1 - p)
            r = r1 + r2
            pmf_Dn[(r, s)] += (1 - p) * prob_combined

    # Memoize the result
    memo[n] = dict(pmf_Dn)
    return memo[n]

# Function to compute the PMF of min(r, s) based on the PMF of D^{(n)}
def compute_pmf_min_rs(n,p):
    memo = {}
    pmf_Dn = compute_pmf_Dn(n, p, memo)

    # Dictionary to hold the PMF of min(r, s)
    pmf_min_rs = defaultdict(float)

    # Sum up probabilities for min(r, s)
    for (r, s), prob in pmf_Dn.items():
        pmf_min_rs[min(r, s)] += prob

    return dict(pmf_min_rs)

def decode_crpcwo_envelope(n,p,pgref):
    pmf_Tn = compute_pmf_min_rs(n,p)
    
    dec_enc = [np.longdouble(0.0)]*(n+1)
    
    for t_in in range(n+1):    
        for t_out in range(n+1):
            
            dec_enc[t_in] += pgref[t_in][t_out]*pmf_Tn[t_out]
            
    return dec_enc

## -------------------------------------------------------------
##              Raccoon-128
## -------------------------------------------------------------

## Key Generation 128-16 
def rpc_ai_keygen_block1(n,t,tri,pgaddp,pgcopy):
    p_rpc_ai_keygen_block1 = 0
    filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
    for tm in range(n+1):
        for tin in filtered_tin:
            p_rpc_ai_keygen_block1 += pgaddp[tin[0],tin[1],tm] * pgcopy[tm,t,t]
    return p_rpc_ai_keygen_block1

def rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode):
    p_rpc_ai_keygen_block2 = 0
    filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
    for tm in range(n+1):
        for tin in filtered_tin:
            for tin_block2 in range(t+1,n+1):
                p_rpc_ai_keygen_block2 += paddnoiseto[tin_block2,tin[0],tin[1],tm] * pgdecode[tm]
    return p_rpc_ai_keygen_block2

def rpc_keygen(n,t,tri,p,gamma1,gamma2,gamma3):
    raccoon_l = 4
    raccoon_k = 5
    raccoon_n = 512 

    # block 1
    print("\nRPC AI - Block 1")
    pgref1 = cardinal_rpc_refresh_envelope(n,p,gamma1)
    pgadd1 = cardinal_rpc_add_envelope(n,p,pgref1)
    pgaddp1 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd1)
    pgcopy1 = cardinal_rpc_gcopy_envelope_pgref(n,p,pgref1)
    p_rpc_ai_keygen_block1 = rpc_ai_keygen_block1(n,t,tri,pgaddp1,pgcopy1)*raccoon_l*raccoon_n
    print("threshold-RPC advantage block 1: 2^",math.log2(p_rpc_ai_keygen_block1))

    # block 2
    print("\nRPC AI - Block 2")
    pgref2 = cardinal_rpc_refresh_envelope(n,p,gamma2)
    pgadd2 = cardinal_rpc_add_envelope(n,p,pgref2)
    pgaddp2 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd2)
    paddnoiseto = cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,2,pgaddp2,pgadd2)
    pgdecode = decode_crpcwo_envelope(n,p,pgref2)
    p_rpc_ai_keygen_block2 = rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode)*raccoon_k*raccoon_n
    print("threshold-RPC advantage block 2: 2^",math.log2(p_rpc_ai_keygen_block2))

    # Gcmatmult
    print("\nRPC - Gcmmult")
    pgref3 = cardinal_rpc_refresh_envelope(n,p,gamma3)
    pgadd3 = cardinal_rpc_add_envelope(n,p,pgref3)
    pgcopy3 = cardinal_rpc_gcopy_envelope_pgref(n,p,pgref3)
    p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_k,pgcopy3) 
    p_rpc_gpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy3,pgadd3,pgref3)
    p_rpc_gsumtree = rpc_gsumtree(n,t,raccoon_l,pgadd3)
    p_rpc_gcmatmult = p_rpc_gcopytree*raccoon_l * raccoon_n + p_rpc_gpolymult*raccoon_k *raccoon_l + p_rpc_gsumtree*raccoon_k*raccoon_n
    print("threshold-RPC advantage Gcmmult: 2^",math.log2(p_rpc_gcmatmult))    

    p_keygen = p_rpc_ai_keygen_block1+p_rpc_gcmatmult+p_rpc_ai_keygen_block2
    return p_keygen

## Signature 128-16 
def rpc_ai_sign_block1(n,t,tri,pgaddp,pgcopy,pgcpolymult,pgadd,pgdecode,raccoon_l,raccoon_n):
    p_rpc_ai_sign_sub_block1 = np.zeros((n+1,n+1,n+1))
    p_rpc_ai_sign_sub_block2 = np.zeros((n+1,n+1))
    p_rpc_ai_sign_block1 = 0
    filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
    # Sub-block 1
    for tin in filtered_tin:
        for tm1 in range(0,n+1):
            for tm2 in range(0,n+1):
                p_rpc_ai_sign_sub_block1[tin[0],tin[1],tm2] += pgaddp[tin[0],tin[1],tm1] * pgcopy[tm1,tm2,t]
    # Sub-block 2
    for tin_s in range(t+1,n+1):
        for tm1 in range(0,n+1):
            for tm2 in range(0,n+1):
                for tm3 in range(0,n+1):
                    p_rpc_ai_sign_sub_block2[tin_s,tm1] += pgadd[tm2,tm1,tm3]*pgdecode[tm3]

    for tin in filtered_tin:
        for tin_s in range(t+1,n+1):
            for tm in range(0,n+1):
                p_rpc_ai_sign_block1 += p_rpc_ai_sign_sub_block1[tin[0],tin[1],tm]*p_rpc_ai_sign_sub_block2[tin_s,tm]

    p_rpc_ai_sign_block1 = p_rpc_ai_sign_block1*raccoon_l*raccoon_n+pgcpolymult*raccoon_l
    return p_rpc_ai_sign_block1

def rpc_sign(n,t,tri,p,gamma1,gamma2,gamma3):
    raccoon_l = 4
    raccoon_k = 5
    raccoon_n = 512 

    # block 1
    print("\nRPC AI - Block 1")
    pgref1 = cardinal_rpc_refresh_envelope(n,p,gamma1)
    pgadd1 = cardinal_rpc_add_envelope(n,p,pgref1)
    pgaddp1 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd1)
    pgcopy1 = cardinal_rpc_gcopy_envelope_pgref(n,p,pgref1)
    pgcpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy1,pgadd1,pgref1)
    pgdecode = decode_crpcwo_envelope(n,p,pgref1)
    p_rpc_ai_sign_block1 = rpc_ai_sign_block1(n,t,tri,pgaddp1,pgcopy1,pgcpolymult,pgadd1,pgdecode,raccoon_l,raccoon_n)
    print("threshold-RPC advantage block 1",math.log2(p_rpc_ai_sign_block1))

    # block 2
    print("\nRPC AI - Block 2")
    pgref2 = cardinal_rpc_refresh_envelope(n,p,gamma2)
    pgadd2 = cardinal_rpc_add_envelope(n,p,pgref2)
    pgaddp2 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd2)
    paddnoiseto = cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,2,pgaddp2,pgadd2)
    pgdecode = decode_crpcwo_envelope(n,p,pgref2)
    p_rpc_ai_sign_block2 = rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode)*raccoon_k*raccoon_n
    print("threshold-RPC advantage block 2: 2^",math.log2(p_rpc_ai_sign_block2))

    # Gcmatmult
    print("\nRPC - Gcmmult")
    pgref3 = cardinal_rpc_refresh_envelope(n,p,gamma3)
    pgadd3 = cardinal_rpc_add_envelope(n,p,pgref3)
    pgcopy3 = cardinal_rpc_gcopy_envelope_pgref(n,p,pgref3)
    p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_k,pgcopy3) 
    p_rpc_gpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy3,pgadd3,pgref3)
    p_rpc_gsumtree = rpc_gsumtree(n,t,raccoon_l,pgadd3)
    p_rpc_gcmatmult = p_rpc_gcopytree*raccoon_l * raccoon_n + p_rpc_gpolymult*raccoon_k *raccoon_l + p_rpc_gsumtree*raccoon_k*raccoon_n
    print("threshold-RPC advantage Gcmmult: 2^",math.log2(p_rpc_gcmatmult))    

    p_keygen = p_rpc_ai_sign_block1+p_rpc_gcmatmult+p_rpc_ai_sign_block2
    return p_keygen

## -------------------------------------------------------------
##              Raccoon-128 - Complexities
## -------------------------------------------------------------

def complexity_RPRefresh(n,gamma):
    return (2*gamma+n,0,gamma)

def complexity_Gadd(n,gamma):
    rprefresh_comp = complexity_RPRefresh(n,gamma)
    return (2*rprefresh_comp[0]+n,0,2*rprefresh_comp[2])

def complexity_Gcopy(n,gamma):
    rprefresh_comp = complexity_RPRefresh(n,gamma)
    return (2*rprefresh_comp[0],0,2*rprefresh_comp[2])

def complexity_Gcmult(n,gamma):
    rprefresh_comp = complexity_RPRefresh(n,gamma)
    return (rprefresh_comp[0],n,rprefresh_comp[2])

def complexity_Gcpmult(n,nR,gamma):
    gcopy_comp = complexity_Gcopy(n,gamma)
    gcmult_comp = complexity_Gcmult(n,gamma)
    gadd_comp = complexity_Gadd(n,gamma)
    nb_gcopy = (nR-1)*nR
    nb_gcmult = nR*nR
    nb_gadd = (nR-1)*nR
    gcpmult_gcopy = tuple(x * nb_gcopy for x in gcopy_comp)
    gcpmult_gcmult = tuple(x * nb_gcmult for x in gcmult_comp)
    gcpmult_gadd = tuple(x * nb_gadd for x in gadd_comp)
    gcpmult_comp = tuple(x + y + z for x, y, z in zip(gcpmult_gcopy, gcpmult_gcmult, gcpmult_gadd))
    return gcpmult_comp

def complexity_Gdecode(n,gamma):
    rprefresh_comp = complexity_RPRefresh(n,gamma)
    sums_add_comp = n-1
    return (rprefresh_comp[0]+sums_add_comp,0,rprefresh_comp[2])

def complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,gamma1,gamma2,gamma3):
    # -- Key Generation
    
    # KG_Gsum
    nb_gadd_b1 = (math.ceil(nb_r//n)-1)*raccoon_l*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma1)
    kg_gsum_b1_comp = tuple(x * nb_gadd_b1 for x in gadd_comp)
    
    # KG_Gcopy
    nb_gcopy_b1 = raccoon_l*raccoon_n
    gcopy_comp = complexity_Gcopy(n,gamma1)
    kg_gcopy_comp = tuple(x * nb_gcopy_b1 for x in gcopy_comp)

    kg_b1_comp = tuple(x + y for x, y in zip(kg_gsum_b1_comp, kg_gcopy_comp))

    # KG_Gcopies^k
    nb_gcopy_b2 = (raccoon_k-1)*raccoon_l*raccoon_n
    gcopy_comp = complexity_Gcopy(n,gamma2)
    kg_gcopies_comp = tuple(x * nb_gcopy_b2 for x in gcopy_comp)

    # KG_Gcpmult^k
    nb_gcpmult = raccoon_k*raccoon_l
    gcpmult_comp = complexity_Gcpmult(n,raccoon_n,gamma2)
    kg_gcpmult_comp = tuple(x * nb_gcpmult for x in gcpmult_comp)

    # KG_Gsum
    nb_gadd_b2 = (raccoon_l-1)*raccoon_k*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma2)
    kg_gsum_b2_comp = tuple(x * nb_gadd_b2 for x in gadd_comp)

    kg_b2_comp = tuple(x + y + z for x, y, z in zip(kg_gcopies_comp, kg_gcpmult_comp, kg_gsum_b2_comp))

    # KG_AddNoiseTo
    nb_gadd_b3 = (math.ceil(nb_r//n))*raccoon_k*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma3)
    kg_addnoiseto_comp = tuple(x * nb_gadd_b3 for x in gadd_comp)   

    # KG_Gdecode
    nb_gdecode = raccoon_k*raccoon_n
    gdecode_comp = complexity_Gdecode(n,gamma3)
    kg_gdecode_comp = tuple(x * nb_gdecode for x in gdecode_comp) 

    kg_b3_comp = tuple(x + y for x, y in zip(kg_addnoiseto_comp, kg_gdecode_comp))

    kg_comp = tuple(x + y + z for x, y, z in zip(kg_b1_comp, kg_b2_comp, kg_b3_comp))

    return kg_comp

def complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,gamma1,gamma2,gamma3):
    # -- Signature
    
    # Sign_Gsum
    nb_gadd_b1 = (math.ceil(nb_r//n)-1)*raccoon_l*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma1)
    sign_gsum_b1_comp = tuple(x * nb_gadd_b1 for x in gadd_comp)
    
    # Sign_Gcopy
    nb_gcopy_b1 = raccoon_l*raccoon_n
    gcopy_comp = complexity_Gcopy(n,gamma1)
    sign_gcopy_comp = tuple(x * nb_gcopy_b1 for x in gcopy_comp)

    # Sign_Gcpmult
    nb_gcpmult_b1 = raccoon_l
    gcpmult_comp = complexity_Gcpmult(n,raccoon_n,gamma1)
    sign_gcpmult_comp = tuple(x * nb_gcpmult_b1 for x in gcpmult_comp)

    # Sign_Gadd
    nb_gadd_b1_2 = raccoon_n*raccoon_l
    sign_gadd_b1_comp = tuple(x * nb_gadd_b1_2 for x in gadd_comp)

    # Sign_Gdecode
    nb_gdecode = raccoon_l*raccoon_n
    gdecode_comp = complexity_Gdecode(n,gamma1)
    sign_gdecode_comp = tuple(x * nb_gdecode for x in gdecode_comp) 

    sign_b1_comp = tuple(x + y + z + u + w for x, y, z, u, w in zip(sign_gsum_b1_comp, sign_gcopy_comp, sign_gcpmult_comp, sign_gadd_b1_comp, sign_gdecode_comp))

    # Sign_Gcopies^k
    nb_gcopy_b2 = (raccoon_k-1)*raccoon_l*raccoon_n
    gcopy_comp = complexity_Gcopy(n,gamma2)
    sign_gcopies_comp = tuple(x * nb_gcopy_b2 for x in gcopy_comp)

    # Sign_Gcpmult^k
    nb_gcpmult = raccoon_k*raccoon_l
    gcpmult_comp = complexity_Gcpmult(n,raccoon_n,gamma2)
    sign_gcpmult_comp = tuple(x * nb_gcpmult for x in gcpmult_comp)

    # Sign_Gsum
    nb_gadd_b2 = (raccoon_l-1)*raccoon_k*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma2)
    sign_gsum_b2_comp = tuple(x * nb_gadd_b2 for x in gadd_comp)

    sign_b2_comp = tuple(x + y + z for x, y, z in zip(sign_gcopies_comp, sign_gcpmult_comp, sign_gsum_b2_comp))

    # Sign_AddNoiseTo
    nb_gadd_b3 = (math.ceil(nb_r//n))*raccoon_k*raccoon_n
    gadd_comp = complexity_Gadd(n,gamma3)
    sign_addnoiseto_comp = tuple(x * nb_gadd_b3 for x in gadd_comp)   

    # Sign_Gdecode
    nb_gdecode = raccoon_k*raccoon_n
    gdecode_comp = complexity_Gdecode(n,gamma3)
    sign_gdecode_comp = tuple(x * nb_gdecode for x in gdecode_comp) 

    sign_b3_comp = tuple(x + y for x, y in zip(sign_addnoiseto_comp, sign_gdecode_comp))

    sign_comp = tuple(x + y + z for x, y, z in zip(sign_b1_comp, sign_b2_comp, sign_b3_comp))

    return sign_comp


## -------------------------------------------------------------
##              Tests
## -------------------------------------------------------------

def cardinal_rpc_refresh_timings():
    n=8
    gamma=50
    p= 2**-8
    start_time = time.time()
    for i in range(10):
        env1 = cardinal_rpc_refresh_envelope(n,p,gamma)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Execution time for (n,gamma)=(",n,",",gamma,") : ",elapsed_time/10," seconds")
    
    n=16
    gamma=100
    p= 2**-8
    start_time = time.time()
    for i in range(10):
        env1 = cardinal_rpc_refresh_envelope(n,p,gamma)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Execution time for (n,gamma)=(",n,",",gamma,") : ",elapsed_time/10," seconds")

    n=20
    gamma=120
    p= 2**-8
    start_time = time.time()
    for i in range(10):
        env1 = cardinal_rpc_refresh_envelope(n,p,gamma)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Execution time for (n,gamma)=(",n,",",gamma,") : ",elapsed_time/10," seconds")

def rpc_adv_gamma_figure():
    n_values = [4,8,16]
    t_values = [2,4,8]
    p_values = [2**-10,2**-16,2**-3]
    gamma_values = list(range(10, 150, 10))
    p_rpc_values = [[] for _ in range(3)]

    for i in range(4):
        for gamma in range(10, 150, 10):
            pgref = cardinal_rpc_refresh_envelope(n_values[i], p_values[0], gamma)  
            p_rpc = 0
            for tin in range(t_values[i]+1, n_values[i]+1):
                p_rpc += pgref[tin, t_values[i]]  
            p_rpc_values[i].append(p_rpc)
    
        plt.plot(gamma_values, p_rpc_values[i], marker='.', label=f'n = {((n_values[i]))}')
    
        threshold_value = comb(n_values[i], t_values[i]+1) * p_values[0]**(t_values[i]+1) * (1-p_values[0])**(n_values[i]-t_values[i]-1)
        plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
    
    plt.xlabel("Number of random values γ")
    current_ylim = plt.ylim()
    plt.ylim(2**-96)
    plt.yscale('log', base=2)
    plt.gca().invert_yaxis()
    plt.ylabel("RPC (n,t,p)")
    plt.legend()
    plt.grid(True)
    plt.savefig("rpc-refresh_p210_bounds.png", format="png")
    plt.close()
    
    p_rpc_values = [[] for _ in range(4)]
    for i in range(4):
        for gamma in range(10, 150, 10):
            pgref = cardinal_rpc_refresh_envelope(n_values[i], p_values[1], gamma)  
            p_rpc = 0
            for tin in range(t_values[i]+1, n_values[i]+1):
                p_rpc += pgref[tin, t_values[i]]  
            print("gamma",gamma,"p_rpc",math.log2(p_rpc))
            p_rpc_values[i].append(p_rpc)
    
        plt.plot(gamma_values, p_rpc_values[i], marker='.', label=f'n = {((n_values[i]))}')
    
        threshold_value = comb(n_values[i], t_values[i]+1) * p_values[1]**(t_values[i]+1) * (1-p_values[1])**(n_values[i]-t_values[i]-1)
        plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
    
    plt.xlabel("Number of random values γ")
    plt.yscale('log', base=2)
    plt.gca().invert_yaxis()
    plt.ylabel("RPC (n,t,p)")
    plt.legend()
    plt.grid(True)
    plt.savefig("rpc-refresh_p216_bounds.png", format="png")   
    plt.close()

    for i in range(3):
        for gamma in range(10, 150, 10):
            pgref = cardinal_rpc_refresh_envelope(n_values[i], p_values[0], gamma)  
            p_rpc = 0
            for tin in range(t_values[i]+1, n_values[i]+1):
                p_rpc += pgref[tin, t_values[i]]  
            p_rpc_values[i].append(p_rpc)
    
        plt.plot(gamma_values, p_rpc_values[i], marker='.', label=f'n = {((n_values[i]))}')
        #plt.plot([n_values[i] * (log(n_values[i], 2))], [eps_RPC_nlogn (n_values[i], t_values[i], p_values[0])], linestyle = '', marker ='x', label="Rnlogn, n = " + str(n_values[i]))
        
        
        threshold_value = comb(n_values[i], t_values[i]+1) * p_values[0]**(t_values[i]+1) * (1-p_values[0])**(n_values[i]-t_values[i]-1)
        plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
    
    plt.xlabel("Number of random values γ")
    current_ylim = plt.ylim()
    plt.ylim(2**-140)
    plt.yscale('log', base=2)
    plt.gca().invert_yaxis()
    plt.ylabel("RPC (n,t,p)")
    plt.legend()
    plt.grid(True)
    plt.savefig("rpc-refresh_p10_bounds.png", format="png")
    plt.close()

def probing_to_rp():
    def eps_rps(p,t,s):
        p_rps = 0
        for i in range(t,s+1):
            p_rps += comb(s,i)*(p**i)*((1-p)**(s-i))
        return p_rps

    # Raccoon's key generation and signature
    # nb shares
    n = 8
    # nb gates
    s_keygen = 84900000+83900000
    s_sign = 102000000+101000000

    # leakage probability
    p = 2**-10

    # corresponding RP advantages
    eps_keygen = eps_rps(p,n-1,s_keygen)
    eps_sign = eps_rps(p,n-1,s_sign)
    print("Trivial RP Security of Raccoon's Key Generation for p = 2^",math.log2(p),": 2^",math.log2(eps_keygen))
    print("Trivial RP Security of Raccoon's Signature for p = 2^",math.log2(p),": 2^",math.log2(eps_sign))

    # n log(n) refresh
    n_values = [4,8,12,16,20] 
    p_values = [2**-10,2**-16]
    for j in range(len(p_values)):
        for i in range(len(n_values)):
            s_nlogn = int(2.5*n_values[i]*math.log2(n_values[i])) # nb additions
            eps_nlogn = eps_rps(p_values[j],n_values[i]//2,s_nlogn)

            print("Trivial RP Security of Refresh n log(n) with n = ",n_values[i]," and p = 2^",math.log2(p_values[j])," : 2^",math.log2(eps_nlogn))
            print("-- Number of random values = ",n_values[i]*math.log2(n_values[i])/2,"\n")

def threshold_rpc_raccoon_keygen():
    n=16
    t=8
    tri=15
    p_values = [2**-24,2**-20,2**-16,2**-12]
    gamma_values_128 = [(60,25,125),(60,30,135)]
    gamma_values_80 = [(40,15,80),(40,15,80),(40,20,85)]
    gamma_values_64 = [(30,10,65),(30,10,70),(30,15,70),(30,20,115)]

    # Security level <= 2^-128
    print("\n\n \t Security Level <= 2^-128 \n")
    for i in range(2):
        res_keygen_128 = rpc_keygen(n,t,tri,p_values[i],*(gamma_values_128[i]))
        print("\nRPC advantage KeyGen for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_128[i][0],gamma_values_128[i][1],gamma_values_128[i][2],") = ",math.log2(res_keygen_128),"\n")
        if math.log2(res_keygen_128)>-128:
            print("ERROR\n")
            sys.exit()

    # Security level <= 2^-80
    print("\n\n \t Security Level <= 2^-80 \n")
    for i in range(3):
        res_keygen_80 = rpc_keygen(n,t,tri,p_values[i],*(gamma_values_80[i]))
        print("\nRPC advantage KeyGen for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_80[i][0],gamma_values_80[i][1],gamma_values_80[i][2],") = ",math.log2(res_keygen_80),"\n")
        if math.log2(res_keygen_80)>-80:
            print("ERROR\n")
            sys.exit()

    # Security level <= 2^-64
    print("\n\n \t Security Level <= 2^-64 \n")
    for i in range(4):
        res_keygen_64 = rpc_keygen(n,t,tri,p_values[i],*(gamma_values_64[i]))
        print("\nRPC advantage KeyGen for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_64[i][0],gamma_values_64[i][1],gamma_values_64[i][2],") = ",math.log2(res_keygen_64),"\n")
        if math.log2(res_keygen_64)>-64:
            print("ERROR\n")
            sys.exit()

def threshold_rpc_raccoon_sign():
    n=16
    t=8
    tri=15
    p_values = [2**-24,2**-20,2**-16,2**-12]
    gamma_values_128 = [(120,30,125),(125,65,130)]
    gamma_values_80 = [(75,20,80),(80,20,80),(85,25,85)]
    gamma_values_64 = [(60,15,65),(65,15,70),(70,20,70),(75,25,85)]

    # Security level <= 2^-128
    print("\n\n \t Security Level <= 2^-128 \n")
    for i in range(2):
        res_sign_128 = rpc_sign(n,t,tri,p_values[i],*(gamma_values_128[i]))
        print("\nRPC advantage Sign for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_128[i][0],gamma_values_128[i][1],gamma_values_128[i][2],") = ",math.log2(res_sign_128),"\n")
        if math.log2(res_sign_128)>-128:
            print("ERROR\n")
            sys.exit()

    # Security level <= 2^-80
    print("\n\n \t Security Level <= 2^-80 \n")
    for i in range(3):
        res_sign_80 = rpc_sign(n,t,tri,p_values[i],*(gamma_values_80[i]))
        print("\nRPC advantage Sign for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_80[i][0],gamma_values_80[i][1],gamma_values_80[i][2],") = ",math.log2(res_sign_80),"\n")
        if math.log2(res_sign_80)>-80:
            print("ERROR\n")
            sys.exit()


    # Security level <= 2^-64
    print("\n\n \t Security Level <= 2^-64 \n")
    for i in range(4):
        res_sign_64 = rpc_sign(n,t,tri,p_values[i],*(gamma_values_64[i]))
        print("\nRPC advantage Sign for p = 2^",math.log2(p_values[i])," and gamma = (",gamma_values_64[i][0],gamma_values_64[i][1],gamma_values_64[i][2],") = ",math.log2(res_sign_64),"\n")
        if math.log2(res_sign_64)>-64:
            print("ERROR\n")
            sys.exit()

def complexities_theorems():
    n=16
    nb_r=32
    raccoon_l=4
    raccoon_k=5
    raccoon_n=512

    kg_gamma_values_128 = [(60,25,125),(60,30,135)]
    kg_gamma_values_80 = [(40,15,80),(40,15,80),(40,20,85)]
    kg_gamma_values_64 = [(30,10,65),(30,10,70),(30,15,70),(30,20,115)]
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_128[0])
    print("Complexity KG (security 128 bits) - p = 2^-24 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_128[1])
    print("Complexity KG (security 128 bits) - p = 2^-20 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_80[0])
    print("Complexity KG (security 80 bits) - p = 2^-24 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_80[1])
    print("Complexity KG (security 80 bits) - p = 2^-20 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_80[2])
    print("Complexity KG (security 80 bits) - p = 2^-16 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_64[0])
    print("Complexity KG (security 64 bits) - p = 2^-24 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_64[1])
    print("Complexity KG (security 64 bits) - p = 2^-20 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_64[2])
    print("Complexity KG (security 64 bits) - p = 2^-16 : ",comp)
    comp = complexity_keygeneration(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*kg_gamma_values_64[3])
    print("Complexity KG (security 64 bits) - p = 2^-12 : ",comp)

    sign_gamma_values_128 = [(120,30,125),(125,65,130)]
    sign_gamma_values_80 = [(75,20,80),(80,20,80),(85,25,85)]
    sign_gamma_values_64 = [(60,15,65),(65,15,70),(70,20,70),(75,25,85)]
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_128[0])
    print("Complexity Sign (security 128 bits) - p = 2^-24 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_128[1])
    print("Complexity Sign (security 128 bits) - p = 2^-20 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_80[0])
    print("Complexity Sign (security 80 bits) - p = 2^-24 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_80[1])
    print("Complexity Sign (security 80 bits) - p = 2^-20 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_80[2])
    print("Complexity Sign (security 80 bits) - p = 2^-16 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_64[0])
    print("Complexity Sign (security 64 bits) - p = 2^-24 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_64[1])
    print("Complexity Sign (security 64 bits) - p = 2^-20 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_64[2])
    print("Complexity Sign (security 64 bits) - p = 2^-16 : ",comp)
    comp = complexity_signature(n,nb_r,raccoon_l,raccoon_k,raccoon_n,*sign_gamma_values_64[3])
    print("Complexity Sign (security 64 bits) - p = 2^-12 : ",comp)



if __name__ == "main" :
  print("main")
  #rpc_adv_gamma_figure()

