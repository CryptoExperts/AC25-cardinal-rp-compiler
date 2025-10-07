#!/usr/bin/env python3

# This file computes the cardinal-RPC envelopes uniform variant of CardSecMult.
# Moreover, it computes it by cutting the number of outputs of MatMult in 4 to 
# obtain better security. 
# In fact, MatMult exposes four outputs of size n**2/4, one per subtree (see 
# Figure 5 of the paper)—instead of a single aggregated output of size n**2. 
# Accordingly, TreeAdd takes four inputs, each of size n**2/4, instead of a 
# single input of size n**2. 
# The implementation targets numbers of shares n that are powers of two.
# This module is used to produce Figure 7 of the full paper (uniform symmetric 
# variant),generating the points labeled `Unif = 4` and `Unif = 8` on the graph.


from threading import Thread
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.special import comb

from partitions import cardinal_rpc_refresh_envelope, cardinal_rpc_add_envelope

from multiprocessing import Pool, Process, Queue
from math import log

################################################################################
############################ MatMult Enveloppes ################################

def precomp_hypergeom(n) :
  """
  Precompute hypergeometric weights used in the envelope convolutions.

  Computes the 3D tensor ``precomp`` with entries
    precomp[i11, i12, l1] = C(i11, l1) * C(n - i11, i12 - l1) / C(n, i12)
  whenever ``C(n, i12)`` is non-zero; otherwise the entry is left at 0.

  Args:
    n (int): Number of share; indices range over 0..n.

  Returns:
    numpy.ndarray: Array of shape ``(n+1, n+1, n+1)`` indexed as
      ``[i11, i12, l1]``.
  """
  precomp = np.zeros((n + 1, n + 1, n + 1))
  for i11 in range (n + 1) :
    for l1 in range (i11 + 1) :
      comb1 = comb(i11, l1)  
      for i12 in range (n + 1) :
        comb2 = comb(n - i11 , i12 - l1)
        denom_comb = comb (n, i12)
        if denom_comb :
          precomp[i11, i12, l1] = (comb1 * comb2) / denom_comb
  
  return precomp

def precompute_single_proba (n, envn_MM, envrefn):
  """
  Precompute threshold probabilities P(|I_x|=ix, |I_y|=iy | |J|=j) 
  for all relevant indices.

  This computes, for each ``j = j1 + j2 + j3 + j4`` and all pairs ``(ix, iy)``,
  the (max-)threshold of the probabilities obtained by combining:
    1) a MatMult gadget with envelopes ``envn_MM`` used with ``n`` shares, and
    2) a refresh gadget with envelopes ``envrefn``, used with ``n`` shares.

  The output contains four 3D tensors (cases 1..4) of shape
  ``(lim_card+1, n+1, n+1)`` where ``lim_card = n**2``. For each tensor
  and for each index triple ``(j, ix, iy)``, we take the **maximum** value
  over all quadruples ``(j1, j2, j3, j4)`` such that
  ``j = j1 + j2 + j3 + j4`` with ``j1, j2, j3, j4 ∈ [0, n^2/4]``.

  Args:
    n (int): Number of shares.
    envn_MM (np.ndarray): 6D array of multiplication-gadget envelopes with
    indices ``[ix, iy, j1, j2, j3, j4]`` where:
      - ``ix`` ranges over ``0..n``
      - ``iy`` ranges over ``0..n``
      - each ``j•`` ranges over ``0..n^2/4``
    envrefn (np.ndarray): 2D array of refresh-gadget envelopes with indices
    ``[i, l]`` for ``i, l ∈ 0..n``.

  Returns:
    List[np.ndarray]: A list ``[case_one, case_two, case_three, case_four]``,
    each of shape ``(n**2 + 1, n + 1, n + 1)``:
      - ``case_one[j, ix, iy] = max_{j1..j4: sum=j} envn_MM[ix, iy, j1, j2, j3, j4]``
      - ``case_two[j, ix, iy] = max … sum_l envrefn[iy, l] * envn_MM[ix, l, j…]``
      - ``case_three[j, ix, iy] = max … sum_l envrefn[ix, l] * envn_MM[l, iy, j…]``
      - ``case_four[j, ix, iy] = max … sum_{l,l1} envrefn[ix, l] * envrefn[iy, l1] * envn_MM[l, l1, j…]``
  """   
  lim_card = n**2
  lim_card_rec = int(n**2 / 4)
  
  case_one = np.zeros((lim_card + 1, n + 1, n + 1))
  for j1 in range (lim_card_rec + 1):
    for j2 in range (lim_card_rec + 1) :
      for j3 in range (lim_card_rec + 1):
        for j4 in range (lim_card_rec + 1) :
          sm_j = j1 + j2 + j3 + j4
          for i11 in range (n + 1):
            for i31 in range (n + 1) :
              case_one [sm_j, i11, i31] = max(envn_MM[i11, i31, j1, j2, j3, j4], case_one [sm_j, i11, i31])   
     
  case_two = np.zeros((lim_card + 1, n + 1, n + 1))
  for j1 in range (lim_card_rec + 1):
    for j2 in range (lim_card_rec + 1) :
      for j3 in range (lim_card_rec + 1):
        for j4 in range (lim_card_rec + 1) :
          sm_j = j1 + j2 + j3 + j4
          for i21 in range (n + 1):
            for i32 in range (n + 1) :
              tmp = 0
              for l in range (n + 1) :
                tmp += (envrefn[i32, l] * envn_MM[i21, l, j1, j2, j3, j4])
              case_two[sm_j, i21, i32] = max (tmp, case_two[sm_j, i21, i32])
              
  case_three = np.zeros((lim_card + 1, n + 1, n + 1))
  for j1 in range (lim_card_rec + 1):
    for j2 in range (lim_card_rec + 1) :
      for j3 in range (lim_card_rec + 1):
        for j4 in range (lim_card_rec + 1) :
          sm_j = j1 + j2 + j3 + j4
          for i12 in range (n + 1):
            for i41 in range (n + 1) :
              tmp = 0
              for l in range (n + 1) :
                tmp += (envrefn[i12,l] * envn_MM[l,i41, j1, j2, j3, j4])
              case_three[sm_j, i12, i41] = max(case_three[sm_j, i12, i41], tmp)
  
  case_four = np.zeros((lim_card + 1, n + 1, n + 1))
  for j1 in range (lim_card_rec + 1):
    for j2 in range (lim_card_rec + 1) :
      for j3 in range (lim_card_rec + 1):
        for j4 in range (lim_card_rec + 1) :
          sm_j = j1 + j2 + j3 + j4
          for i22 in range (n + 1):
            for i42 in range (n + 1) :
              tmp = 0
              for l in range (n + 1) :
                for l1 in range (n + 1) :
                  tmp += (envrefn[i22, l] * envrefn[i42, l1] 
                          * envn_MM[l, l1, j1, j2, j3, j4])
 
              case_four[sm_j,i22,i42] = max(case_four[sm_j,i22,i42], tmp)
                         
  single_proba = [case_one, case_two, case_three, case_four]         
  
  return single_proba        
        
def precompute_single_proba_sym_case (n, envn_MM, envrefn):   
  """
  Precompute symmetric-case thresholds P(|I_x|=ix, |I_y|=iy | |J|=j).

  Symmetric MatMult case: for each ``j = j1 + j2 + j3 + j4`` and all 
  ``(ix, iy)``, compute 
  ``sum_{l,l1} envrefn[ix,l] * envn_MM[l,l1,j1,j2,j3,j4] * envrefn[iy,l1]``,
  then take the maximum over all quadruples ``(j1,j2,j3,j4)`` with fixed
  ``j``. The result is identical for the four cases, so we return
  ``[case_one, case_one, case_one, case_one]``.

  Args:
    n (int): Number of shares.
    envn_MM (np.ndarray): 6D array of multiplication-gadget envelopes with
    indices ``[l, l1, j1, j2, j3, j4]`` where ``l, l1 ∈ [0..n]`` and each 
    ``j• ∈ [0..n^2/4]``.
    envrefn (np.ndarray): 2D refresh-gadget envelope, shape ``(n+1, n+1)``,
    indexed as ``[i, l]``.

  Returns:
    List[np.ndarray]: A list ``[case_one, case_one, case_one, case_one]``,
    where each item has shape ``(n**2 + 1, n + 1, n + 1)`` and
    ``case_one[j, ix, iy]`` stores the max over quadruples with sum
    ``j`` of ``envrefn[ix,:] * envn_MM[:,:,j...] * envrefn[iy,:]``.
  """  
  
  lim_card = n**2
  lim_card_rec = int(n**2 / 4)
  
  case_one = np.zeros((lim_card + 1, n + 1, n + 1))
  for j1 in range (lim_card_rec + 1):
    for j2 in range (lim_card_rec + 1) :
      for j3 in range (lim_card_rec + 1):
        for j4 in range (lim_card_rec + 1) :
          sm_j = j1 + j2 + j3 + j4
          for i11 in range (n + 1):
            for i31 in range (n + 1) :
              tmp = 0
              for l in range (n + 1) :
                for l1 in range (n + 1) :
                  tmp += (envrefn[i11, l] * envrefn[i31, l1] 
                          * envn_MM[l, l1, j1, j2, j3, j4])  
              
              case_one[sm_j, i11, i31] = max (case_one[sm_j, i11, i31], tmp)
              
  single_proba = [case_one, case_one, case_one, case_one]         
  return single_proba   

def precomp1 (n, single_proba_three, single_proba_four, j3, j4, hypergeom, queue) :
  """
  Precompute, for fixed |J3|=j3 and |J4|=j4, the accumulated thresholds
  over i41 ∈ [0..i4] with i42 = i4 - i41, for all i4 ∈ [0..n], i21 ∈ [0..n],
  and i22 ∈ [0..n].

  Concretely, for each (end=i4, l4, i14) with l4 ≤ i14 ≤ end, this performs:
      prec1[j3, j4, end, i21, i22] +=
          single_proba_three[j3, i21, i14]
        * single_proba_four[j4, i22, (end + l4 - i14)]
        * hypergeom[i14, (end + l4 - i14), l4]

  Only the slice at indices (j3, j4, ·, ·, ·) is populated; the other
  (j3', j4') entries in the returned tensor remain zero. The result tensor
  (with the populated slice) is put into the provided queue. 

  This corresponds to some factorized parts of the proof of lemma 14 
  in the appendix of the paper.

  Args:
    n (int): Number of shares.
    single_proba_three: 3D tensor indexed as [|J3|, i21, i41=i14],
      providing P(i21, i41, |J3|).
    single_proba_four: 3D tensor indexed as [|J4|, i22, i42],
      providing P(i22, i42, |J4|).
    j3 (int): Fixed |J3| index for this computation.
    j4 (int): Fixed |J4| index for this computation.
    hypergeom: 3D table of the hypergeometric law indexed as
      [i41, i42, i4=l4], i.e., hypergeom[i14, end + l4 - i14, l4], with n as 
      population size.
    queue: A multiprocessing-compatible queue. The tuple (j3, j4, prec1)
      is enqueued, where prec1 has shape (n^2+1, n^2+1, n+1, n+1, n+1) and
      only the (j3, j4) slice is filled.

  Returns:
    None
  """
  
  lim_card = n**2
  prec1 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1, n + 1))
  for end in range (n + 1) :
    for l4 in range (end + 1) :
      for i14 in range(l4, end + 1) :
        for i21 in range (n + 1) :
          for i22 in range (n + 1) :
            prec1[j3, j4, end, i21, i22] += single_proba_three[j3, i21, i14] * single_proba_four[j4, i22, end + l4 - i14] * hypergeom[i14, end + l4 - i14, l4]
  
  queue.put((j3, j4, prec1))            
  return

def precomp2 (n, single_proba_one, single_proba_two, j1, j2, hypergeom, queue) :
  """
  Precompute, for fixed |J1|=j1 and |J2|=j2, the accumulated thresholds
  over i31 ∈ [0..i3] with i32 = i3 - i31, for all i3 ∈ [0..n], i11 ∈ [0..n],
  and i12 ∈ [0..n].

  Concretely, for each (end=i3, l3, i13=i31) with l3 ≤ i13 ≤ end, this performs:
      prec2[j1, j2, end, i11, i12] +=
          single_proba_one[j1, i11, i13]
        * single_proba_two[j2, i12, (end + l3 - i13)]
        * hypergeom[i13, (end + l3 - i13), l3]

  Only the slice at indices (j1, j2, ·, ·, ·) is populated in the returned
  tensor. The result tensor (with the populated slice) is put into the provided
  queue.

  This corresponds to some other factorized parts of the proof of lemma 14 
  in the appendix of the paper.

  Args:
    n (int): Number of shares.
    single_proba_one: 3D tensor indexed as [|J1|, i11, i31=i13],
      providing P(i11, i31, |J1|).
    single_proba_two: 3D tensor indexed as [|J2|, i12, i32],
      providing P(i12, i32, |J2|).
    j1 (int): Fixed |J1| index for this computation.
    j2 (int): Fixed |J2| index for this computation.
    hypergeom: 3D table of the hypergeometric law indexed as
      [i31, i32, i3=l3], i.e., hypergeom[i13, end + l3 - i13, l3].
    queue: A multiprocessing-compatible queue. The tuple (j1, j2, prec2)
      is enqueued, where prec2 has shape (n^2+1, n^2+1, n+1, n+1, n+1) and
      only the (j1, j2) slice is filled.

  Returns:
    None
  """
  lim_card = n**2
  prec2 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1, n + 1))
  for end in range (n + 1) :
    for l3 in range (end + 1) :
      for i13 in range (l3, end + 1) :          
        for i12 in range (n + 1) :
          for i11 in range (n + 1) :  
            prec2[j1, j2, end, i11, i12] += single_proba_one[j1, i11, i13] * single_proba_two[j2, i12, end + l3 - i13] * hypergeom[i13, end + l3 - i13, l3]
  
  queue.put((j1, j2, prec2))            
  return 

def initial_case_MM (p) :
  """
  Compute probability envelopes for the cardinal-RPC MatMult gadget with n=2.

  Builds the 6D tensor ``case`` of shape ``(n+1, n+1, 2, 2, 2, 2)`` where the
  first two axes correspond to ``|I_x|`` and ``|I_y|`` (ranging from 0..n), and
  the last four binary axes correspond to the cardinalities ``(|J_1|, |J_2|,
  |J_3|, |J_4|) ∈ {0,1}^4``. Entries are populated using closed-form
  expressions derived from ``v = (1 - p)^3`` and its complements, with cases
  chosen by ``j = j1 + j2 + j3 + j4 ∈ {0,1,2,≥3}``.

  Args:
    p (float): Leakage probability parameter (0 ≤ p ≤ 1).

  Returns:
    numpy.ndarray: Array of shape ``(n+1, n+1, 2, 2, 2, 2)`` containing the
      probability envelopes ``case[|I_x|, |I_y|, j1, j2, j3, j4]`` for all
      indices.
  """

  n = 2
  
  inv_p = 1 - p
  v = inv_p**3
  v_inv = 1 - v
  v_inv_sq = v_inv**2
  v_sq = v**2
  v0 = v * v_inv
  v1 = 2 * v0
  v2 = v_sq + v1 + v_inv_sq
    
  case = np.zeros((n + 1, n + 1, 2, 2, 2, 2))
  for j1 in range (n) :
    for j2 in range (n) :
      for j3 in range (n) :
        for j4 in range (n) :
          smj = j1 + j2 + j3 + j4 
          if (smj == 0) : 
            case[0, 0, j1, j2, j3, j4] = v_sq**2 
            case[0, 1, j1, j2, j3, j4] = v_sq * v1
            case[0, 2, j1, j2, j3, j4] = v_sq * v_inv_sq
            case[1, 0, j1, j2, j3, j4] = case[0, 1, j1, j2, j3, j4]
            case[1, 1, j1, j2, j3, j4] = v1**2
            case[1, 2, j1, j2, j3, j4] = v1 * v_inv_sq
            case[2, 0, j1, j2 ,j3, j4] = case[0, 2, j1, j2, j3, j4]
            case[2, 1, j1, j2, j3, j4] = case[1, 2, j1, j2, j3, j4]
            case[2, 2, j1, j2, j3, j4] = v_inv_sq**2

          elif (smj == 1) :
            case[1, 1, j1, j2, j3, j4] = v_sq
            case[1, 2, j1, j2, j3, j4] = v0
            case[2, 1, j1, j2, j3, j4] = case[1,2, j1, j2, j3, j4]
            case[2, 2, j1, j2, j3, j4] = v_inv_sq
          
          elif (smj == 2) :
            case[1, 2, j1, j2, j3, j4] = v / 3
            case[2, 1, j1, j2, j3, j4] = v / 3
            case[2, 2, j1, j2, j3, j4] = (1 / 3) + (2 / 3) * (1 - v)
            
          else :
            case[2, 2, j1, j2, j3, j4] = 1
  
  
  return case

def compute_proba(n, j1, j2, j3, j4, ix, iy, prec1, prec2, hypergeom) :
  """
  Compute the probability envelope value for MatMult with 2·n shares at fixed
  (|J1|, |J2|, |J3|, |J4|) = (j1, j2, j3, j4) and (|Ix|, |Iy|) = (ix, iy).

  This evaluates a convolution-style sum over internal indices combining the
  precomputed tensors ``prec1`` and ``prec2`` with hypergeometric weights:
  
      pr = Σ_{i1=kx..ixt} Σ_{l1=0..i1} Σ_{i11=l1..i1} [
               Σ_{l2=0..(ix-i1)} Σ_{i12=l2..(ix-i1)} Σ_{i3=ky..iyt}
                 prec1[j3, j4, (iy - i3), (i1 + l1 - i11), (ix - i1 + l2 - i12)]
               * prec2[j1, j2, i3, i11, i12]
               * hypergeom[i12, (ix - i1 + l2 - i12), l2]
               * hypergeom[i11, (i1 + l1 - i11), l1]

  with boundaries:
      ixt = min(ix, n),   kx = max(0, ix - n)
      iyt = min(iy, n),   ky = max(0, iy - n)

  Args:
    n (int): Base share parameter; the MatMult gadget uses ``2 * n`` shares.
    j1 (int): Cardinality ``|J1|``.
    j2 (int): Cardinality ``|J2|``.
    j3 (int): Cardinality ``|J3|``.
    j4 (int): Cardinality ``|J4|``.
    ix (int): Cardinality ``|Ix|``.
    iy (int): Cardinality ``|Iy|``.
    prec1: 5D tensor of shape ``(n^2+1, n^2+1, n+1, n+1, n+1)`` indexed as
      ``[j3, j4, end, i21, i22]``.
    prec2: 5D tensor of shape ``(n^2+1, n^2+1, n+1, n+1, n+1)`` indexed as
      ``[j1, j2, end, i11, i12]``.
    hypergeom: 3D table of hypergeometric weights of shape ``(n+1, n+1, n+1)``,
      indexed as ``[a, b, c]``.

  Returns:
    float: The computed probability envelope value ``pr``.
  """
  pr = 0
  ixt = min(ix, n)
  kx = max (0, ix - n)
  iyt = min (iy,n)
  ky = max (0, iy - n)
 
  for i1 in range (kx, ixt + 1) :
    for l1 in range (i1 + 1) :
      for i11 in range (l1, i1 + 1) :              
        sub_pr = 0
        for l2 in range (ix - i1 + 1) :
          for i12 in range (l2, ix - i1 + 1) :
            for i3 in range (ky, iyt + 1) :
              sub_pr += prec1[j3, j4, iy - i3, i1 + l1 - i11, ix - i1 + l2 - i12] * prec2[j1, j2, i3 , i11, i12]  * hypergeom [i12, ix - i1 + l2 - i12, l2]          
        sub_pr *= hypergeom[i11, i1 + l1 - i11, l1]
        pr += sub_pr
  return pr

def final_induction_envelopes(n, p, start_env, ix, iy, queue) : 
  """
  Final induction step: Take into account the 3rd wire leaked during the copy of 
  the shares x_i and y_i for a fixed (|Ix|, |Iy|) = (ix, iy), 
  and clamp each entry to 1.

  For each (j1, j2, j3, j4), this computes:
      env[j1,j2,j3,j4] = Σ_{lx=0..ix} Σ_{ly=0..iy}
          comb(n - ix + lx, lx) * comb(n - iy + ly, ly)
        * p^{lx+ly} * (1 - p)^{2n - ix - iy}
        * start_env[ix - lx, iy - ly, j1, j2, j3, j4]
  and then applies min(1, ·).

  Args:
    n (int): Base share parameter (MatMult uses 2*n shares overall).
    p (float): Leakage probability parameter.
    start_env: 6D tensor of shape
        (2*n + 1, 2*n + 1, lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1),
        with lim_card = n**2. Indexed as [ix, iy, j1, j2, j3, j4].
    ix (int): Target |Ix|.
    iy (int): Target |Iy|.
    queue: Multiprocessing-compatible queue. The tuple (ix, iy, env) is enqueued,
        where env has shape (lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1).

  Returns:
    None
  """
  lim_card = n**2
  env = np.zeros((lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1))
  for j1 in range(lim_card + 1) :
    for j2 in range (lim_card + 1) :
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          for lx in range (ix + 1) :
            for ly in range (iy + 1) :
              comb1 = comb(n - ix + lx, lx)
              comb2 = comb(n - iy + ly, ly)
              env[j1, j2, j3, j4] += comb1 * comb2 * p**(lx + ly) * (1 - p)**(2 * n - ix - iy) * start_env[ix - lx, iy - ly, j1, j2, j3, j4]
          env[j1, j2, j3, j4] = min (1,  env[j1, j2, j3, j4])

  queue.put((ix, iy, env))
  return 

def induction_envelopes(n, prec1, prec2, ix, iy, hypergeom, queue) : 
  """
  Induction step for MatMult envelopes at fixed (|Ix|, |Iy|) = (ix, iy).

  For all (j1, j2, j3, j4) in [0..n^2], compute:
      pr_ix_iy = compute_proba(n, j1, j2, j3, j4, ix, iy, prec1, prec2, hypergeom)
  and store env[ix, iy, j1, j2, j3, j4] = min(1, pr_ix_iy).

  Args:
    n (int): Base share parameter (MatMult uses 2*n shares overall).
    prec1: 5D tensor (n^2+1, n^2+1, n+1, n+1, n+1), indexed [j3, j4, end, i21, i22].
    prec2: 5D tensor (n^2+1, n^2+1, n+1, n+1, n+1), indexed [j1, j2, end, i11, i12].
    ix (int): Target |Ix|.
    iy (int): Target |Iy|.
    hypergeom: 3D table of hypergeometric weights, shape (n+1, n+1, n+1).
    queue: Multiprocessing-compatible queue. Enqueues (ix, iy, env) where
        env has shape (2*n+1, 2*n+1, n^2+1, n^2+1, n^2+1, n^2+1).

  Returns:
    None
  """
  lim_card = n**2
  env = np.zeros((2 * n + 1, 2 * n + 1, lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1))
  
  for j1 in range (lim_card + 1) :
    for j2 in range (lim_card + 1) :
      for j3 in range (lim_card + 1) :                  
        for j4 in range (lim_card + 1) :      
          pr_ix_iy = compute_proba(n, j1, j2, j3, j4, ix, iy, prec1, prec2, hypergeom)
          env[ix, iy, j1, j2, j3, j4] = min(1, pr_ix_iy)
    
  queue.put((ix, iy, env))
  return  
  
def compute_partition_J(n) :
  """
  Count the number of 4-tuples (|J1|, |J2|, |J3|, |J4|) with bounded parts
  that sum to a given |J|, for all |J| ∈ [0..n^2].

  Each part is constrained to 0..(n/2)^2. For each j in 0..n^2, this returns
  the count of quadruples (j1, j2, j3, j4) such that j1 + j2 + j3 + j4 = j.

  Args:
    n (int): Base share parameter; parts are bounded by (n/2)^2.

  Returns:
    list[int]: A list `card_part` of length n^2 + 1 where `card_part[j]`
      equals the number of valid quadruples summing to j.
  """
  nh = int(n/2)
  nh_sq = nh**2
  card_part = []
  for j in range (n**2 + 1) :
    cpt = 0
    for j1 in range (nh_sq + 1) :
      for j2 in range (nh_sq + 1) :
        for j3 in range (nh_sq + 1) :
          for j4 in range (nh_sq + 1) :
            if (j1 + j2 + j3 + j4 == j) :
              cpt += 1
    card_part.append(cpt)
  
  return card_part 

def compute_envn_MM (n_lim,p,l_nb_iter,case,cores) :
  """
  Iteratively compute MatMult probability envelopes up to target share size 
  `n_lim`.

  Starting from the base case `n = 2` (`initial_case_MM(p)`), this loop doubles 
  `n` at each stage, computes refresh envelopes and precomputations *
  (prec1/prec2), builds intermediate envelopes via `induction_envelopes`, 
  applies a final leakage folding via `final_induction_envelopes`, and proceeds 
  until `n < n_lim` is no longer satisfied. Precomputations are cached to disk 
  as `Uni_prec1{2*n}.npy` and `Uni_prec2{2*n}.npy`.

  Args:
    n_lim (int): Target maximum number of shares (the loop doubles n from 2).
    p (float): Leakage probability parameter.
    l_nb_iter (list[int]): Number of iteration `nb_iter` used in
      `cardinal_rpc_refresh_envelope(n, p, nb_iter)` according to `n`.
    case (str): Either "Asym" for asymmetric precomputation
        (`precompute_single_proba`) or any other value to select the symmetric
        variant (`precompute_single_proba_sym_case`).
    cores (int) : Number of cores (for the parallelisation).

  Returns:
    numpy.ndarray: The final 6D envelope tensor for the last `n` reached,
    shaped as 
    (2*n + 1, 2*n + 1, lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1),
    with lim_card = n**2.
  """
  n = 2
  envn_MM = initial_case_MM(p)
  queue_precomp = Queue()
  queue_precomp1 = Queue()
  queue = Queue()
  precomp = []
  pgref = 0
  
  
  while (n < n_lim) :
    
    nb_iter = l_nb_iter.pop()
    #Compute the probability envelopes for the gadget with n' = 2 * n.
    
    #The maximal cardinality an output set of MatMult with n shares can have.
    lim_card = n**2
    
    #Instantiate the probability envelopes.
    final_envn_MM = np.zeros((2 * n + 1, 2 * n + 1, lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1))
    
    #Compute the probability envelopes of the refresh gadget with n shares. 
    pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)
   

    #Make the Precomputation
    if (case == "Asym") :
      precomp = precompute_single_proba(n, envn_MM, pgref)
    else :
      precomp = precompute_single_proba_sym_case(n, envn_MM, pgref)
    
    single_proba_one = precomp[0]
    single_proba_two = precomp[1]
    single_proba_three = precomp[2]
    single_proba_four = precomp[3]
    
    prec1 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1, n + 1))
    prec2 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1, n + 1))
    hypergeom = precomp_hypergeom(n)
    
    processes = []
    
    if (os.path.isfile("Uni_prec1"+str(2 * n)+".npy") ) :
      prec1 = np.load("Uni_prec1"+str(2 * n)+".npy")
      
    else :
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          processes.append(Process(target=precomp1, args=(n, single_proba_three, single_proba_four, j3, j4, hypergeom, queue_precomp1)))
      
      nb_process = 0
      while (nb_process < len(processes)) :
        lim = min(cores + nb_process, len(processes))
        for i in range (nb_process, lim) :
          pro = processes[i]
          pro.daemon = True
          pro.start()
      
        for _ in range (nb_process, lim) :
          (j3, j4, prec1j3j4) = queue_precomp1.get()
          prec1[j3, j4] += prec1j3j4[j3, j4]        
        
        
        for i in range (nb_process, lim) : 
          processes[i].join()
          processes[i].terminate()
          processes[i].close()
          nb_process += 1
      
      
      #prec1 = precomp1(n, single_proba_three, single_proba_four)
      np.save("Uni_prec1"+str(2*n)+".npy",prec1)
    
    processes = []
    
    
    if (os.path.isfile("Uni_prec2"+str(2 * n)+".npy")) :
      prec2 = np.load("Uni_prec2"+str(2 * n)+".npy")
    
    else :  
      for j1 in range (lim_card + 1) :
        for j2 in range (lim_card + 1) :
          processes.append(Process(target=precomp2, args=(n, single_proba_one, single_proba_two, j1, j2, hypergeom, queue_precomp)))
      
      nb_process = 0
      while (nb_process < len(processes)) :
        lim = min(cores + nb_process, len(processes))
        for i in range (nb_process, lim) :
          pro = processes[i]
          pro.daemon = True
          pro.start()
        
        
        for _ in range (nb_process, lim) :
          (j1, j2, prec2j1j2) = queue_precomp.get()
          prec2[j1, j2] += prec2j1j2[j1, j2]
      
        for i in range (nb_process, lim) : 
          processes[i].join()
          processes[i].terminate()
          processes[i].close()
          nb_process += 1
        
      np.save ("Uni_prec2"+str(2 * n)+".npy", prec2)
    
    #Computation
    processes = []
    for ix in range (2 * n + 1) :
      for iy in range (2 * n + 1) :    
        processes.append(Process(target=induction_envelopes, args=(n, prec1, prec2, ix, iy, hypergeom, queue)))
    
    for pro in processes :
      pro.daemon = True
      pro.start()
      
    for pro in processes : 
      (ix, iy, pro_envn_MM) = queue.get()
      final_envn_MM[ix,iy] += pro_envn_MM[ix,iy]
        
            
    for pro in processes : 
      pro.join()
    

    new_final_envn_MM = np.zeros((2 * n + 1, 2 * n + 1, lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1))

    #Computation
    processes = []
    for ix in range (2 * n + 1) :
      for iy in range (2 * n + 1) :    
        processes.append(Process(target=final_induction_envelopes, args=(n, p, final_envn_MM, ix, iy, queue)))
    
    for pro in processes :
      pro.daemon = True
      pro.start()
      
    for pro in processes : 
      (ix, iy, pro_envn_MM) = queue.get()
      new_final_envn_MM[ix,iy] += pro_envn_MM
                    
    for pro in processes : 
      pro.join()
    

    #remove_partial_save(n)
    os.remove("Uni_prec1"+str(2 * n)+".npy")
    os.remove("Uni_prec2"+str(2 * n)+".npy")
    n = 2 * n
    envn_MM = final_envn_MM

  return envn_MM

################################################################################   
############################# TreeAdd Enveloppes ###############################

def initial_case_TA(n, envgadd) :
  """
  Base case envelopes for TreeAdd.

  Args:
    n (int): Number of shares.
    envgadd: 3D envelope tensor for the add gadget, indexed as
      ``envgadd[i1, i2, j]`` with each index in ``[0..n]``.

  Returns:
    numpy.ndarray: Array of shape ``(2*n + 1, n + 1)``.
  """
  envn = np.zeros((2 * n + 1, n + 1))
  for i in range (2 * n + 1) :
    it = min (n , i)
    k = max (0, i - n) 
    for j in range (n + 1) :
      for i1 in range (k, it + 1) :
        envn[i,j] += envgadd[i1, i - i1, j]

  return envn

def induction_envn_TA(nb_row, n, envn_TA, envgadd) :
  """
  Inductive convolution step for TreeAdd.

  Given current envelopes ``envn_TA`` (one stage), produces the next-stage
  envelopes by convolving two copies of ``envn_TA`` via ``envgadd``:
      envn[i, j] = Σ_{i1, l1, l2} envn_TA[i1, l1] * envn_TA[i - i1, l2]
                                * envgadd[l1, l2, j]
  with bounds ``i ∈ [0..size_i]``, ``size_i = 2 * n * nb_row``,
  ``i1 ∈ [k..it]``, where ``it = nb_row * n`` and ``k = max(0, i - nb_row * n)``.
  Each ``envn[i, j]`` is clamped by ``min(1, ·)``.

  Args:
    nb_row (int): Current number of rows (tree width factor).
    n (int): Number of shares per operand.
    envn_TA: 2D array of shape ``(nb_row * 2 * n + 1, n + 1)`` from the
      previous stage.
    envgadd: 3D envelope tensor for the add gadget, indexed as
      ``envgadd[l1, l2, j]``.

  Returns:
    numpy.ndarray: Next-stage envelopes of shape ``(2 * n * nb_row + 1, n + 1)``.
  """

  size_i = nb_row * n * 2
  envn = np.zeros((size_i + 1, n + 1))
  for i in range (size_i + 1) :
    it = min(i, nb_row * n)
    k = max (0, i - nb_row * n)
    for j in range(n + 1) :
      for i1 in range (k , it + 1) :
        for l1 in range (n + 1) :
          for l2 in range(n + 1) : 
            envn[i,j] += envn_TA[i1,l1] * envn_TA[i - i1, l2] * envgadd[l1, l2, j]
      envn[i,j] = min (1, envn[i,j])
  
 
  return envn 

def prec_TArec (n, envgadd) :
  """
  Recursive precomputation for TreeAdd (n > 4).

  Starts from the base case ``initial_case_TA`` and repeatedly applies
  ``induction_envn_TA`` while doubling ``nb_row`` until ``nb_row < n/8``
  is no longer satisfied.

  Args:
    n (int): Number of shares per operand.
    envgadd: 3D add-gadget envelope tensor, indexed as ``envgadd[i1, i2, j]``.

  Returns:
    numpy.ndarray: Envelope matrix after the final recursion step.
  """
  nb_row = 1
  envn_TA = initial_case_TA(n, envgadd)
  
  while (nb_row < int(n / 8)) :    
    nb_row = 2 * nb_row
    envn_TA = induction_envn_TA(nb_row, n, envn_TA, envgadd)

  return envn_TA

def TAprec1 (n, envgadd) :
  """
  First-stage 4-way precomputation for TreeAdd. Only used to compute TreeAdd 
  with n = 4.

  Computes a 5D tensor:
      env[i1, i2, i3, i4, j] = Σ_{l, l1} envgadd[i1, i2, l] * envgadd[i3, i4, l1]
                                           * envgadd[l, l1, j]
  and clamps each entry by ``min(1, ·)``.

  Args:
    n (int): Number of shares per operand.
    envgadd: 3D add-gadget envelope tensor, indexed as ``envgadd[a, b, c]``.

  Returns:
    numpy.ndarray: Array of shape ``(n+1, n+1, n+1, n+1, n+1)``.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for j in range (n + 1) :
    for i1 in range (n + 1) :
      for i2 in range (n + 1) :
        for i3 in range (n + 1) : 
          for i4 in range (n + 1) : 
            for l in range (n + 1) :
              for l1 in range (n + 1) :
                env[i1, i2, i3, i4, j] += envgadd[i1, i2, l] * envgadd[i3, i4, l1] * envgadd[l, l1, j]
            env[i1, i2, i3, i4, j] = min (1, env[i1, i2, i3, i4, j])
  return env
  
def TAprec2 (n, envTArec, env_prec, index, size_env) :
  """
  Second-stage precomputation: contract ``envTArec`` into one axis of ``env_prec``.

  For each output axis position, adds:
    - if ``index == 1``: envTArec[i1, l] * env_prec[l, i2, i3, i4, j]
    - if ``index == 2``: envTArec[i2, l] * env_prec[i1, l, i3, i4, j]
    - if ``index == 3``: envTArec[i3, l] * env_prec[i1, i2, l, i4, j]
    - if ``index == 4``: envTArec[i4, l] * env_prec[i1, i2, i3, l, j]
  then clamps each entry by ``min(1, ·)``.

  Args:
    n (int): Number of shares per operand.
    envTArec: 2D tensor from ``prec_TArec``.
    env_prec: 5D tensor produced by a previous precomputation step.
    index (int): Which axis (1..4) of ``env_prec`` to contract with ``envTArec``.
    size_env (tuple[int, int, int, int, int]): Output tensor shape.

  Returns:
    numpy.ndarray: Array of shape ``size_env``.
  """
  in_size = int (n**2 / 4)
  env = 0
  env = np.zeros(size_env)
  for j in range (n + 1) :
    for i1 in range (size_env[0]) :
      for i2 in range (size_env[1]) :
        for i3 in range (size_env[2]) : 
          for i4 in range (size_env[3]) :
            for l in range (n + 1) :
              if (index == 1) : 
                env[i1, i2, i3, i4, j] += envTArec[i1, l] * env_prec[l, i2, i3, i4, j]
              if (index == 2) : 
                env[i1, i2, i3, i4, j] += envTArec[i2, l] * env_prec[i1, l, i3, i4, j]
              if (index == 3) : 
                env[i1, i2, i3, i4, j] += envTArec[i3, l] * env_prec[i1, i2, l, i4, j]
              if (index == 4) : 
                env[i1, i2, i3, i4, j] += envTArec[i4, l] * env_prec[i1, i2, i3, l, j]      
            env[i1, i2, i3, i4, j] = min (1, env[i1, i2, i3, i4, j])
  return env

def compute_envn_TA (n, p, nb_iter) :
  """
  Compute TreeAdd probability envelopes.

  Pipeline:
    1) ``pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)``
    2) ``envgadd = cardinal_rpc_add_envelope(n, p, pgref)``
    3) ``prec1 = TAprec1(n, envgadd)``
    4) If ``n <= 4``: return ``prec1`` (early base-case path).
    5) Otherwise:
       a) ``envTArec = prec_TArec(n, envgadd)``
       b) Repeatedly call ``TAprec2`` with ``index = 1,2,3,4`` to expand shapes.

  Args:
    n (int): Number of shares.
    p (float): Leakage probability parameter.
    nb_iter (int): Number of Iteration used in the refresh gadget.

  Returns:
    numpy.ndarray: Final TreeAdd envelope tensor (shape depends on ``n``).
  """
  
  pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)
  envgadd = cardinal_rpc_add_envelope(n,p,pgref)
  prec1 = TAprec1 (n, envgadd)
  
  #Works only for n = 4.
  if (n <= 4) :
    return prec1
  
  in_size = int(n**2 / 4)
  
  envTArec = prec_TArec(n, envgadd)
  
  size_env = (in_size + 1, n + 1, n + 1, n + 1, n + 1)
  prec1 = TAprec2(n, envTArec, prec1, 1, size_env)
  
  size_env = (in_size + 1, in_size + 1, n + 1, n + 1, n + 1) 
  prec1 = TAprec2(n, envTArec, prec1, 2, size_env) 
  
  size_env = (in_size + 1, in_size + 1, in_size + 1, n + 1, n + 1)
  prec1 = TAprec2(n, envTArec, prec1, 3, size_env)
  
  size_env = (in_size + 1, in_size + 1, in_size + 1, in_size + 1, n + 1) 
  envn_TA = TAprec2(n, envTArec, prec1, 4, size_env) 

  return envn_TA

################################################################################
########################### CardSecMult envelopes ##############################

def compute_envn (n, envn_MM, envn_TA) :
  """
  Compose MatMult and TreeAdd envelopes to obtain CardSecMult envelopes.

  For each (ix, iy, j), this computes:
      envn[ix, iy, j] = Σ_{j1=0..⌊n^2/4⌋} Σ_{j2} Σ_{j3} Σ_{j4}
          envn_MM[ix, iy, j1, j2, j3, j4] * envn_TA[j1, j2, j3, j4, j]

  Args:
    n (int): Base share parameter (envelopes indexed on 0..n).
    envn_MM: 6D tensor with shape ``(n+1, n+1, lim+1, lim+1, lim+1, lim+1)``,
      where ``lim = ⌊n^2/4⌋``; indexed as ``[ix, iy, j1, j2, j3, j4]``.
    envn_TA: 5D tensor with shape ``(lim+1, lim+1, lim+1, lim+1, n+1)``,
      indexed as ``[j1, j2, j3, j4, j]``.

  Returns:
    numpy.ndarray: 3D tensor ``envn`` of shape ``(n+1, n+1, n+1)``,
      containing CardSecMult envelopes at indices ``[ix, iy, j]``.
  """
  lim_card = int(n**2 / 4)
  envn = np.zeros((n + 1, n + 1, n + 1))
  for j in range (n + 1) :
    for ix in range (n + 1) :
      for iy in range (n + 1) :
        pr = 0
        for j1 in range (lim_card + 1) :
          for j2 in range (lim_card + 1) :
            for j3 in range (lim_card + 1) :
              for j4 in range (lim_card + 1) :  
                pr += envn_MM[ix, iy, j1, j2, j3, j4] * envn_TA[j1, j2, j3, j4, j]    
        envn[ix,iy,j] = pr
  return envn

def cardinal_rpc_mult_uni_envelopes (n, p, l_nb_iter,  nb_iter_TA, case, cores):
  """
  Build unified CardSecMult envelopes (cardinal RPC) by composing MatMult and 
  TreeAdd.

  Pipeline:
    1) ``envn_MM = compute_envn_MM(n, p, l_nb_iter, case, cores)``
    2) ``envn_TA = compute_envn_TA(n, p, nb_iter_TA)``
    3) ``envn = compute_envn(n, envn_MM, envn_TA)``

  Args:
    n (int): Number of share.
    p (float): Leakage probability.
    l_nb_iter (list[int]): Number of iteration `nb_iter` used in
    `cardinal_rpc_refresh_envelope(n, p, nb_iter)` according to `n` for the 
    computation of Matmult enveloppes.
    nb_iter_TA (int): Number of iteration used in the refresh gadget for 
    TreeAdd enveloppes computation.
    case (str): Variant selector for MatMult precomputation 
    (e.g., ``"Asym"`` or symmetric).
    cores (int) : Number of cores.

  Returns:
    numpy.ndarray: 3D tensor of CardSecMult envelopes with shape ``(n+1, n+1, n+1)``.
  """
  envn_MM = compute_envn_MM(n, p, l_nb_iter, case, cores)
  envn_TA = compute_envn_TA(n, p, nb_iter_TA)
  envn = compute_envn (n, envn_MM, envn_TA)
  return envn

  





