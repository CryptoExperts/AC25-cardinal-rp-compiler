#!/usr/bin/env python3

# This file computes the cardinal-RPC envelopes non-uniform variant of 
# CardSecMult.
# Moreover, it computes it by cutting the number of outputs of MatMult in 4 to 
# obtain better security. 
# In fact, MatMult exposes four outputs of size n**2/4, one per subtree (see 
# Figure 5 of the paper)—instead of a single aggregated output of size n**2. 
# Accordingly, TreeAdd takes four inputs, each of size n**2/4, instead of a 
# single input of size n**2. 
# The implementation targets numbers of shares n that are powers of two.
# This module is used to produce Figure 7 of the full paper (non-uniform 
# variant),generating the points labeled `Unif = 4` and `Unif = 8` on the graph.
#
# The Figure 7 is notably built with the function ``compute_graph``.


from threading import Thread
#from queue import Queue
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import comb

from partitions import cardinal_rpc_refresh_envelope, cardinal_rpc_add_envelope, cardinal_rpc_gcmult_envelope_pgref, cardinal_rpc_gcopy_envelope_pgref

from multiprocessing import Pool, Process, Queue
from math import log

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


################################################################################
############################ MatMult Enveloppes ################################

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


def precomp1 (n, single_proba_three, single_proba_four, j3, j4, queue) :
  """
  Precompute, for fixed |J3|=j3 and |J4|=j4, the accumulated thresholds
  over i41 ∈ [0..i4] with i42 = i4 - i41, for all i4 ∈ [0..n], i21 ∈ [0..n],
  and i22 ∈ [0..n].

  Concretely, for each (end=i4, i14, i4 - i14), this performs:
      prec1[j3, j4, end, i21, i22] +=
          single_proba_three[j3, i21, i14]
        * single_proba_four[j4, i22, (end - i14)]

  Only the slice at indices (j3, j4, ·, ·, ·) is populated; the other
  (j3', j4') entries in the returned tensor remain zero. The result tensor
  (with the populated slice) is put into the provided queue. 

  Args:
    n (int): Number of shares.
    single_proba_three: 3D tensor indexed as [|J3|, i21, i41=i14],
      providing P(i21, i41, |J3|).
    single_proba_four: 3D tensor indexed as [|J4|, i22, i42],
      providing P(i22, i42, |J4|).
    j3 (int): Fixed |J3| index for this computation.
    j4 (int): Fixed |J4| index for this computation.
    queue: A multiprocessing-compatible queue. The tuple (j3, j4, prec1)
      is enqueued, where prec1 has shape (n^2+1, n^2+1, n+1, n+1, n+1) and
      only the (j3, j4) slice is filled.

  Returns:
    None
  """
  lim_card = n**2
  prec1 = np.zeros((lim_card + 1, lim_card + 1, 2 * n + 1, n + 1, n + 1))  
  for end in range (n + 1) :
    for i14 in range(end + 1) :
      for i21 in range (n + 1) :
        for i22 in range (n + 1) :
          prec1[j3, j4, end, i21, i22] += single_proba_three[j3, i21, i14] * single_proba_four[j4, i22, end - i14]   
  
  queue.put((j3, j4, prec1))            
  return

def precomp2 (n, single_proba_one, single_proba_two, j1, j2, queue) :
  """
  Precompute, for fixed |J1|=j1 and |J2|=j2, the accumulated thresholds
  over i31 ∈ [0..i3] with i32 = i3 - i31, for all i3 ∈ [0..n], i11 ∈ [0..n],
  and i12 ∈ [0..n].

  Concretely, for each (end=i3, i13, i3 - i13), this performs:
      prec2[j1, j2, end, i11, i12] +=
          single_proba_one[j1, i11, i13]
        * single_proba_two[j2, i12, (end - i13)]

  Only the slice at indices (j1, j2, ·, ·, ·) is populated in the returned
  tensor. The result tensor (with the populated slice) is put into the provided
  queue.

  Args:
    n (int): Number of shares.
    single_proba_one: 3D tensor indexed as [|J1|, i11, i31=i13],
      providing P(i11, i31, |J1|).
    single_proba_two: 3D tensor indexed as [|J2|, i12, i32],
      providing P(i12, i32, |J2|).
    j1 (int): Fixed |J1| index for this computation.
    j2 (int): Fixed |J2| index for this computation.
    queue: A multiprocessing-compatible queue. The tuple (j1, j2, prec2)
      is enqueued, where prec2 has shape (n^2+1, n^2+1, n+1, n+1, n+1) and
      only the (j1, j2) slice is filled.

  Returns:
    None
  """
  lim_card = n**2
  prec2 = np.zeros((lim_card + 1, lim_card + 1, 2 * n + 1, n + 1, n + 1))
  for i3 in range(2 * n + 1) :
    for i13 in range (max(0, i3 - n), min(i3, n) + 1) :
      for i11 in range (n + 1) :
        for i12 in range (n + 1) : 
          prec2[j1, j2, i3, i11, i12] += single_proba_one[j1, i11, i13] * single_proba_two[j2, i12, i3 - i13]
  
  queue.put((j1, j2, prec2))            
  return 

def precomp3 (n, single_proba_three, single_proba_four, j3, j4, queue) :
  """
  Precompute, for fixed |J3|=j3 and |J4|=j4, the partial convolution over
  (i4, i14) that aggregates single-probability envelopes into a slice (j3, j4).

  For each i21, i22 ∈ [0..n], this accumulates
      Σ_{i4=n..2n} Σ_{i14=i4-n..n}
        single_proba_three[j3, i21, i14] *
        single_proba_four[j4, i22, (i4 - i14)]

  and stores the result into ``prec3[j3, j4, i21, i22]``. The function enqueues
  the tuple ``(j3, j4, prec3)`` where ``prec3`` has shape
  ``(n^2+1, n^2+1, n+1, n+1)``; only the slice at indices ``(j3, j4, :, :)``
  is populated.

  Args:
    n (int): Number of shares.
    single_proba_three: 3D tensor with shape (|J|, n+1, n+1), indexed as
      ``[j3, i21, i14]``.
    single_proba_four: 3D tensor with shape (|J|, n+1, n+1), indexed as
      ``[j4, i22, i42]`` with ``i42 = i4 - i14``.
    j3 (int): Fixed |J3| index for this computation.
    j4 (int): Fixed |J4| index for this computation.
    queue: A multiprocessing-compatible queue to receive ``(j3, j4, prec3)``.

  Returns:
    None
  """
  lim_card = n**2
  prec3 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1))  
  for i4 in range (n, 2 * n + 1) :
    for i14 in range(i4 - n, n + 1) :
      for i21 in range (n + 1) :
        for i22 in range (n + 1) :
          prec3[j3, j4, i21, i22] += single_proba_three[j3, i21, i14] * single_proba_four[j4, i22, i4 - i14]   
  
  queue.put((j3, j4, prec3))            
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
          
          elif (smj == 2) and (j1 != j4) :
            if (j1 == 1) :
              if (j2 == 1) :
                case[2, 1, j1, j2, j3, j4] = v
                case[2, 2, j1, j2, j3, j4] = 1 - v
              else :
                case[1, 2, j1, j2, j3, j4] = v
                case[2, 2, j1, j2, j3, j4] = 1 - v
            else : 
              if (j2 == 1) :
                case[1, 2, j1, j2, j3, j4] = v
                case[2, 2, j1, j2, j3, j4] = 1 - v
              else :
                case[2, 1, j1, j2, j3, j4] = v
                case[2, 2, j1, j2, j3, j4] = 1 - v
            
          else :
            case[2, 2, j1, j2, j3, j4] = 1
  
  
  return case

def compute_proba(n, ix, iy, j1, j2, j3, j4, prec1, prec2, prec3) :
  """
  Compute one probability-envelope entry for MatMult with ``2·n`` shares.

  Evaluates the convolution that combines precomputed slices:
  - ``prec1[j3, j4, end, i21, i22]`` (depends on ``j3, j4``),
  - ``prec2[j1, j2, i3, i11, i12]`` (depends on ``j1, j2``), and
  - ``prec3[j3, j4, i21, i22]`` used in the branches where carry terms exceed ``n``,
  
  to obtain ``P(|Ix|=ix, |Iy|=iy, |J1|=j1, |J2|=j2, |J3|=j3, |J4|=j4)``.

  The computation proceeds by cases depending on whether ``ix`` and/or ``iy``
  cross ``n``. When both are strictly less than ``n``, it reduces to:
  
      pr = Σ_{i1=0..ix} Σ_{i11=0..i1} Σ_{i12=0..(ix-i1)} Σ_{i3=0..iy}
              prec1[j3, j4, (iy - i3), (i1 - i11), (ix - i1 - i12)]
            * prec2[j1, j2, i3, i11, i12]
  
  For the three other regions (``ix ≥ n`` and/or ``iy ≥ n``), the code adjusts
  the summation bounds and, when the residual exceeds ``n``, uses the pre-folded
  term ``prec3[j3, j4, i21, i22]`` instead of ``prec1`` (see branches in code).

  Args:
    n (int): Base share parameter (the MatMult gadget uses ``2 * n`` shares).
    ix (int): Cardinality ``|Ix|``.
    iy (int): Cardinality ``|Iy|``.
    j1 (int): Cardinality ``|J1|``.
    j2 (int): Cardinality ``|J2|``.
    j3 (int): Cardinality ``|J3|``.
    j4 (int): Cardinality ``|J4|``.
    prec1 (numpy.ndarray): 5D tensor of shape ``(n^2+1, n^2+1, n+1, n+1, n+1)``,
        indexed as ``[j3, j4, end, i21, i22]``.
    prec2 (numpy.ndarray): 5D tensor of shape ``(n^2+1, n^2+1, n+1, n+1, n+1)``,
        indexed as ``[j1, j2, i3, i11, i12]``.
    prec3 (numpy.ndarray): 4D tensor of shape ``(n^2+1, n^2+1, n+1, n+1)``,
        indexed as ``[j3, j4, i21, i22]``; used when residual indices exceed ``n``.

  Returns:
    float: The computed probability envelope value ``pr``.
  """
  pr = 0
  if (ix < n and iy < n) :
    for i1 in range (ix + 1) :
      for i11 in range (i1 + 1) :
        i21 = i1 - i11
        for i12 in range (ix - i1 + 1 ) :
          i22 = ix - i1 - i12
          for i3 in range (iy + 1) :
            pr += (prec1[j3, j4, iy - i3, i21, i22] * 
                   prec2[j1, j2, i3, i11, i12])                       
  
  elif (ix >= n and iy < n) :
    for i1 in range (ix - n, 2 * n + 1) :
      mi = min (i1, n)
      ma = max (0, i1 - n)
      if (ix - mi < n) :
        for i11 in range (ma, mi + 1) :
          for i12 in range (ix - mi + 1) :
            for i3 in range (iy + 1) :
              pr += (prec1[j3, j4, iy - i3, i1 - i11, ix - mi - i12] * 
                     prec2[j1, j2, i3, i11, i12])
      else :
        for i11 in range (ma, mi + 1) :
          for i2 in range (n, 2 * n + 1) :
            for i12 in range (i2 - n, n + 1) :
              for i3 in range (iy + 1) :
                pr += (prec1[j3, j4, iy - i3, i1 - i11, i2 - i12] * 
                       prec2[j1, j2, i3, i11, i12])
  
  elif (ix < n and iy >= n) :
    for i1 in range (ix + 1) :
      for i11 in range (i1 + 1) :
        for i12 in range (ix - i1 + 1) :
          for i3 in range (iy - n, 2 * n + 1) :
            mi = min(i3, n)
            if (iy - mi < n) :
              pr += (prec1[j3, j4, iy - mi, i1 - i11, ix - i1 - i12] *
                     prec2[j1, j2, i3, i11, i12])
            else :
              pr += (prec2[j1, j2, i3, i11, i12] * 
                     prec3[j3, j4, i1 - i11, ix - i1 - i12])
                     
                            
  else :
    for i1 in range (ix - n, 2 * n + 1) :
      mix = min (i1, n)
      ma_x = max(0, i1 - n)
      if (ix - mix < n) :
        for i11 in range (ma_x, mix + 1) :
          for i12 in range (ix - mix + 1) :
            for i3 in range (iy - n, 2 * n + 1) :
              mi = min(i3, n)
              if (iy - mi < n) :
                pr += (prec1[j3, j4, iy - mi, i1 - i11, ix - mix - i12] *
                       prec2[j1, j2, i3, i11, i12])
              else :
                pr += (prec2[j1, j2, i3, i11, i12] * 
                       prec3[j3, j4, i1 - i11, ix - mix - i12]) 
      
      else :
        for i11 in range (ma_x, mix + 1) :
          for i2 in range (n, 2 * n + 1) :
            for i12 in range (i2 - n, n + 1) :
              for i3 in range (iy - n, 2 * n + 1) :
                mi = min(i3, n)
                if (iy - mi < n) :
                  pr += (prec1[j3, j4, iy - mi, i1 - i11, i2 - i12] *
                       prec2[j1, j2, i3, i11, i12])
                else :
                  pr += (prec2[j1, j2, i3, i11, i12] * 
                         prec3[j3, j4, i1 - i11, i2 - i12])              
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


def induction_envelopes(n, prec1, prec2, prec3, ix, iy, queue) :
  """
  Induction step for MatMult envelopes at fixed (|Ix|, |Iy|) = (ix, iy).

  For all (j1, j2, j3, j4) in [0..n^2], compute:
      pr_ix_iy = compute_proba(n, j1, j2, j3, j4, ix, iy, prec1, prec2, prec3)
  and store env[ix, iy, j1, j2, j3, j4] = min(1, pr_ix_iy).

  Args:
    n (int): Base share parameter (MatMult uses 2*n shares overall).
    prec1: 5D tensor (n^2+1, n^2+1, n+1, n+1, n+1), indexed [j3, j4, end, i21, i22].
    prec2: 5D tensor (n^2+1, n^2+1, n+1, n+1, n+1), indexed [j1, j2, end, i11, i12].
    prec3: 4D tensor (n^2+1, n^2+1, n+1, n+1), indexed [j3, j4, i21, i22].
    ix (int): Target |Ix|.
    iy (int): Target |Iy|.
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
          pr_ix_iy = compute_proba(n, ix, iy, j1, j2, j3, j4, prec1, prec2, prec3)
          env[ix, iy, j1, j2, j3, j4] += min(1, pr_ix_iy)
    
  queue.put((env, ix, iy))
  return  

def compute_envn_MM (n_lim, p, nb_iter, case, cores) :
  """
  Compute MatMult probability envelopes for a number of shares ``n_lim``.

  Starting from the base case ``n = 2`` (via ``initial_case_MM(p)``), this
  routine doubles ``n`` at each iteration and builds the 6D envelope tensor
  for MatMult with ``2*n`` shares. At each stage, it:
    1) Computes refresh envelopes for the current ``n``:
         ``pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)``.
    2) Precomputes single-probability envelopes for MatMult × Refresh using
       either the asymmetric or symmetric variant, depending on ``case``:
         - ``precompute_single_proba(n, envn_MM, pgref)`` if ``case == "Asym"``
         - ``precompute_single_proba_sym_case(n, envn_MM, pgref)`` otherwise
       yielding four tensors: single_proba_one/two/three/four.
    3) Builds the three precomputation :
         - ``prec1`` of shape ``(lim_card+1, lim_card+1, 2*n+1, n+1, n+1)``
         - ``prec2`` of shape ``(lim_card+1, lim_card+1, 2*n+1, n+1, n+1)``
         - ``prec3`` of shape ``(lim_card+1, lim_card+1, n+1, n+1)``
       These are produced in parallel using the worker functions
       ``precomp1``, ``precomp2``, and ``precomp3`` and cached to disk as
       ``prec1{2*n}.npy``, ``prec2{2*n}.npy``, and ``prec3{2*n}.npy`` 
       if not present.
    4) Computes the intermediate envelopes ``final_envn_MM`` by dispatching
       one ``induction_envelopes`` job per ``(ix, iy) ∈ [0..2n]²``; results are
       gathered from a multiprocessing queue.
    5) Applies a final folding step per ``(ix, iy)`` using
       ``final_induction_envelopes(n, p, final_envn_MM, ix, iy, queue)`` to
       obtain ``new_final_envn_MM``.

  After processing a stage, the intermediate cache files
  ``prec1{2*n}.npy``, ``prec2{2*n}.npy``, and ``prec3{2*n}.npy`` are removed,
  ``n`` is doubled, and the loop continues while ``n < n_lim``. The function
  returns the last ``final_envn_MM`` computed.

  Args:
    n_lim (int): Target maximum number of shares; the loop doubles ``n``
      starting from 2 while ``n < n_lim``.
    p (float): Leakage probability parameter.
    nb_iter (int): Number of iterations used in
      ``cardinal_rpc_refresh_envelope``.
    case (str): Precomputation variant. Use ``"Asym"`` for the asymmetric
      pipeline; any other value selects the symmetric precomputation.
    cores (int) : Number of cores to be used.

  Returns:
    numpy.ndarray: 6D envelope tensor for MatMult at the last stage processed,
      with shape ``(2*n + 1, 2*n + 1, lim_card + 1, lim_card + 1, lim_card + 1, 
      lim_card + 1)``, where ``n`` is the final value reached by the loop and
      ``lim_card = n**2`` for that stage.
  """
  n = 2
  envn_MM = initial_case_MM(p)
  queue_precomp = Queue()
  queue_precomp1 = Queue()
  queue = Queue()
  precomp = []
  
  while (n < n_lim) :
    #Compute the probability envelopes for the gadget with n' = 2 * n.
    
    #The maximal cardinality an output set of MatMult with n shares can have.
    lim_card = n**2
    
    #Instantiate the probability envelopes.
    final_envn_MM = np.zeros((2 * n + 1, 2 * n + 1, lim_card + 1, lim_card + 1, 
                              lim_card + 1, lim_card + 1))
    
    #Compute the probability envelopes of the refresh gadget with n shares. 
    pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)
    
    #Make the Precomputation
    if (case == "Asym") :
      precomp = precompute_single_proba(n, envn_MM, pgref)
    else :
      #case = "Sym"
      precomp = precompute_single_proba_sym_case(n, envn_MM, pgref)
    single_proba_one = precomp[0]
    single_proba_two = precomp[1]
    single_proba_three = precomp[2]
    single_proba_four = precomp[3]
    #prec1 = []
    prec1 = np.zeros((lim_card + 1, lim_card + 1, 2 * n + 1, n + 1, n + 1))
    prec2 = np.zeros((lim_card + 1, lim_card + 1, 2 * n + 1, n + 1, n + 1))
    prec3 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1))
    
    
    processes = []
    
    if (os.path.isfile("prec1"+str(2 * n)+".npy") ) :
      prec1 = np.load("prec1"+str(2 * n)+".npy")
      
    else :
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          processes.append(Process(target=precomp1, args=(n, single_proba_three, single_proba_four, j3, j4, queue_precomp1)))
      
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
      
      np.save("prec1"+str(2*n)+".npy",prec1)
    
    processes = []
    
    
    if (os.path.isfile("prec2"+str(2 * n)+".npy")) :
      prec2 = np.load("prec2"+str(2 * n)+".npy")
    
    else :  
      for j1 in range (lim_card + 1) :
        for j2 in range (lim_card + 1) :
          processes.append(Process(target=precomp2, args=(n, single_proba_one, single_proba_two, j1, j2, queue_precomp)))
      
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
        
      np.save ("prec2"+str(2 * n)+".npy", prec2)
      
    processes = []
    if (os.path.isfile("prec3"+str(2 * n)+".npy")) :
      prec3 = np.load("prec3"+str(2 * n)+".npy")
    
    else :  
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          processes.append(Process(target=precomp3, args=(n, single_proba_three, single_proba_four, j3, j4, queue_precomp)))
      
      nb_process = 0
      while (nb_process < len(processes)) :
        lim = min(cores + nb_process, len(processes))
        for i in range (nb_process, lim) :
          pro = processes[i]
          pro.daemon = True
          pro.start()
        
        
        for _ in range (nb_process, lim) :
          (j3, j4, prec3j3j4) = queue_precomp.get()
          prec3[j3, j4] += prec3j3j4[j3, j4]
      
        for i in range (nb_process, lim) : 
          processes[i].join()
          processes[i].terminate()
          processes[i].close()
          nb_process += 1
        
      np.save ("prec3"+str(2 * n)+".npy", prec3)

    
    
    #Computation
    
    processes = []
    for ix in range (2 * n + 1) :
      for iy in range (2 * n + 1) :    
        processes.append(Process(target=induction_envelopes, args=(n, prec1, prec2, prec3, ix, iy, queue)))
    
    for pro in processes :
      pro.daemon = True
      pro.start()
      
    for pro in processes : 
      (pro_envn_MM, ix, iy) = queue.get()
      final_envn_MM[ix,iy] = pro_envn_MM[ix,iy]
        
            
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
    os.remove("prec1"+str(2 * n)+".npy")
    os.remove("prec2"+str(2 * n)+".npy")
    os.remove("prec3"+str(2 * n)+".npy")
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

def compute_envn_mult_4card(n, p, nb_iter, case, cores) :
  """
  Build unified CardSecMult envelopes (cardinal RPC) by composing MatMult and 
  TreeAdd.

  Pipeline:
    1) ``envn_MM = compute_envn_MM(n, p, nb_iter, case, cores)``
    2) ``envn_TA = compute_envn_TA(n, p, nb_iter)``
    3) ``envn = compute_envn(n, envn_MM, envn_TA)``

  Args:
    n (int): Number of share.
    p (float): Leakage probability.
    nb_iter (int): Number of iteration used in the refresh gadget for 
    MatMult and TreeAdd enveloppes computation.
    case (str): Variant selector for MatMult precomputation 
    (e.g., ``"Asym"`` or symmetric).
    cores (int) : Number of cores.

  Returns:
    numpy.ndarray: 3D tensor of CardSecMult envelopes with shape ``(n+1, n+1, n+1)``.
  """
  envn_MM = compute_envn_MM(n, p, nb_iter, case, cores)
  envn_TA = compute_envn_TA(n, p, nb_iter)
  envn = compute_envn (n, envn_MM, envn_TA)
  return envn

def print_envn (n, envn) :
  """
  Print the CardSecMult gadget enveloppe.

  Args:
    n (int): Number of shares.
    envn (numpy.ndarray): 3D tensor of shape ``(n+1, n+1, n+1)`` indexed as
      ``envn[ix, iy, j]``.

  Returns:
    None
  """  
  for j in range (n + 1) :
    print("For  |J| = " + str(j) + " : \n")
    first_row = "        |"
    for iy in range (n + 1) :
      if (iy < 10) :
        first_row += "iy = 0" + str(iy) + "      |"
      
      else :
        first_row += "iy = " + str(iy) + "      |"
    print(first_row) 
    for ix in range (n + 1) :
      row = ""
      if (ix < 10) :
        row += "ix = 0"+ str(ix) + " |"
      else :  
        row += "ix = "+ str(ix) + " |"
      for iy in range (n + 1) :
        var = "%e" % envn[ix, iy, j]
        row += var + " |"
      print(row)
    print("\n")  

################################################################################
############################# Building graphs ##################################

def compute_graph (n,t,p, eps_for_nb_iter, l_nb_iter, str_sym) :
  """
  Plots the empirical Threshold-RPC advantage ``eps_for_nb_iter`` as a function
  of ``l_nb_iter`` (interpreted here as the number of random values used),
  and overlays the theoretical threshold line:
      2 * C(n, t+1) * p^{t+1} * (1 - p)^{n - t - 1}
  using the same color as the plotted series (dashed), which corresponds at the 
  smaller security level we can hope.

  Args:
    n (int): Number of shares.
    t (int): Threshold parameter.
    p (float): Leakage probability.
    eps_for_nb_iter (list[float]): Computed Threshold-RPC
      advantages corresponding to each gamma in ``l_nb_iter``.
    l_nb_iter (list[int]): List of random values used.
    str_sym (str): Label suffix for the curve (e.g., "Sym", "Asym", "Unif").

  Returns:
    None
  """
  plt.plot(l_nb_iter, eps_for_nb_iter, marker='.', label=str_sym + ", n = "+str(n))
  
  threshold_value = 2 * comb(n, t +1) * p**(t + 1) * (1-p)**(n-t-1)
  plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
  plt.legend()

  

  



      
