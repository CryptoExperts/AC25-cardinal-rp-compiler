#!/usr/bin/env python3

# This file compute the random probing security of our masked AES encryption, 
#  build with our compiler described in Section 6 of the paper. In addition,
#  it enables to optimize automatically the gamma taken (i.e. the number of 
#  iteration in RPRefresh) for each block Subbytes, AddRoundKey and 
#  MixColumns.
  


################################################################################
################################# Packages #####################################

import matplotlib.pyplot as plt
from scipy.special import comb
from math import log
import numpy as np
import os
from multiprocessing import Pool, Process, Queue, shared_memory


from partitions import (cardinal_rpc_refresh_envelope, 
                        cardinal_rpc_add_envelope, 
                        cardinal_rpc_gcmult_envelope_pgref, 
                        cardinal_rpc_gcopy_envelope_pgref)
from mult_gen import compute_envn_mult, compute_RPC_threshold                      

import time

################################################################################
################################  Utils ########################################

def env_cadd (n, p, env_ref) :
  """
  Compute the cardinal RPC envelope of a gadget that adds a public constant
  to an n-share masked value (Algorithm 8 in the full paper).

  Args:
    n: Number of shares.
    p: Leakage rate parameter.
    env_ref: Cardinal RPC envelope of the refresh gadget.

  Returns:
    The cardinal RPC envelope of the “addition by constant” gadget.

  Notes:
    The formula follows directly from Lemma 11 of the full paper.
  """

  env = np.zeros((n + 1, n + 1))
  for tin in range (n + 1) :
    for tout in range (n) :
      env[tin, tout] = ((1 - p) * env_ref[tin, tout] + 
                             p * env_ref[tin, tout + 1])
    env[tin, n] = env_ref[tin, n] 
  return env

def precomp_hypergeom(N) :
  """
  Compute and return a 3D array of hypergeometric probability coefficients for 
  all parameters up to N.

  Each value precomp[K, n, k] corresponds to:
    (C(K, k) * C(N - K, n - k)) / C(N, n)

  Args:
    N (int): Maximum population size.

  Returns:
    np.ndarray: A (N+1, N+1, N+1) array of hypergeometric probabilities.
  """

  precomp = np.zeros((N + 1, N + 1, N + 1))
  for K in range (N + 1) :
    for n in range (N + 1) :
      for k in range (max(0, n + K - N), min(K, n) + 1) :
        comb1 = comb(K, k)  
        comb2 = comb(N - K , n - k)
        denom_comb = comb (N, n)
        if denom_comb :
          precomp[K, n, k] = (comb1 * comb2) / denom_comb
  
  return precomp
  
def proceed_para (n, target_file, param, nb_param_env, cores) :
  """
  Run the function `target_file` in parallel over all (ind1, ind2) pairs 
  from 0 to n and collect results in an array.

  The size of the output array depends on nb_param_env.

  Args:
    n (int): Maximum index value.
    target_file (function): Function to run in each process.
    param (list): Base list of parameters (modified in place).
    nb_param_env (int): Number of dimensions for the result array.
    cores (int): Number of parallel processes.

  Returns:
    np.ndarray: The filled result array, or 0 if nb_param_env is invalid.

  Notes : 
    Most of the times, the arrays computed are cardinal RPC enveloppe of a 
    gadget, and the pair (ind1, ind2) represents the first two inputs of the 
    gadget.
  """

  processes = []
  queue = Queue()
  param.append(0)
  param.append(0)
  param.append(queue)
  env = 0
  
  match nb_param_env :
    case 3 :
      env = np.zeros((n + 1, n + 1, n + 1))
    case 4 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1))
    case 5 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
    case 6 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
    case 7 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
    case _ :
      env = 0 
          
  for ind1 in range (n + 1) :
    for ind2 in range (n + 1) :
      param[-3] = ind1
      param[-2] = ind2     
      processes.append(Process(target=target_file, args=tuple(param)))

  nb_process = 0
  while (nb_process < len(processes)) :
    lim = min(cores + nb_process, len(processes))
    for i in range (nb_process, lim) :
      pro = processes[i]
      pro.daemon = True
      pro.start()
      
    for _ in range (nb_process, lim) :
      (ind1, ind2, env_prec) = queue.get()
      env[ind1, ind2] = env_prec
        
        
    for i in range (nb_process, lim) : 
      processes[i].join()
      processes[i].terminate()
      processes[i].close()
      nb_process += 1

  return env

################################################################################
###########################  Subytes Enveloppes ################################

###  Enveloppes for x --> x^{254} in GF(256) ###

def expo_precomp11 (n, envhypergeom, env_sq) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

      ----------------------> j1 
      | 
      |
  i1 ----O---------> ()² ----> j2

  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    envhypergeom : The cardinal RPC enveloppe of the copy using a permutation of 
                   the shares on 1 of the two branches.
    env_sq : The cardinal RPC enveloppe of the squaring gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  precomp = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
            precomp[i1, j1, j2] += (envhypergeom[i1, j1, l] * env_sq[l, j2])
        precomp[i1, j1, j2] = min(1, precomp[i1, j1, j2])                                 
  return precomp

def expo_precomp12 (n, envhypergeom, env_mult) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  i1 ----------> x --------> j1
                 ^
                 |
  i2 ----------> | --O-----> j2

  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    envhypergeom : The cardinal RPC enveloppe of the copy using a permutation of 
                   the shares on 1 of the two branches.
    env_mult : The cardinal RPC enveloppe of the multiplication gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.  
  """
  
  precomp = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1):
      for j1 in range (n + 1) :
        for j2 in range (n + 1) :
          for l in range (n + 1) :
            precomp[i1, i2, j1, j2] += (env_mult[i1, l, j1] * 
                                        envhypergeom[i2, l, j2])
          precomp[i1, i2, j1, j2] = min(1, precomp[i1, i2, j1, j2])                                 
  return precomp

def expo_precomp13 (n, env_prec2, env_ref) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  i1 ----------> x --------------> j1
                 ^
                 |
  i2 ----------> | --O---> R ----> j2

  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    env_prec2 : The cardinal RPC enveloppe computed in the previous function 
                (*expo_precomp12*)
    env_ref : The cardinal RPC enveloppe of the refresh gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.  
  """

  precomp = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for j2 in range (n + 1) :
      for j1 in range (n + 1) :
        for i2 in range (n + 1) :
          for l in range (n + 1) :
            precomp[i1, i2, j1, j2] += (env_prec2[i1, i2, j1, l] * 
                                        env_ref[l, j2])
          precomp[i1, i2, j1, j2] = min(1, precomp[i1, i2, j1, j2])                                 
  return precomp

def env_topref_mult (n, env_mult, env_ref) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  i1 ----> R ----> x ----> j1
                   ^ 
  i2 --------------|

  Args : 
    n : Number of shares.
    env_mult : The cardinal RPC enveloppe of the multiplication gadget.
    env_ref : The cardinal RPC enveloppe of the refresh gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.

  Notes : 
    Adding a refresh before the multiplication can help sometimes to increase 
    the security of the overall AES. In fact, in the multiplication gadget, the 
    shares are not refreshed all together, only half sharings are refreshed 
    together. Hence, it can gives better results to refresh all the shares 
    together before.   
  """

  env = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for l1 in range (n + 1) :
          env[i1, i2, j1] += env_ref[i1, l1] *  env_mult[l1, i2, j1]
        env[i1, i2, j1] = min(1, env[i1, i2, j1]) 
  return env

def precomp_quad (n, envn_sq) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :
  
  i ----> (.)² ----> (.)² ----> j

  which gives us the masked secret at power 4.

  Args : 
    n : Number of shares.
    env_mult : The cardinal RPC enveloppe of the squaring gadget.
  
  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  envn = np.zeros((n + 1, n + 1))
  for i in range (n + 1) :
    for j in range (n + 1) :
      for i1 in range (n + 1) :
        envn[i,j] += envn_sq[i, i1] * envn_sq[i1, j]
      envn[i,j] = min(1, envn[i,j])
  return envn   

def expo_precomp1 (n, prec_meta1, prec_meta2) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  i ----> meta1 ----> meta2 --------> j1 
                |      ^    |-------> j2   
                |      |    
                --------

  where meta1, meta2 are two cardinal RPC enveloppes (computed by expo_precomp11 
  and expo_precomp12 ).

  Args : 
    n : Number of shares.
    prec_meta1 : Cardinal RPC enveloppe of the first gadget M1,
    prec_meta2 : Cardinal RPC enveloppe of the second gadget M2,


  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for i2 in range (n + 1) :
          for i3 in range (n + 1) :
            env[i, j1, j2] += prec_meta1[i, i2, i3] * prec_meta2[i2, i3, j1, j2]
        env[i, j1, j2] = min (1, env[i, j1, j2])
  return env
  
def expo_precomp21(n, envn_sq, envn_mult) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

    i1 ----> (.)² ----> x ----> j
                        ^
                        |
    i2 ------------------

  where (.)² denotes the squaring gadget and x denotes the multiplication gadget.

  Args : 
    n : Number of shares.
    envn_sq : Cardinal RPC enveloppe of the squaring gadget.
    envn_mult : Cardinal RPC enveloppe of the multiplication gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  envn = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) : 
    for i2 in range (n + 1) :
      for j in range (n + 1) : 
        for i3 in range (n + 1) : 
            envn[i1, i2 ,j] += envn_sq[i1, i3] * envn_mult[i3, i2, j]
        envn[i1, i2 ,j] = min(1, envn[i1, i2, j])
  return envn 

def expo_precomp2(n, envn_prec1, envn_prec21) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

      i ----> p1 ----> p2 ----> j
                 |     ^
                 |     | 
                 -------

  where p1 has enveloppe `envn_prec1` and p2 has enveloppe `envn_prec21`.

  Args : 
    n : Number of shares.
    envn_prec1 : Cardinal RPC enveloppe of p1.
    envn_prec21 : Cardinal RPC enveloppe of p2.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  envn = np.zeros((n + 1, n + 1))
  for i in range (n + 1) : 
    for j in range (n + 1) :
      for i1 in range (n + 1) : 
        for i2 in range (n + 1) :
          envn[i,j] += envn_prec21[i1, i2, j] * envn_prec1[i, i1, i2]
      envn[i,j] = min (1, envn[i,j])
  return envn    
  
def env_expo_254 (n, p, envn_sq, envn_mult, pgref) :
  """
  Compute the cardinal RPC envelope of the Exponentiation part of the Subbytes
  block (see Figure 11 of the full paper).

  Args :
    n : The Number of shares.
    p : Probability Leakage rates.
    envn_sq : Cardinal RPC envelope of the squaring gadget.
    envn_mult : Cardinal RPC envelope of the multiplication gadget.
    pgref : Cardinal RPC envelope of the refresh gadget.
  
  Returns : 
    Cardinal RPC envelope of the gadget computing x^{254} for a secret x.  
  """

  hypergeom = precomp_hypergeom(n)
  envhypergeom = env_perm(n, hypergeom)
  envhypergeom = full_env_perm(n, p, envhypergeom)

  new_env_mult = env_topref_mult (n, envn_mult, pgref)

  envn_prec11 = expo_precomp11(n, envhypergeom, envn_sq)
  envn_prec12 = expo_precomp12(n, envhypergeom, new_env_mult)

  envn_prec1 = expo_precomp1(n, envn_prec11, envn_prec12)
  
  #eps1 = expo_precomp1_RPC (n, 0, envn_prec1)
  #print("Advantage eps threshold-RPC, Precomp1 ", log(eps1 ,2))


  envn_quad = precomp_quad (n, envn_sq)  
  envn_pow_sixteen = precomp_quad (n, envn_quad)
  
  envn_prec11_bis = expo_precomp11(n, envhypergeom, envn_quad)
  envn_prec1_bis = expo_precomp1(n, envn_prec11_bis, envn_prec12)
  
  #eps1 = expo_precomp1_RPC (n, 0, envn_prec1_bis)
  #print("Advantage eps threshold-RPC, Precomp1bis ", log(eps1 ,2))

  envn_prec21 = expo_precomp21(n, envn_pow_sixteen, envn_mult)
  envn_prec2 = expo_precomp2(n, envn_prec1_bis, envn_prec21)
  
  envn_prec31 = expo_precomp21(n, envn_prec2, envn_mult)
  envn_prec3 = expo_precomp2(n, envn_prec1, envn_prec31)

  return envn_prec3  



### Enveloppes for Affine(x) ###

def aff_prec11 (n, env_cmult, env_ref, hypergeom) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i ---------> c* ----> j1
          |
          ---O---> R -----> j2

  where `c*` denotes a constant multiplication gadget, `R` denotes a refresh 
  gadget, and O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    env_cmult : Cardinal RPC enveloppe of the constant multiplication gadget.
    env_ref : Cardinal RPC enveloppe of the refresh gadget.
    hypergeom : Precomputed hypergeometric coefficients.

  Returns : 
    The cardinal RPC enveloppe of the above scheme (without taking into account 
    the leakage of the third wire in the copy).
  """

  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) : 
        for i1 in range (i + 1) :
          for i2 in range (i - i1, i + 1) :
            env[i, j1, j2] += hypergeom[i2, i1, i1 + i2 - i] * env_cmult[i1, j1] * env_ref[i2, j2]
        env[i, j1, j2] = min(1, env[i, j1, j2])
  return env

def aff_prec1 (n, p, prec11) :
  """
  Add the third wire involved in the copy made in the previous case (which was 
  not done in the function aff_prec11).

  Args : 
    n : Number of shares.
    p : Leakage rate.
    prec11 : The cardinal RPC enveloppe without taking into account the leak of 
    the third wire.


  Returns : 
    The cardinal RPC enveloppe of the scheme described above.  
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
          env[i, j1, j2] += (comb(i, l) * p**l * (1 - p)**(n - i) * 
                             prec11[i - l, j1, j2]) 
        env[i, j1, j2] = min(1, env[i, j1, j2])
  return env
  
def aff_prec2 (n, envn_sq, envn_add) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

      i1 ----> (.)² --------> + ----> j
                              ^
                              |
      i2 ---------------------

  where (.)² denotes the squaring gadget and + denotes the addition gadget.

  Args : 
    n : Number of shares.
    envn_sq : Cardinal RPC enveloppe of the squaring gadget.
    envn_add : Cardinal RPC enveloppe of the addition gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  envn = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j in range (n + 1) :
        for i3 in range (n + 1) :
          envn[i1, i2, j] += envn_add[i3, i2, j] * envn_sq[i1, i3]
        envn[i1, i2, j] = min(1, envn[i1, i2, j])
  return envn
  
def aff_prec3 (n, envn_cmult, envn_prec2) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

      i1 ----------------> p2 ----> j
                           ^
                           |
      i2 ----> c* ---------|
                   
  where `c*` denotes the constant multiplication gadget and `p2` 
  corresponds to the scheme defined in `aff_prec2`.

  Args : 
    n : Number of shares.
    envn_cmult : Cardinal RPC enveloppe of the constant multiplication gadget.
    envn_prec2 : Cardinal RPC enveloppe of the previous scheme (aff_prec2).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  envn = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) : 
    for i2 in range (n + 1) :
      for j in range (n + 1) :
        for i3 in range (n + 1) :
          envn[i1, i2, j] += envn_prec2[i1, i3, j] * envn_cmult[i2, i3]
        envn[i1, i2, j] = min(1, envn[i1, i2, j])
  return envn


def aff_prec_compo (n, env_prec1, env_prec2, env_prec3) :
  """
  Compute the cardinal RPC enveloppe of the composition pattern in the Affine 
  part of AES : 

  i ----> p1 --------------> p2 ----> j1
             |               ^
             |               |
             ----> p3 -------
                      |
                      |
                      --------------> j2             

  where p1 outputs (i2, i3) and take as input i;
        p3 outputs (i1, j2) and take as input i3;
        p2 outputs (j1) and take as input (i2, i1).

  **Warning : ** p1, p2, and p3 are not the cardinal RPC enveloppe of aff_preci.      

  Args : 
    n : Number of shares.
    env_prec1 : Cardinal RPC enveloppe of p1.
    env_prec2 : Cardinal RPC enveloppe of p2.
    env_prec3 : Cardinal RPC enveloppe of p3.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """ 
  env = np.zeros((n + 1, n + 1, n + 1))
  
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for i1 in range (n + 1) :
          for i2 in range (n + 1) :
            tmp = 0
            for i3 in range (n + 1) :
              tmp += env_prec3[i3, i1, j2] * env_prec1[i, i2, i3]
            env[i, j1, j2] += tmp * env_prec2[i2, i1, j1] 
        env[i, j1, j2] = min (1, env[i, j1, j2])
  return env

def aff_prec_compo_para (n, env_prec1, env_prec2, env_prec3, i, j1, queue) :
  """
  The parallelized version of aff_prec_compo, parallelize according to i and j1.
  """
  env = np.zeros((n + 1))
  
  for j2 in range (n + 1) :
    for i1 in range (n + 1) :
      for i2 in range (n + 1) :
        tmp = 0
        for i3 in range (n + 1) :
          tmp += env_prec3[i3, i1, j2] * env_prec1[i, i2, i3]
        env[j2] += tmp * env_prec2[i2, i1, j1] 
    env[j2] = min (1, env[j2])
  queue.put((i, j1, env))
  return

def aff_final_env (n, env_prec, env_prec2, envn_cadd) :
  """
  Compute the cardinal RPC enveloppe of the following affine pipeline : 

  i ----> p1 -------------
             |            |
             -----------> p2 ----> cadd ----> j
           

  where cadd denotes the addition-by-constant.

  Args : 
    n : Number of shares.
    env_prec : Cardinal RPC enveloppe of p1.
    env_prec2 : Cardinal RPC enveloppe of p2.
    envn_cadd : Cardinal RPC enveloppe of cadd.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  env = np.zeros((n + 1, n + 1))
  for i in range (n + 1) : 
    for j in range (n + 1) :
      for i1 in range (n + 1) : 
        tmp = 0
        for i2 in range (n + 1) : 
          for i3 in range (n + 1) : 
            tmp += env_prec2[i2, i3, i1] * env_prec[i, i2, i3]                    
        env[i,j] += tmp * envn_cadd[i1, j]
      env[i,j] = min(1, env[i,j])
  return env

def env_perm (n, hypergeom) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  ------------->
      |
      |
      ----O---->
  
  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    hypergeom : Distribution of the hypergeometric law
  
  Returns : 
    The cardinal RPC enveloppe of the above scheme ** if we don't take into 
    account the third leaked wires of the scheme**.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (i + 1) :
      for j2 in range (i- j1, i + 1) :
        env[i, j1, j2] = hypergeom[j2, j1, j1 + j2 - i]
  return env

def full_env_perm (n, p, env_perm) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  ------------->
      |
      |
      ----O---->
  
  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    p : probability leakage rate
    env_perm : The cardinal RPC enveloppe returned by env_perm.
  
  Returns : 
    The cardinal RPC enveloppe of the above scheme, ** we take into 
    account the third leaked wires of the scheme**.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
          env[i, j1, j2] += (comb(n - i + l, l) * p**l * (1 - p)**(n - i) * 
                            env_perm[i - l, j1, j2])
  return env
  
def env_copy_simple_prev (n, env_ref, hypergeom) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  ----------------->
      |
      |
      ----O----R--->
  
  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    env_ref : Cardinal RPC enveloppe of the refresh gadget.
    hypergeom : Distribution of the hypergeometric law
  
  Returns : 
    The cardinal RPC enveloppe of the above scheme ** if we don't take into 
    account the third leaked wires of the scheme**.
  """
  env = np.zeros((n + 1, n + 1,  n + 1))
  for i in range (n + 1) :  
    for j1 in range (i + 1) :
      for j2 in range (n + 1) :
        for l in range (i - j1, i + 1) :
          env[i, j1, j2] += hypergeom[l, j1, l + j1 - i] * env_ref[l, j2]
        env[i, j1, j2] = min(1, env[i, j1, j2])
  return env

def env_copy_simple (n, p, env_ref, hypergeom) :
  """
  Compute the cardinal RPC enveloppe of the following scheme : 

  ----------------->
      |
      |
      ----O----R--->
  
  where O denotes a random permutation applied on the shares.

  Args : 
    n : Number of shares.
    p : Probability leakage rate.
    env_ref : Cardinal RPC enveloppe of the refresh gadget.
    hypergeom : Distribution of the hypergeometric law
  
  Returns : 
    The cardinal RPC enveloppe of the above scheme, **we take into 
    account the third leaked wires of the scheme**.
  """  
  prec = env_copy_simple_prev (n, env_ref, hypergeom)
  env = np.zeros((n + 1, n + 1,  n + 1))
  for tin in range (n + 1) :
    for tout1 in range (n + 1) :
      for tout2 in range (n + 1) :
        for l in range (tin + 1) :
          env[tin, tout1, tout2] += (comb(n - l, tin - l) * p**(tin - l) * 
                                    (1 - p)**(n - tin) * prec[l, tout1, tout2]) 
        env[tin, tout1, tout2] = min(1, env[tin, tout1, tout2])
  return env

def env_affine(n, p, env_ref, envn_sq, envn_add, envn_cmult, envn_cadd, hypergeom, cores) : 
  """
  Compute the cardinal RPC envelope of the Affine part of the Subbytes
  block (see Figure 12 of the full paper).

  Args :
    n : The Number of shares.
    p : Probability Leakage rates.
    env_ref : Cardinal RPC envelope of the refresh gadget.
    envn_sq : Cardinal RPC envelope of the squaring gadget.
    envn_add : Cardinal RPC envelope of the addition gadget.
    envn_cmult : Cardinal RPC envelope of the multiplication by a constant 
    gadget.
    envn_cadd : Cardinal RPC envelope of the addition by a constant gadget.
    hypergeom : Probability distribution of the Hypergeometric law.
    cores : Number of cores.
  
  Returns : 
    Cardinal RPC envelope of the Affine part of Subbytes.  
  """


  env_copy = env_copy_simple(n, p, env_ref, hypergeom)
  envn_perm = env_perm(n, hypergeom)
  envn_perm = full_env_perm(n, p, envn_perm) 
  
  envn_prec11 = aff_prec11 (n, envn_cmult, env_ref, hypergeom)
  envn_prec1 = aff_prec1(n, p, envn_prec11)

  envn_prec2 = aff_prec2 (n, envn_sq, envn_add)
  envn_prec3 = proceed_para(n, aff_prec_compo_para,[n, envn_prec1, envn_prec2, envn_prec1], 3, cores)
  envn_prec4 = proceed_para(n, aff_prec_compo_para, [n, envn_prec3, envn_prec2, env_copy], 3, cores)
  envn_prec5 = proceed_para(n, aff_prec_compo_para, [n, envn_prec4, envn_prec2, envn_prec1], 3, cores)
  envn_prec6 = proceed_para(n, aff_prec_compo_para, [n, envn_prec5, envn_prec2, envn_prec1], 3, cores)
  envn_prec7 = proceed_para(n, aff_prec_compo_para, [n, envn_prec6, envn_prec2, envn_prec1], 3, cores)
  envn_prec8 = proceed_para(n, aff_prec_compo_para, [n, envn_prec7, envn_prec2, envn_perm], 3, cores)  
  
  envn_prec3 = aff_prec3(n, envn_cmult, envn_prec2)
  env = aff_final_env(n , envn_prec8, envn_prec3, envn_cadd)
  return env


def env_subbytes (n, env_exp, env_aff) :
  """
  Compute the cardinal RPC envelope of Subbytes.

  Args : 
    n : Number of shares.
    env_exp : Cardinal RPC envelope of the Exponentiation gadget.
    env_aff : Cardinal RPC envelope of the Affine gadget.
  
  Returns : 
    Cardinal RPC envelope of Subbytes.
  """
  env = np.zeros((n + 1, n + 1)) 
  for i in range (n + 1) : 
    for j in range (n + 1) : 
      for i1 in range (n + 1) : 
        env[i,j] += env_aff[i1, j] * env_exp[i, i1]
      env[i,j] = min(1, env[i,j])
  return env
  
  
def compute_RPC_threshold_inner (n, t, envn) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *one input secret and one output secret* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 1 input and 1 output secret.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for i in range (t + 1, n + 1) :
        smj += envn[i, j]
    
    if (res < smj) :
      res = smj    
  
  return res  

################################################################################
########################  MixColumns Enveloppes ################################

#TODO : See if it is important, or we can remove it.
def mc_prechypergeom (n, hypergeom) :
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        if (j1 + j2 >= i and j1 + j2 - i <= n) :
          env[i, j1, j2] = hypergeom[j1, j2, j1 + j2 - i]
  return env

def mc_hypergeom_ref (n, env_hypergeom, pgref) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i ---------------------> j1
            |
            |
            ----O----- R ----> j2

  Args : 
    n : Number of shares.
    env_hypergeom : Cardinal RPC enveloppe of the copy based on hypergeometric 
                    distribution.
    pgref : Cardinal RPC enveloppe of the refresh gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
            env[i, j1, j2] +=  env_hypergeom[i, j1, l] * pgref[l, j2]
  return env 

def mc_sb_hypergeom (n, env_hypergeom, env_sb) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i ---- (.)² --------------> j1
                     |
                     |
                     ----O------> j2

  Args : 
    n : Number of shares.
    env_hypergeom : Cardinal RPC enveloppe of the copy based on hypergeometric 
                    distribution.
    pgref : Cardinal RPC enveloppe of the refresh gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
          env[i, j1, j2] += env_sb[i, l] * env_hypergeom[l, j1, j2]
  return env    



def mc_prec1 (n, env_hypergeom, env_cmult) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i ----------------------> j1
            |
            |
            ----O----- c* ----> j2

  where O denotes a random permutation applied on the shares and 
  c* denotes a constant multiplication gadget.

  Args : 
    n : Number of shares.
    env_hypergeom : Cardinal RPC enveloppe of the copy based on the 
                    hypergeometric distribution.
    env_cmult : Cardinal RPC enveloppe of the constant multiplication gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
          env[i, j1, j2] += env_hypergeom[i, j1, l] * env_cmult[l, j2]
  return env

def mc_prec2 (n, prec1, env_add) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

  i1 ---> p1 ------------------> j1
            |
            --------- 
                    |
  i2 -------------> + ----> j2

  where + denotes the addition gadget.

  Args : 
    n : Number of shares.
    prec1 : Cardinal RPC enveloppe of the previous scheme (mc_prec1).
    env_add : Cardinal RPC enveloppe of the addition gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for l in range (n + 1) :
            env[i1, i2, j1, j2] += prec1[i1, j1, l] * env_add[l, i2, j2]
  return env

def mc_prec3 (n, prec1, prec2) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

    i1 ------------> p2 --------> j1
                     | |
                ------ ---------> j2
               |
    i2 ----> p1 ----------------> j3

  Args : 
    n : Number of shares.
    prec1 : Cardinal RPC enveloppe of the first composed scheme (p1).
    prec2 : Cardinal RPC enveloppe of the second composed scheme (p2).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for j3 in range (n + 1) :
            for l in range (n + 1) :
              env[i1, i2, j1, j2, j3] += prec2[i1, l, j1, j2] * prec1[i2, l, j3]
  return env

def mc_prec4 (n, env_hypergeomref, prec3) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

    i1 ---------------> href --------> j1
                            |
                            v
    i2 -------------------> p3 --------> j2
                               |-------> j3
                               |-------> j4

  where `href` denotes the copy based on the hypergeometric distribution
  followed by a refresh (O then R), and `p3` is the composed scheme defined
  in *mc_prec3*.

  Args : 
    n : Number of shares.
    env_hypergeomref : Cardinal RPC enveloppe of the copy with refresh (O then R).
    prec3 : Cardinal RPC enveloppe of the previous composed scheme (p3).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for j3 in range (n + 1) :
            for j4 in range (n + 1) :
              for l in range (n + 1) :
                env[i1, i2, j1, j2, j3, j4] += (env_hypergeomref[i1, j1, l] * 
                                                prec3 [l, i2, j2, j3, j4])
  return env

def mc_prec4_para (n, env_hypergeomref, name_prec3, i1, i2, queue) :
  """
  Parallelized version of *mc_prec4*.
  """
  existing_shm = shared_memory.SharedMemory(name=name_prec3)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1)
  prec3 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
  
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for j1 in range (n + 1) :
    for j2 in range (n + 1):
      for j3 in range (n + 1) :
        for j4 in range (n + 1) :
          for l in range (n + 1) :
                env[j1, j2, j3, j4] += (env_hypergeomref[i1, j1, l] * 
                                        prec3 [l, i2, j2, j3, j4])
  queue.put((i1, i2, env))
  existing_shm.close()
  return 

def mc_prec5 (n, prec4, env_add) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

    i1 ---------> p4 ---------------
                  ^  |              |
                  |  |--------------|-------------> j2      
    i2 ------------  |              +-------------> j1
                     |--------------|-------------> j3      
                     |              |
                     |--------------  


  where `p4` is the composed scheme defined in *mc_prec4*.

  Args : 
    n : Number of shares.
    prec4 : Cardinal RPC enveloppe of the previous composed scheme (p4).
    env_add : Cardinal RPC enveloppe of the addition gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """  
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for j3 in range (n + 1) :
            for l in range (n + 1) :
              for l1 in range (n + 1) :
                env[i1, i2, j1, j2, j3] += (prec4[i1, i2, l, j2, j3, l1] * 
                                            env_add[l, l1, j1])
  return env

def mc_prec5_para (n, name_prec4, env_add, i1, i2, queue) :
  """
  Parallelized version of *mc_prec5*.
  """
  existing_shm = shared_memory.SharedMemory(name=name_prec4)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1, n + 1)
  prec4 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
  
  env = np.zeros((n + 1, n + 1, n + 1))
  for j1 in range (n + 1) :
    for j2 in range (n + 1):
      for j3 in range (n + 1) :
        for l in range (n + 1) :
          for l1 in range (n + 1) :
            env[j1, j2, j3] += (prec4[i1, i2, l, j2, j3, l1] * 
                                env_add[l, l1, j1])
  queue.put((i1, i2, env))
  existing_shm.close()
  return

def mc_prec6 (n, prec5, env_add) :
  r"""
  Compute the cardinal RPC enveloppe of the following scheme :

    i1 -----------> p5 -----------------------------> j1
                  ^  | \
                  |  |  \------> + -----------------> j2
                  |  |           ^
                  |  ------------|------------------> j3
                  |              |
                  |              |
    i2 -----------|--------------+
                  |
    i3 ------------ 

  where `p5` is the composed scheme defined in *mc_prec5*, producing (j1, j3)
  and an intermediate `l` that is combined with `i2` through the addition gadget
  to yield `j2`.

  Args : 
    n : Number of shares.
    prec5 : Cardinal RPC enveloppe of the previous composed scheme (p5).
    env_add : Cardinal RPC enveloppe of the addition gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """  
  
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for i3 in range (n + 1) :
        for j1 in range (n + 1) :
          for j2 in range (n + 1):
            for j3 in range (n + 1) :
              for l in range (n + 1) :
                env[i1, i2, i3, j1, j2, j3] += (prec5[i1, i3, j1, l, j3] * 
                                                env_add[l, i2, j2])
  return env

def mc_prec6_para (n, name_prec5, env_add, i1, i2, queue) :
  """
  Parallelized version of *mc_prec6*.
  """
  existing_shm = shared_memory.SharedMemory(name=name_prec5)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1)
  prec5 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)

  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i3 in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1):
        for j3 in range (n + 1) :
          for l in range (n + 1) :
            env[i3, j1, j2, j3] += (prec5[i1, i3, j1, l, j3] * 
                                    env_add[l, i2, j2])
  queue.put((i1, i2, env))
  existing_shm.close()
  return

def mc_prec7 (n, prec6, env_hypergeomref) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

    i1 --------------------------                              
                                 |
                ---------------> p6 ----------------> j1
              /                 /   |--------------> j2
    i2 ----------O--- R --------    |--------------> j3

  where `O` denotes a random permutation on the shares and `R` a refresh.
  The block `p6` is the composed scheme defined in *mc_prec6*.

  Args : 
    n : Number of shares.
    prec6 : Cardinal RPC enveloppe of the previous composed scheme (p6).
    env_hypergeomref : Cardinal RPC enveloppe of the copy with refresh (O then R).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1) :
          for j3 in range (n + 1) :
            for l in range (n + 1) :
              for l1 in range (n + 1) :
                env[i1, i2, j1, j2, j3] += (env_hypergeomref[i2, l, l1] * 
                                            prec6[i1, l, l1, j1, j2, j3])
  return env

def mc_prec7_para (n, name_prec6, env_hypergeomref, i1, i2, queue) :
  """
  Parallelized version of *mc_prec7*.
  """
  existing_shm = shared_memory.SharedMemory(name=name_prec6)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1, n + 1)
  prec6 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
  
  env = np.zeros((n + 1, n + 1, n + 1))
  for j1 in range (n + 1) :
    for j2 in range (n + 1) :
      for j3 in range (n + 1) :
        for l in range (n + 1) :
          for l1 in range (n + 1) :
            env[j1, j2, j3] += (env_hypergeomref[i2, l, l1] * 
                                prec6[i1, l, l1, j1, j2, j3])
  queue.put((i1, i2, env))
  existing_shm.close()              
  return

def mc_prec8 (n, env_hypergeomref, env_cmult) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i ---------------- c* -----> j1
              |
              | 
              ---O----- R -------> j2


  where `O` denotes a random permutation applied on the shares, 
  `R` a refresh, and `c*` the constant multiplication gadget.

  Args : 
    n : Number of shares.
    env_hypergeomref : Cardinal RPC enveloppe of the copy with refresh (O then R).
    env_cmult : Cardinal RPC enveloppe of the constant multiplication gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """

  env = np.zeros((n + 1, n + 1, n + 1))
  for i in range (n + 1) :
    for j1 in range (n + 1) :
      for j2 in range (n + 1) :
        for l in range (n + 1) :
          env[i, j1, j2] += env_hypergeomref[i, l, j2] * env_cmult[l, j1]
  return env

def mc_prec9 (n, prec8, env_add) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i1 -----------> p8 --------- + --------------> j1
                        |          ^
                        |          |
                        -----------|---------------> j2
                                   |
                                   |
      i2 ---------------------------

  where `p8` is the composed scheme defined in *mc_prec8*, and `+` denotes
  the addition gadget.

  Args : 
    n : Number of shares.
    prec8 : Cardinal RPC enveloppe of the previous composed scheme (p8).
    env_add : Cardinal RPC enveloppe of the addition gadget.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
    
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n +1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1) :
          for l in range (n + 1) :
            env[i1, i2, j1, j2] += prec8[i1, l, j2] * env_add[l, i2, j1]
  return env

def mc_prec10 (n, prec9, prec8) :
  """
  Compute the cardinal RPC enveloppe of the following scheme :

      i1 -------------> p9 -------> j1 
                        ^  |------> j2
                        |
                  -------                          
                  |
      i2 ----> p8 -----------------> j3

  where `p8` and `p9` are the composed schemes defined in *mc_prec8* and
  *mc_prec9* respectively.

  Args : 
    n : Number of shares.
    prec9 : Cardinal RPC enveloppe of the previous composed scheme (p9).
    prec8 : Cardinal RPC enveloppe of the composed scheme (p8).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for j3 in range (n + 1) :
            for l in range (n + 1) :
              env[i1, i2, j1, j2, j3] += prec8[i2, l, j3] * prec9[i1, l, j1, j2]
  return env


def mc_prec11 (n, prec10, prec7) :
  r"""
  Compute the cardinal RPC enveloppe of the following scheme :

      i1 ---------> p10 ------------
                   /    |           \   |--------> j1
      i2 ----------     |            p7 ---------> j2
                        |           /   |--------> j3
                        ------------    |--------> j4

  where `p10` and `p7` are the composed schemes defined in *mc_prec10*
  and *mc_prec7*, respectively.

  Args : 
    n : Number of shares.
    prec10 : Cardinal RPC enveloppe of the composed scheme (p10).
    prec7 : Cardinal RPC enveloppe of the composed scheme (p7).

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j1 in range (n + 1) :
        for j2 in range (n + 1):
          for j3 in range (n + 1) :
            for j4 in range (n + 1) :
              for l1 in range (n + 1) :
                for l2 in range (n + 1) :
                  env[i1, i2, j1, j2, j3, j4] += (prec10[i1, i2, j1, l1, l2] * 
                                                  prec7[l1, l2, j2, j3, j4])
  return env

def mc_prec11_para (n, name_prec10, name_prec7, i1, i2, queue) :
  """
  Parallelized version of *mc_prec11*.
  """
  existing_shm = shared_memory.SharedMemory(name=name_prec10)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1)
  prec10 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
  
  existing_shm2 = shared_memory.SharedMemory(name=name_prec7)
  shape2 = (n + 1, n + 1, n + 1, n + 1, n + 1)
  prec7 = np.ndarray(shape2, dtype=np.float64, buffer=existing_shm2.buf)


  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for j1 in range (n + 1) :
    for j2 in range (n + 1):
      for j3 in range (n + 1) :
        for j4 in range (n + 1) :
          for l1 in range (n + 1) :
            for l2 in range (n + 1) :
              env[j1, j2, j3, j4] += (prec10[i1, i2, j1, l1, l2] * 
                                      prec7[l1, l2, j2, j3, j4])
  queue.put((i1, i2, env))
  existing_shm.close()
  existing_shm2.close()
  return


def env_start_mixc (n, p, env_add, env_cmult, hypergeom, pgref, cores) :
  """
  Compute the cardinal RPC envelope of the start part of MixColumn (i.e. one 
  blue frame on the Figure 13 of the full paper version)

  Args : 
    n : Number of shares.
    p : Probability Leakage rate.
    env_add : Cardinal RPC envelope of the addition gadget.
    env_cmult : Cardinal RPC envelope of the multiplication by constant gadget.
    hypergeom : Hypergeometric Distribution.
    pgref : Cardinal RPC envelope of the refresh gadget.
    cores : Number of cores.

  Returns : 
    Cardinal RPC envelope of the blue frame on the Figure 13.
  """
  envhypergeom = mc_prechypergeom(n, hypergeom)
  envhypergeom = full_env_perm(n, p, envhypergeom)
  #envhypergeom = mc_sb_hypergeom(n, envhypergeom, env_sb)
  envhypergeom_ref = mc_hypergeom_ref (n, envhypergeom, pgref)

  prec1 = mc_prec1(n, envhypergeom, env_cmult)
  prec2 = mc_prec2(n, prec1, env_add)
  prec3 = mc_prec3(n, prec1, prec2)

  shm = shared_memory.SharedMemory(create=True, size=prec3.nbytes)
  shared_arr = np.ndarray(prec3.shape, dtype=prec3.dtype, buffer=shm.buf)
  np.copyto(shared_arr, prec3)
  
  prec4 = proceed_para(n, mc_prec4_para, [n, envhypergeom_ref, shm.name], 6, cores)
  shm.close()
  shm.unlink()
  #prec4 = mc_prec4(n, envhypergeom_ref, prec3)
  
  shm = shared_memory.SharedMemory(create=True, size=prec4.nbytes)
  shared_arr = np.ndarray(prec4.shape, dtype=prec4.dtype, buffer=shm.buf)
  np.copyto(shared_arr, prec4)

  prec5 = proceed_para(n, mc_prec5_para, [n, shm.name, env_add], 5, cores)
  
  shm.close()
  shm.unlink()
  #prec5 = mc_prec5(n, prec4, env_add)
  
  shm = shared_memory.SharedMemory(create=True, size=prec5.nbytes)
  shared_arr = np.ndarray(prec5.shape, dtype=prec5.dtype, buffer=shm.buf)
  np.copyto(shared_arr, prec5)

  prec6 = proceed_para(n, mc_prec6_para, [n, shm.name, env_add], 6, cores)
  
  shm.close()
  shm.unlink()
  #prec6 = mc_prec6(n, prec5, env_add)
  

  shm = shared_memory.SharedMemory(create=True, size=prec6.nbytes)
  shared_arr = np.ndarray(prec6.shape, dtype=prec6.dtype, buffer=shm.buf)
  np.copyto(shared_arr, prec6)

  prec7 = proceed_para(n, mc_prec7_para, [n, shm.name, envhypergeom_ref], 5, cores)

  shm.close()
  shm.unlink()
  
  #prec7 = mc_prec7(n, prec6, envhypergeom_ref)
  
  prec8 = mc_prec8(n, envhypergeom_ref, env_cmult)
  prec9 = mc_prec9(n, prec8, env_add)
  prec10 = mc_prec10(n, prec9, prec8)
  
  shm = shared_memory.SharedMemory(create=True, size=prec10.nbytes)
  shared_arr = np.ndarray(prec10.shape, dtype=prec10.dtype, buffer=shm.buf)
  np.copyto(shared_arr, prec10)

  shm2 = shared_memory.SharedMemory(create=True, size=prec7.nbytes)
  shared_arr2 = np.ndarray(prec7.shape, dtype=prec7.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, prec7)

  prec11 = proceed_para(n, mc_prec11_para, [n, shm.name, shm2.name], 6, cores)
  
  shm2.close()
  shm2.unlink()

  shm.close()
  shm.unlink()


  #prec11 = mc_prec11(n, prec10, prec7)

  return prec11


def compute_RPC_threshold_start_mc(n, t, env) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *two input secrets and four output secrets* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 2 input and 4 output secrets.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  eps = 0
  for j1 in range (t + 1) : 
    for j2 in range (t + 1) :
      for j3 in range (t + 1) :
        for j4 in range (t + 1) :
          smj = 0
          for i1 in range (n + 1) :
            for i2 in range (n + 1) :
              if (i1 > t or i2 > t) :
                smj += env[i1, i2, j1, j2, j3, j4]
          eps = max(eps, smj)         
  return eps

################################################################################
##############################  Ark + Sb #######################################

def env_ark_subbytes (n, env_add_ark, env_sb) :
  """
  Compute the cardinal RPC enveloppe of AddRoundKey followed by SubBytes.

  Args : 
    n : Number of shares.
    env_add_ark : Cardinal RPC enveloppe of AddRoundKey
    env_sb : Cardinal RPC enveloppe of SubBytes.

  Returns : 
    The cardinal RPC enveloppe of AddRoundKey + SubBytes.
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for j in range (n + 1) :
        for i3 in range (n + 1) :
          env[i1, i2, j] += env_add_ark[i1, i2, i3] * env_sb[i3, j]
        env[i1, i2, j] = min (1, env[i1, i2, j])
  return env 

def compute_RPC_threshold_ark_sb (n, t, env_ark_sb) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *two input secrets and one output secret* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 2 input and 1 output secrets.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for i1 in range (n + 1) :
      for i2 in range (n + 1) :
        if (i1 > t or i2 > t) :
          smj += env_ark_sb[i1, i2, j]
    
    if (res < smj) :
      res = smj    
  
  return res

################################################################################
##############################  Ark + Sb + Ark #################################

def env_ark_subbytes_ark (n, env_add_ark, env_ark_sb) :
  r"""
  Compute the cardinal RPC enveloppe of :

    i1 -----> ARK -----> SB -----> ARK -----> j
             /                    /
    i2 ------                    /
                                /
    i3 -------------------------
    

  Args : 
    n : Number of shares.
    env_add_ark : Cardinal RPC enveloppe of AddRoundKey
    env_ark_sb : Cardinal RPC enveloppe of ARK + SB.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) :
      for i3 in range (n + 1) :
        for j in range (n + 1) :
          for i4 in range (n + 1) :
            env[i1, i2, i3, j] += env_ark_sb[i1, i2, i4] * env_add_ark [i4, i3, j]
          env[i1, i2, i3, j] = min (1, env[i1, i2, i3, j])
  return env

################################################################################
############################  End_MC + Ark + Sb ################################

def env_endmixc_ark_sb (n, env_add_mc, env_ark_sb) :
  r"""
  Compute the cardinal RPC enveloppe of :

    i1 -----> EndMC -----> ARK -----> SB-----> j
             /            /
    i2 ------            /
                        /
    i3 -----------------
    

  Args : 
    n : Number of shares.
    env_add_mc : Cardinal RPC enveloppe of EndMC.
    env_ark_sb : Cardinal RPC enveloppe of ARK + SB.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) : 
      for i3 in range (n + 1) :
        for j in range (n + 1) :
          for l in range (n + 1) :
            env[i1, i2, i3, j] += env_add_mc[i1, i2, l] * env_ark_sb[l, i3, j]
          env[i1, i2, i3,j] = min(1, env[i1, i2, i3, j])
  return env  

def compute_RPC_threshold_endmc_ark_sb (n, t, env_endmc_ark_sb) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *three input secrets and one output secret* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 3 input and 1 output secrets.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for i1 in range (n + 1) :
      for i2 in range (n + 1) :
        for i3 in range (n + 1) :
          if (i1 > t or i2 > t or i3 > t) :
            smj += env_endmc_ark_sb[i1, i2, i3, j]
    
    if (res < smj) :
      res = smj    
  
  return res

################################################################################
#########################  End_MC + Ark + Sb + Ark #############################

def env_endmixc_ark_sb_ark (n, env_add_mc, env_ark_sb_ark) :
  r"""
  Compute the cardinal RPC enveloppe of :

    i1 -----> EndMC -----> ARK -----> SB-----> ARK -----> j
             /            /                   /
    i2 ------            /                   /
                        /                   /
    i3 -----------------                   /
                                          /
    i4------------------------------------
    

  Args : 
    n : Number of shares.
    env_add_mc : Cardinal RPC enveloppe of EndMC.
    env_ark_sb_ARK : Cardinal RPC enveloppe of ARK + SB + ARK.

  Returns : 
    The cardinal RPC enveloppe of the above scheme.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for i1 in range (n + 1) :
    for i2 in range (n + 1) : 
      for i3 in range (n + 1) :
        for i4 in range (n + 1) :
          for j in range (n + 1) :
            for l in range (n + 1) :
              env[i1, i2, i3, i4, j] += env_add_mc[i1, i2, l] * env_ark_sb_ark[l, i3, i4, j]
            env[i1, i2, i3, i4, j] = min(1, env[i1, i2, i3, i4, j])
  return env  

def compute_RPC_threshold_endmc_ark_sb_ark (n, t, env_endmc_ark_sb_ark) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *four input secrets and one output secret* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 4 input and 1 output secrets.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for i1 in range (n + 1) :
      for i2 in range (n + 1) :
        for i3 in range (n + 1) :
          for i4 in range (n + 1) : 
            if (i1 > t or i2 > t or i3 > t or i4  > t) :
              smj += env_endmc_ark_sb_ark[i1, i2, i3, i4, j]
    
    if (res < smj) :
      res = smj    
  
  return res



def compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, gamma_ark, t, cores) :
  r"""
  Args :
    n : Number of shares.
    p : Leakage rate.
    logp : Theblogarithm in base 2 of p.
    gamma_sb : Gamma used for the Subbytes part and the multiplication gadget.
    l_gamma_sb : List of gamma used for the multiplication gadget (in the MatMult part)
    gamma_mc : Gamma used for the MixColumns part.
    gamma_ark : Gamma used for the AddRoundKey part.
    t : Threshold for RPC notions.
    cores : Number of cores.

  Returns : 
    Advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC of a full AES.
  """
  hypergeom = precomp_hypergeom(n)
  
  pgref_mc = cardinal_rpc_refresh_envelope(n, p, gamma_mc)
  pgref_ark = cardinal_rpc_refresh_envelope(n, p, gamma_ark)
  pgref_sb = cardinal_rpc_refresh_envelope(n, p, gamma_sb)
  
  envn_add_mc = cardinal_rpc_add_envelope (n, p, pgref_mc)
  envn_add_ark = cardinal_rpc_add_envelope (n, p, pgref_ark)
  envn_add_sb = cardinal_rpc_add_envelope (n, p, pgref_sb)
  
  envn_cmult_mc = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref_mc)
  envn_cmult_sb = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref_sb)
  envn_cadd_sb = env_cadd(n, p, pgref_sb)
  
  envn_sq_sb = envn_cmult_sb
  
  envn_mult = 0
  str_file = "mult_n"+ str(n) + "_p"+ str(logp) + "_gamma" + str(gamma_sb)+"_lgammasb" + str(l_gamma_sb)  + ".npy"
  if(os.path.isfile(str_file)) : 
    envn_mult = np.load(str_file)
  else :
    envn_mult = compute_envn_mult(n, p, l_gamma_sb, gamma_sb, cores)
    np.save(str_file, envn_mult)


  #Subbytes enveloppes. 
  env_expo = env_expo_254 (n, p, envn_sq_sb, envn_mult, pgref_sb)
  env_aff = env_affine (n, p, pgref_sb, envn_sq_sb, envn_add_sb, envn_cmult_sb, 
                        envn_cadd_sb, hypergeom, cores)  
  env_sb = env_subbytes (n, env_expo, env_aff)

  #AddRoundKey + Subbytes enveloppes.
  env_ark_sb = env_ark_subbytes (n, envn_add_ark, env_sb)
  
  #AddRoundKey + Subbytes + AddRoundKey enveloppes.
  env_ark_sb_ark = env_ark_subbytes_ark (n, envn_add_ark, env_ark_sb)
  
  #Last addition of MixColums + AddRoundKey + Subbytes enveloppes.
  env_endmc_ark_sb = env_endmixc_ark_sb (n, envn_add_mc, env_ark_sb)
  
  #Last addition of MixColums + AddRoundKey + Subbytes +AddRoundKey enveloppes.
  env_endmc_ark_sb_ark = env_endmixc_ark_sb_ark (n, envn_add_mc, env_ark_sb_ark)
  
  #MixColumns without the last addition enveloppes.
  env_start_mc = env_start_mixc (n, p, envn_add_mc, envn_cmult_mc, hypergeom, 
                                 pgref_mc, cores) 

  eps1 = compute_RPC_threshold_endmc_ark_sb_ark(n, t, env_endmc_ark_sb_ark)
  eps2 = compute_RPC_threshold_start_mc(n, t, env_start_mc)
  eps3 = compute_RPC_threshold_endmc_ark_sb(n, t, env_endmc_ark_sb)
  eps4 = compute_RPC_threshold_ark_sb(n,t, env_ark_sb)

  #There is in total 232 blocks to compute.
  eps_AES_encrypt = 232 * max (eps1, eps2,  eps3, eps4)

  return eps_AES_encrypt

def compute_RPC_threshold_copy(n, t, env_copy) :
  r"""
  Derive an advantage $\epsilon$ for the $(t, p, \epsilon)$-threshold RPC 
  security of a gadget with *one input secret and two output secrets* from its 
  cardinal RPC enveloppe.

  Args : 
    n : Number of shares.
    t : Threshold used for the threshold RPC security.
    envn : cardinal RPC enveloppe with 1 input and 2 output secrets.

  Returns : 
    Advantage $\epsilon$ of the threshold RPC security.
  """
  eps = 0
  for tout1 in range (t + 1) :
    for tout2 in range (t + 1) :
      epso1o2 = 0
      for tin in range (t + 1, n + 1) : 
        epso1o2 += env_copy[tin, tout1, tout2]
      eps = max (epso1o2, eps) 
  return eps



def compare_cRPC_tRPC_AES (n, p, thr, cores) :
  r"""
  Compute the "optimal" threshold RPC security for the masked AES using 
  two different approaches:

    1. Derive the threshold RPC security of the AES directly from the 
       threshold RPC of the individual gadgets composing it.
    2. Derive the threshold RPC security of the AES from the cardinal RPC 
       security of the gadgets and their composition.

  In both approaches, the optimization is performed over the different 
  gamma values used in the refresh gadgets.

  Args : 
    n : Number of shares.
    p : Probability leakage rate.
    thr : Threshold parameter used in the gamma selection. When computing 
          the final threshold RPC, the result corresponds to the best 
          threshold RPC security within this margin (thr).
    cores : Number of CPU cores used for parallelization.
  
  Returns : 
    1. The advantage $\epsilon$ for the first technique.
    2. The optimized gamma for the first technique.
    3. The advantage $\epsilon$ for the second technique.
    4. The optimized gamma for the second technique.
  """

  ##############################################################################
  ############################# Threshold RPC ##################################
  gamma = 500
  l_gamma_mult = [500] * n
  eps_tRPC = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
  
  eps_tRPC, gamma = optimize_gamma_tRPC (n, p, thr, eps_tRPC, cores)
  eps_tRPC, l_gamma = optimize_l_gamma_mult_tRPC (n, p, thr, gamma, eps_tRPC, cores)

  ##############################################################################
  ############################## Cardinal RPC ##################################
  
  gamma_ark = 500
  gamma_sb = 500
  gamma_mc = 500
  l_gamma_sb = [500] * n
  t = n // 2
  eps_cRPC = compute_RPC_AES(n, p, int(log(p, 2)), gamma_sb, l_gamma_sb, 
                             gamma_mc, gamma_ark, t, cores)

  eps_cRPC, gamma_ark = optimize_gamma_ark(n, p, eps_cRPC, thr, cores)
  eps_cRPC, gamma_mc = optimize_gamma_mc (n, p, eps_cRPC, thr, gamma_ark, cores)
  eps_cRPC, gamma_sb = optimize_gamma_sb(n, p, eps_cRPC, thr, gamma_ark, 
                                         gamma_mc, cores)
  eps_cRPC, l_gamma_sb = optimize_gamma_l_sb(n, p, eps_cRPC, thr, gamma_ark, 
                                              gamma_mc, gamma_sb, cores)
  
  return (eps_tRPC, gamma, l_gamma, eps_cRPC, gamma_ark, gamma_mc, gamma_sb, 
          l_gamma_sb)

def optimize_gamma_tRPC (n, p, thr, eps_witness, cores) :
  r"""
  Optimize the "gamma" to use to btain the best security level for the threshold 
  RPC security of the full AES.

  Args :
    n : Number of shares.
    p : Probability Leakage rate.
    thr : threshold, our security must be within thr of the best result.
    eps_witness : the best security result we can obtained. In practice it is 
                  not the best, it s just one where the gamma taken is big.
    cores : Number of cores to be used.

  Returns : 
    The security result $\epsilon$ and the gamma used to obtain this security 
    result.
  """

  logp = int(log(p, 2))
  l_gamma_mult = [500] * n
  t = n // 2

  #We start with 50.
  gamma = 50
  eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
  
  gamma1 = gamma
  eps1 = eps

  if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
    while (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr and not(gamma == 0)) : 
      eps1 = eps
      gamma1 = gamma
      gamma = gamma // 2
      eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
    if (gamma == 0) :
      if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
        return eps, gamma
      else :
        return eps1, gamma + 1
    
  else :
    while (np.abs(log(eps1, 2) - log(eps_witness, 2)) >= thr) :
      eps = eps1
      gamma1 = 2 * gamma1
      eps1 = compute_tRPC_AES(n, p, gamma1, l_gamma_mult, cores)
    gamma = gamma1 // 2

  #We have gamma which is upper the threshold and gamma1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma == gamma1) and not (gamma == (gamma1 - 1))) :
    gamma2 = (gamma + gamma1) // 2
    eps2 = compute_tRPC_AES(n, p, gamma2, l_gamma_mult, cores)
    if (np.abs(log(eps2, 2) - log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma1 = gamma2
    else :
      eps = eps2
      gamma = gamma2
    
  if (gamma == gamma1) :
    return eps, gamma

  if (gamma == (gamma1 - 1)):
    return eps1, gamma1

def optimize_l_gamma_mult_tRPC (n, p, thr, gamma, eps_witness, cores) :
    r"""
    Optimize the list of "gamma" used in the multiplication gadget to obtain the 
    best security level for the threshold RPC security of the full AES, after 
    optimized the gamma for the other gadget in the previous function.

    Args :
      n : Number of shares.
      p : Probability Leakage rate.
      thr : threshold, our security must be within thr of the best result.
      gamma : The optimal gamma, from the previous function.
      eps_witness : the best security result we can obtained. In practice it is 
                  not the best, it s just one where the gamma taken is big.
      cores : Number of cores to be used.

    Returns : 
      The security result $\epsilon$ and the gamma used to obtain this security 
      result.
    """ 
    nbis = n // 2 + 2
    if (n%2 == 0) :
      nbis = n//2 + 1
    
    
    l_gamma_mult = [500] * (nbis)
    l_gamma_mult[0] = 0
    l_gamma_mult[1] = 0

    for i in range (2, len(l_gamma_mult)) :
      l_gamma_mult[i] = 0
      eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
      while (np.abs(log(eps, 2) - log(eps_witness, 2)) > thr) :
        l_gamma_mult[i] += 3
        eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
      
      if(not(l_gamma_mult[i] == 0)) :
        l_gamma_mult[i] -= 2
        eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)
        while (np.abs(log(eps, 2) - log(eps_witness, 2)) > thr) :
          l_gamma_mult[i] += 1
          eps = compute_tRPC_AES(n, p, gamma, l_gamma_mult, cores)

      
    return eps, l_gamma_mult


def compute_tRPC_AES (n, p, gamma, l_gamma_mult, cores) :
  """
  Compute the threshold RPC security of the AES, using only the threshold RPC 
  security of the base gadget.

  Args :
    n : Number of shares
    p : Probability Leakage rate
    gamma : gamma used in every base gadget except the MatMult part of the 
            multiplication gadget.
    l_gamma : List of gamma used in the MatMult part of the multiplication gadget.
    cores : Number of cores.

  Returns : 
    Threshold RPC security of the overall AES.  
  """

  t = n // 2

  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  pgadd = cardinal_rpc_add_envelope(n, p, pgref)
  pgcopy = cardinal_rpc_gcopy_envelope_pgref (n, p, pgref)
  pgcmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  pgcadd = env_cadd(n, p, pgref)
  pgmult = compute_envn_mult(n, p, l_gamma_mult, gamma, cores)


  eps_ref = compute_RPC_threshold_inner(n, t, pgref)
  eps_add = compute_RPC_threshold(n, pgadd, t)
  eps_copy = compute_RPC_threshold_copy(n, t, pgcopy)
  eps_cmult = compute_RPC_threshold_inner(n, t, pgcmult)
  eps_cadd = compute_RPC_threshold_inner(n, t, pgcadd)
  eps_mult = compute_RPC_threshold(n, pgmult, t)

  eps = max (eps_ref, eps_add, eps_copy, eps_cmult, eps_cadd, eps_mult)
  return 9936 * eps






################################################################################
########################### Optimisation choice Gamma ##########################

def optimize_gamma_ark (n, p, eps_witness, thr, cores) :
  """
  Finds an “optimal” value of gamma_ark for the compute_RPC_AES function.  
  This gamma_ark parameter is used only in the AddRoundKey part of AES, i.e.,
  the addition gadget of AddRoundKey.

  Idea
  ----
  We search for a value of gamma_ark such that the difference between log2(eps)
  (returned by compute_RPC_AES) and log2(eps_witness) remains below the
  threshold `thr`.

  The search strategy is:
    1. Start with gamma_ark = 50.
    2. If this initial value already gives an eps close enough to eps_witness
       (|log2(eps) - log2(eps_witness)| < thr), repeatedly **halve** gamma_ark
       to find the smallest value that still satisfies the threshold.
    3. Otherwise, if the initial value is not close enough, repeatedly
       **double** gamma_ark until the threshold is satisfied.
    4. Once two consecutive values are found such that one satisfies the
       threshold and the other does not, perform a **binary search** between
       them to locate the smallest gamma_ark that meets the criterion.

  Args:
    n : Number of shares.
    p : Leakage probability (used to derive logp).
    eps_witness : Target RPC security value to approximate.
    thr : Allowed threshold on |log2(eps) - log2(eps_witness)|.
    cores : Number of CPU cores used by compute_RPC_AES (parallelization).

  Returns:
    eps1 : The eps value obtained for the chosen gamma_ark1.
    gamma_ark1 : The smallest gamma_ark found such that the log-difference
                with eps_witness remains below the threshold `thr`.
  """

  logp = int(log(p, 2))
  gamma_mc = 500
  gamma_sb = 500
  l_gamma_sb = [500] * n
  t = n // 2

  #We start with 50.
  gamma_ark = 50

  eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, gamma_ark, 
                        t, cores)
  gamma_ark1 = gamma_ark

  eps1 = eps

  if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
    while (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr and not(gamma_ark == 0)) : 
      eps1 = eps
      gamma_ark1 = gamma_ark
      gamma_ark = gamma_ark1 // 2
      eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    if (gamma_ark == 0) :
      if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
        return eps, gamma_ark
      else :
        return eps1, gamma_ark + 1
    
  else :
    while (np.abs(log(eps1, 2) - log(eps_witness, 2)) >= thr) :
      eps = eps1
      gamma_ark1 = 2 * gamma_ark1
      eps1 = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                             gamma_ark1, t, cores)
    gamma_ark = gamma_ark1 // 2

  #We have gamma_ark which is upper the threshold and gamma_ark1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma_ark == gamma_ark1) and not (gamma_ark == (gamma_ark1 - 1))) :
    gamma_ark2 = (gamma_ark + gamma_ark1) // 2
    eps2 = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                           gamma_ark2, t, cores)
    if (np.abs(log(eps2, 2) - log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma_ark1 = gamma_ark2
    else :
      eps = eps2
      gamma_ark = gamma_ark2
    
  if (gamma_ark == gamma_ark1) :
    return eps, gamma_ark

  if (gamma_ark == (gamma_ark1 - 1)):
    return eps1, gamma_ark1
  
def optimize_gamma_mc (n, p, eps_witness, thr, gamma_ark, cores) :
  """
  Finds an “optimal” value of gamma_mc for the compute_RPC_AES function.  
  This gamma_mc parameter is used only in the MixColumns part of AES, i.e.,
  in the linear gadget implementing MixColumns.

  Idea
  ----
  We search for a value of gamma_mc such that the difference between log2(eps)
  (returned by compute_RPC_AES) and log2(eps_witness) remains below the
  threshold `thr`.

  The search strategy is the same than in the previous function.

  Args:
    n : Number of shares.
    p : Leakage probability.
    eps_witness : Target RPC security value to approximate.
    thr : Allowed threshold on |log2(eps) - log2(eps_witness)|.
    gamma_ark : Fixed gamma parameter used in the AddRoundKey part of AES
                (kept constant during this optimization).
    cores : Number of cores used by compute_RPC_AES (parallelization).

  Returns:
    eps1 : The eps value obtained for the chosen gamma_mc1
    gamma_mc1 : The smallest gamma_mc found such that the log-difference
                with eps_witness remains below the threshold `thr`.
"""

  logp = int(log(p, 2))
  gamma_sb = 500
  l_gamma_sb = [500] * n
  t = n // 2

  #We start with 20.
  gamma_mc = 20

  eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, gamma_ark, 
                        t, cores)
  gamma_mc1 = gamma_mc
  eps1 = eps

  if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
    while (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr and not (gamma_mc == 0)) : 
      eps1 = eps
      gamma_mc1 = gamma_mc
      gamma_mc = gamma_mc1 // 2
      eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    if (gamma_mc == 0) :
      if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
        return eps, gamma_mc
      else :
        return eps1, gamma_mc + 1
    
  else :
    while (np.abs(log(eps1, 2) - log(eps_witness, 2)) >= thr) :
      eps = eps1
      gamma_mc1 = 2 * gamma_mc1
      eps1 = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc1, 
                            gamma_ark, t, cores)
    gamma_mc = gamma_mc1 // 2

  #We have gamma_mc which is upper the threshold and gamma_mc1 which is under 
  #the threshold, we apply dichotomy.

  while (not (gamma_mc == gamma_mc1) and not (gamma_mc == (gamma_mc1 - 1))) :
    gamma_mc2 = (gamma_mc + gamma_mc1) // 2
    eps2 = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc2, 
                           gamma_ark, t, cores)
    if (np.abs(log(eps2, 2) - log(eps_witness, 2))< thr) :
      gamma_mc1 = gamma_mc2
      eps1 = eps2
    else :
      gamma_mc = gamma_mc2
      eps = eps2
    
  if (gamma_mc == gamma_mc1) :
    return eps, gamma_mc

  if (gamma_mc == (gamma_mc1 - 1)):
    return eps1, gamma_mc1  
   
def optimize_gamma_sb (n, p, eps_witness, thr, gamma_ark, gamma_mc, cores) :
  """
  Finds an “optimal” value of gamma_sb for the compute_RPC_AES function.  
  This gamma_sb parameter is used in the SubBytes part of AES, i.e., in the
  gadget implementing the S-box except the MatMult part of the multiplication 
  gadget.

  Idea
  ----
  We search for a value of gamma_sb such that the difference between log2(eps)
  (returned by compute_RPC_AES) and log2(eps_witness) remains below the
  threshold `thr`.

  The search strategy is the same than in the previous function.
  
  Args:
    n : Number of shares.
    p : Leakage probability.
    eps_witness : Target RPC security value to approximate.
    thr : Allowed threshold on |log2(eps) - log2(eps_witness)|.
    gamma_ark : Fixed gamma parameter used in the AddRoundKey part of AES
                (kept constant during this optimization).
    gamma_mc : Fixed gamma parameter used in the MixColumns part of AES
               (kept constant during this optimization).
    cores : Number of cores used by compute_RPC_AES (parallelization).

  Returns:
    eps1 : The eps value obtained for the chosen gamma_sb1.
    gamma_sb1 : The smallest gamma_sb found such that the log-difference
                with eps_witness remains below the threshold `thr`.
"""
  logp = int(log(p, 2))
  l_gamma_sb = [500] * n
  t = n // 2

  #We start with 10.
  gamma_sb = 10

  eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, gamma_ark, 
                        t, cores)
  gamma_sb1 = gamma_sb
  eps1 = eps


  if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
    while (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr and not (gamma_sb == 0)) : 
      gamma_sb1 = gamma_sb
      eps1 = eps
      gamma_sb = gamma_sb1 // 2
      eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    if (gamma_sb == 0) :
      if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
        return eps, gamma_sb
      else :
        return eps1, gamma_sb + 1
    
  else :
    while (np.abs(log(eps1, 2) - log(eps_witness, 2)) >= thr) :
      eps = eps1
      gamma_sb1 = 2 * gamma_sb1
      eps1 = compute_RPC_AES(n, p, logp, gamma_sb1, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    gamma_sb = gamma_sb1 // 2

  #We have gamma_sb which is upper the threshold and gamma_sb1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma_sb == gamma_sb1) and not (gamma_sb == (gamma_sb1 - 1))) :
    gamma_sb2 = (gamma_sb + gamma_sb1) // 2
    eps2 = compute_RPC_AES(n, p, logp, gamma_sb2, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    if (np.abs(log(eps2, 2) - log(eps_witness, 2))< thr) :
      gamma_sb1 = gamma_sb2
      eps1 = eps2
    else :
      gamma_sb = gamma_sb2
      eps = eps2
    
  if (gamma_sb == gamma_sb1) :
    return eps, gamma_sb

  if (gamma_sb == (gamma_sb1 - 1)):
    eps = compute_RPC_AES(n, p, logp, gamma_sb1, l_gamma_sb, gamma_mc, 
                          gamma_ark, t, cores)
    return eps1, gamma_sb1

def optimize_gamma_l_sb (n, p, eps_witness, thr, gamma_ark, gamma_mc, gamma_sb, 
                         cores) :
  """
  Finds an “optimal” value of l_gamma_sb for the compute_RPC_AES function.  
  This l_gamma_sb parameter is used in the SubBytes part of AES, in the MatMult 
  part of the multiplication gadget.
  
  Args:
    n : Number of shares.
    p : Leakage probability.
    eps_witness : Target RPC security value to approximate.
    thr : Allowed threshold on |log2(eps) - log2(eps_witness)|.
    gamma_ark : Fixed gamma parameter used in the AddRoundKey part of AES
                (kept constant during this optimization).
    gamma_mc : Fixed gamma parameter used in the MixColumns part of AES
               (kept constant during this optimization).
    gamma_mc : Fixed gamma parameter used in the SubBytes part of AES except the
               MatMult part of the multiplication gadget. 
               (kept constant during this optimization).
    cores : Number of cores used by compute_RPC_AES (parallelization).

  Returns:
    eps : The eps value obtained for the chosen l_gamma_sb.
    l_gamma_sb : The smallest l_gamma_sb found such that the log-difference
                with eps_witness remains below the threshold `thr`.
"""


  nh = n // 2 + (n % 2)
  logp = int(log(p, 2))
  l_gamma_sb = [500] * (nh + 1)
  l_gamma_sb[0] = 0
  l_gamma_sb[1] = 0
  t = n // 2

  for i in range (2, nh + 1) :
    l_gamma_sb[i] = 0
    eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, gamma_ark, 
                          t, cores)
    while (np.abs(log(eps, 2) - log(eps_witness, 2)) >= thr) : 
      l_gamma_sb[i] += 5
      eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                            gamma_ark, t, cores)
    if (l_gamma_sb[i] == 0) :
      continue

    l_gamma_sb1 = np.copy(l_gamma_sb)
    l_gamma_sb[i] = l_gamma_sb1[i] - 5

    while (not (l_gamma_sb[i] == l_gamma_sb1[i]) and (not (l_gamma_sb[i] == (l_gamma_sb1[i] - 1)))) :
      l_gamma_sb2 = l_gamma_sb
      l_gamma_sb2[i] = (l_gamma_sb[i] + l_gamma_sb1[i]) // 2
      eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb2, gamma_mc, 
                            gamma_ark, t, cores)
      if (np.abs(log(eps, 2) - log(eps_witness, 2)) < thr) :
        l_gamma_sb1[i] = l_gamma_sb2[i]
      else :
        l_gamma_sb[i] = l_gamma_sb2[i]

    if (l_gamma_sb[i] == (l_gamma_sb1[i] - 1)):
      l_gamma_sb[i] += 1

  
  return eps, l_gamma_sb

if __name__ == "__main__" :
  print("main")


  


  
