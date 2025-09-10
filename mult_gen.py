#!/usr/bin/env python3

import numpy as np
import os
import glob

import matplotlib.pyplot as plt
from scipy.special import comb

from partitions import cardinal_rpc_refresh_envelope, cardinal_rpc_add_envelope, cardinal_rpc_gcmult_envelope_pgref, cardinal_rpc_gcopy_envelope_pgref

from multiprocessing import Pool, Process, Queue
from math import log



#********************** MatMult Enveloppes **********************

def precomp_hypergeom(nx, ny) :
  """
  Precompute the hypergeometric distribution for ⌊nx / 2⌋, ⌈nx / 2⌉,
                                                                                                  ⌊ny / 2⌋, ⌈ny / 2⌉.
                                                 
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :return: Array containing the four hypergeometric distribution.
  """
  
  nxl = int(nx / 2)
  nyl = int(ny / 2)
  nxr = nx - nxl
  nyr = ny - nyl
  m = max(nxr, nyr)
  
  nb_pop = [nxl, nxr, nyl, nyr]
  precomp = np.zeros((4, m + 1, m + 1, m + 1))
  for l in range (len(nb_pop)) :
    n = nb_pop[l]
    for i11 in range (n + 1) :
      for l1 in range (i11 + 1) :
        comb1 = comb(i11, l1)  
        for i12 in range (n + 1) :
          comb2 = comb(n - i11 , i12 - l1)
          denom_comb = comb (n, i12)
          if denom_comb :
            precomp[l, i11, i12, l1] = (comb1 * comb2) / denom_comb
  
  return precomp


def precompute_single_proba_sym_case (nx, ny, env_MM, env_ref_nx, env_ref_ny):       
  
  """
  Compute the probability of the following circuit :
  
  ```
  x ---|Ref(nx)|--- 
                   |
                    -------|UnifMatMult(nx,ny)|-------> 
                   | 
  y ---|Ref(ny)|--- 
  ```
  
  knowing the envelopes of the refresh gadget as well as the envelope of 
  UnifMatMult. This circuit corresponds to one of the four branch of a bigger 
  MatMult.
  
  
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param env_MM: Cardinal-RPC envelopes of UnifMatMult instantiated with nx, ny.
  :param env_ref_nx: Cardinal-RPC envelopes of RPRefresh instantiated with nx.
  :param env_ref_ny: Cardinal-RPC envelopes of RPRefresh instantiated with ny.
  :return: The Cardinal-RPC envelopes of the circuit described above. It is an
           array 3D 'case[j, ix, iy]' where :
            - j stands for the cardinality of the output leaked.
            - ix stands for the cardinality of the input shares of x leaked.
            - iy stands for the cardinality of the input shares of y leaked.  
  """
   
  lim_card = nx * ny + 1
  lim_x = nx + 1
  lim_y = ny + 1

  case = np.zeros((lim_card, lim_x, lim_y))
  
  #No refresh to x and y because only one share.
  if (nx == 1 and ny == 1) :
    for j in range (lim_card):
      for ix in range (lim_x):
        for iy in range (lim_y) :
          case [j, ix, iy] += env_MM[j,ix,iy]
  
  #No refresh to x because only one share.
  elif (nx == 1) :
    for j in range (lim_card):
      for ix in range (lim_x):
        for iy in range (lim_y) :
          for l1 in range (lim_y) :
            case [j, ix, iy] += (env_ref_ny[iy, l1] * env_MM[j, ix, l1])
  
  #No refresh to y because only one share.
  elif (ny == 1) :
    for j in range (lim_card):
      for ix in range (lim_x):
        for iy in range (lim_y) :
          for l in range (lim_x) :
              case [j, ix, iy] += (env_ref_nx[ix, l] * env_MM[j,l,iy])  
  
  else :
    for j in range (lim_card):
      for ix in range (lim_x):
        for iy in range (lim_y) :
          for l in range (lim_x) :
            for l1 in range (lim_y) :
              case [j, ix, iy] += (env_ref_nx[ix, l] * env_ref_ny[iy, l1] * 
                                   env_MM[j,l,l1])  
 
  return case          

def precomp1 (nx, ny, single_proba_three, single_proba_four, j3, j4, hypergeom, queue) :
  """
  Compute : 
    Σ_{l4=0}^{end} 	Σ_{i14=l4}^{end} ε_{j3}(*1, i14).ε_{j4}(*2, end + l4 - i14).
                                     H(⌈ny / 2⌉  , i14, end + l4 - i14, l4)
                                     
  For end in range [0, ⌈ny / 2⌉] and *1, *2 in range [0, ⌊nx / 2⌋] and [0, ⌈nx / 2⌉]
  
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param single_proba_three: Cardinal-RPC envelopes used for ε_{j3}. 
  :param single_proba_four: Cardinal-RPC envelopes used for ε_{j4}.   
  :param j3: Cardinal of the leaked output shares of the third branch of
                  UnifMatMult (See Figure TODO).
  :param j4: Cardinal of the leaked output shares of the fourth branch of
                  UnifMatMult (See Figure TODO).
  :param hypergeom: The hypergeometric distribution.
  :param queue: A queue to stack results (use of parallelisation).
  :return: An array giving the results of the sum for all end, *1, *2.                 
  """
  
  
  nxl = int (nx / 2)
  nyl = int (ny / 2)
  nxr = nx - nxl
  nyr = ny - nyl
  
  lim_card = nxl * nyr + nxr * nyr
  prec1 = np.zeros((lim_card + 1, nyr + 1, nxl + 1, nxr + 1))
  j34 = j3 + j4
  for end in range (nyr + 1) :
    for l4 in range (end + 1) :
      for i14 in range(l4, end + 1) :
        for i21 in range (nxl + 1) :
          for i22 in range (nxr + 1) :
            prec1[j34, end, i21, i22] += single_proba_three[j3, i21, i14] * single_proba_four[j4, i22, end + l4 - i14] * hypergeom[3, i14, end + l4 - i14, l4]
  
  queue.put((j3, j4, prec1))            
  return

def precomp2 (nx, ny, single_proba_one, single_proba_two, j1, j2, hypergeom, queue) :
  """
  Compute : 
    Σ_{l3=0}^{end} 	Σ_{i13=l3}^{end} ε_{j1}(*1, i13).ε_{j2}(*2, end + l3 - i13).
                                     H(⌊ny / 2⌋  , i13, end + l3 - i13, l3)
                                     
  For end in range [0, ⌊ny / 2⌋ ] and *1, *2 in range [0, ⌊nx / 2⌋] and [0, ⌈nx / 2⌉]
  
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param single_proba_one: Cardinal-RPC envelopes used for ε_{j1}. 
  :param single_proba_two: Cardinal-RPC envelopes used for ε_{j2}.   
  :param j1: Cardinal of the leaked output shares of the first branch of
                  UnifMatMult (See Figure TODO).
  :param j2: Cardinal of the leaked output shares of the second branch of
                  UnifMatMult (See Figure TODO).
  :param hypergeom: The hypergeometric distribution.
  :param queue: A queue to stack results (use of parallelisation).
  :return: An array giving the results of the sum for all end, *1, *2.                 
  """
  
  
  nxl = int (nx / 2)
  nyl = int (ny / 2)
  nxr = nx - nxl
  nyr = ny - nyl
  
  lim_card = nxl * nyl + nxr * nyl
  prec2 = np.zeros((lim_card + 1, nyl + 1, nxl + 1, nxr + 1))
  j12 = j1 + j2
  for end in range (nyl + 1) :
    for l3 in range (end + 1) :
      for i13 in range (l3, end + 1) :          
        for i12 in range (nxr + 1) :
          for i11 in range (nxl + 1) :
            prec2[j12, end, i11, i12] += single_proba_one[j1, i11, i13] * single_proba_two[j2, i12, end + l3 - i13] * hypergeom[2, i13, end + l3 - i13, l3]
  
  queue.put((j1, j2, prec2))            
  return
 
def initial_case_MM (p) :
  """
  Compute the cardinal-RPC envelopes of UnifMatMult when nx = ny = 1
  (Base case n°1), see Proof of Lemma TODO in Appendix TODO for details.
  
  :param p: The leakage probability.
  :return: The cardinal-RPC envelopes of UnifMatMult with nx = ny = 1.It is an
           array 3D 'case[j, ix, iy]' where :
            - j stands for the cardinality of the output leaked.
            - ix stands for the cardinality of the input shares of x leaked.
            - iy stands for the cardinality of the input shares of y leaked.  
  """
  
  case = np.zeros((2, 2, 2))
  case[0, 0, 0] = (1 - p)**2
  case[0, 0, 1] = p * (1 - p)
  case[0, 1, 0] = p * (1 - p)
  case[0, 1, 1] = p**2
  case[1, 0, 0] = 0
  case[1, 0, 1] = 0
  case[1, 1, 0] = 0
  case[1, 1, 1] = 1
  
  return case

def special_case_MM_nx (p) :
  """
  Compute the cardinal-RPC envelopes of UnifMatMult when nx = 2 and ny = 1
  (Base case n°2), see Proof of Lemma TODO in Appendix TODO for details.
  
  :param p: The leakage probability.
  :return: The cardinal-RPC envelopes of UnifMatMult with nx = 2 and  ny = 1.
           It is an array 3D 'case[j, ix, iy]' where :
            - j stands for the cardinality of the output leaked.
            - ix stands for the cardinality of the input shares of x leaked.
            - iy stands for the cardinality of the input shares of y leaked.  
  """
  
  
  case = np.zeros((3, 3, 2))
  case[0, 0, 0] = (1 - p)**3 
  case[0, 0, 1] = p * (1 - p)**2
  case[0, 1, 0] = 2 * p * (1 - p)**2
  case[0, 1, 1] = 2 * p**2 * (1 - p)
  case[0, 2, 0] = p**2 * (1 - p)
  case[0, 2, 1] = p**3
  
  case[1, 0, 0] = 0
  case[1, 0, 1] = 0
  case[1, 1, 0] = 0
  case[1, 1, 1] = (1 - p)
  case[1, 2, 0] = 0
  case[1, 2, 1] = p
  
  case[2, 0, 0] = 0
  case[2, 0, 1] = 0
  case[2, 1, 0] = 0
  case[2, 1, 1] = 0
  case[2, 2, 0] = 0
  case[2, 2, 1] = 1
  
  return case
 
def special_case_MM_ny (p) :
  """
  Compute the cardinal-RPC envelopes of UnifMatMult when nx = 1 and ny = 2
  (Base case n°3), see Proof of Lemma TODO in Appendix TODO for details.
  
  :param p: The leakage probability.
  :return: The cardinal-RPC envelopes of UnifMatMult with nx = 1 and  ny = 2.
           It is an array 3D 'case[j, ix, iy]' where :
            - j stands for the cardinality of the output leaked.
            - ix stands for the cardinality of the input shares of x leaked.
            - iy stands for the cardinality of the input shares of y leaked.  
  """

  case = np.zeros((3, 2, 3))
  case[0, 0, 0] = (1 - p)**3 
  case[0, 0, 1] = 2 * p * (1 - p)**2
  case[0, 0, 2] = p**2 * (1 - p)
  case[0, 1, 0] = p * (1 - p)**2
  case[0, 1, 1] = 2 * p**2 * (1 - p)
  case[0, 1, 2] = p**3

  case[1, 0, 0] = 0
  case[1, 0, 1] = 0
  case[1, 0, 2] = 0
  case[1, 1, 0] = 0
  case[1, 1, 1] = (1 - p)
  case[1, 1, 2] = p

  
  case[2, 0, 0] = 0
  case[2, 0, 1] = 0
  case[2, 0, 2] = 0
  case[2, 1, 0] = 0
  case[2, 1, 1] = 0
  case[2, 1, 2] = 1
  
  return case     
 
def compute_proba(nx, ny, j, ix, iy, prec1, prec2, hypergeom) :
  """
  Compute the overall envelopes ε_{j}(ix, iy) of UnifMatMult following the 
  formula in Proof of Lemma TODO in Appendix TODO.
  
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param j: Cardinality of output shares leaked.
  :param ix: Cardinality of input shares from x leaked.
  :param iy: Cardinality of input shares from y leaked.
  :param prec1: Outputs of the function 'precomp1', it is an array of the two 
                last sum in the Formula.
  :param prec2: Outputs of the function 'precomp2', it is an array of the third 
                and fourth last sum in the Formula.
  :param hypergeom: Array containing the hypergeometric distribution.
  :return: ε_{j}(ix, iy)
  """
  nxl = int (nx / 2)
  nyl = int (ny / 2)
  nxr = nx - nxl
  nyr = ny - nyl
 
  lim_card12 = nxl * nyl + nxr * nyl
  lim_card34 = nxl * nyr + nxr * nyr
  
  ixt = min(ix, nxl)
  kx = max (0, ix - nxr)
  iyt = min (iy, nyl)
  ky = max (0, iy - nyr)
 
  pr = 0
  for j12 in range (max(0, j - lim_card34), min(j , lim_card12) + 1) :
    tmp = 0
    j34 = j - j12    
    for i1 in range (kx, ixt + 1) :
      for l1 in range (i1 + 1) :
        for i11 in range (l1, i1 + 1) :              
          sub_pr = 0
          for l2 in range (ix - i1 + 1) :
            for i12 in range (l2, ix - i1 + 1) :
              for i3 in range (ky, iyt + 1) : 
                sub_pr += prec1[j34, iy - i3, i1 + l1 - i11, ix - i1 + l2 - i12] * prec2[j12, i3, i11, i12]  * hypergeom [1, i12, ix - i1 + l2 - i12, l2]          
          sub_pr *= hypergeom[0, i11, i1 + l1 - i11, l1]
          pr += sub_pr 
  return pr


#TODO : J'ai ajouté ça, voir ce que ça donne avec .
def final_induction_enveloppes(nx, ny, p, start_env) : 
  nxh = nx // 2
  nyh = ny // 2
  lim_card = nx * ny + 1
  env = np.zeros((lim_card, nx + 1, ny + 1))
  for j in range(lim_card) :
    for ix in range (nx + 1) :
      for iy in range (ny + 1) :
        for lx in range (ix + 1) :
          for ly in range (iy + 1) :
            #comb1 = comb(ix, lx)
            #comb2 = comb(iy, ly)
            comb1 = comb(nx - ix + lx, lx)
            comb2 = comb(ny - iy + ly, ly)
            env[j, ix, iy] += comb1 * comb2 * p**(lx + ly) * (1 - p)**(nx + ny - ix - iy) * start_env[j, ix - lx, iy - ly]
        env[j, ix, iy] = min (1,  env[j, ix, iy])
  return env

def induction_envelopes(nx, ny, prec1, prec2, ix, iy, hypergeom, queue) :
  """
  Compute the cardinal-RPC envelopes  ε_{j}(ix, iy) of UnifMatMult for all j in
  range [0, nx.ny].
  
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param prec1: Outputs of the function 'precomp1'.
  :param prec2: Outputs of the function 'precomp2'.
  :param ix: Cardinality of input shares from x leaked.
  :param iy: Cardinality of input shares from y leaked.
  :param hypergeom: Array containing the hypergeometric distribution.
  :param queue: A queue to stack results (use of parallelisation).
  :return : ε_{j}(ix, iy) for all j in range [0, nx.ny]
  """ 
  lim_card = nx * ny + 1
  env = np.zeros((lim_card, nx + 1, ny + 1))
  card_part = compute_partition_J(nx, ny)
  
  for j in range (lim_card) :      
    pr_ix_iy = compute_proba(nx, ny, j, ix, iy, prec1, prec2, hypergeom)
    env[j, ix, iy] += min(pr_ix_iy / card_part[j], 1)

  queue.put((env, ix, iy))
  return  
  
def compute_partition_J(nx, ny) :
  """
  For all j in range [0, nx.ny], compute the number of 4-uplet (j1, j2, j3, j4)
  such that j1 + j2 + j3 + j4 = j. We have the constraints that :
    - 0 <= j1 <=  ⌊nx / 2⌋ . ⌊ny / 2⌋
    - 0 <= j2 <=  ⌈nx / 2⌉ . ⌊ny / 2⌋
    - 0 <= j3 <=  ⌊nx / 2⌋ . ⌈ny / 2⌉ 
    - 0 <= j4 <=  ⌈nx / 2⌉ . ⌈ny / 2⌉
    
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :return : The number of 4-uplet (j1, j2, j3, j4) for all j in range [0, nx.ny] 
  """
  nxl = int(nx /2)
  nxr = nx - nxl
  nyl = int(ny / 2)
  nyr = ny - nyl
  
  card_part = []
  for j in range (nx * ny + 1) :
    cpt = 0
    for j1 in range (nxl * nyl + 1) :
      for j2 in range (nxr * nyl + 1) :
        for j3 in range (nxl * nyr + 1) :
          for j4 in range (nxr * nyr + 1) :
            if (j1 + j2 + j3 + j4 == j) :
              cpt += 1
    card_part.append(cpt)
  
  return card_part 
  
#Compute the probability envelopes for MatMult.
def compute_envn_MM (nx, ny, p, gamma_l) :
  """
  Compute the cardinal-RPC envelopes of UnifMatMult using the formula in Proof
  of Lemma TODO in Appendix TODO.
  Optimisation : We use a list of gamma (for the envelopes of refresh gadget 
                 RPRefresh) according to the number of shares currently used in 
                 the induction step. In this way, we lower the number of gamma 
                 used when the number of shares decreased. This upgrade the 
                 trade-off security/complexity.
                 
  :param nx: Number of shares for the secret x.
  :param ny: Number of shares for the secret y.
  :param p: The leakage rate in [0, 1].
  :param gamma_l: The list of gamma used in RPRefresh according to the number of
                  shares(optimisation).
  :return: The cardinal-RPC envelopes of UnifMatMult used with nx input shares 
           for x and ny input shares for ny.
  """  
  if (os.path.isfile(str(nx) + "_" + str(ny) + ".npy") ) :
      env = np.load(str(nx) + "_" + str(ny) +".npy")
      return env
  
  if (nx == 1 and ny == 1) :
    env = initial_case_MM(p)
    np.save(str(nx) + "_" + str(ny) + ".npy", env)
    return env
  
  
  if (nx == 1) :
    return special_case_MM_ny (p)
    
  if (ny == 1) :
    return special_case_MM_nx (p)
    
  env_MM = np.zeros((nx * ny + 1, nx + 1, ny + 1))
  processes = []
  queue = Queue()
    
  nxl = int (nx / 2)
  nyl = int (ny / 2)
  nxr = nx - nxl
  nyr = ny - nyl
  lim_card1 = nxl * nyl
  lim_card2 = nxr * nyl
  lim_card3 = nxl * nyr 
  lim_card4 = nxr * nyr
  
  pgrefxl = cardinal_rpc_refresh_envelope(nxl, p, gamma_l[nxl])
  pgrefyl = cardinal_rpc_refresh_envelope(nyl, p, gamma_l[nyl])  
  pgrefxr = cardinal_rpc_refresh_envelope(nxr, p, gamma_l[nxr])
  pgrefyr = cardinal_rpc_refresh_envelope(nyr, p, gamma_l[nyr])      
    
  env_LL = compute_envn_MM (nxl, nyl, p, gamma_l)
  prec_LL = precompute_single_proba_sym_case(nxl, nyl, env_LL, pgrefxl, 
                                               pgrefyl)
    
  env_LR = compute_envn_MM (nxr, nyl, p, gamma_l)
  prec_LR = precompute_single_proba_sym_case(nxr, nyl, env_LR, pgrefxr, 
                                               pgrefyl)
    
  env_RL = compute_envn_MM (nxl, nyr, p, gamma_l)
  prec_RL = precompute_single_proba_sym_case(nxl, nyr, env_RL, pgrefxl, 
                                                 pgrefyr)
    
  env_RR = compute_envn_MM (nxr, nyr, p, gamma_l)
  prec_RR = precompute_single_proba_sym_case(nxr, nyr, env_RR, pgrefxr, 
                                               pgrefyr)
    
  prec1 = np.zeros((lim_card3 + lim_card4 + 1, nyr + 1, nxl + 1, nxr + 1))
  prec2 = np.zeros((lim_card1 + lim_card2 + 1, nyl + 1, nxl + 1, nxr + 1))
  hypergeom = precomp_hypergeom(nx, ny)
    
  #Precomputation 1 : 
    
  for j3 in range (lim_card3 + 1) :
    for j4 in range (lim_card4 + 1) :
      processes.append(Process(target=precomp1, 
                               args=(nx, ny, prec_RL, prec_RR, j3, j4, 
                                     hypergeom, queue)))
  cores = 2 * 192
  nb_process = 0
  while (nb_process < len(processes)) :
    lim = min(cores + nb_process, len(processes))
    for i in range (nb_process, lim) :
      pro = processes[i]
      pro.daemon = True
      pro.start()
      
    for _ in range (nb_process, lim) :
      (j3, j4, prec1_para) = queue.get()
      prec1[j3 + j4] += prec1_para[j3 + j4]        
          
    for i in range (nb_process, lim) : 
      processes[i].join()
      processes[i].terminate()
      processes[i].close()
      nb_process += 1
    
  #End of Precomputation 1.
  #Precomputation 2 :
  processes = []
  for j1 in range (lim_card1 + 1) :
    for j2 in range (lim_card2 + 1) :
      processes.append(Process(target=precomp2, 
                               args=(nx, ny, prec_LL, prec_LR, j1, j2, 
                                     hypergeom, queue)))
      
  cores = 2 * 192
  nb_process = 0
  while (nb_process < len(processes)) :
    lim = min(cores + nb_process, len(processes))
    for i in range (nb_process, lim) :
      pro = processes[i]
      pro.daemon = True
      pro.start()
        
        
    for _ in range (nb_process, lim) :
      (j1, j2, prec2_para) = queue.get()
      prec2[j1 +  j2] += prec2_para[j1 + j2]
      
    for i in range (nb_process, lim) : 
      processes[i].join()
      processes[i].terminate()
      processes[i].close()
      nb_process += 1
  #End of Precomputation2  
  #Computation
  processes = []
  for ix in range (nx + 1) :
    for iy in range (ny + 1) :    
      processes.append(Process(target=induction_envelopes, 
                               args=(nx, ny, prec1, prec2, ix, iy, hypergeom, 
                                     queue)))
    
  for pro in processes :
    pro.daemon = True
    pro.start()
      
  for pro in processes : 
    (pro_env_MM, ix, iy) = queue.get()
    for j in range (nx * ny + 1) :  
      env_MM[j,ix,iy] = min(1, pro_env_MM[j,ix,iy])  
            
  for pro in processes : 
    pro.join()

  env_MM = final_induction_enveloppes(nx, ny, p, env_MM)

  np.save(str(nx) + "_" + str(ny) + ".npy", env_MM)         
  return env_MM




#********************** TreeAdd Enveloppes **********************

#Base case for TreeAdd.
def initial_case_TA(n, p, envgadd) :
  """
  Compute the cardinal-RPC envelopes of the base case of BasicTreeAdd. 
  It is the circuit :
  
  ```
  |n-sharing|---
                \\
                 + --->  
                /
  |n-sharing|---
  ``` 
  
  We keep in mind that the output of BasicTreeAdd is a n.n sharing (in this 
  case a 2.n-sharing), this is why we have to split the 2.n inputs shares in two 
  parts (i1 and i - i1 in the function) during the computation.
  
  :param n: The number of shares.
  :param p: The leakage rate in [0,1].
  :param envgadd: The cardinal-RPC envelopes of the addition gadget.
  :return: The cardinal-RPC envelopes of the circuit above, where the two 
           n-sharing are seen as a unique 2.n-sharing. 
  """
  envn = np.zeros((2 * n + 1, n + 1))
  for i in range (2 * n + 1) :
    it = min (n , i)
    k = max (0, i - n) 
    for j in range (n + 1) :
      for i1 in range (k, it + 1) :
        envn[i,j] += envgadd[i1, i - i1, j]
      envn[i,j] = min(1, envn[i,j])
        
  return envn


"""
Non parallelized version.
def env_fin_TA (n, ell, env_TA1, env_TA2, envgadd) :
  ell_rec = ell // 2
  env_TA = np.zeros((ell * n + 1, n + 1))
  for tin in range (ell * n + 1) :
    for tout in range (n + 1) :
      for tin1 in range (max(0, tin - (ell - ell_rec) * n), min(ell_rec * n, tin) + 1) :
        for i1 in range (n + 1) :
          for i2 in range (n + 1) :
            env_TA[tin, tout] += env_TA1[tin1, i1] * env_TA2[tin - tin1, i2] * envgadd[i1, i2, tout]
      env_TA[tin, tout] = min(1, env_TA[tin, tout])
  
  return env_TA 
"""
def env_fin_TA (n, ell, env_TA1, env_TA2, envgadd, tin, queue) :
  """
  Compute the cardinal-RPC envelopes of the circuit :
  
  ```
  |(⌊ell/2⌋.n)-sharing|--- |BasicTreeAdd|
                                        \\
                                         + --->  
                                        /
  |(⌈ell/2⌉.n)-sharing|--- |BasicTreeAdd|
  ```
  
  This circuit is the induction step of BasicTreeAdd.
  :param n: The number of shares.
  :param ell: The number of n-sharing of the output matrix of UnifMatMult used 
              for the induction.
  :param envn_TA1: The cardinal-RPC envelopes of BasicTreeAdd used with an 
                   (⌊ell/2⌋.n)-sharing input.
  :param envn_TA1: The cardinal-RPC envelopes of BasicTreeAdd used with an 
                   (⌈ell/2⌉.n)-sharing input.
  :param envgadd: The cardinal-RPC envelopes of the addition gadget.
  :param tin: Number of input shares leaked among the (ell.n) shares input.
  :param queue: A queue to stack results (use of parallelisation).
  :return: The cardinal-RPC envelopes of the circuit above, where the two 
           sharings in input are seen as a unique (ell.n)-sharing.                
  """
  ell_rec = ell // 2
  env_TA = np.zeros((n + 1))
  for tout in range (n + 1) :
    for tin1 in range (max(0, tin - (ell - ell_rec) * n), min(ell_rec * n, tin) + 1) :
      for i1 in range (n + 1) :
        for i2 in range (n + 1) :
          env_TA[tout] += env_TA1[tin1, i1] * env_TA2[tin - tin1, i2] * envgadd[i1, i2, tout]
    env_TA[tout] = min(1, env_TA[tout])
  
  queue.put((tin, env_TA))
  return



#Compute the probability envelopes of TreeAdd.        
def compute_envn_TA (n, ell, p, gamma) :
  """
  Compute the cardinal-RPC envelopes of BasicTreeAdd used with a 
  (ell.n)-sharing inputs.
  
  :param n: The number of shares.
  :param ell: The number of n-sharing of the output matrix of UnifMatMult used 
              for the induction.
  :param p: The leakage rate in [0, 1].
  :param gamma: The gamma used for the instantiation of RPRefresh gadget.
  :return: The cardinal-RPC envelopes of BasicTreeAdd.
  
  """
  if (ell == 2) :  
    pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
    envgadd = cardinal_rpc_add_envelope(n, p, pgref)
    env_TA = initial_case_TA(n, p, envgadd)
    return env_TA
    
  if (ell == 3) :
    pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
    envgadd = cardinal_rpc_add_envelope(n, p, pgref)
    env_add = initial_case_TA(n, p, envgadd)
    env_TA = np.zeros((3 * n + 1, n + 1))
    for tin in range (3 * n + 1) :
      for tout in range (n + 1) :
        for tin1 in range (max(0, tin - n), min(2 * n, tin) + 1) :
            for i in range (n + 1) : 
              env_TA[tin, tout] += (env_add[tin1, i] * envgadd[i, tin - tin1, tout])
        env_TA[tin, tout] = min(1, env_TA[tin, tout])
    return env_TA 
   
  ell_rec = ell // 2
   
  env_TA1 = compute_envn_TA(n, ell_rec, p, gamma)
  env_TA2 = compute_envn_TA(n, ell - ell_rec, p, gamma)  
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  envgadd = cardinal_rpc_add_envelope(n, p, pgref)

   
  env_TA = np.zeros((ell * n + 1, n + 1))
  processes = []
  queue = Queue()
  
  for tin in range (ell * n + 1) :
    processes.append(Process(target=env_fin_TA, 
                               args=(n, ell, env_TA1, env_TA2, envgadd, tin, 
                                     queue)))
  
  cores = 2 * 192
  nb_process = 0
  while (nb_process < len(processes)) :
    lim = min(cores + nb_process, len(processes))
    for i in range (nb_process, lim) :
      pro = processes[i]
      pro.daemon = True
      pro.start()
        
        
    for _ in range (nb_process, lim) :
      (tin , env_TA_tin) = queue.get()
      env_TA[tin] = env_TA_tin
      
    for i in range (nb_process, lim) : 
      processes[i].join()
      processes[i].terminate()
      processes[i].close()
      nb_process += 1
 


  #env_TA = env_fin_TA (n, ell, env_TA1, env_TA2, envgadd)
   
  return env_TA


#********************** CardSecMult envelopes **********************
def compute_envn (n, envn_MM, envn_TA) :
  """
  Compute the Cardinal-RPC envelopes of CardSecMult from the envelopes of 
  UnifMatMult and BasicTreeAdd thanks to Lemma TODO.
  
  :param n: The number of shares.
  :param envn_MM: The cardinal-RPC envelopes of UnifMatMult.
  :param envn_TA: The cardinal-RPC envelopes of BasicTreeAdd.
  :return: The cardinal-RPC envelopes of CardSecMult.
  """
  envn = np.zeros((n + 1, n + 1, n + 1))
  for j in range (n + 1) :
    for ix in range (n + 1) :
      for iy in range (n + 1) :
        pr = 0
        for l in range (n**2 + 1) :
          pr += envn_MM[l, ix, iy] * envn_TA[l,j]    
        
        envn[ix,iy,j] = min(pr, 1)
  return envn
    
#********************** RPC-threshold from cardinal-RPC **********************
def compute_RPC_threshold (n, envn, t) :
  """
  Compute the RPC-threshold security from the cardinal-RPC envelopes.
  It consists to compute ,for each output shares leaked cardinality j, the 
  probability that we require at least more than t shares of one secret input.
  
  :param n: The number of shares.
  :param envn: The envelopes of the cardinal-RPC security of the gadget 
               (multiplication gadget here)
  :param t: The threshold t of the threshold-RPC security.
  :return: Advantage ε of the threshold-RPC security. 
  """
  eps = 0
  for j in range (t + 1) :
    smj = 0 
    for ix in range (n + 1) :
      for iy in range (n + 1) :
        if (ix  > t or iy > t) :
          smj += envn[ix,iy,j]
    
    if (eps < smj) :
      eps = smj    
  return eps  

#********************** RPS from cardinal-RPC **********************
def compute_RPM_threshold (n, envn) :
  """
  Compute the RP security from the cardinal-RPC envelopes.
  It consists to compute, when there is no output shares leaked (i.e. j = 0), 
  the probability that we require all the shares of at least one secret input.
  
  :param n: The number of shares.
  :param envn: The envelopes of the cardinal-RPC security of the gadget 
               (multiplication gadget here)
  :return: Advantage ε of the RP security. 
  """
  eps = 0 
  j = 0
  for ix in range (n + 1) :
    eps += envn[ix, n, j]    
    
    
  for iy in range (n) :
    eps += envn[n, iy, j]
     
  return eps 


def compute_envn_mult (n,p, gamma_l, gamma_TA) :
  """
  Compute the Cardinal-RPC envelopes of CardSecMult.
  :param n: The number of shares.
  :param p: The leakage rate in [0,1].
  :param gamma_l: The list of gamma used for the refresh gadget RPRefresh in 
                  UnifMatMult.
  :param gamma_TA: The gamma used for the refresh gadget RPRefresh in 
                   BasicTreeAdd.
  :return: The Cardinal-RPC envelopes of CardSecMult.
  """
  env_MM = compute_envn_MM(n, n, p, gamma_l)
  envn_TA = compute_envn_TA(n, n, p, gamma_TA)
  envn = compute_envn (n, env_MM, envn_TA)
  for file in glob.glob("*.npy"):
    os.remove(file)
  return envn
  
  
#********************** Main **********************

def main() :
  p = 2**-10
  gamma_l = [0, 0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 24, 26, 28, 30, 32, 35, 37, 40]
  n = 10
  #gamma = 19  

  envn =  compute_envn_mult(n, p, gamma_l, gamma_l[n])
  eps_RPM_threshold = compute_RPM_threshold(n, envn)
  print("log_2(eps) = ", log(eps_RPM_threshold, 2))

  
#Subsection 4.3
def compute_graph () :
  #p = 2**-20
  #gamma_l = [0, 0, 2, 8, 12, 16, 19, 23, 27, 30, 34, 37, 40, 44, 48, 54, 55, 61]  
  
  p = 2**-10
  gamma_l = [0, 0, 2, 6, 8, 10, 13, 15, 18, 19, 23, 28, 28, 33, 38, 38, 43, 48, 48, 53]
  
  #p = 2**-15
  #gamma_l = [0, 0, 2, 7, 10, 12, 15, 18, 21, 24, 28, 29, 33, 38, 38, 43, 48, 48, 53]
  eps = []
  eps_JMB24 = []
  eps_BFO23 = []
  n_values = []
  
  for n in range (2, 19) :
    print("n = ", n)
    n_values.append(n)
  
    envn =  compute_envn_mult(n, p, gamma_l, gamma_l[n])
      
    eps_RPM_threshold = compute_RPM_threshold(n, envn)
    print("log_2(eps) = ", log(eps_RPM_threshold, 2))
    print()
    eps.append(eps_RPM_threshold)
    eps_JMB24.append((2 * p)**(0.3 * n)) 
    eps_BFO23.append((1 - (1 - p)**(8*n) + 1 - (1 - (3*p)**(0.5))**(n - 1))**n)
  
  plt.title ("Security (p, ε)-RPS of multiplications,p = 2^"+str(int(log(p,2))))
  plt.xlabel("Number of shares n")
  plt.ylabel("RPM (p, ε)")
  plt.yscale('log', base=2)
  plt.gca().invert_yaxis()
  plt.grid(True)
  
  plt.plot(n_values, eps, color ="red", marker = 'x', linestyle = '', label="CardSecMult-Unif-OPTI")
  plt.plot(n_values, eps_JMB24, color ="purple", marker = 'x', linestyle = '', label="JMB24")
  plt.plot(n_values, eps_BFO23, color ="blue", marker = 'x', linestyle = '', label="BFO23")
  plt.legend()
  plt.savefig("n_graph_p" + str(log(p, 2)) + ".png")
  plt.close()

#Subsection 4.3 
def find_plateau_RPM(n, p, start_nb_iter) :
  nb_iter = start_nb_iter
  eps_log = 0
  thr = 0.05
  
  #while True :
  
  #  l_nb_iter = [nb_iter for _ in range (n + 1)]
  #  l_nb_iter[0] = 0
  #  l_nb_iter[1] = 0
    
    
  #  envn = compute_envn_mult (n,p, l_nb_iter, nb_iter)
  #  eps_RPM_threshold = compute_RPM_threshold(n,envn)
      
  #  if (np.abs(log(eps_RPM_threshold, 2) - eps_log) < thr) :
  #    break
      
  #  eps_log = log(eps_RPM_threshold, 2) 
  #  nb_iter += 5
  
  #nb_iter -= 10
  #nb_iter = max(0, nb_iter)
  while True :
    l_nb_iter = [nb_iter for _ in range (n + 1)]
    l_nb_iter[0] = 0
    l_nb_iter[1] = 0
    
    envn = compute_envn_mult (n,p, l_nb_iter, nb_iter)
    eps_RPM_threshold = compute_RPM_threshold(n,envn)
    
    if (np.abs(log(eps_RPM_threshold, 2) - eps_log) < thr) :
      break
    
    eps_log = log(eps_RPM_threshold, 2) 
    nb_iter += 1 
 
  return nb_iter

#Table 2 - Subsection 4.3
def find_plateau_RPM_n (p_values, lim_n) : 
  gamma_n_p = []
  for i in range (len(p_values)) :
    p = p_values[i]
    path = "./results/Plateau/"
    str_gamma_n = path + "plateau_gamma_n_p" + str(p) + "_limn"+str(lim_n) + ".npy"
    gamma_n = [0, 0]

    if (os.path.isfile(str_gamma_n)):
      gamma_n = np.load(str_gamma_n)

    else :
      gamma = 2
      for n in range (2, lim_n) :
        print("n = ", n)
        gamma = find_plateau_RPM(n, p, max(2, gamma))
        gamma_n.append(gamma)
      
      np.save(str_gamma_n, gamma_n)
    
    gamma_n_p.append(gamma_n)
  return gamma_n_p


    

if __name__ == "__main__":
  print("main")
  main()
  #compute_graph()
  #find_plateau_RPM_n(2**-10)
  #find_plateau_RPM_n(2**-15)
  #find_plateau_RPM_n(2**-20)

