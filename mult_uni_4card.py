#!/usr/bin/env python3


from threading import Thread
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.special import comb

from partitions import cardinal_rpc_refresh_envelope, cardinal_rpc_add_envelope

from multiprocessing import Pool, Process, Queue
from math import log


#********************** MatMult Enveloppes **********************

def precomp_hypergeom(n) :
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





#Input :
# - n : The number of shares of the child nodes in the tree.
# - envn_MM : Array 3D of probability envelopes for the gadget with n shares. 
#          The first dimension is |J|.
#          The second dimension is |I_x|. 
#          The third dimension is |I_y|. 
# - envrefn : Array 3D of probability envelopes for the refresh gadget with n shares. 
#             The first dimension is |J|.
#             The second dimension is |I_x| or |I_y|.
#
# Compute the threshold of P(|J| = j, |Ix| = ix, |Iy| = iy) for all j in [0, 4^{log(n)}] and (ix,iy) in [0,n/2]²
def precompute_single_proba (n, envn_MM, envrefn):   
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
        

#Input :
# - n : The number of shares of the child nodes in the tree.
# - envn_MM : Array 3D of probability envelopes for the gadget with n shares. 
#          The first dimension is |J|.
#          The second dimension is |I_x|. 
#          The third dimension is |I_y|. 
# - envrefn : Array 3D of probability envelopes for the refresh gadget with n shares. 
#             The first dimension is |J|.
#             The second dimension is |I_x| or |I_y|.
#
# Compute the threshold of P(|J| = j, |Ix| = ix, |Iy| = iy) for all j in [0, 4^{log(n)}] and (ix,iy) in [0,n/2]², in the symmetric version of MatMult.
def precompute_single_proba_sym_case (n, envn_MM, envrefn):   
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

   
#Precompute, for all i4 in [0, n], |J3|, |J4|, i12 in [0,n] and i22 in [0,n],
#  the sum for i41 from 0 to i4, with i42 = i4 - i41, 
#  of the threshold of P(i12, i41, |J3|) . P(i22, i42, |J4|).
def precomp1 (n, single_proba_three, single_proba_four, j3, j4, hypergeom, queue) :
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

#Precompute, for all i3 in [0, n], |J1|, |J2|, j,  i11 in [0,n] and i21 in [0,n],
#  the sum for i31 from 0 to i3, with i32 = i3 - i31, 
#  of the threshold of P(i11, i31, |J3|) . P(i21, i32, |J4|). prec1(j, i12, i22).
def precomp2 (n, single_proba_one, single_proba_two, j1, j2, hypergeom, queue) :
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

#Compute the probability envelopes of the cardinal-RPC gadget MatMult 
#with n = 2.
def initial_case_MM (p) :
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

#Compute the envelopes for |Ix| = ix, |Iy| = iy and |J1|, |J2|, |J3|, |J4| for 
#the MatMult gadgets with 2 * n shares. 
def compute_proba(n, j1, j2, j3, j4, ix, iy, prec1, prec2, hypergeom) :
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

#TODO : J'ai ajouté ça, voir ce que ça donne avec .
def final_induction_envelopes(n, p, start_env, ix, iy, queue) : 
  lim_card = n**2
  env = np.zeros((lim_card + 1, lim_card + 1, lim_card + 1, lim_card + 1))
  for j1 in range(lim_card + 1) :
    for j2 in range (lim_card + 1) :
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          for lx in range (ix + 1) :
            for ly in range (iy + 1) :
              #comb1 = comb(ix, lx)
              #comb2 = comb(iy, ly)
              comb1 = comb(n - ix + lx, lx)
              comb2 = comb(n - iy + ly, ly)
              env[j1, j2, j3, j4] += comb1 * comb2 * p**(lx + ly) * (1 - p)**(2 * n - ix - iy) * start_env[ix - lx, iy - ly, j1, j2, j3, j4]
          env[j1, j2, j3, j4] = min (1,  env[j1, j2, j3, j4])

  queue.put((ix, iy, env))
  return 


#Induction formulas to compute the MatMult envelopes.
def induction_envelopes(n, prec1, prec2, ix, iy, hypergeom, queue) : 
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
  
#TODO  
#Compute the number of elements : 
#  {|J1|, |J2|, |J3|, |J4| in [(n/2)²] s.t. |J1| + |J2| + |J3| + |J4| = |J|}
#for all |J| in [n²].
def compute_partition_J(n) :
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

#Compute the probability envelopes for MatMult.
def compute_envn_MM (n_lim,p,l_nb_iter,case) :
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
      
      #Don't know what to make.
      cores = 2 * 192
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
      
      #Don't know what to make.
      cores = 2 * 192
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
   
#********************** TreeAdd Enveloppes **********************
#Base case for TreeAdd.
def initial_case_TA(n, envgadd) :
  envn = np.zeros((2 * n + 1, n + 1))
  for i in range (2 * n + 1) :
    it = min (n , i)
    k = max (0, i - n) 
    for j in range (n + 1) :
      for i1 in range (k, it + 1) :
        envn[i,j] += envgadd[i1, i - i1, j]

  return envn

#Induction formulas for TreeAdd.
def induction_envn_TA(nb_row, n, envn_TA, envgadd) :
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

#n > 4
def prec_TArec (n, envgadd) :
  nb_row = 1
  envn_TA = initial_case_TA(n, envgadd)
  
  while (nb_row < int(n / 8)) :    
    nb_row = 2 * nb_row
    envn_TA = induction_envn_TA(nb_row, n, envn_TA, envgadd)

  return envn_TA

def TAprec1 (n, envgadd) :
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


#Compute the probability envelopes of TreeAdd.        
def compute_envn_TA (n, p, nb_iter) :
  pgref = cardinal_rpc_refresh_envelope(n, p, nb_iter)
  envgadd = cardinal_rpc_add_envelope(n,p,pgref)
  prec1 = TAprec1 (n, envgadd)
  
  #TODO Works only for n = 4 i guess
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

#Print the probability envelopes of TreeAdd.
def print_envn_TA(nb_row, n, envn_TA) :
  sm = np.zeros(n + 1)
  
  first_row = "         |"
  for j in range (n + 1) :
    if (j < 10) :
      first_row += "|J| = 0" + str(j) + "     |"
      
    else :
      first_row += "|J| = " + str(j) + "     |"
  print(first_row) 
  for i in range (nb_row * n *2 + 1) :
    row = ""
    if (i < 10) :
      row += "|I| = 0"+ str(i) + " |"
    else :  
      row += "|I| = "+ str(i) + " |"
    for j in range (n + 1) :
      sm[j] += envn_TA[i,j]
      var = "%e" % envn_TA[i,j]
      row += var + " |"
    print(row)
  print("\n")
  
  last_row = "         |"
  for j in range (n + 1) :
    last_row += "%e" % sm[j]
    #last_row += str(sm[j])
    last_row += " |"
  print(last_row)



#********************** NewRefSecMult envelopes **********************

#Compute the envelopes of NewRefSecMult.
def compute_envn (n, envn_MM, envn_TA) :
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

#Print the envelopes of NewRefSecMult
def print_envn (n, envn) :  
  for j in range (n + 1) :
    print("Pour |J| = " + str(j) + " : \n")
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
 

#********************** RPC-threshold from cardinal-RPC **********************
def compute_RPC_threshold (n, envn, t) :
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for ix in range (n + 1) :
      for iy in range (n + 1) :
        if (ix  > t or iy > t) :
          smj += envn[ix,iy,j]
    
    if (res < smj) :
      res = smj    
  return res  

def compute_RP (n, envn) :
  res = 0
  for ix in range (n + 1) :
    res += envn[ix, n, 0]
  
  for iy in range (n) :
    res += envn [n, iy, 0]
    
  return res  

#Compute the graph of RPC-Threshold security for the parameters (n,t,p) in function of gamma.
def compute_graph (n,t,p, eps_for_nb_iter, l_nb_iter, str_sym) :
  plt.plot(l_nb_iter, eps_for_nb_iter, marker='.', label=str_sym + ", n = "+str(n))
  
  threshold_value = 2 * comb(n, t +1) * p**(t + 1) * (1-p)**(n-t-1) #Victor : Add a factor 2 because of the 2 secrets values.
  plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
  plt.legend(prop={'size': 6})
  
  #plt.title ("Security (n,t,p)-RPC, t = " + str(t) + ",p = 2^"+str(int(log(p,2))))
  #plt.xlabel("Number of random values γ")
  #plt.ylabel("RPC (n,t,p)")
  #plt.yscale('log', base=2)
  #plt.ylim(2**(log(threshold_value, 2) - 4))
  #plt.gca().invert_yaxis()
  #plt.legend()
  #plt.grid(True)
  
  
  #plt.savefig("RPC-threshold_n"+str(n)+"_t"+str(t)+"_p"+str(int(log(p, 2)))+".png")
  #plt.close()

    

def cardinal_rpc_mult_uni_envelopes (n, p, l_nb_iter,  nb_iter_TA, case) :
  envn_MM = compute_envn_MM(n, p, l_nb_iter, case)
  envn_TA = compute_envn_TA(n, p, nb_iter_TA)
  envn = compute_envn (n, envn_MM, envn_TA)
  return envn

  
#********************** Main **********************

      
def main () :
  #n = 16 in ~ 5h for MM, for only one set of parameters (n,p,nb_iter).
  
  n = 8
  k = int(log(n,2))
  p = 2**-6
  t = int(n / 2)
  case = "Sym"
  
  l_nb_iter = []
  eps_nb_iter = np.zeros((40))
  for nb_iter in range (1,41) :
    print("nb_iter = ", nb_iter)
    l_nb_iter.append(nb_iter) 
  
    l_nb_iter_MM = [nb_iter for _ in range (k - 1)]
  #************ MatMult Enveloppes ************ 
    envn_MM = compute_envn_MM(n, p, l_nb_iter_MM, case)
  
  
  #If I want to save the variables into file "filename", just uncomment this :
  
  #filename = "Param_" + str(n) + "_sh.txt"
  #filename_envn_MM = "envn_MM_" + str(n) + "_sh.npy"
  #save_variables(filename, n, p, nb_iter, filename_envn_MM, envn_MM)
  
  #************ TreeAdd Enveloppes ************
  
    envn_TA = compute_envn_TA(n, p, nb_iter)
  
  #************ NewRefSecMult Enveloppes ************
    envn = compute_envn (n, envn_MM, envn_TA)
  
  #************ RPC-threshold ************
    eps_RPC_threshold = compute_RPC_threshold(n,envn, t)
    eps_nb_iter[nb_iter - 1] = eps_RPC_threshold 
      
  #************ Save Results ************* 
  np.save(case + "_Uni_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy", eps_nb_iter)
  
#main()      
  
def mainbis () :
  print("ok")
  n = 4
  p = 2**-11
  t = int(n / 2)
  case = "Sym"
  nb_iter = 1100
  l_nb_iter = [1900, 900]
  l_nb_iter2 = [1900, 1900]
    
  #************ MatMult Enveloppes ************ 
  envn_MM = compute_envn_MM(n, p, l_nb_iter, case)
  envn_MM2 = compute_envn_MM(n, p, l_nb_iter2, case)
  #************ TreeAdd Enveloppes ************ 
  envn_TA = compute_envn_TA(n, p, nb_iter)
  
  #************ NewRefSecMult Enveloppes ************
  envn = compute_envn (n, envn_MM, envn_TA)
  envn2 = compute_envn (n, envn_MM2, envn_TA)
  
  #************ RPC-threshold ************
  eps_RPC_threshold = compute_RPC_threshold(n, envn, t)
  eps_RPC_threshold2 = compute_RPC_threshold(n, envn2, t)
  
  print("eps1 = ", log(eps_RPC_threshold, 2))
 # print("eps2 = ", eps_RPC_threshold2)


def find_plateau () :
  n = 4
  p = 2**-36
  t = int(n / 2)
  case = "Sym"
  l_nb_iter = []
  eps = []
  
  for nb_iter in range (21) :  
    print("nb_iter = ", nb_iter)
    l_nb_iter.append(nb_iter)
    
    
    #************ MatMult Enveloppes ************ 
    envn_MM = compute_envn_MM(n, p, [nb_iter, nb_iter], case)
  
    #************ TreeAdd Enveloppes ************ 
    envn_TA = compute_envn_TA(n, p, nb_iter)
  
    #************ NewRefSecMult Enveloppes ************
    envn = compute_envn (n, envn_MM, envn_TA)
  
    #************ RPC-threshold ************
    eps_RPC_threshold = compute_RPC_threshold(n, envn, t)
    eps.append(eps_RPC_threshold)
    
  
  plt.plot(l_nb_iter, eps, marker = 'x', linestyle='')
  plt.yscale('log', base=2)
  plt.gca().invert_yaxis()
  plt.grid(True)
  plt.show()
  
#find_plateau()

def compute_SNIref_RPC(n, p) :
  nb_iter = -1
  t = int (n /2)
  case = "Sym"
  
  k = log(n,2)
  l_nb_iter = [nb_iter for _ in range (k - 1)]
  #pgref = compute_envrefn(n,p)
  #print_envn_ref(n, pgref)
  #eps_RPC_threshold = compute_RPC_threshold_ref (n, pgref, t)
  
  
  envn_MM = compute_envn_MM(n, p, l_nb_iter, case)
  envn_TA = compute_envn_TA(n, p, nb_iter)
  envn = compute_envn (n, envn_MM, envn_TA)
    
  eps_RPC_threshold = compute_RPC_threshold(n,envn, t)
  
  return eps_RPC_threshold  


def compute_RPC_threshold_ref (n, envn, t) :
  res = 0
  for j in range (t + 1) :
    smj = 0 
    for ix in range (n + 1) :
      if (ix  > t) :
        smj += envn[ix,j]
    
    if (res < smj) :
      res = smj    
  return res


#Print the probability envelopes of TreeAdd.
def print_envn_ref(n, envn_TA) :
  sm = np.zeros(n + 1)
  
  first_row = "         |"
  for j in range (n + 1) :
    if (j < 10) :
      first_row += "|J| = 0" + str(j) + "     |"
      
    else :
      first_row += "|J| = " + str(j) + "     |"
  print(first_row) 
  for i in range (n + 1) :
    row = ""
    if (i < 10) :
      row += "|I| = 0"+ str(i) + " |"
    else :  
      row += "|I| = "+ str(i) + " |"
    for j in range (n + 1) :
      sm[j] += envn_TA[i,j]
      var = "%e" % envn_TA[i,j]
      row += var + " |"
    print(row)
  print("\n")


  
#Compute my final RPC graph. 
def compute_final_graph() :
  n = 4                                 # n = 4 to ...
  p = 2**-12                            # Vary p.
  l_nb_iter = []                        # Loop on the gamma values of the refresh gadget.
  for nb_iter in range (1,41) :
    l_nb_iter.append(nb_iter) 
  
  
  plt.title ("Security (n,t,p)-RPC of NRSM, t = (n / 2),p = 2^"+str(int(log(p,2))))
  plt.xlabel("Number of random values γ")
  plt.ylabel("RPC (n,t,p)")
  plt.yscale('log', base=2)
  plt.ylim(2**-100)
  plt.gca().invert_yaxis()
  plt.grid(True)
  
  
  while (os.path.isfile("Sym_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_nb_iter = np.load("Sym_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    t = int(n / 2)
    compute_graph(n,t,p, eps_nb_iter, l_nb_iter, "Sym")
    n = n * 2

  n = 4
  while (os.path.isfile("Asym_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_nb_iter = np.load("Asym_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    t = int(n / 2)
    compute_graph(n,t,p, eps_nb_iter, l_nb_iter, "Asym")
    n = n * 2
    
  n = 4
  while (os.path.isfile("Asym_Uni_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_nb_iter = np.load("Asym_Uni_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    t = int(n / 2)
    compute_graph(n,t,p, eps_nb_iter, l_nb_iter, "UniAsym")
    n = n * 2

  n = 4
  while (os.path.isfile("Sym_Uni_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_nb_iter = np.load("Sym_Uni_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    t = int(n / 2)
    compute_graph(n,t,p, eps_nb_iter, l_nb_iter, "UniSym")
    n = n * 2


  #Put in comment if we don't want the SNI refresh gadget case on the graph.
  n = 4
  eps_SNI_n4 = compute_SNIref_RPC(n, p)
  gamma = randoms_used(n)
  plt.plot(gamma, eps_SNI_n4, color="purple", marker = "x", linestyle='', label = "SNI,n = 4")
  
  n = 8
  eps_SNI_n8 = compute_SNIref_RPC(n, p)
  gamma = randoms_used(n) 
  plt.plot(gamma,eps_SNI_n8, color="yellow", marker = "x", linestyle='', label = "SNI, n = 8")
  plt.legend()
  
  
  #plt.show()
  plt.savefig("NRSM_SNI_TresholdRPC_p"+str(int(log(p,2))))
  plt.close()
  

#To test a lot of things.
def compareJMB24() :
  n = 8
  k = log(n, 2)
  nb_iter = 50
  l_nb_iter = [nb_iter for _ in range (k - 1)]
  t = int (n / 2)
  
  eps_comparison_JMB24 = np.zeros((50))
  eps_JMB24_n4 = np.zeros((50))
  eps_JMB24_n8 = np.zeros((50))
  eps_BFO23_n4 = np.zeros((50))
  eps_BFO23_n8 = np.zeros((50))
  
  #For n = 4, RP :
  #p_start = -36
  #p_end = -19
  
  #For n = 8, RP :
  #p_start = -20
  #p_end = -12
  
  #For n = 4, RPC :
  #p_start = -66
  #p_end = -34
  
  #For n = 8, RPC :
  p_start = -31
  p_end = -17
  
  
  prange = np.logspace(p_start, p_end, 50, base = 2)
  for i in range (50) : 
    p = prange[i]
    ind_BFO23 = 1 - (3 * p)**(1/2)
    envn_MM = compute_envn_MM(n, p, l_nb_iter, "Sym")
    envn_TA = compute_envn_TA(n, p, nb_iter)
    envn = compute_envn (n, envn_MM, envn_TA)
    
    eps_RPC_threshold = compute_RPC_threshold(n,envn, t)
    eps_comparison_JMB24[i] = eps_RPC_threshold
    #eps_RP = compute_RP(n, envn)
    #eps_comparison_JMB24[i] = eps_RP
    eps_JMB24_n4[i] = (2 * p)**(0.3 * n) + p**(0.6*n)
    eps_JMB24_n8[i] = (2 * p)**(0.6 * n) + p**(1.2*n)
    eps_BFO23_n4[i] = 1 - (1 - p)**(32) + 1 - (ind_BFO23)**(3)
    eps_BFO23_n8[i] = 1 - (1 - p)**(64) + 1 - (ind_BFO23)**(7)
    
    #p += 0.001
    #p += 2**-27
  
  #epsn4_Asym_JMB24 = np.load("epsn4_Asym_JMB24.npy")  
  #epsn8_Asym_JMB24 = np.load("epsn8_Asym_JMB24.npy")
  #epsn8_SymUnif_JMB24 = np.load("epsn8_SymUnif_JMB24.npy")
  #epsn8_SymUnif_JMB24_zoom = np.load("epsn8_SymUnif_JMB24_zoom.npy")
  
  #plt.plot(prange, epsn8_Asym_JMB24, label = "AdvNRSM-Asym, n = 8")
  #plt.plot(prange, epsn8_SymUnif_JMB24, label = "AdvNRSM-SymUnif, n = 8")
  #plt.plot(prange, epsn8_SymUnif_JMB24_zoom, label = "AdvNRSM-SymUnif, n = 8")
  #plt.plot(prange, eps_comparison_JMB24, label = "AdvNRSM-SymUnif, n = 8")
  #plt.plot(prange, eps_JMB24_n8, label = "AdvJMB24, n = 8")
  #plt.plot(prange, eps_BFO23_n8, label = "AdvBFO23, n = 8")
  
  #plt.plot(prange, epsn4_Asym_JMB24, label = "AdvNRSM-Asym, n = 4")
  plt.plot(prange, eps_comparison_JMB24, label = "AdvNRSM-SymUnif, n = "+str(n), marker= 'x')
  plt.plot(prange, eps_JMB24_n4, label = "AdvJMB24, n = " + str(n), marker = 'x')
  plt.yscale('log', base=2)
  plt.xscale('log', base=2)
  plt.xlim(2**p_start, 2**p_end)
  plt.grid(True)
  #plt.plot(prange, eps_BFO23_n4, label = "AdvBFO23, n = 4")
  plt.legend()
  plt.xlabel("Leakage rate p")
  plt.ylabel("Upper bound of the RPM Advantage")
  plt.title ("RPC advantage bound of multiplication gadget given by JMB24 and me")
  #plt.show()
  plt.savefig("RPCsec_mevsJMB24_n"+str(n)+".png")
  
  #np.save("epsn8_SymUnif_JMB24_zoom.npy", eps_comparison_JMB24)




if __name__ == "__main__":
  print("main")
  #mainbis()
  #compute_graph()
  #find_plateau_RPM_n(2**-10)
  #find_plateau_RPM_n(2**-15)
  #find_plateau_RPM_n(2**-20)

