#!/usr/bin/env python3


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


#********************** MatMult Enveloppes **********************
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


def precomp1 (n, single_proba_three, single_proba_four, j3, j4, queue) :
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
  lim_card = n**2
  prec3 = np.zeros((lim_card + 1, lim_card + 1, n + 1, n + 1))  
  for i4 in range (n, 2 * n + 1) :
    for i14 in range(i4 - n, n + 1) :
      for i21 in range (n + 1) :
        for i22 in range (n + 1) :
          prec3[j3, j4, i21, i22] += single_proba_three[j3, i21, i14] * single_proba_four[j4, i22, i4 - i14]   
  
  queue.put((j3, j4, prec3))            
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

#Compute the envelopes for |Ix| = ix, |Iy| = iy and |J1|, |J2|, |J3|, |J4| for 
#the MatMult gadgets with 2 * n shares. 
def compute_proba(n, ix, iy, j1, j2, j3, j4, prec1, prec2, prec3) :
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
def induction_envelopes(n, prec1, prec2, prec3, ix, iy, queue) : 
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




#Compute the probability envelopes for MatMult.
def compute_envn_MM (n_lim, p, nb_iter, case) :
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
      
      np.save("prec1"+str(2*n)+".npy",prec1)
    
    processes = []
    
    
    if (os.path.isfile("prec2"+str(2 * n)+".npy")) :
      prec2 = np.load("prec2"+str(2 * n)+".npy")
    
    else :  
      for j1 in range (lim_card + 1) :
        for j2 in range (lim_card + 1) :
          processes.append(Process(target=precomp2, args=(n, single_proba_one, single_proba_two, j1, j2, queue_precomp)))
      
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
        
      np.save ("prec2"+str(2 * n)+".npy", prec2)
      
    processes = []
    if (os.path.isfile("prec3"+str(2 * n)+".npy")) :
      prec3 = np.load("prec3"+str(2 * n)+".npy")
    
    else :  
      for j3 in range (lim_card + 1) :
        for j4 in range (lim_card + 1) :
          processes.append(Process(target=precomp3, args=(n, single_proba_three, single_proba_four, j3, j4, queue_precomp)))
      
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
  

#Save the variables used for compute a probability envelope of MatMult as well as the 
#probability envelope itself.
def save_variables (filename, n, p , nb_iter, filename_envn_MM, envn_MM) :
  np.save(filename_envn_MM, envn_MM)
  with open (filename, "w") as f : 
    f.write("Parameters : \n")
    f.write("n = " + str(n) + "\n")
    f.write("p = " + str(p) + "\n")
    f.write("nb_iter = " + str(nb_iter) + "\n")
    f.write("filename_envn_MM = " + filename_envn_MM + "\n")
       


#Load the variables saved by the function |save_variables|.  
def load_variables(filename) :
  n = 0
  p = 0.0
  nb_iter = 0
  filename_envn_MM = ""
  
  with open(filename, "r") as f :
    f.readline()
    line = f.readline()
    line = line[4 : -1]
    n = int(line)
    line = f.readline()
    line = line[4 : -1]
    p = float(line)   
    line = f.readline()
    line = line[10 : -1]
    nb_iter = int(line)
    line = f.readline()
    line = line[16 : -1]
    filename_envn_MM = line  
  
  envn_MM = np.load(filename_envn_MM)
  return (n, p, nb_iter, envn_MM)
  
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

def compute_envn_mult_4card(n, p, nb_iter, case) :
  #************ MatMult Enveloppes ************ 
  envn_MM = compute_envn_MM(n, p, nb_iter, case)
  
  #************ TreeAdd Enveloppes ************
  envn_TA = compute_envn_TA(n, p, nb_iter)
  
  #************ NewRefSecMult Enveloppes ************
  envn = compute_envn (n, envn_MM, envn_TA)
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


#Compute the graph of RPC-Threshold security for the parameters (n,t,p) in function of gamma.
def compute_graph (n,t,p, eps_for_nb_iter, l_nb_iter, str_sym) :
  plt.plot(l_nb_iter, eps_for_nb_iter, marker='.', label=str_sym + ", n = "+str(n))
  
  threshold_value = 2 * comb(n, t +1) * p**(t + 1) * (1-p)**(n-t-1)
  plt.axhline(y=threshold_value, color=plt.gca().lines[-1].get_color(), linestyle='--')
  plt.legend()

  
#********************** Main **********************

      
def main () :
  n = 4
  p = 2**-6
  t = int(n / 2)
  case = "Asym"
  
  nb_iter = 40
  
  
  #************ MatMult Enveloppes ************ 
  envn_MM = compute_envn_MM(n, p, nb_iter, case)
  
  #************ TreeAdd Enveloppes ************
  envn_TA = compute_envn_TA(n, p, nb_iter)
  
  #************ NewRefSecMult Enveloppes ************
  envn = compute_envn (n, envn_MM, envn_TA)
  
  #************ RPC-threshold ************
  eps_RPC_threshold = compute_RPC_threshold(n,envn, t)
  print("eps  = ", log(eps_RPC_threshold, 2))
  #************ Save Results ************* 
  #np.save(case + "_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy", eps_nb_iter)

#main()      
  
#Compute my final RPC graph. Section 5.3 
def compute_final_graph(n_values, p, lim_gamma) :
  path = "./results/gamma_func_4card/"
  
  
  
  n = 4                                 
  p = 2**-6                           # p = 2^{-6}, 2^{-12}, 2^{-18}
  ylim_graph = 2**-25                 #     2^{-25}  2^{-56}  2^{-86}
  gamma = []                        # Loop on the gamma values of the refresh gadget.
  for g in range (1,41) :
    gamma.append(g) 
  
  
  #plt.title ("Security (n,t,p)-RPC of NRSM, t = (n / 2), p = 2^"+str(int(log(p,2))))
  fig, ax = plt.subplots(figsize=(5.4, 4), dpi = 1200)
  
  ax.set_xlabel(r"Number of random values $\gamma$")
  ax.set_ylabel("RPC (n,t,p)")
  ax.set_yscale('log', base=2)
  ax.set_ylim(ylim_graph)
  ax.set_xlim(0, 40)
  ax.invert_yaxis()
  ax.grid(True)
  
  while (os.path.isfile(path + "Asym_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_gamma = np.load(path + "Asym_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    for i in range(len(eps_gamma)) : 
      eps_gamma[i] = min(1, eps_gamma[i])
    t = int(n / 2)
    compute_graph(n, t, p, eps_gamma, gamma, "Asym")
    n = n * 2  
  
  n = 4
  while (os.path.isfile(path + "Sym_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_gamma = np.load(path + "Sym_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    for i in range(len(eps_gamma)) : 
      eps_gamma[i] = min(1, eps_gamma[i])
    t = int(n / 2)
    compute_graph(n, t, p, eps_gamma, gamma, "Sym")
    n = n * 2

  n = 4
  while (os.path.isfile(path + "Sym_Uni_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")) :
    eps_gamma = np.load(path + "Sym_Uni_4card_eps_n"+str(n)+"_p"+str(int(log(p,2))) + ".npy")
    for i in range(len(eps_gamma)) : 
      eps_gamma[i] = min(1, eps_gamma[i])
    t = int(n / 2)
    compute_graph(n,t,p, eps_gamma, gamma, "Unif")
    n = n * 2

  fig.tight_layout()
  fig.savefig("NRSM_4card_TresholdRPC_p"+str(int(log(p,2)))+".pdf", bbox_inches="tight")  
  plt.close()

#Graphe final 
#compute_final_graph()  


if __name__ == "__main__":
  print("main")
  #main()
  #compute_graph()
  #find_plateau_RPM_n(2**-10)
  #find_plateau_RPM_n(2**-15)
  #find_plateau_RPM_n(2**-20)



      
