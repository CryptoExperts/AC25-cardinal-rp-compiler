#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.special import comb
from math import log
import numpy as np
import os
from multiprocessing import Pool, Process, Queue, shared_memory

from partitions import (cardinal_rpc_refresh_envelope, 
                        cardinal_rpc_add_envelope, 
                        cardinal_rpc_gcmult_envelope_pgref,
                        cardinal_rpc_gcopy_envelope_pgref,
                        cardinal_rpc_gcopy_envelope_pgref_prec)
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

#ref following by add gadget.
def env_ref_add(n, env_ref, env_add):
  env = np.zeros((n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout in range (n + 1) :
        for l1 in range (n + 1) :
          for l2 in range (n + 1) :
            env[tin1, tin2, tout] += env_ref[tin1, l1] * env_ref[tin2, l2] * env_add[l1, l2, tout]
  return env 
                        

#cRPC part

def cRPC_copy(n, p) :
  env = np.zeros((n + 1, n + 1, n + 1))
  for tin in range (n + 1) :
    for tout1 in range (tin + 1) : 
      for tout2 in range (tin + 1) :
        for l in range (tin + 1) :
          if (tin <= (tin - l + tout1  + tout2)) :
            env[tin, tout1, tout2] += (comb(n - l, tin - l) * p**(tin - l) * 
                                       (1 - p)**(n - tin))
        env[tin, tout1, tout2] = min(1, env[tin, tout1, tout2])
  return env
          

def cRPC_prec1 (n, env_add, env_copy) :
  env = np.zeros((n + 1,  n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (tin1 + 1) :
            env[tin1, tin2, tout1, tout2] += env_add[l, tin2, tout1] * env_copy[tin1, l, tout2]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env
  
def cRPC_prec2 (n , env_prec1, env_copy) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout3 in range (n + 1) :
        for tout1 in range (n + 1) :
          for tout2 in range (n + 1) :
            for l in range (tin2 + 1) :
              env[tin1, tin2, tout1, tout2, tout3] += env_prec1[tin1, l, tout1, tout2] * env_copy[tin2, l, tout3]
            env[tin1, tin2, tout1, tout2, tout3] = min(1, env[tin1, tin2, tout1, tout2, tout3])
  return env

def cRPC_prec3 (n, env_prec2, env_add) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) : 
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l1 in range (n + 1) :
            for l2 in range (n + 1) :
              env[tin1, tin2, tout1, tout2] += env_prec2[tin1, tin2, tout1, l1, l2] * env_add[l1, l2, tout2]
          env[tin1, tin2, tout1, tout2] = min (1, env[tin1, tin2, tout1, tout2])
  return env

def cRPC_prec4 (n, env_prec3, env_cmult) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (n + 1) :
            env[tin1, tin2, tout1, tout2] += env_cmult[tin2, l] * env_prec3[tin1, l, tout1, tout2]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env
  
def cRPC_butterfly_stage1 (n, p, gamma):
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  env_copy = cRPC_copy(n, p)
  
  env_prec1 = cRPC_prec1 (n, env_add, env_copy)
  env_prec2 = cRPC_prec2 (n , env_prec1, env_copy)
  env_prec3 = cRPC_prec3 (n, env_prec2, env_add)
  env_butterfly = cRPC_prec4 (n, env_prec3, env_cmult)
  
  return env_butterfly

def cRPC_butterfly_v2_stage1(n, p, gamma):
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_radd = env_ref_add(n, pgref, env_add)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  env_copy = cRPC_copy(n, p)

  
  env_prec1 = cRPC_prec1 (n, env_radd, env_copy)
  env_prec2 = cRPC_prec2 (n , env_prec1, env_copy)
  env_prec3 = cRPC_prec3 (n, env_prec2, env_radd)
  env_butterfly = cRPC_prec4 (n, env_prec3, env_cmult)
  
  return env_butterfly
  
#ucRPC part

def precomp_hypergeom(N) :
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

def ucRPC_prec1 (n, env_add, hypergeom) :
  env = np.zeros((n + 1,  n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range(n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          if (tout2 <= tin1) :
            for l in range (max(tin1 - tout2, 0), tin1 + 1) :
              env[tin1, tin2, tout1, tout2] += env_add[l, tin2, tout1] * hypergeom[l, tout2, l + tout2 - tin1]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env

def ucRPC_prec11 (n, p, prec1) :
  env = np.zeros((n + 1,  n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range(n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (tin1 + 1) :
            env[tin1, tin2, tout1, tout2] += (p**l * (1 - p)**(n - tin1) * 
                                              prec1[tin1 - l, tin2, tout1, tout2])
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env
  
def ucRPC_prec2 (n , env_prec1, hypergeom) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout3 in range (n + 1) :
        if (tout3 <= tin2) :
          for tout1 in range (n + 1) :
            for tout2 in range (n + 1) :
              for l in range (max(0, tin2 - tout3), tin2 + 1) :
                env[tin1, tin2, tout1, tout2, tout3] += env_prec1[tin1, l, tout1, tout2] * hypergeom[l, tout3, l + tout3 - tin2]
              env[tin1, tin2, tout1, tout2, tout3] = min(1, env[tin1, tin2, tout1, tout2, tout3])
  return env

def ucRPC_prec21 (n , p, prec2) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for tout3 in range (n + 1) :
            for l in range (tin2 + 1) :
              env[tin1, tin2, tout1, tout2, tout3] += (p**l * 
              (1 - p)**(n - tin2) * prec2[tin1, tin2 - l, tout1, tout2, tout3])
            env[tin1, tin2, tout1, tout2, tout3] = min(1, env[tin1, tin2, tout1, tout2, tout3])
  return env

def ucRPC_prec3 (n, env_prec2, env_add) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) : 
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l1 in range (n + 1) :
            for l2 in range (n + 1) :
              env[tin1, tin2, tout1, tout2] += env_prec2[tin1, tin2, tout1, l1, l2] * env_add[l1, l2, tout2]
          env[tin1, tin2, tout1, tout2] = min (1, env[tin1, tin2, tout1, tout2])
  return env

def ucRPC_prec4 (n, env_prec3, env_cmult) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (n + 1) :
            env[tin1, tin2, tout1, tout2] += env_cmult[tin2, l] * env_prec3[tin1, l, tout1, tout2]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env
  
def ucRPC_butterfly_stage1 (n, p, gamma):
  hypergeom = precomp_hypergeom(n)
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  
  env_prec1 = ucRPC_prec1 (n, env_add, hypergeom)
  env_prec1 = ucRPC_prec11 (n, p, env_prec1)

  env_prec2 = ucRPC_prec2 (n , env_prec1, hypergeom)
  env_prec2 = ucRPC_prec21(n, p, env_prec2)

  env_prec3 = ucRPC_prec3 (n, env_prec2, env_add)
  env_butterfly = ucRPC_prec4 (n, env_prec3, env_cmult)
  
  return env_butterfly

def ucRPC_butterfly_v2_stage1 (n, p, gamma):
  hypergeom = precomp_hypergeom(n)
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_radd = env_ref_add(n, pgref, env_add)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  
  env_prec1 = ucRPC_prec1 (n, env_radd, hypergeom)
  env_prec1 = ucRPC_prec11 (n, p, env_prec1)

  env_prec2 = ucRPC_prec2 (n , env_prec1, hypergeom)
  env_prec2 = ucRPC_prec21(n, p, env_prec2)

  env_prec3 = ucRPC_prec3 (n, env_prec2, env_radd)
  env_butterfly = ucRPC_prec4 (n, env_prec3, env_cmult)
  
  return env_butterfly

#RPC_threshold :
def tRPC_env_tree_copy (n) :
  env = np.zeros((n + 1, n + 1, n + 1))
  for tin in range (n + 1) :
    for tout1 in range (tin + 1) :    
      for tout2 in range (max(0, tin - tout1),tin + 1) :
        env[tin, tout1, tout2] = comb (n, tin) * comb(tin, tout1) * comb(tout1, tout2 - tin + tout1) / (comb(n, tout1) * comb(n, tout2))
  return env       




#Butterfly Stage 2 :

def st2_prec1 (n, env_butt_st1) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tin3 in range (n + 1) :
        for tout1 in range (n + 1) :
          for tout2 in range (n + 1) :
            for tout3 in range (n + 1) :
              for l in range (n + 1) :
                env[tin1, tin2, tin3, tout1, tout2, tout3] += env_butt_st1[tin1, tin2, l, tout3] * env_butt_st1[l, tin3, tout1, tout2]
              env[tin1, tin2, tin3, tout1, tout2, tout3] = min(env[tin1, tin2, tin3, tout1, tout2, tout3], 1)
  return env
  
def st2_prec1_para (n, name, tin1, tin2, queue) :
  existing_shm = shared_memory.SharedMemory(name=name)
  shape = (n + 1, n + 1, n + 1, n + 1)
  env_butt_st1 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin3 in range (n + 1) :
    for tout1 in range (n + 1) :
      for tout2 in range (n + 1) :
        for tout3 in range (n + 1) :
          for l in range (n + 1) :
            env[tin3, tout1, tout2, tout3] += env_butt_st1[tin1, tin2, l, tout3] * env_butt_st1[l, tin3, tout1, tout2]
          env[tin3, tout1, tout2, tout3] = min(env[tin3, tout1, tout2, tout3], 1)
  existing_shm.close()  
  queue.put((tin1, tin2, env))
  return 
  
def st2_prec2 (n, env_butt_st1, env_prec1) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tin3 in range (n + 1) :
        for tin4 in range (n + 1) :
          for tout1 in range (n + 1) :
            for tout2 in range (n + 1) :
              for tout3 in range (n + 1) :
                for tout4 in range (n + 1) :
                  for l in range (n + 1) :
                    env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4] += env_prec1[tin1, tin2, tin3, tout1, tout2, l] * env_butt_st1[l, tin4, tout3, tout4]
                  env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4] = min(1, env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4])
  return env
  
def st2_prec2_para (n, name, name2, tin1, tin2, queue) :
  existing_shm = shared_memory.SharedMemory(name=name)
  shape = (n + 1, n + 1, n + 1, n + 1)
  env_butt_st1 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)  
  
  existing_shm2 = shared_memory.SharedMemory(name=name2)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1, n + 1)
  env_prec1 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm2.buf)
  
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin3 in range (n + 1) :
    for tin4 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for tout3 in range (n + 1) :
            for tout4 in range (n + 1) :
              for l in range (n + 1) :
                env[tin3, tin4, tout1, tout2, tout3, tout4] += env_prec1[tin1, tin2, tin3, tout1, tout2, l] * env_butt_st1[l, tin4, tout3, tout4]
              env[tin3, tin4, tout1, tout2, tout3, tout4] = min(1, env[tin3, tin4, tout1, tout2, tout3, tout4])
  existing_shm.close()
  existing_shm2.close()
  queue.put((tin1, tin2, env))
  return
  
def st2_prec3 (n, env_butt_st1, env_prec2) :
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tin3 in range (n + 1) :
        for tin4 in range (n + 1) :
          for tout1 in range (n + 1) :
            for tout2 in range (n + 1) :
              for tout3 in range (n + 1) :
                for tout4 in range (n + 1) :
                  for l in range (n + 1) :
                    for l2 in range (n  + 1) :
                      env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4] += env_butt_st1[tin3, tin4, l, l2] * env_prec2[tin1, tin2, l, l2, tout1, tout2, tout3, tout4]
                    env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4] = min(1, env[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4])                            
  return env

def st2_prec3_para (n, name, name2, tin1, tin2, queue) :
  existing_shm = shared_memory.SharedMemory(name=name)
  shape = (n + 1, n + 1, n + 1, n + 1)
  env_butt_st1 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)

  existing_shm2 = shared_memory.SharedMemory(name=name2)
  shape = (n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1)
  env_prec2 = np.ndarray(shape, dtype=np.float64, buffer=existing_shm2.buf)
  
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin3 in range (n + 1) :
    for tin4 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for tout3 in range (n + 1) :
            for tout4 in range (n + 1) :
              for l in range (n + 1) :
                for l2 in range (n  + 1) :
                  env[tin3, tin4, tout1, tout2, tout3, tout4] += env_butt_st1[tin3, tin4, l, l2] * env_prec2[tin1, tin2, l, l2, tout1, tout2, tout3, tout4]
                env[tin3, tin4, tout1, tout2, tout3, tout4] = min(1, env[tin3, tin4, tout1, tout2, tout3, tout4])                            
  existing_shm.close()
  existing_shm2.close()
  queue.put((tin1, tin2, env))
  return


def proceed_para(target_file, param, nb_param_env) :
  processes = []
  queue = Queue()
  cores = 192
  param.append(0)
  param.append(0)
  param.append(queue)
  n = param[0]
  
  env = 0
  match nb_param_env :
    case 6 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
    case 8 :
      env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1, n + 1))
    case _ :
      env = 0 
  
  
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      param[-3] = tin1
      param[-2] = tin2
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

def cRPC_butterfly_stage2 (n, p, gamma):
  env_butt_st1 = cRPC_butterfly_stage1 (n, p, gamma)
  
  #env_prec1 = st2_prec1 (n, env_butt_st1)
  #env_prec2 = st2_prec2 (n, env_butt_st1, env_prec1)
  #env_prec3 = st2_prec3 (n, env_butt_st1, env_prec2)
  
  shm = shared_memory.SharedMemory(create=True, size=env_butt_st1.nbytes)
  shared_arr = np.ndarray(env_butt_st1.shape, dtype=env_butt_st1.dtype, buffer=shm.buf)
  np.copyto(shared_arr, env_butt_st1)
  
  env_prec1 = proceed_para(st2_prec1_para, [n, shm.name], 6)
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec1.nbytes)
  shared_arr2 = np.ndarray(env_prec1.shape, dtype=env_prec1.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec1)
  
  env_prec2 = proceed_para(st2_prec2_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec2.nbytes)
  shared_arr2 = np.ndarray(env_prec2.shape, dtype=env_prec2.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec2)
  
  env_prec3 = proceed_para(st2_prec3_para, [n, shm.name, shm2.name], 8)
  
  #print(env_prec3)
  
  shm2.close()
  shm2.unlink()
  
  shm.close()
  shm.unlink()
  
  return env_prec3


def ucRPC_butterfly_stage2 (n, p, gamma):
  env_butt_st1 = ucRPC_butterfly_stage1 (n, p, gamma)
  
  #env_prec1 = st2_prec1 (n, env_butt_st1)
  #env_prec2 = st2_prec2 (n, env_butt_st1, env_prec1)
  #env_prec3 = st2_prec3 (n, env_butt_st1, env_prec2)
  
  shm = shared_memory.SharedMemory(create=True, size=env_butt_st1.nbytes)
  shared_arr = np.ndarray(env_butt_st1.shape, dtype=env_butt_st1.dtype, buffer=shm.buf)
  np.copyto(shared_arr, env_butt_st1)
  
  env_prec1 = proceed_para(st2_prec1_para, [n, shm.name], 6)
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec1.nbytes)
  shared_arr2 = np.ndarray(env_prec1.shape, dtype=env_prec1.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec1)
  
  env_prec2 = proceed_para(st2_prec2_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec2.nbytes)
  shared_arr2 = np.ndarray(env_prec2.shape, dtype=env_prec2.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec2)
  
  env_prec3 = proceed_para(st2_prec3_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm.close()
  shm.unlink()
  
  return env_prec3
  
  
def cRPC_butterfly_v2_stage2 (n, p, gamma):
  env_butt_st1 = cRPC_butterfly_v2_stage1 (n, p, gamma)
  
  #env_prec1 = st2_prec1 (n, env_butt_st1)
  #env_prec2 = st2_prec2 (n, env_butt_st1, env_prec1)
  #env_prec3 = st2_prec3 (n, env_butt_st1, env_prec2)
  
  shm = shared_memory.SharedMemory(create=True, size=env_butt_st1.nbytes)
  shared_arr = np.ndarray(env_butt_st1.shape, dtype=env_butt_st1.dtype, buffer=shm.buf)
  np.copyto(shared_arr, env_butt_st1)
  
  env_prec1 = proceed_para(st2_prec1_para, [n, shm.name], 6)
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec1.nbytes)
  shared_arr2 = np.ndarray(env_prec1.shape, dtype=env_prec1.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec1)
  
  env_prec2 = proceed_para(st2_prec2_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec2.nbytes)
  shared_arr2 = np.ndarray(env_prec2.shape, dtype=env_prec2.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec2)
  
  env_prec3 = proceed_para(st2_prec3_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm.close()
  shm.unlink()
  
  return env_prec3


def ucRPC_butterfly_v2_stage2 (n, p, gamma):
  env_butt_st1 = ucRPC_butterfly_v2_stage1 (n, p, gamma)
  
  #env_prec1 = st2_prec1 (n, env_butt_st1)
  #env_prec2 = st2_prec2 (n, env_butt_st1, env_prec1)
  #env_prec3 = st2_prec3 (n, env_butt_st1, env_prec2)
  
  shm = shared_memory.SharedMemory(create=True, size=env_butt_st1.nbytes)
  shared_arr = np.ndarray(env_butt_st1.shape, dtype=env_butt_st1.dtype, buffer=shm.buf)
  np.copyto(shared_arr, env_butt_st1)
  
  env_prec1 = proceed_para(st2_prec1_para, [n, shm.name], 6)
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec1.nbytes)
  shared_arr2 = np.ndarray(env_prec1.shape, dtype=env_prec1.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec1)
  
  env_prec2 = proceed_para(st2_prec2_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm2 = shared_memory.SharedMemory(create=True, size=env_prec2.nbytes)
  shared_arr2 = np.ndarray(env_prec2.shape, dtype=env_prec2.dtype, buffer=shm2.buf)
  np.copyto(shared_arr2, env_prec2)
  
  env_prec3 = proceed_para(st2_prec3_para, [n, shm.name, shm2.name], 8)
  
  shm2.close()
  shm2.unlink()
  
  shm.close()
  shm.unlink()
    
  return env_prec3

  
#Main
def RPM_butterfly (n, env_butterfly) : 
  eps = 0
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      if (tin1 == n or tin2 == n) :
        eps += env_butterfly[tin1, tin2, 0, 0]
  return eps     

def RPM_butterfly_st2 (n, env_butterfly_st2) : 
  eps = 0
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tin3 in range (n + 1) :
        for tin4 in range (n + 1) :
          if (tin1 == n or tin2 == n or tin3 == n or tin4 == n) :
            eps += env_butterfly_st2[tin1, tin2, tin3, tin4, 0, 0, 0, 0]
  return eps     

def tRPC_butterfly (n, env_butterfly, t) : 
  eps = 0
  for tout1 in range (t + 1) :
    for tout2 in range (t + 1) :  
      tmp = 0
      for tin1 in range (n + 1) :
        for tin2 in range (n + 1) :
          if (tin1 > t or tin2 > t) :
            tmp += env_butterfly[tin1, tin2, tout1, tout2]
      eps = max(eps, min(tmp, 1))
  return eps
  
def tRPC_butterfly_st2 (n, env_butterfly, t) : 
  eps = 0
  for tout1 in range (t + 1) :
    for tout2 in range (t + 1) :
      for tout3 in range (t + 1) :
        for tout4 in range (t + 1) :
          
          tmp = 0
          for tin1 in range (n + 1) :
            for tin2 in range (n + 1) :
              for tin3 in range (n + 1) :
                for tin4 in range (n + 1) :
                  if (tin1 > t or tin2 > t or tin3 > t or tin4 > t) :
                    tmp += env_butterfly[tin1, tin2, tin3, tin4, tout1, tout2, tout3, tout4]
          eps = max(eps, min(tmp, 1))
  return eps   
      


def RPC_add (n, t, env_add) :
  eps = 0
  for tout in range (t + 1) :
    tmp = 0
    for tin1 in range (n + 1) :
      for tin2 in range (n + 1) :
        if (tin1 > t or tin2 > t) :
          tmp += env_add[tin1, tin2, tout]
    eps = max(eps, min(tmp, 1))
  
  return eps

def RPC_cmult (n, t, env_cmult) :
  eps = 0
  for tout in range (t + 1) :
    tmp = 0
    for tin in range (t + 1, n + 1) :
      tmp += env_cmult[tin, tout]
    eps = max(min(tmp, 1), eps)
  return eps
  
def RPC_copy (n, t, env_copy) :
  eps = 0
  for tout1 in range (t + 1) :
    for tout2 in range (t + 1) :
      tmp = 0
      for tin in range (t + 1, n + 1) :
        tmp += env_copy[tin, tout1, tout2]
      eps = max(min(tmp, 1), eps)
  return eps

def RPC_threshold_butterfly_stage1 (n, p, gamma, t) :
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  #env_copy = cardinal_rpc_gcopy_envelope_pgref(n, p, pgref)
  env_copy = cardinal_rpc_gcopy_envelope_pgref_prec(n, pgref)

  eps_copy = RPC_copy(n, t, env_copy)
  eps_add = RPC_add(n, t, env_add)
  eps_cmult = RPC_cmult(n, t, env_cmult) 
  eps = max(eps_add, eps_cmult, eps_copy)
  return 5 * eps
  #return 7 * eps
    
def RPC_threshold_butterfly_stage2 (n, p, gamma, t) :
  pgref = cardinal_rpc_refresh_envelope(n, p, gamma)
  env_add = cardinal_rpc_add_envelope (n, p, pgref)
  env_cmult = cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)
  #env_copy = cardinal_rpc_gcopy_envelope_pgref(n, p, pgref)
  env_copy = cardinal_rpc_gcopy_envelope_pgref_prec(n, pgref)

  eps_copy = RPC_copy(n, t, env_copy)
  eps_add = RPC_add(n, t, env_add)
  eps_cmult = RPC_cmult(n, t, env_cmult)

  
  eps = max(eps_add, eps_cmult, eps_copy)
  return 20 * eps
  #return 28 * epss


#Figure section 4.4, choose between the 3 logp.
def compare_security_stage2 (logp, n_lim):
  p = 2**logp 
  gamma = 200
  eps1 = []
  eps2 = []
  eps3 = []
  eps4 = []
  eps5 = []
  n_values = []
  #n_lim = 11

  path = "./results/butterfly_results_stage2/"

  str_prec1 = path + "eps1_"+str(n_lim - 1)+"_p"+str(logp)+".npy"
  str_prec2 = path + "eps2_"+str(n_lim - 1)+"_p"+str(logp)+".npy"
  str_prec3 = path + "eps3_"+str(n_lim - 1)+"_p"+str(logp)+".npy"
  str_prec4 = path + "eps4_"+str(n_lim - 1)+"_p"+str(logp)+".npy"
  str_prec5 = path + "eps5_"+str(n_lim - 1)+"_p"+str(logp)+".npy"

  if (os.path.isfile(str_prec1)
      and os.path.isfile(str_prec2)
      and os.path.isfile(str_prec3)
      and os.path.isfile(str_prec4)
      and os.path.isfile(str_prec5)) :
    
    eps1 = np.load(str_prec1)
    eps2 = np.load(str_prec2)
    eps3 = np.load(str_prec3)
    eps4 = np.load(str_prec4)
    eps5 = np.load(str_prec5)

    for n in range (3, n_lim) :
      n_values.append(n)
     
  else :
    for n in range (3, n_lim) :
      print("n = ", n)
      n_values.append(n)
      t = n // 2

      cRPC_butterfly = cRPC_butterfly_stage2 (n, p, gamma)
      ucRPC_butterfly = ucRPC_butterfly_stage2 (n, p, gamma)
      cRPC_butterfly_v2 = cRPC_butterfly_v2_stage2 (n, p, gamma)
      ucRPC_butterfly_v2 = ucRPC_butterfly_v2_stage2 (n, p, gamma)   

      eps_cRPC = tRPC_butterfly_st2 (n, cRPC_butterfly, t)
      eps_ucRPC = tRPC_butterfly_st2 (n, ucRPC_butterfly, t)  
      eps_cRPC_v2 = tRPC_butterfly_st2 (n, cRPC_butterfly_v2, t)
      eps_ucRPC_v2 = tRPC_butterfly_st2 (n, ucRPC_butterfly_v2, t)
      eps_tRPC_v2 =  RPC_threshold_butterfly_stage2 (n, p, gamma, t)

      eps1.append(min(1, eps_cRPC))
      eps2.append(min(1, eps_ucRPC))
      eps3.append(min(1, eps_cRPC_v2))
      eps4.append(min(1, eps_ucRPC_v2))
      eps5.append(min(1, eps_tRPC_v2))


    np.save(str_prec1, eps1)
    np.save(str_prec2, eps2)  
    np.save(str_prec3, eps3)  
    np.save(str_prec4, eps4)
    np.save(str_prec5, eps5)  

  fig, ax = plt.subplots(figsize=(5.4, 4), dpi = 300)

  ax.plot(n_values, eps5, marker='x', linestyle='', label="tRPC_v2")
  ax.plot(n_values, eps1, marker = 'x', linestyle='', label="cRPC_v1")
  ax.plot(n_values, eps3, marker = 'x', linestyle='', label="cRPC_v2")
  ax.plot(n_values, eps2, marker = 'x', linestyle='', label="ucRPC_v3")
  ax.plot(n_values, eps4, marker = 'x', linestyle='', label="ucRPC_v4")
  
  ax.set_yscale('log', base=2)
  ax.grid(True)
  ax.invert_yaxis()
  ax.legend()
  ax.set_xlabel("Number of shares")
  ax.set_ylabel(r"Adversary Advantage $\varepsilon$ of the threshold RPC")
  
  fig.tight_layout()
  fig.savefig("BNet"+str(logp)+".pdf", bbox_inches="tight")
  plt.close(fig) 



import itertools
from partitions import process_partitions, precompute_binomial_coefficients


#General-RPC :

def generate_set(n) :  
  elts = list(range(1, n + 1))
  all_subsets = list(itertools.chain.from_iterable(
    itertools.combinations(elts, r) for r in range(len(elts) + 1)
  ))
  return all_subsets

def compute_probas_x_outputs(n, sets_list, cardinal_sets_list, probas, binom_coeffs):
    # probabilities that the internal leakage and the given outputs
    # make it possible to recover x input shares
    nb_sets = 1 << n
    env = np.zeros((nb_sets, nb_sets))
    corres = dict()
  
    for ind_I, I in enumerate(sets_list) :
      corres[str(I)] = ind_I
      for ind_J, J in enumerate(sets_list) :
        if (not set(I).issubset(set(J))) :
          continue
        
        binom_n_t_out = binom_coeffs[(n, len(J))]  
        for ind_cs, cs in enumerate(cardinal_sets_list):    
            # Multihypergeometric law
            # 1. We enumerate all the (k1,k2,k_ell) with ell the number of cardinals cs
                    # ki = number of known outputs in the set of cardinal ci
            valid_combinations = [
                combo for combo in itertools.product(*[range(c + 1) for c in cs])
                if sum(combo) == len(J)
            ]
            for combo in valid_combinations:
              # 2. for each of them, we compute s_k = sum(k_i, tq k_i = c_i)
              s_k = sum(k for k, c in zip(combo, cs) if k == c)
              if (s_k != len(I)) :
                continue
                
              # 3. we update probas_x_outputs[s_k] with probas[ind_cs]*(see multihypergeometric law)
              proba_this_combo = 1
              for i, k_i in enumerate(combo):    
                proba_this_combo *= binom_coeffs[(cs[i],k_i)]
              proba_this_combo = proba_this_combo/(binom_n_t_out * binom_coeffs[(len(J), len(I))])
              
              env[ind_I, ind_J] += probas[ind_cs] * proba_this_combo 

    return env, corres

def env_gen_ref(n, p, gamma) :  
  binom_coeffs = precompute_binomial_coefficients(n)
  cardinal_sets_list, probas, p = process_partitions(n, gamma, p, binom_coeffs)
  sets_list = generate_set(n)
  env1, corres = compute_probas_x_outputs(n, sets_list, cardinal_sets_list, probas, binom_coeffs)
  
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets))
  
  for ind_I, I in enumerate(sets_list) :
    for ind_J, J in enumerate(sets_list) :  
      final_pr = 0
      Z = set(I).intersection(set(J))
      elts = list(Z)
      subsets_I = list(
                    itertools.chain.from_iterable(
                      itertools.combinations(elts, r) for r in range(len(elts) + 1)
                    )
                  )
      for I1 in subsets_I :
        ind_I1 = corres[str(I1)]
        final_pr += p**(len(I) - len(I1)) * (1 - p)**(n - len(I)) * env1[ind_I1, ind_J]
        
      
      env[ind_I, ind_J] = min(final_pr, 1)  

  return env

def env_gen_cmult (n, p, env_ref) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I, I in enumerate(sets_list) :
    for ind_J, J in enumerate(sets_list) :  
      final_pr = 0
      for ind_I1, I1 in enumerate(sets_list) :
        
        if (not set(J).issubset(set(I1))) :
          continue

        final_pr += env_ref[ind_I, ind_I1] * p**(len(I1) - len(J)) * (1 - p)**(n - len(I1))
      env[ind_I, ind_J] = min(1, final_pr)
  return env

def prec_env_add (n, p, env_ref) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I2, I2 in enumerate(sets_list) :
    for ind_J, J in enumerate(sets_list) :
      for ind_I3, I3 in enumerate(sets_list) :
        if (not set(J).issubset(set(I3))) :
          continue

        final_pr = 0
        for ind_I4, I4 in enumerate(sets_list) :
          if (not set(J).issubset(set(I4))) :
            continue          
          final_pr += env_ref[ind_I2, ind_I4] * p**(len(I3) + len(I4) - 2 * len(J)) * (1 - p)**(2 * n - len(I3) - len(I4))
        env[ind_I2, ind_J, ind_I3] = final_pr
  return env          

def env_gen_add (n, p, env_ref) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  prec = prec_env_add (n, p, env_ref) 
  
  for ind_I1, I1 in enumerate(sets_list) :
    for ind_I2, I2 in enumerate(sets_list) :
      for ind_J, J in enumerate(sets_list) :  
        final_pr = 0
        for ind_I3, I3 in enumerate(sets_list) :
          if (not set(J).issubset(set(I3))) :
            continue
        
          final_pr += env_ref[ind_I1, ind_I3] * prec[ind_I2, ind_J, ind_I3]
        env[ind_I1, ind_I2, ind_J] = min(1, final_pr)
  return env

def env_gen_copy (n, p) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I, I in enumerate(sets_list) :
    for ind_J1, J1 in enumerate(sets_list) :
      for ind_J2, J2 in enumerate(sets_list) :
        J1uJ2 = set(J1).union(set(J2))
        sI = set(I)
        #if (sI == J1uJ2) :
        #  env[ind_I, ind_J1, ind_J2] = 1
        for ind_L, L in enumerate(sets_list) :
          LuJ1uJ2 = set(L).union(J1uJ2)  
          if (sI == LuJ1uJ2) :
            env[ind_I, ind_J1, ind_J2] += p**(len(L)) * (1 - p)**(n - len(LuJ1uJ2))
        env[ind_I, ind_J1, ind_J2] = min(1, env[ind_I, ind_J1, ind_J2])  
  return env

"""
def print_env_gen(n, env) :
  sets_list = generate_set(n) 
  strJ="        "
  for ind_J, J in enumerate(sets_list) :
    strJ+= "| " + str(J) + " "
    #for ind_I, I in enumerate(sets_list) : 
  strJ+=" |"
  print(strJ)
  
  for ind_I, I in enumerate(sets_list) :
    strJ="| " + str(I)+" "
    for ind_J, J in enumerate(sets_list) :
      strJ+="| " + f"{env[ind_I, ind_J]:.2e}" +" "
    strJ+= "|"
    print(strJ)
"""       


def env_gen_prec1 (n, env_cmult, env_copy) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I, I in enumerate(sets_list) :
    for ind_J1, J1 in enumerate(sets_list) :
      for ind_J2, J2 in enumerate(sets_list) :
        pr = 0
        
        for ind_I1, I1 in enumerate(sets_list) :
          pr += env_cmult[ind_I, ind_I1] * env_copy[ind_I1, ind_J1, ind_J2]
 
        env[ind_I, ind_J1, ind_J2] = min(1, pr)
        
  return env    

def env_gen_prec2 (n, env_add, env_prec1) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I1, I1 in enumerate(sets_list) :
    for ind_I2, I2 in enumerate(sets_list) :
      for ind_J1, J1 in enumerate(sets_list) :
        for ind_J2, J2 in enumerate(sets_list) : 
          pr = 0
          for ind_tmp1, tmp1 in enumerate(sets_list) :
            pr += env_prec1[ind_I2, ind_tmp1, ind_J2] * env_add[ind_I1, ind_tmp1, ind_J1]
          env[ind_I1, ind_I2, ind_J1, ind_J2] = min(1, pr)
  return env
              
def env_gen_prec3 (n, env_copy, env_prec2):
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  
  for ind_I1, I1 in enumerate(sets_list) :
    for ind_I2, I2 in enumerate(sets_list) :
      for ind_J1, J1 in enumerate(sets_list) :
        for ind_J2, J2 in enumerate(sets_list) :
          for ind_J3, J3 in enumerate(sets_list) :
            pr = 0
            for ind_tmp1, tmp1 in enumerate(sets_list) :
              pr += env_copy[ind_I1, ind_tmp1, ind_J3] * env_prec2[ind_tmp1, ind_I2, ind_J1, ind_J2]
            env[ind_I1, ind_I2, ind_J1, ind_J2, ind_J3] = min(1, pr)
            
  return env  

def env_gen_prec4 (n, env_add, env_prec3) :
  nb_sets = 1 << n
  env = np.zeros((nb_sets, nb_sets, nb_sets, nb_sets))
  sets_list = generate_set(n)
  for ind_I1, I1 in enumerate(sets_list) :
    for ind_I2, I2 in enumerate(sets_list) :
      for ind_J1, J1 in enumerate(sets_list) :
        for ind_J2, J2 in enumerate(sets_list) :
          pr = 0
          for ind_tmp1, tmp1 in enumerate(sets_list) :
            for ind_tmp2, tmp2 in enumerate(sets_list) :
              pr += env_prec3[ind_I1, ind_I2, ind_J1, ind_tmp1, ind_tmp2] * env_add[ind_tmp1, ind_tmp2, ind_J2]
          env[ind_I1, ind_I2, ind_J1, ind_J2] = min(1, pr)
  return env

def gRPC_to_tRPC (n, t, env) :
  nb_sets = 1 << n
  sets_list = generate_set(n)
  eps = 0
  for ind_J1, J1 in enumerate(sets_list) :
    if (len(J1) > t) :
      continue
      
    for ind_J2, J2 in enumerate(sets_list) :
      if (len(J2) > t) :
        continue
      
      tmp = 0
      for ind_I1, I1 in enumerate(sets_list) :
        for ind_I2, I2 in enumerate(sets_list) :
          if (len(I1) > t or len(I2) > t) : 
            tmp += env[ind_I1, ind_I2, ind_J1, ind_J2]
      eps = max(min(1, tmp), eps)
      
  return eps
  
def env_gen_st1(n, p, gamma) :
  env_ref = env_gen_ref(n, p, gamma)
  env_cmult = env_gen_cmult (n, p, env_ref)
  env_add = env_gen_add (n, p, env_ref)
  env_copy = env_gen_copy (n, p)

  env_prec1 = env_gen_prec1 (n, env_cmult, env_copy)
  env_prec2 = env_gen_prec2 (n, env_add, env_prec1)
  env_prec3 = env_gen_prec3 (n, env_copy, env_prec2)
  
  final_env = env_gen_prec4 (n, env_add, env_prec3)
 
  #np.save("st1_gRPC.npy", final_env) 
  return final_env
  
#Figure section 4.4, choose between the 3 logp.
def compare_security_stage1 (logp, n_lim):
  p = 2**logp
  gamma = 200
  
  eps_cRPCv1 = []
  eps_cRPCv2 = []
  eps_ucRPCv3 = []
  eps_ucRPCv4 = []
  eps_gRPC = []
  eps_tRPC = []
  n_values = []
  
  for n in range (2, n_lim) :
    print("n = ", n)
    n_values.append(n)
    t = int(n // 2)
  
    cRPC_butterfly_v1 = cRPC_butterfly_stage1 (n, p, gamma)
    cRPC_butterfly_v2 = cRPC_butterfly_v2_stage1 (n, p, gamma)
    ucRPC_butterfly_v3 = ucRPC_butterfly_stage1 (n, p, gamma)
    ucRPC_butterfly_v4 = ucRPC_butterfly_v2_stage1 (n, p, gamma)
    gen_butterfly = env_gen_st1(n, p, gamma)
  
    eps1 = min(1, RPC_threshold_butterfly_stage1 (n, p, gamma, t))
    eps2 = min(1, tRPC_butterfly (n, cRPC_butterfly_v1, t))
    eps3 = min(1, tRPC_butterfly (n, cRPC_butterfly_v2, t))
    eps4 = min(1, gRPC_to_tRPC (n, t, gen_butterfly)) 
    eps5 = min(1, tRPC_butterfly (n, ucRPC_butterfly_v3, t))  
    eps6 = min(1, tRPC_butterfly (n, ucRPC_butterfly_v4, t))

    eps_tRPC.append(eps1)
    eps_cRPCv1.append(eps2)
    eps_cRPCv2.append(eps3)
    eps_gRPC.append(eps4)
    eps_ucRPCv3.append(eps5)
    eps_ucRPCv4.append(eps6)
  
  fig, ax = plt.subplots(figsize=(5.4, 4), dpi = 300)

  ax.plot(n_values, eps_tRPC, marker = 'x', linestyle='', label="tRPC_v2")
  ax.plot(n_values, eps_cRPCv1, marker = 'x', linestyle='', label="cRPC_v1")
  ax.plot(n_values, eps_cRPCv2, marker = 'x', linestyle='', label="cRPC_v2")
  ax.plot(n_values, eps_gRPC, marker = 'x', linestyle='', label="gRPC_v1")  
  ax.plot(n_values, eps_ucRPCv3, marker = 'x', linestyle='', label="ucRPC_v3")
  ax.plot(n_values, eps_ucRPCv4, marker = 'x', linestyle='', label="ucRPC_v4")

  ax.set_yscale('log', base=2)
  ax.grid(True)
  ax.invert_yaxis()
  ax.legend()
  ax.set_xlabel("Number of shares")
  ax.set_ylabel(r'Adversary Advantage $\varepsilon$ of the threshold RPC security')
  ax.set_xticks(n_values)        # impose que les ticks soient exactement 2,3,4
  ax.set_xticklabels(n_values) 
  #ax.set_title(rf'Random probing security of Butterfly Network, $p=2^{{{logp}}}$')
  fig.tight_layout()
  fig.savefig("st1_BNet"+str(logp)+".pdf", bbox_inches="tight")
  plt.close()  


def test ():
  n = 3
  p= 2**-5

  env_copy_card = cRPC_copy(n, p)
  env_copy_gen = env_gen_copy(n, p)


  for I_card in range (n + 1) :
    for J1_card in range (n + 1) :
      for J2_card in range (n + 1) : 
        pb = 0
        sets_list = generate_set(n)
        for ind_J1, J1 in enumerate(sets_list) :
          for ind_J2, J2 in enumerate(sets_list) :
            pbJ1J2 = 0
            for ind_I, I in enumerate(sets_list) :
              if(len(I) == I_card and len(J1) == J1_card and len(J2) == J2_card) :
                pbJ1J2 += env_copy_gen[ind_I, ind_J1, ind_J2]
            pb = max(pb, pbJ1J2)
  
        if (pb > env_copy_card[I_card, J1_card, J2_card]) :
          print("I_card = ", I_card)
          print("J1_card = ", J1_card)
          print("J2_card = ", J2_card)
          print("pb = ", pb)
          print("env_copy_card = ", env_copy_card[I_card, J1_card, J2_card])
          print()



  

if __name__ == "__main__":
  print("main")
  test()
  #compare_security_stage1(-10, 5)


