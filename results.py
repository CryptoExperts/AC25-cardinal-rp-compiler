#!/usr/bin/env python3

# This file contains the results that are exhibited in the paper  
# 
# "Masked Circuit Compiler in the Cardinal Random Probing Composability 
# Framework"
# 
# Each function is responsible for one or many graphs of the paper and will be 
# highlighted above the function. 

################################## Packages ####################################

import numpy as np
from math import log
import os
import matplotlib.pyplot as plt
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

from AES import (compute_RPC_AES, optimize_gamma_ark, optimize_gamma_mc, 
optimize_gamma_sb, optimize_gamma_l_sb, compare_cRPC_tRPC_AES)
from complexity import (comp_AES_enc, comp_AES_enc_JMB24, comp_AES_enc_bfo23,
                        graph_complexity, comp_mult, comp_mult_JMB24, 
                        comp_mult_bfo23, nbRandBits)
from mult_4card import (compute_envn_mult_4card, compute_RPC_threshold, 
                        compute_graph)
from mult_uni_4card import cardinal_rpc_mult_uni_envelopes
from mult_gen import (find_plateau_RPM_n, compute_envn_mult, 
                      compute_RPM_threshold)
from butterfly_network import compare_security_stage1, compare_security_stage2


################################# Functions ####################################


"""  
Function to obtain graph of Figure 4.
:param p_values: List of probability leakage rates.
:n_lim: A threshold value of number of shares n.
:return : A graph for each probability leakages rates of the advantage of the 
Threshold RPC security for different versions of the stage 1 of the butterfly 
network, according to the number of shares n until a threshold |n_lim|.
"""
def graph_bnet_st1(p_values, n_lim):
  for p in p_values :
    logp = int(log(p, 2))
    compare_security_stage1(logp, n_lim)

"""  
Function to obtain graph of Figure 16.
:param p_values: List of probability leakage rates.
:n_lim: A threshold value of number of shares n.
:return : A graph for each probability leakages rates of the advantage of the 
Threshold RPC security for different versions of the stage 2 of the butterfly 
network, according to the number of shares n until a threshold |n_lim|.
"""
def graph_bnet_st2(p_values, n_lim):
  for p in p_values :
    logp = int(log(p, 2))
    compare_security_stage2(logp, n_lim)  


"""
Function to obtain graph of Figure 7.
:param n_values: List of number of shares.
:param p_values: List of probability leakage rates.
:lim_gamma: A threshold on the gamma values used in the refresh gadget.
:return : A graph, for each probability leakage rates p in |p_values|, on the 
Threshold RPC obtained by the multiplication gadget according to the number of 
gamma used and the number of shares n.
"""
def NRSM_4card_graph (n_values, p_values, lim_gamma) :
  list_gamma = [i for i in range(lim_gamma)]
  for p in p_values :   
    logp = int(log(p, 2))
    
    fig, ax = plt.subplots(figsize=(5.4, 4), dpi = 1200)  
    ax.set_xlabel(r"Number of random values $\gamma$")
    ax.set_ylabel("RPC (n,t,p)")
    ax.set_yscale('log', base=2)
    ax.set_xlim(0, lim_gamma - 1)
    ax.invert_yaxis()
    ax.grid(True)
    
    for n in n_values :
      print("n = ", n)
      t = n // 2

      path = "./results/gamma_func_4card/"
      filename_asym = (path + "Asym_4card_eps_n" + str(n) + "_p"+str(logp) + 
                       "_limgamma" + str(lim_gamma) + ".npy")
      filename_sym = (path + "Sym_4card_eps_n" + str(n) + "_p" + str(logp) + 
                      "_limgamma" + str(lim_gamma) + ".npy")
      filename_sym_unif = (path + "Sym_Uni_4card_eps_n" + str(n) + "_p" + 
                           str(logp) + "_limgamma" + str(lim_gamma) + ".npy")

      eps_asym_4card = []
      eps_sym_4card = []
      eps_sym_unif_4card = []

      if (os.path.isfile(filename_asym) and os.path.isfile(filename_sym) and 
          os.path.isfile(filename_sym_unif)) :
        eps_asym_4card = np.load(filename_asym)
        eps_sym_4card = np.load(filename_sym)
        eps_sym_unif_4card = np.load(filename_sym_unif)
      
      else :
        for gamma in range (lim_gamma) :
          print("gamma = ", gamma)
          l_gamma = [gamma] * (n + 1) 
    

          #Computation of the different enveloppes.
          env_asym_4card = compute_envn_mult_4card(n, p, gamma, "Asym")
          env_sym_4card = compute_envn_mult_4card(n, p, gamma, "Sym")
          env_sym_unif_4card = cardinal_rpc_mult_uni_envelopes(n, p, l_gamma, 
                                                         gamma, "Sym")
    
          #Obtaining advantage of the threshold RPC from the enveloppes.
          eps_asym_4card.append(min(1, 
                         compute_RPC_threshold(n, env_asym_4card, t)))
          
          eps_sym_4card.append(min(1, 
                        compute_RPC_threshold(n, env_sym_4card, t)))
          
          eps_sym_unif_4card.append(min(1, 
                             compute_RPC_threshold(n, env_sym_unif_4card, t)))
  
        np.save(filename_asym, eps_asym_4card)
        np.save(filename_sym, eps_sym_4card)
        np.save(filename_sym_unif, eps_sym_unif_4card)

      #Computation of the graphs.
      compute_graph(n, t, p, eps_asym_4card, list_gamma, "Asym")
      compute_graph(n, t, p, eps_sym_4card, list_gamma, "Sym")
      compute_graph(n, t, p, eps_sym_unif_4card, list_gamma, "Unif")

    fig.tight_layout()
    fig.savefig("NRSM_4card_TresholdRPC_p"+str(logp)+".pdf", 
                bbox_inches="tight")  
    plt.close(fig)

"""
Function to obtain graph of Figure 9.
:param p_values: List of probability leakage rates.
:param l_sec_level: List of security levels.
:return : A graph, for each leakage rates in |p_values|, of the complexity (in 
random, addition and multiplication) required to reach the different 
security level of the list |l_sec_lev| of the Random probing security of the 
multiplication gadget.
"""
def histo_mult_complexity (p_values, l_sec_level) :
  #If you want, you can go further in the range of n, but it will take more time.
  lim_n = 19

  #List of gamma values considered "optimal" for the refresh gadget (see Section 
  #5.3 of the paper, paragraph "Fine-tuning of the RPRefresh gadget"), valid for 
  #all leakage rates p.
  gamma_n = find_plateau_RPM_n(p_values, lim_n)
  for p in p_values :
    i = p_values.index(p)
    l_gamma_mult = gamma_n[i].astype(int)
    logp = int(log(p, 2))

    #Our complexity in Randoms, additions, multiplications to reach the desired 
    #security levels.
    c_rand = []
    c_add = []
    c_mult = []
    c_randbit = []

    #JMB24's complexity in Randoms, additions, multiplications to reach the 
    #desired security levels.
    c_rand_JMB24 = []
    c_add_JMB24 = []
    c_mult_JMB24 = []

    #BFO23's complexity in Randoms, additions, multiplications to reach the 
    #desired security levels.
    c_rand_BFO23 = []
    c_add_BFO23 = []
    c_mult_BFO23 = []

    security = []

    for sec_level in l_sec_level :
      security.append(str(-1  * int(log(sec_level, 2))))
      path = "./results/mult/"
      filename = (path + "mult_p" + str(logp) + "_seclev" + 
                  str(int(log(sec_level, 2))) + ".npy")
      n = 2
      eps = 1

      #For the figures, we have precomputed the values.
      if (os.path.isfile(filename)) :
        results = np.load(filename)
        n = int(results[0])
        eps = results[1]
      
      else : 
        t = n // 2
        filename_mult = (path + "env_mult_n"+str(n) + "_p" + str(logp) + 
                         "_lgamma" + str(l_gamma_mult) + ".npy")
        env_mult = None
        #For the figures, we have precomputed the enveloppes for the 
        #multiplication gadget.
        if (os.path.isfile(filename_mult)) :
          env_mult = np.load(filename_mult)
        else :
          ######################################################################
          ################## Computation of the enveloppes #####################
          env_mult = compute_envn_mult(n, p, l_gamma_mult, l_gamma_mult[n])
          np.save(filename_mult, env_mult)
        
        ########################################################################
        #################### Compuatation of the RPM advantage #################
        eps = compute_RPM_threshold(n, env_mult)  
        
        #Continue until we obtain an adequate advantage, 
        #increasing the number of shares n.
        while (eps > sec_level) :
          n += 1
          filename_mult = (path + "env_mult_n"+str(n) + "_p" + str(logp) + 
                           "_lgamma" + str(l_gamma_mult) + ".npy")
          t = n // 2
          env_mult = compute_envn_mult(n, p, l_gamma_mult, l_gamma_mult[n])
          if (os.path.isfile(filename_mult)) :
            env_mult = np.load(filename_mult)
          else :
            env_mult = compute_envn_mult(n, p, l_gamma_mult, l_gamma_mult[n])
            np.save(filename_mult, env_mult)
          eps = compute_RPM_threshold(n, env_mult)        
        np.save(filename, [n, eps])
  
      ##########################################################################
      ##################### Computation of the complexity ######################
      crand, cadd, cmult, crandbit = comp_mult(n, l_gamma_mult, l_gamma_mult[n])
      crandbits = (crandbit + nbRandBits(n)) // 8
   
      c_rand.append(crand)
      c_randbit.append(crandbits)
      c_add.append(cadd)
      c_mult.append(cmult)
    
      ##########################################################################
      ###################### JMB24's multiplication gadget #####################
      
      n_JMB24 = 2
      while ((2 * p)**(0.3 * n_JMB24) > sec_level and n < 31) :
        n_JMB24 += 1
      crand, cadd, cmult = comp_mult_JMB24(n_JMB24)
      if(n_JMB24 < 31) :
        c_rand_JMB24.append(crand)
        c_add_JMB24.append(cadd)
        c_mult_JMB24.append(cmult)
      else :
        c_rand_JMB24.append(np.nan)
        c_add_JMB24.append(np.nan)
        c_mult_JMB24.append(np.nan)

      ##########################################################################
      ###################### BFO23's multiplication gadget #####################
      n_BFO23 = 2
      p1 = 1 - (1 - p)**(8 * n_BFO23)
      p2 = 1 - (1 - (3 * p)**(1/2))**(n_BFO23 - 1)
      eps = (p1 + p2)**n_BFO23
      while (eps > sec_level and n_BFO23 <= 1000) :
        n_BFO23 += 1
        p1 = 1 - (1 - p)**(8 * n_BFO23)
        p2 = 1 - (1 - (3 * p)**(1/2))**(n_BFO23 - 1)
        eps = min((p1 + p2)**n_BFO23, eps)


      if (n_BFO23 <= 1000) :
        crand, cadd, cmult = comp_mult_bfo23(n_BFO23)

        c_rand_BFO23.append(crand)
        c_add_BFO23.append(cadd)
        c_mult_BFO23.append(cmult)
      else :
        c_rand_BFO23.append(np.nan)
        c_add_BFO23.append(np.nan)
        c_mult_BFO23.append(np.nan)

    ############################################################################
    ##################### Graph Computation for p ##############################
    graph_complexity(p, logp, l_sec_level, c_rand, c_add, c_mult, c_randbit, 
                     c_rand_JMB24, c_add_JMB24, c_mult_JMB24, c_rand_BFO23, 
                     c_add_BFO23, c_mult_BFO23, "histo_p" +str(logp) +".pdf", 
                     security)  







"""
Function to obtain graph of Figure 14.
:param p_values: List of probability leakage rates.
:param l_sec_level: List of security levels.
:return : A graph, for each leakage rates in |p_values|, of the complexity (in 
random, addition and multiplication) required to reach the different 
security level of the list |l_sec_lev| of the Threshold RPC security of the 
AES for our version, JMB24's version and BFO23's version.
"""
def histo_AES_complexity (p_values, l_sec_level) :
  for p in p_values :
    logp = int(log(p, 2))  
  
    #Our complexity in Randoms, additions, multiplications to reach the desired 
    #security levels.
    c_rand = []
    c_add = []
    c_mult = []
    c_randbit = []

    #JMB24's complexity in Randoms, additions, multiplications to reach the 
    #desired security levels.
    c_rand_JMB24 = []
    c_add_JMB24 = []
    c_mult_JMB24 = []

    #BFO23's complexity in Randoms, additions, multiplications to reach the 
    #desired security levels.
    c_rand_BFO23 = []
    c_add_BFO23 = []
    c_mult_BFO23 = []

    security = []

    for sec_level in l_sec_level :
      security.append(str(-1  * int(log(sec_level, 2))))

      #################### Determinations of the parameters ####################
      #################### to use for reach the security    ####################
      #################### level for my version             ####################

      path = "./results/AES/"
      str_params = (path + "param_AES_p" + str(logp) + "_seclev" + 
                    str(int(log(sec_level, 2))) + ".npy")
      str_l_g_sb = (path + "l_g_sb_p" + str(logp) + "_seclev" + 
                    str(int(log(sec_level, 2))) + ".npy")
    
      n = 2
      t = n // 2
      gamma_mc = 500
      gamma_ark = 500
      gamma_sb = 500
      l_gamma_sb = [500] * n
      eps = 0

      if (os.path.isfile(str_params) and os.path.isfile(str_l_g_sb)) : 
        params = np.load(str_params)
        n = params[0]
        gamma_ark = params[1]
        gamma_mc = params[2]
        gamma_sb = params[3]
        eps = params[4]
        l_gamma_sb = np.load(str_l_g_sb)
      

      else :    
        eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                              gamma_ark, t)
    
        while (eps > sec_level) :
          n += 1
          t = n // 2
          l_gamma_sb.append(500)
          eps = compute_RPC_AES(n, p, logp, gamma_sb, l_gamma_sb, gamma_mc, 
                                gamma_ark, t)
      
        eps, gamma_ark = optimize_gamma_ark(n,p, eps, 0.1)
        eps, gamma_mc = optimize_gamma_mc (n, p, eps, 0.1, gamma_ark)
        eps, gamma_sb = optimize_gamma_sb(n, p, eps, 0.1, gamma_ark, gamma_mc)
        eps, l_gamma_sb = optimize_gamma_l_sb(n, p, eps, 0.1, gamma_ark, 
                                              gamma_mc, gamma_sb)
      
        params = [n, gamma_ark, gamma_mc, gamma_sb, eps]
        np.save(str_params, params)
        np.save(str_l_g_sb, l_gamma_sb)

      n = int(n)
      print("n = ", n)
      print("gamma_ark = ", gamma_ark)
      print("gamma_mc = ", gamma_mc)
      print("gamma_sb = ", gamma_sb)
      print("l_gamma_sb = ", l_gamma_sb)
      print("log(eps, 2) = ", log(eps, 2))

      ##########################################################################
      ######################## Complexity computation ##########################

      crand, cadd, cmult, crand_bit  = comp_AES_enc(n, gamma_ark, l_gamma_sb, 
                                                    gamma_sb, gamma_mc)
    
      c_rand.append(crand)
      c_add.append(cadd)
      c_mult.append(cmult)
      c_randbit.append(crand_bit // 8)

      ##########################################################################
      #################### Determinations of the parameters ####################
      #################### to use for reach the security    ####################
      #################### level for JMB24's version        ####################

      n_JMB24 = 2
      while (4 * (5.3 * p)**(0.3 * n_JMB24) > sec_level) :      
        n_JMB24 += 1

      ##########################################################################
      ######################## Complexity computation ##########################
      crand, cadd, cmult = comp_AES_enc_JMB24(n_JMB24)

      c_rand_JMB24.append(crand)
      c_add_JMB24.append(cadd)
      c_mult_JMB24.append(cmult)

      ##########################################################################
      #################### Determinations of the parameters ####################
      #################### to use for reach the security    ####################
      #################### level for BFO23's version        ####################

      n_BFO23 = 2
      p1 = 1 - (1 - p)**(8 * n_BFO23)
      p2 = 1 - (1 - (3 * p)**(1/2))**(n_BFO23 - 1)
      eps = (18 * p + 2 * (p1 + p2))**n_BFO23
      while (eps > sec_level and n_BFO23 <= 100) :
        n_BFO23 += 1
        p1 = 1 - (1 - p)**(8 * n_BFO23)
        p2 = 1 - (1 - (3 * p)**(1/2))**(n_BFO23 - 1)
        eps = (18 * p + 2 * (p1 + p2))**n_BFO23
    
    
      ##########################################################################
      ######################## Complexity computation ########################## 
      if (eps <= sec_level) :
        crand, cadd, cmult = comp_AES_enc_bfo23(n_BFO23)
        c_rand_BFO23.append(crand)
        c_add_BFO23.append(cadd)
        c_mult_BFO23.append(cmult)
      else :
        c_rand_BFO23.append(np.nan)
        c_add_BFO23.append(np.nan)
        c_mult_BFO23.append(np.nan)

    graph_complexity(p, logp, l_sec_level, c_rand, c_add, c_mult, c_randbit, 
                     c_rand_JMB24, c_add_JMB24, c_mult_JMB24, c_rand_BFO23, 
                     c_add_BFO23, c_mult_BFO23, 
                     "AES_histo_p" +str(logp) +".pdf", security)
  


def complexity_cRPCvstRPC (n, p, thr) :
  logp = int(log(p, 2))
  res_tuple = compare_cRPC_tRPC_AES(n, p, thr)
  
  eps_tRPC = res_tuple[0]
  gamma = res_tuple[1]
  l_gamma_mult = res_tuple[2]

  cr_trpc, ca_trpc, cm_trpc, crb_trpc = comp_AES_enc(n, gamma, l_gamma_mult, 
                                                     gamma, gamma)

  eps_cRPC = res_tuple[3]
  gamma_ark = res_tuple[4]
  gamma_mc = res_tuple[5]
  gamma_sb = res_tuple[6]
  l_gamma_sb = res_tuple[7]

  cr_crpc, ca_crpc, cm_crpc, crb_crpc  = comp_AES_enc(n, gamma, l_gamma_mult, 
                                                       gamma, gamma_mc)

  fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
  
  x1 = int(log(eps_tRPC, 2))
  x2 = int(log(eps_cRPC, 2))
  w = 1
  
  
  ax[0].bar(x1, cr_trpc, width = w, label = "tRPC", color = 'C2')  
  ax[0].bar(x1, crb_trpc, bottom = cr_trpc, label = "tRPC - rb", 
            color = 'lightgreen', width = w)
  ax[0].bar(x2, cr_crpc, width = w, label = "cRPC", color = 'C0')
  ax[0].bar(x2, crb_crpc, bottom  = cr_crpc, width = w, label = "cRPC - rb", 
            color = 'skyblue')
  
  
  ax[0].set_ylabel("Number of randoms")
  ax[0].set_xlabel("Security Level")
  ax[0].set_title(r'\#Rand')
  ax[0].legend(fontsize = "x-small")

  ax[0].set_xticks([x1, x2]) 

  #ax[0].set_yscale('log', base=2) 

  #ax[0].set_xticks(x) 
  #ax[0].set_xticklabels(security)

  ax[1].bar(x1, ca_trpc, width = w, label = "tRPC", color = 'C2')  
  ax[1].bar(x2, ca_crpc, label = "cRPC", width = w, color = 'C0')
  ax[1].set_ylabel("Number of additions")
  ax[1].set_xlabel("Security Level")
  ax[1].set_title(r'\#Add')
  ax[1].legend(fontsize = "small")  
  
  ax[1].set_xticks([x1, x2]) 

  #ax[1].set_yscale('log', base=2) 
  #ax[1].set_xticks(x)  
  #ax[1].set_xticklabels(security)
   

  ax[2].bar(x1, cm_trpc, width = w, label = "tRPC", color = 'C2')  
  ax[2].bar(x2, cm_crpc, label = "cRPC", width = w, color='C0')
  ax[2].set_ylabel("Number of multiplications")
  ax[2].set_xlabel("Security Level")
  ax[2].set_title(r'\#Mult')
  ax[2].legend(fontsize = "small")  
  
  #ax[2].set_yscale('log', base=2) 
  ax[2].set_xticks([x1, x2]) 
  #ax[2].set_xticklabels(security)  
  
  fig.tight_layout()
  fig.savefig("cRPCvstRPC_n"+str(n)+"_p"+str(logp)+".pdf", bbox_inches="tight")
  plt.close(fig)
  


if __name__ == "__main__" :
  
  #Figure 4
  graph_bnet_st1([2**-3, 2**-5, 2**-10], 5)

  #Figure 16
  #graph_bnet_st2([2**-3, 2**-5, 2**-10], 11)

  #Figure 7 
  NRSM_4card_graph ([4, 8], [2**-6, 2**-12], 41)
  
  #Figure 8
  #gamma_n_p = find_plateau_RPM_n([2**-10, 2**-15, 2**-20], 19)
  #i = 0 
  #for p in [2**-10, 2**-15, 2**-20] :
  #  print("p = 2^"+ str(int(log(p, 2))) + ", gamma_n = " + str(gamma_n_p[i]))
  #  i += 1

  #Figure 9 
  histo_mult_complexity ([2**-10], [2**-64, 2**-80])
  histo_mult_complexity([2**-15, 2**-20], [2**-64, 2**-80, 2**-128])

  #Figure 14 
  histo_AES_complexity([2**-20], [2**-64, 2**-80, 2**-128])
  histo_AES_complexity ([2**-16], [2**-64, 2**-80])

  #Last paragraph of the core paper.
  #complexity_cRPCvstRPC (8, 2**-20, 0.1)



