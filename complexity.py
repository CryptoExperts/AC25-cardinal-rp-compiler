#!/usr/bin/env python3

################################################################################
################################ Packages ######################################
import numpy as np
from math import log, ceil
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


################################################################################
########################### Permutations Complexity ############################

def nbP(nx, ny, k) :
  """
  Args:
    nx (int): Number of shares used for the first secret variable 'x'.
    ny (int): Number of sahres used for the second secret variable 'y'.
    k (int) : An integer > 1
  
  Returns: 
    The number of permutations of size k used in UnifMatMult (appendix H, 
    algorithm 12 of the paper).

  Notes:
    Explore recursively all invocations of UnifMatMult and check when nx or ny is 
    equal to k. This corresponds to the use of permutations of size k.
  """
  
  if (k > max (nx, ny)) :
    return 0
  
  nb = 0
  
  nxhl = nx // 2
  nxhr = nx - nxhl
  nyhl = ny // 2
  nyhr = ny - nyhl
  
  #Base case
  if (k == nxhl or k == nxhr) :
    nb += 2
  
  if (k == nyhl or k == nyhr) :
    nb += 2
  
  #Recursive Call
  nb += nbP(nxhl, nyhl, k) + nbP(nxhr, nyhl, k) + nbP(nxhl, nyhr, k) + nbP(nxhr, nyhr, k)
  
  return nb


def comp_perm (n) :
  """
  Args:
    n (int):  the size of the permutation.

  Returns:  
    The number of random bits necessary to generate the permutation.

  Notes: 
    Compared to the |Knuth-Yates| shuffling presented in Algorithm 10 of the 
    paper, we omit the rejection that can happened when we draw an uniform in a 
    range {1, ..., i} where i is not necessary a power of two.
  """
  
  nb_rand_bits = 0 
  for l in range (2, n  + 1) :
     nb_rand_bits += int(np.ceil(log(l, 2)))
  return nb_rand_bits

def nbRandBits (n) :
  """
  Args:
    n (int): Number of shares.
  
  Returns: 
    Compute the number of random bits necessary when we apply |UnifMatMult| 
    with |n| shares.
  """  
  
  nb = 0
  for k in range (2, n + 1) :
    cperm_k = comp_perm(k)
    nb += cperm_k * nbP(n, n, k)
  return nb
  
#******** Complexity of the multiplication ***************

def comp_ref(n, gamma) :
  """
  Compute the complexity (random, addition) of RPRefresh.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns: 
    The random field complexity, the addition complexity and the random bits 
    complexity.
  
  Notes:
    If |gamma| is equal to zero, then there is no refresh gadget used.
  """

  if (n == 1) :
    return 0, 0, 0
  
  nb_rand = gamma
  nb_add = 2 * gamma + n
  if (gamma == 0) :
    nb_add = 0

  nb_rand_bit =  gamma * (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_rand_bit
  
def comp_add (n, gamma) :
  """
  Compute the complexity (random, addition) of Addition gadget of the compiler.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns:
    The random field complexity, the addition complexity and the random bits 
    complexity.

  Notes:
    If |gamma| is equal to zero, then there is no refresh gadget used.
  """

  if (n == 1) :
    return 0, n, 0
  
  
  nb_rand = 2 * gamma
  nb_add = 4 * gamma + 3 * n
  if (gamma == 0) :
    nb_add = 2 * n
  nb_rand_bit =  2 * gamma *  (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_rand_bit

def comp_cadd (n, gamma) :
  """
  Compute the complexity (random, addition) of the Addition by a constant gadget
  of the compiler.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns: 
    The random field complexity, the addition complexity and the random bits 
    complexity.
  
  Notes:
    If |gamma| is equal to zero, then there is no refresh gadget used.
  """
  
  if (n == 1) :
    return 0, 1, 0
  
  nb_rand = gamma
  nb_add = 2 * gamma + n + 1
  if (gamma == 0) :
    nb_add = 1
  nb_rand_bit =  gamma *  (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_rand_bit
  
def comp_cmult (n, gamma):
  """
  Compute the complexity (random, addition, multiplication) of the 
  Multiplication by a constant gadget of the compiler.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns: 
    The random, addition and multiplication complexity.
  """

  if (n == 1) :
    return 0, 0, 1, 0
  
  nb_rand = gamma
  nb_add = 2 * gamma + n
  if(gamma == 0) :
    nb_add = 0
  nb_mult = n
  nb_rand_bit =  gamma *  (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_mult, nb_rand_bit


def comp_matmult (nx, ny, l_gamma) : 
  """
  Compute the complexity (random, addition, multiplication) of MatMultSym 
  (Algorihtm 11 of the paper) of the compiler.
  
  Args:
    nx (int): Number of shares of the first value x.
    ny (int): Number of shares of the second secret value y.
    l_gamma(list): List of gamma used in RPZeroEnc according to the number of 
                  shares.
  
  Returns:
    The random, addition, multiplication complexity.
  """
  
  if nx == 1 and ny == 1 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 1
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 2 and ny == 1 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 2
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 1 and ny == 2 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 2
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 2 and ny == 2 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 4
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
    
  nxhl = nx // 2
  nxhr = nx - nxhl
  nyhl = ny // 2
  nyhr = ny - nyhl
  
  nb_rand1, nb_add1, tmp, nb_rand_bit1 = comp_matmult(nxhl, nyhl, l_gamma)
  nb_rand2, nb_add2, tmp, nb_rand_bit2 = comp_matmult(nxhr, nyhl, l_gamma)
  nb_rand3, nb_add3, tmp, nb_rand_bit3 = comp_matmult(nxhl, nyhr, l_gamma)
  nb_rand4, nb_add4, tmp, nb_rand_bit4 = comp_matmult(nxhr, nyhr, l_gamma)
  
  nb_rand_rec = nb_rand1 + nb_rand2 + nb_rand3 + nb_rand4
  nb_add_rec = nb_add1 + nb_add2 + nb_add3 + nb_add4
  nb_rand_bit_rec = nb_rand_bit1 + nb_rand_bit2 + nb_rand_bit3 + nb_rand_bit4
  
  nb_rand5, nb_add5, nb_rand_bit5 = comp_ref(nxhl, l_gamma[nxhl])
  nb_rand6, nb_add6, nb_rand_bit6 = comp_ref(nxhr, l_gamma[nxhr])
  nb_rand7, nb_add7, nb_rand_bit7 = comp_ref(nyhl, l_gamma[nyhl])
  nb_rand8, nb_add8, nb_rand_bit8 = comp_ref(nyhr, l_gamma[nyhr])
  
  nb_rand_ref = 2 * (nb_rand5 + nb_rand6 + nb_rand7 + nb_rand8)
  nb_add_ref = 2 * (nb_add5 + nb_add6 + nb_add7 + nb_add8)
  nb_rand_bit_ref = 2 * (nb_rand_bit5 + nb_rand_bit6 + nb_rand_bit7 + nb_rand_bit8)
  
  
  nb_rand = nb_rand_ref + nb_rand_rec
  nb_add = nb_add_ref + nb_add_rec
  nb_mult = nx * ny
  nb_rand_bit = nb_rand_bit_ref + nb_rand_bit_rec
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit

def comp_tree_add (k, n, gamma) :
  """
  Compute the complexity (random, addition) of TreeAdd of the compiler.
  
  Args:
    k (int): Number of n-sharing to add together.
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns:
    The random, addition complexity.
  """

  if k == 1 :
    return 0, 0, 0
  
  if k == 2 :
    nb_rand, nb_add, nb_rand_bit = comp_add(n, gamma)
    return nb_rand, nb_add, nb_rand_bit
  
  #k > 2
  khdo = k // 2
  khup = k - khdo
  
  nb_rand1, nb_add1, nb_rand_bit1 = comp_tree_add(khdo, n, gamma)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(khup, n, gamma)
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, gamma)
  
  nb_rand = nb_rand1 + nb_rand2 + nb_rand_add
  nb_add = nb_add1 + nb_add2 + nb_add_add
  nb_rand_bit = nb_rand_bit1 + nb_rand_bit2 + nb_rand_bit_add
  return nb_rand , nb_add, nb_rand_bit

def comp_mult (n, l_gamma, gamma) :
  """
  Compute the complexity (random, addition, multiplication) of CardSecMult.
  
  Args:
    n (int): Number of shares
    l_gamma (list): List of gamma used in MatMultSym according to the number of 
                    shares.
    gamma (int): gamma used in TreeAdd.
  
  Returns:
    The random, addition, multiplication complexity.
  """

  nb_rand, nb_add, nb_mult, nb_rand_bit = comp_matmult(n, n, l_gamma)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(n, n, gamma)
  
  nb_rand_fin = nb_rand + nb_rand2
  nb_add_fin = nb_add + nb_add2
  nb_mult_fin = nb_mult
  nb_rand_bit_fin = nb_rand_bit + nb_rand_bit2  
  return nb_rand_fin, nb_add_fin, nb_mult_fin, nb_rand_bit_fin
  
def comp_mult_unif (n, l_gamma, gamma) :
  """
  Compute the complexity (random, addition, multiplication) of CardSecMult.
  
  Args:
    n (int): Number of shares
    l_gamma (list): List of gamma used in UnifMatMult according to the number of 
                    shares.
    gamma (int): gamma used in TreeAdd.
  
  Returns
    The random, addition, multiplication complexity.
  """

  #Warning : It gives us the number of random bits of MatMultSym, it lacks the 
  # random bits used in the permutation. This is why we add |nbRandBits(n)| in 
  # the following.
  nb_rand, nb_add, nb_mult, nb_rand_bit = comp_matmult(n, n, l_gamma)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(n, n, gamma)
  
  nb_rand_fin = nb_rand + nb_rand2
  nb_add_fin = nb_add + nb_add2
  nb_mult_fin = nb_mult
  nb_rand_bit_fin = nb_rand_bit + nb_rand_bit2 + nbRandBits(n) 
  return nb_rand_fin, nb_add_fin, nb_mult_fin, nb_rand_bit_fin
  


def comp_addrk (n, gamma_ark) :
  """
  Compute the complexity of the step AddRoundKey in AES, with input the state of
  16 bytes where each byte is masked by n shares.
  
  Args:
    n (int): The number of shares.
    gamma_ark (int): The gamma used in the refresh of the step AddRoundKey.
  
  Returns: 
    The random, addition complexity of AddRoundKey.
  """

  nb_rand, nb_add, nb_rand_bit = comp_add(n, gamma_ark)
  nb_rand_fin = 16 * nb_rand
  nb_add_fin = 16 * nb_add
  nb_rand_bit_fin = 16 * nb_rand_bit
  
  return nb_rand_fin, nb_add_fin, nb_rand_bit_fin

def comp_expo_sb (n, l_gamma_sb, gamma_sb) :
  """
  Compute the complexity of the exponentiation in the SubBytes step of AES, for 
  a single masked byte with n shares.
  
  Args:
    n (int) : Number of shares.
    l_gamma_sb (list): List of gamma for UnifMatMult in CardSecMult.
    gamma_sb (int): The gamma used in the linear step of the exponentiation 
                   (including TreeAdd)
  Returns: 
    Complexity of the Exponentiation of the SubBytes step for a single byte.
  """

  #There is in the Exponentiation gadget (see Figure 11 of the paper):
  # - 2 refresh gadgets.
  # - 7 squaring gadgets (which can be seen as multiplication by a constant 
  #   gadget in F_{256})
  # - 4 permutations.
  # - 4 multiplication gadgets.

  nb_rand_ref, nb_add_ref,  nb_rand_bit_ref = comp_ref(n, gamma_sb)
  nb_rand_sq, nb_add_sq, nb_mult_sq, nb_rand_bit_sq = comp_cmult (n, gamma_sb)
  nb_rand_mult, nb_add_mult, nb_mult_mult, nb_rand_bit_mult = (
    comp_mult_unif(n, l_gamma_sb, gamma_sb))
  
  nb_rand = 4 * nb_rand_mult + 7 * nb_rand_sq + 2 * nb_rand_ref
  nb_add = 4 * nb_add_mult + 7 * nb_add_sq
  nb_mult = 4 * nb_mult_mult + 7 * nb_mult_sq
  nb_rand_bit = (4 * nb_rand_bit_mult + 7 * nb_rand_bit_sq + 
                 4 * comp_perm(n) + 2 * nb_rand_bit_ref)
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  
def comp_aff_sb (n, gamma_sb) :
  """
  Compute the complexity of the affine function in the SubBytes step of AES, for
  a single masked byte with n shares.
  
  Args:
    n (int): Number of shares
    gamma_sb (int): The gamma used in the affine function for the refresh gadget.
  
  Returns: 
    Complexity of the affine function of the SubBytes step for a single byte.
  """
  
  #There is in the Affine gadget (see Figure 12 of the paper):
  # - 6 refresh gadgets.
  # - 7 multiplication by a constant gadgets.
  # - 7 squaring gadgets (which can be seen as multiplication by a constant 
  #   gadget in F_{256})
  # - 7 additions gadgets.
  # - 7 permutations.
  # - 1 addition by a constant gadget.

  
  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n, gamma_sb)
  nb_rand_cadd, nb_add_cadd, nb_rand_bit_cadd = comp_cadd(n, gamma_sb)  
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, gamma_sb)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult, nb_rand_bit_cmult = (
    comp_cmult(n, gamma_sb))

  nb_rand = (6 * nb_rand_ref + 14 * nb_rand_cmult + 7 * nb_rand_add + 
             nb_rand_cadd)
  nb_add = 6 * nb_add_ref + 14 * nb_add_cmult + 7 * nb_add_add + nb_add_cadd
  nb_mult = 14 * nb_mult_cmult
  nb_rand_bit = (6 * nb_rand_bit_ref + 14 * nb_rand_bit_cmult + 
                 7 * nb_rand_bit_add + nb_rand_bit_cadd + 7 * comp_perm(n))
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  
def comp_sb (n, l_gamma_sb, gamma_sb) :
  """
  Compute the complexity of SubBytes, with input the 16 masked bytes of the 
  state.
  
  Args:
    n (int): Number of shares.
    l_gamma_sb (list): List of gamma used in the refresh of UnifMatMult in 
                       CardSecMult,  according to the number of shares.
    gamma_sb (int): The gamma used in the linear part of SubBytes 
                    (including TreeAdd).
  
  Returns: 
    Complexity (random, addition, multiplication) of SubBytes for 16 masked 
    bytes.
  """

  nb_rand_exp, nb_add_exp, nb_mult_exp, nb_rand_bit_exp = comp_expo_sb(n, l_gamma_sb, gamma_sb)
  nb_rand_aff, nb_add_aff, nb_mult_aff, nb_rand_bit_aff = comp_aff_sb(n, gamma_sb)
   
  nb_rand = 16 * (nb_rand_exp + nb_rand_aff)
  nb_add = 16 * (nb_add_exp + nb_add_aff)
  nb_mult = 16 * (nb_mult_exp + nb_mult_aff)
  nb_rand_bit = 16 * (nb_rand_bit_exp + nb_rand_bit_aff)
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  

def comp_mc (n, gamma_mc) :
  """
  Compute the complexity (random, addition, mult) of the MixColumns step, with 
  input 16 masked bytes.
  
  Args:
    n (int): Number of shares.
    gamma_mc (int): The gamma used for the refresh gadget in MixColumn.
  
  Returns: 
    Complexity (random, addition, multiplication) of MixColumn.
  """

  #There is in the MixColumns gadget (see Figure 13 of the paper):
  # - 8 refresh gadgets.
  # - 8 multiplication by a constant gadgets.
  # - 12 additions gadgets.
  # - 12 permutations.

  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n, gamma_mc)
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, gamma_mc)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult, nb_rand_bit_cmult = comp_cmult(n, gamma_mc)
 
  nb_rand = 12 * nb_rand_add + 8 * nb_rand_cmult + 8 * nb_rand_ref
  nb_add = 12 * nb_add_add + 8 * nb_add_cmult + 8 * nb_rand_add + 8 * nb_add_ref
  nb_mult = 8 * nb_mult_cmult
  nb_rand_bit = (12 * nb_rand_bit_add + 8 * nb_rand_bit_cmult + 
                 8 * nb_rand_bit_ref + 12 * comp_perm(n))
  
  return 4 * nb_rand, 4 * nb_add, 4 * nb_mult, 4 * nb_rand_bit
  
def comp_AES_enc (n, gamma_ark, l_gamma_sb, gamma_sb, gamma_mc) :
  """
  Compute the complexity (random, addition, multiplication) of a complete AES 
  encryption.
  
  Args:
    n (int): Number of shares
    ark (int): Gamma for the AddRoundKey step.
    l_gamma_sb (list): Gamma used in UnifMatMult from CardSecMult.
    gamma_sb (int): Gamma for the SubBytes step.
    gamma_mc (int): Gamma for the MixColumns step.
  """
  
  nb_rand_ark, nb_add_ark, nb_rand_bit_ark = comp_addrk(n, gamma_ark)
  nb_rand_sb, nb_add_sb, nb_mult_sb, nb_rand_bit_sb = comp_sb(n, l_gamma_sb, gamma_sb)
  nb_rand_mc, nb_add_mc, nb_mult_mc, nb_rand_bit_mc = comp_mc(n, gamma_mc)
  
  nb_rand = 11 * nb_rand_ark + 10 * nb_rand_sb + 9 * nb_rand_mc
  nb_add = 11 * nb_add_ark + 10 * nb_add_sb + 9 * nb_add_mc
  nb_mult = 10 * nb_mult_sb + 9 * nb_mult_mc
  nb_rand_bit = 11 * nb_rand_bit_ark + 10 * nb_rand_bit_sb + 9 * nb_rand_bit_mc
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit

################################################################################
###################### Complexity of JMB24's gadgets ###########################

def comp_ref_JMB24 (n) :
  """
  Complexity (random, addition) of the SR-SNI gadget used in JMB24.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of the refresh gadget.
  """

  nb_add = n**2 - n
  nb_rand = int(nb_add / 2)
  
  return nb_rand, nb_add

  
def comp_MatMult_JMB24 (nx, ny) :
  """
  Complexity (random, addition, multiplication) of the MatMult step of the 
  multiplication gadget of JMB24.
  
  Args:
    nx (int): Number of shares for the 1st secret of the multiplication gadget.
    ny (int): Number of shares for the 2nd secret of the multiplication gadget.
  
  Returns: 
    Comlpexity of MatMult.
  """
  if nx == 1 and ny == 1 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 1
    return nb_rand, nb_add, nb_mult
  
  else :
    nb_rand = 0
    nb_add = 0
    nb_mult = 0
    
    nxhl = nx // 2
    nxhr = nx - nxhl
    nyhl = ny // 2
    nyhr = ny - nyhl
    
    nb_rand_nxhl, nb_add_nxhl = comp_ref_JMB24(nxhl)
    nb_rand_nyhl, nb_add_nyhl = comp_ref_JMB24(nyhl)
    nb_rand_nxhr, nb_add_nxhr = comp_ref_JMB24(nxhr)
    nb_rand_nyhr, nb_add_nyhr = comp_ref_JMB24(nyhr)    
    
    if nxhl != 0 and nyhl != 0 :  
      nb_rand_rec, nb_add_rec, nb_mult_rec = comp_MatMult_JMB24 (nxhl, nyhl)
      
      nb_rand += nb_rand_nxhl + nb_rand_nyhl + nb_rand_rec
      nb_add += nb_add_nxhl + nb_add_nyhl + nb_add_rec
      nb_mult += nb_mult_rec
    
    if nxhl != 0 :
      nb_rand_rec, nb_add_rec, nb_mult_rec = comp_MatMult_JMB24 (nxhl, nyhr)
      
      nb_rand += nb_rand_nxhl + nb_rand_nyhr + nb_rand_rec
      nb_add += nb_add_nxhl + nb_add_nyhr + nb_add_rec
      nb_mult += nb_mult_rec
      
    if nyhl != 0 :
      nb_rand_rec, nb_add_rec, nb_mult_rec = comp_MatMult_JMB24 (nxhr, nyhl)
      
      nb_rand += nb_rand_nxhr + nb_rand_nyhl + nb_rand_rec
      nb_add += nb_add_nxhr + nb_add_nyhl + nb_add_rec
      nb_mult += nb_mult_rec      

    nb_rand_rec, nb_add_rec, nb_mult_rec = comp_MatMult_JMB24 (nxhr, nyhr)
      
    nb_rand += nb_rand_nxhr + nb_rand_nyhr + nb_rand_rec
    nb_add += nb_add_nxhr + nb_add_nyhr + nb_add_rec
    nb_mult += nb_mult_rec
    
    return nb_rand, nb_add, nb_mult
    
def comp_comp_JMB24 (n) :
  """
  Complexity (random, addition) of the compression step of the multiplication 
  gadget from JMB24.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of Comp from JMB24. 
  """

  tmp = n**2 - n
  nb_rand = int(tmp / 2)
  nb_add = tmp + n * (n - 1)
  
  return nb_rand, nb_add
  
def comp_mult_JMB24 (n) :
  """
  Complexity (random, addition, multiplication) of the multiplication gadget of 
  JMB24.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of the multiplication gadget of JMB24.
  """
  nb_rand_MM, nb_add_MM, nb_mult_MM = comp_MatMult_JMB24(n, n)
  nb_rand_comp, nb_add_comp = comp_comp_JMB24 (n)
  
  nb_rand = nb_rand_MM + nb_rand_comp
  nb_add = nb_add_MM + nb_add_comp
  nb_mult = nb_mult_MM
  
  return  nb_rand, nb_add, nb_mult

def comp_squaring_JMB24 (n) :
  """
  Complexity of the squaring gadget of JMB24 (n multiplications).

  Args:
    n (int): Number of shares.
  
  Returns:
    Complexity of the squaring gadgets of JMB24.
  """
  
  nb_mult = n
  return nb_mult
  
def comp_sb_JMB24 (n) :
  """
  Complexity of the SubBytes step of JMB24 for 16 masked bytes.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of SubBytes for 16 masked bytes.
  """
  
  #Exponentiation part
  nb_rand_ref, nb_add_ref = comp_ref_JMB24(n)
  nb_rand_mult, nb_add_mult, nb_mult_mult = comp_mult_JMB24(n)
  nb_mult_squa = comp_squaring_JMB24(n)
  
  nb_rand = 8 * nb_rand_ref + 4 * nb_rand_mult
  nb_add = 8 * nb_add_ref + 4 * nb_add_mult
  nb_mult = 4 * nb_mult_mult + 7 * nb_mult_squa
  
  #Linear part
  nb_add += 7 * n + 1
  nb_mult += 14 * n

  return 16 * nb_rand, 16 * nb_add, 16 * nb_mult

def comp_addrk_JMB24 (n) :
  """
  Complexity of the AddRoundKey step of JMB24 for 16 masked bytes.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of AddRoundKey for 16 masked bytes.  
  """

  nb_add = n
  return 16 * nb_add
  
def comp_mc_JMB24 (n) :
  """
  Complexity of the MixColumn step of JMB24 for 16 masked bytes. As MixColumn is 
  not described in JMB24 paper, we take the same circuit than in the paper and 
  use JMB24's compiler. 

  Args: 
    n (int): Number of shares.
 
  Returns: 
    Complexity of MixColumn for 16 masked bytes.
 """
 
  #48 addition gadgets, 32 cmult gadgets
  nb_add = n * 48 + n * 32
  nb_mult = n * 32
  return nb_add, nb_mult
 
def comp_AES_enc_JMB24 (n) :
  """
  Complexity (random, addition, multiplication) of a full AES encryption with 16 
  masked bytes with n shares.
  
  Args:
    n (int): Number of shares.
  
  Returns: 
    Complexity of AES encryption using JMB24 compiler.
  """

  nb_add_addrk = comp_addrk_JMB24(n)
  nb_rand_sb, nb_add_sb, nb_mult_sb = comp_sb_JMB24(n)
  nb_add_mc, nb_mult_mc = comp_mc_JMB24(n)
  
  nb_rand_ref, nb_add_ref = comp_ref_JMB24(n)
  #31 refresh gadget, 11 addrk, 10 Sbox, 10 SR, 9 Mc
  
  nb_rand = 31 * nb_rand_ref + 10 * nb_rand_sb 
  nb_add = 31 * nb_add_ref + 11 * nb_add_addrk + 10 * nb_add_sb + 9 * nb_add_mc
  nb_mult = 10 * nb_mult_sb + 9 * nb_mult_mc
  
  return nb_rand, nb_add, nb_mult
  

################################################################################
###################### Complexity of BFO23's gadgets ###########################

def comp_pref_bfo23(n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns: 
    Complexity of the pRef gadget of BFO23.
  """
  
  nb_rand = n
  nb_add = 2 * n
  return nb_rand, nb_add

def comp_add_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns: 
    Complexity of the Add gadget of BFO23.
  """
  
  #Refresh gadget after addition gadget.
  nb_rand_ref, nb_add_ref = comp_pref_bfo23(n)
  nb_add = n + nb_add_ref
  return nb_rand_ref, nb_add

def comp_copy_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the Copy gadget of BFO23.
  """
  
  #Refresh gadget after copy gadget on the 2 outputs.
  nb_rand_ref, nb_add_ref = comp_pref_bfo23(n)
  return 2 * nb_rand_ref, 2 * nb_add_ref

def comp_cmult_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the cMult gadget of BFO23.
  """
  
  #refresh gadget after cmult gadget.
  nb_rand_ref, nb_add_ref = comp_pref_bfo23(n)
  nb_mult = n
  return nb_rand_ref, nb_add_ref, nb_mult

def comp_cadd_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the cAdd gadget of BFO23.
  """
  
  #refresh gadget after cadd gadget.
  nb_rand_ref, nb_add_ref = comp_pref_bfo23(n)
  return nb_rand_ref, nb_add_ref + 1  
  
def comp_mult_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the mult gadget of BFO23.
  """
  print("n = ", n)

  nb_rand = int(n * (n - 1) / 2)
  L = ceil(log(n + 1, 2)) 
  nb_add = n * ((1 << L) + n - 2)
  nb_mult = n**2
  
  #Refresh gadget after multiplication gadget.
  nb_rand_ref, nb_add_ref = comp_pref_bfo23(n)
  
  return nb_rand + nb_rand_ref, nb_add + nb_add_ref, nb_mult

def comp_addrk_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the AddRoundKey gadget of BFO23.
  """  

  #16 addition gadget.
  nb_rand_add, nb_add_add = comp_add_bfo23(n)
  nb_rand = 16 * nb_rand_add
  nb_add = 16 * nb_add_add
  
  return nb_rand, nb_add

def comp_expo_sb_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the Exponentiation part of SubBytes gadget of BFO23.
  """  
  
  #For one secret value, 4 multiplication gadget, 7 squaring gadget. 
  
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult = comp_cmult_bfo23(n)    
  nb_rand_mult, nb_add_mult, nb_mult_mult = comp_mult_bfo23(n)
  nb_rand_copy, nb_add_copy = comp_copy_bfo23(n)
  
  nb_rand = 4 * (nb_rand_mult + nb_rand_copy) + 7 * nb_rand_cmult
  nb_add = 4 * (nb_add_mult + nb_add_copy) + 7 * nb_add_cmult
  nb_mult = 4 * nb_mult_mult + 7 * nb_mult_cmult
  
  return nb_rand, nb_add, nb_mult
  
def comp_aff_sb_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the Affine  part of SubBytes gadget of BFO23.
  """

  #For one secret value, 8 addition gadgets, 14 cmult gadget, 1 cadd_gadget.
  
  nb_rand_cadd, nb_add_cadd = comp_cadd_bfo23(n)  
  nb_rand_add, nb_add_add = comp_add_bfo23(n)
  nb_rand_copy, nb_add_copy = comp_copy_bfo23(n)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult = comp_cmult_bfo23(n)

  nb_rand = 14 * nb_rand_cmult + 8 * nb_rand_add + 7 * nb_rand_copy + nb_rand_cadd
  nb_add = 14 * nb_add_cmult + 8 * nb_add_add + 7 * nb_add_copy + nb_add_cadd
  nb_mult = 14 * nb_mult_cmult
  
  return nb_rand, nb_add, nb_mult
  
def comp_sb_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the SubBytes gadget of BFO23.
  """

  #16 sbox gadgets composed of exp followed by aff.
  nb_rand_exp, nb_add_exp, nb_mult_exp = comp_expo_sb_bfo23(n)
  nb_rand_aff, nb_add_aff, nb_mult_aff = comp_aff_sb_bfo23(n)
   
  nb_rand = 16 * (nb_rand_exp + nb_rand_aff)
  nb_add = 16 * (nb_add_exp + nb_add_aff)
  nb_mult = 16 * (nb_mult_exp + nb_mult_aff)
  
  return nb_rand, nb_add, nb_mult
  
def comp_mc_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the MixColumns gadget of BFO23.
  """
    
  #48 addition gadgets, 32 cmult gadgets , 48 copy gadgets
  nb_rand_add, nb_add_add = comp_add_bfo23(n)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult = comp_cmult_bfo23(n)
  nb_rand_copy, nb_add_copy = comp_copy_bfo23(n)
 
  nb_rand = 48 * (nb_rand_add + nb_rand_copy) + 32 * nb_rand_cmult
  nb_add =  48 * (nb_add_add + nb_add_copy) + 32 * nb_add_cmult
  nb_mult = 32 * nb_mult_cmult
  
  return nb_rand, nb_add, nb_mult
  
def comp_AES_enc_bfo23 (n) :
  """
  Args:
    n (int) Number of shares.
  
  Returns:   
    Complexity of the AES gadget of BFO23.
  """

  nb_rand_ark, nb_add_ark = comp_addrk_bfo23(n)
  nb_rand_sb, nb_add_sb, nb_mult_sb = comp_sb_bfo23(n)
  nb_rand_mc, nb_add_mc, nb_mult_mc = comp_mc_bfo23(n)
  
  nb_rand = 11 * nb_rand_ark + 10 * nb_rand_sb + 9 * nb_rand_mc
  nb_add = 11 * nb_add_ark + 10 * nb_add_sb + 9 * nb_add_mc
  nb_mult = 10 * nb_mult_sb + 9 * nb_mult_mc
  
  return nb_rand, nb_add, nb_mult

################################################################################  
########################### Graphes Constructions ##############################



def graph_complexity(p, logp, l_sec_level, crand, cadd, cmult, 
                     crandbit, crand_JMB24, cadd_JMB24, cmult_JMB24, 
                     crand_BFO23, cadd_BFO23, cmult_BFO23, filename, security ) :
  """
  Args:
    p (float) : The leakage rate.
    logp (int) : The logarithme in base 2 of p (usually we take p as a power of 
                  2).
    l_sec_level (list) : The list of security level we want to obtain.
    crand: Random complexity of our gadget.
    cadd: Addition complexity of our gadget.
    cmult : Multiplication complexity of our gadget.
    crandbit : Random Bits complexity of our gadget.
    crand_JMB24 : Random complexity of JMB24's gadget.
    cadd_JMB24 : Addition complexity of JMB24's gadget.
    cmult_JMB24 : Multiplication complexity of JMB24's gadget.
    crand_BFO23 : Random complexity of BFO23's gadget.
    cadd_BFO23 : Addition complexity of BFO23's gadget.
    cmult_BFO23 : Multiplication complexity of BFO23's gadget.
    filename : The name of the file to save the graphs.

  Returns:
    Graphs with the different complexity reached for the different levels of 
    security for us, JMB24 gadgets and BFO23 gadgets
  """


  fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=1200)
  
  x = np.arange(len(l_sec_level))
  w = 0.25
  
  
  ax[0].bar(x - w, crand_BFO23, width = w, label = "BFO23", color = 'green')  
  ax[0].bar(x, crand_JMB24, width = w, label = "JMB24", color = 'C1')
  ax[0].bar(x + w, crand, width = w, label = "Our Work - Random Fields", color = 'C0')
  ax[0].bar(x + w, crandbit, bottom  = crand, width = w, label = "Our work - Random Bits", color = 'skyblue')
  
  ax[0].set_ylabel("Number of randoms")
  ax[0].set_xlabel("Security Level")
  #ax1.set_yscale('log', base=2) 
  ax[0].set_title(r'\#Rand')
  ax[0].legend(loc='upper left', fontsize = "x-small")

  ax[0].set_xticks(x) 
  ax[0].set_xticklabels(security)

  ax[1].bar(x - w, cadd_BFO23, width = w, label = "BFO23", color = 'green')  
  ax[1].bar(x, cadd_JMB24, label = "JMB24", width = w, color = 'C1')
  ax[1].bar(x + w, cadd, label = "Our Work", width = w, color= 'C0')
  ax[1].set_ylabel("Number of additions")
  ax[1].set_xlabel("Security Level")
  ax[1].set_title(r'\#Add')
  ax[1].legend(loc='upper left', fontsize = "small")  
  
  #ax2.set_yscale('log', base=2) 
  ax[1].set_xticks(x)  
  ax[1].set_xticklabels(security)
   

  ax[2].bar(x - w, cmult_BFO23, width = w, label = "BFO23", color = 'green')  
  ax[2].bar(x, cmult_JMB24, label = "JMB24", width = w, color = 'C1')
  ax[2].bar(x + w, cmult, label = "Our Work", width = w, color = 'C0')  
  ax[2].set_ylabel("Number of multiplications")
  ax[2].set_xlabel("Security Level")
  ax[2].set_title(r'\#Mult')
  ax[2].legend(loc='upper left', fontsize = "small")  
  
  #ax.set_yscale('log', base=2) 
  ax[2].set_xticks(x) 
  ax[2].set_xticklabels(security)  
  
  fig.tight_layout()
  fig.savefig(filename, bbox_inches="tight")
  plt.close(fig)


  
#Nothing interesting.
def compare_RPE() :
  logp = -9.5
  p = 2**logp
  sec_level = 2**80
  
  #Complexity RPE :
  Nadd_RPE = 21962863500
  #Ncopy_RPE = 11307611100
  Nmult_RPE = 217890000
  Nrand_RPE = 10872746400

  n = 20
  gamma_ark = 20
  gamma_mc = 20
  gamma_sb = 21
  l_gamma_sb = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  
  Nrand, Nadd, Nmult, tmp  = comp_AES_enc (n, gamma_ark, l_gamma_sb, gamma_sb, gamma_mc)
  
  print("Rand :     RPE : " + str(log(Nrand_RPE, 2)) + ",         my work : " + str(log(Nrand, 2)))
  print("Add :      RPE : " + str(log(Nadd_RPE, 2)) + ",         my work : " + str(log(Nadd, 2)))
  print("Mult :     RPE : " + str(log(Nmult_RPE, 2)) + ",         my work : " + str(log(Nmult, 2)))
  

if __name__ == "__main__" :
  print("main")



