
[results.py]:results.py
[mult_gen.py]:mult_gen.py
[mult_4card.py]:mult_4card.py
[mult_uni_4card.py]:mult_uni_4card.py
[partitions.py]:partitions.py
[AES.py]:AES.py
[complexity.py]:complexity.py
[butterfly_network.py]:butterfly_network.py




# AC25 - Cardinal RP Compiler

This repository contains all the scripts necessary to reproduce the results presented in the paper:

"Masked Circuit Compiler in the Cardinal Random Probing Composability Framework" by Sonia Belaïd, Victor Normand and Matthieu Rivain, published at Asiacrypt 2025.

In the following, we may refer to the full version of this publication, which includes an appendix and minor revisions to the main text. In particular, additional figures have been included, some in the main body and others in the appendix, which is why we refer to figures from the full version rather than those in the published version. The full version is available here:

https://ia.cr/2025/1747

This project reuses the [partitions.py] file from another CryptoExperts 
MIT-licensed repository.

## Dependencies

This script requires : 
  - Python 3.x
  - `numpy`, `matplotlib`, `scipy`, `sympy`
  - `cm-super` package

To install dependencies, run :

`pip install numpy matplotlib scipy sympy`

`sudo apt install cm-super`

## Usage

To obtain the graphs used in the paper, please run the following command : 

`python3 results.py`


By default, this script uses $16$ cores to run, if you want to increase or 
reduce the number of cores, you are invited to specify it at the end of the 
file [results.py] :

```
if __name__ == "__main__" :
  
  #Number of cores to be used, change according to your ressources.
  cores = 16

  #Figure 4
  graph_bnet_st1([2**-3, 2**-5, 2**-10], 5)

  #Figure 16
  graph_bnet_st2([2**-3, 2**-5, 2**-10], 11, cores)

  #Figure 7 
  NRSM_4card_graph ([4, 8], [2**-6, 2**-12], 41, cores)
  
  #Figure 8
  gamma_n_p = find_plateau_RPM_n([2**-10, 2**-15, 2**-20], 19, cores)
  i = 0 
  for p in [2**-10, 2**-15, 2**-20] :
    print("p = 2^"+ str(int(log(p, 2))) + ", gamma_n = " + str(gamma_n_p[i]))
    i += 1

  #Figure 9 
  histo_mult_complexity ([2**-10], [2**-64, 2**-80], cores)
  histo_mult_complexity([2**-15, 2**-20], [2**-64, 2**-80, 2**-128], cores)

  #Figure 14 
  histo_AES_complexity([2**-20], [2**-64, 2**-80, 2**-128], cores)
  histo_AES_complexity ([2**-16], [2**-64, 2**-80], cores)

  #Last sentence of the main body.
  complexity_cRPCvstRPC (8, 2**-20, 0.1, cores)
```

Moreover, if you want to obtain only some graphs precisely, please comment the 
adequate function, each one is responsible for a figure of the paper, which is 
commented above the function.
If you intend to change any parameter values, check the function’s documentation 
in [results.py] for a better understanding of what each parameter does.


## Organization of the Repository

  1. [results.py] : Contains the **results** that are exhibited in the paper  
    
        "Masked Circuit Compiler in the Cardinal Random Probing Composability 
    Framework"
        
        Each function is responsible for one or many graphs of the paper and 
    will be highlighted above the function. 

  2. [mult_gen.py] : computes the cardinal-RPC envelopes uniform variant of 
  *CardSecMult* (i.e. the **multiplication gadget**) for every number of shares 
  $n \geq 2$. We consider in this file 
  that the output of *UnifMatMult* is given as a single $n^2$ instead of the 
  file [mult_4card.py] and [mult_uni_4card.py] which consider 4 
  $(\frac{n^2}{4})$ sharings at the output of *UnifMatMult*.
  Consequently, we consider a single input sharing of size $n^2$ at the input of 
  TreeAdd. 

      This file is the **core** file which will be used to compute the envelope of 
  the multiplication gadget used in our masked AES implementation, the function 
  `compute_envn_mult` gives the uniformly cardinal RPC envelope of the 
  multiplication gadget (i.e. implemented with *UnifMatMult*). The formula to 
  compute the cardinal RPC envelope of the multiplication gadget are exhibited 
  in the appendix of the full paper version (see Lemma 14). 
  Then this multiplication gadget is used in the bar chart of **Figure 9** and 
  **Figure 14** of the full paper.
  Moreover, functions `find_plateau_RPM_n` and `find_plateau_RPM` are the one 
  used to compute **Figure 8** of the paper. 

  3. [partitions.py] : gives the cardinal-RPC enveloppes of the **linear gadget** 
  used as well as the **refresh gadget**, code taken from 

      https://github.com/CryptoExperts/EC25-random-probing-Raccoon
  
      **Warning :** We slightly modify some element, in particular, the way to 
  obtain the copy gadget. 
   
  4. [AES.py] : Compute the 
  **random probing security of our masked AES encryption**, 
  build with our compiler described in *Section 6*. In addition,
  it enables to **optimize automatically the $\gamma$** taken (i.e. the number of 
  iteration in *RPRefresh*) for each block *Subbytes*, *AddRoundKey* and 
  *MixColumns*.

  5. [complexity.py] : Computes the **complexity** of the different masked *AES* 
  (*our*, the one of *JMB24* and the one of *BFO3*), as well as the 
  different complexity of the multiplication gadget that we compare (*our*, 
  the one of *JMB24* and the one of *BFO23*). 


  6. [butterfly_network.py] : Enables the computation of the *cardinal RPC*/ 
  *uniformly cardinal RPC*/ *general RPC* envelopes for variants of the 
  **1-stage Butterfly Network** and the **2-stage Butterfly Network**. 
  Then it enables to compute the **threshold RPC security** of the 1-stage and 
  2-stage butterfly network derived from the different envelopes computed, in 
  addition to the threshold RPC of the base gadget.
  Concretely the security of the *1-stage butterfly network* is analyzed in the 
  final function `compare_security_stage1` (responsible of **Figure 4** of the 
  full paper) while the security of the *2-stage butterfly network* is analyzed 
  in the final function `compare_security_stage2` (responsible of **Figure 16** 
  of the full paper).


  7. [mult_4card.py] : This file computes the cardinal-RPC envelopes 
  **non-uniform variant of CardSecMult**.
  Moreover, it computes it by **cutting the number of outputs of MatMult in 4** 
  to obtain better security. In fact, *MatMult* exposes four outputs of size 
  $\frac{n^2}{4}$, one per subtree (see **Figure 5** of the full paper)—instead of a 
  single aggregated output of size $n^2$. 
  Accordingly, *TreeAdd* takes four inputs, each of size $\frac{n^2}{4}$, 
  instead of a single input of size $n^2$. 
  The implementation targets numbers of shares $n$ that are powers of two.
  This file is used to produce **Figure 7** of the full paper (non-uniform 
  variant), generating the points labeled `Unif = 4` and `Unif = 8` on the 
  graph. The **Figure 7** is notably built with the function ``compute_graph``.
  
  8. [mult_uni_4card.py] : This file computes the cardinal-RPC envelopes 
  **uniform variant of CardSecMult**.
  Moreover, it computes it by **cutting the number of outputs of MatMult in 4** 
  to obtain better security. 
  In fact, *MatMult* exposes four outputs of size $\frac{n^2}{4}$, 
  one per subtree (see **Figure 5** of the full paper)—instead of a single aggregated 
  output of size $n^2$. 
  Accordingly, *TreeAdd* takes four inputs, each of size $\frac{n^2}{4}$, 
  instead of a single input of size $n^2$. 
  The implementation targets numbers of shares $n$ that are powers of two.
  This file is used to produce **Figure 7** of the full paper (uniform symmetric 
  variant),generating the points labeled `Unif = 4` and `Unif = 8` on the graph.

  9. A repository `results` containing all the precomputations made for the 
  graphs (which can be removed without any issues).

## License

This project is distributed under the **MIT License**.

##

For any questions, please refer to the **paper** or contact the **authors**.
 


  
  




  
  
  
  
