Artifact of the paper : "Masked Circuit Compiler in the Cardinal Random Probing Composability Framework"

Authors : Sonia Belaïd & Victor Normand & Matthieu Rivain


Organization of the file :

  1. 'partition.py' : gives the cardinal-RPC enveloppes of the linear gadget 
  used as well as the refresh gadget, not a code from us, code taken from 
  https://github.com/CryptoExperts/EC25-random-probing-Raccoon.
  
  2. 'mult_gen.py' : computes the cardinal-RPC envelopes uniform variant of 
  *CardSecMult* for every number of shares $n \geq 2$. We consider in this file 
  that the output of *UnifMatMult* is given as a single $n^2$ instead of the 
  file ``mult_4card.py`` and ``mult_uni_4card.py`` which consider 4 
  $(\frac{n^2}{4})$ sharings at the output of *UnifMatMult*.
  Consequently, we consider a single input sharing of size $n^2$ at the input of 
  TreeAdd.

      This file is the **core** file which will be used to compute the envelope of 
  the multiplication gadget used in our masked AES implementation, the function 
  `compute_envn_mult` gives the uniformly cardinal RPC envelope of the 
  multiplication gadget (i.e. implemented with *UnifMatMult*). 
  Then this multiplication gadget is used in the bar chart of **Figure 9** and 
  **Figure 14** of the full paper.
  Moreover, functions `find_plateau_RPM_n` and `find_plateau_RPM` are the one 
  used to compute **Figure 8** of the paper.  
  
  3. 'mult_4_card.py' : This file computes the cardinal-RPC envelopes 
  **non-uniform variant** of *CardSecMult*.
  Moreover, it computes it by **cutting the number of outputs of MatMult in 4** 
  to obtain better security. In fact, *MatMult* exposes four outputs of size 
  $\frac{n^2}{4}$, one per subtree (see **Figure 5** of the paper)—instead of a 
  single aggregated output of size $n^2$. 
  Accordingly, *TreeAdd* takes four inputs, each of size $\frac{n^2}{4}$, 
  instead of a single input of size $n^2$. 
  The implementation targets numbers of shares $n$ that are powers of two.
  This file is used to produce **Figure 7** of the full paper (non-uniform 
  variant), generating the points labeled `Unif = 4` and `Unif = 8` on the 
  graph. The **Figure 7** is notably built with the function ``compute_graph``.
  
  4. 'mult_uni_4_card.py' : This file computes the cardinal-RPC envelopes 
  **uniform variant** of *CardSecMult*.
  Moreover, it computes it by **cutting the number of outputs of MatMult in 4** 
  to obtain better security. 
  In fact, *MatMult* exposes four outputs of size $\frac{n^2}{4}$, 
  one per subtree (see **Figure 5** of the paper)—instead of a single aggregated 
  output of size $n^2$. 
  Accordingly, *TreeAdd* takes four inputs, each of size $\frac{n^2}{4}$, 
  instead of a single input of size $n^2$. 
  The implementation targets numbers of shares $n$ that are powers of two.
  This file is used to produce **Figure 7** of the full paper (uniform symmetric 
  variant),generating the points labeled `Unif = 4` and `Unif = 8` on the graph.

  5. 'butterfly_network.py' : Enables the computation of the *cardinal RPC*/ 
  *uniformly cardinal RPC*/ *general RPC* envelopes for variants of the 
  *1-stage Butterfly Network* and the *2-stage Butterfly Network*. 
  Then it enables to compute the **threshold RPC security** of the 1-stage and 
  2-stage butterfly network derived from the different envelopes computed, in 
  addition to the threshold RPC of the base gadget.
  Concretely the security of the *1-stage butterfly network* is analyzed in the 
  final function `compare_security_stage1` (responsible of **Figure 4** of the 
  full paper) while the security of the *2-stage butterfly network* is analyzed 
  in the final function `compare_security_stage2` (responsible of **Figure 16** 
  of the full paper).

  6. 'complexity.py' : Computes the complexity of the different masked *AES* 
  (**our**, the one of **JMB24** and the one of **BFO3**), as well as the 
  different complexity of the multiplication gadget that we compare (**our**, 
  the one of **JMB24** and the one of **BFO23**).  

  7. 'AES.py' : Compute the 
  **random probing security of our masked AES encryption**, 
  build with our compiler described in *Section 6*. In addition,
  it enables to optimize automatically the $\gamma$ taken (i.e. the number of 
  iteration in *RPRefresh*) for each block *Subbytes*, *AddRoundKey* and 
  *MixColumns*.
  
  8. 'results.py' : Contains the results that are exhibited in the paper  
    
        "Masked Circuit Compiler in the Cardinal Random Probing Composability 
    Framework"
        
        Each function is responsible for one or many graphs of the paper and will be 
    highlighted above the function. 




  
  
  
  
