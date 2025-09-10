Artifact of the paper : "Efficient RP Compiler from our New Multiplication and the Use of Permutations"

Authors : Sonia Belaïd & Victor Normand & Matthieu Rivain


Organization of the file : There is many files you should know :

  1. 'partition.py' gives the cardinal-RPC enveloppes of the linear gadget used as well as the refresh gadget, not a code from us, code taken on TODO.
  2. 'mult_gen.py' compute the *cardinal-RPC* enveloppes of **CardSecMult** used with **UnifMatMult** (variants 3 of the paper), where the output cardinal of **UnifMatMult** $t_{out}$ is in range $[0, n^2]$. In particular, we do not split the output in $4$ distinct smaller outputs as it increases the complexity of the computation too much as explained in Remark TODO. We start by computing the *cardinal-RPC* enveloppe of **UnifMatMult**. Then it computes the *cardinal-RPC* enveloppes **BasicTreaAdd**. In fact, since there is only one output at the output of **UnifMatMult**, the first permutation in **TreeAdd** is useless. Then we compute the *cardinal-RPC* enveloppes of **CardSecMult** using Theorem TODO of the paper.
  3. 'mult_4_card.py' : TODO
  4. 'mult_uni_4_card.py' : TODO
  5. 'AES.py' compute the random probing security of our masked AES implementation, build with our compiler described in Section 5. Eventually, it compares the trade-off between random probing security and complexity of our AES implementation and the ones from TODO.
  
  
  
  
