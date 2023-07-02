**GDNMF**

note: The function 'dnmtf_RES'  is the previous name of GDNMF, which is called in GDNMG().

**Input**

X: data matrix p * n
Y: label matrix c * n
paras: parameters
---- label_ratio: ratio of known labels
---- num_cluster: number of classes 
---- lr: learning rate
---- max_L_left: number of layers for basis decomposition
---- max_L_right: number of layers for feature decomposition

---- decay_rate
---- alpha: hyper parameter

**Output**
[ACC, NMI, Purity]

Refer to demo.m 

