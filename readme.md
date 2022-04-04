# Official code for "[Rich Feature Construction for the Optimization-Generalization Dilemma](https://arxiv.org/pdf/2203.15516.pdf)"

## Overview
In Machine Learning, defining a generalized goal (e.g. the invariant goal in out-of-distribution generalization) and finding a path to the goal (e.g. the many optimization tricks) are two key problems. Usually, there is a dilemma between the two. i.e. either the generalization goal is weak/poor or the optimization process is hard. This optimization-generalization dilemma is especially obvious in the out-of-distribution area. This work tries to solve this dilemma by creating a RICH and SIMPLE representation, such that the optimization process becomes easier with the representation. As a result, we can pursue a stronger generalization goal.



### A short Story
Two common questions, in many areas, are: "where is our goal?" and "how to reach the goal from our current position?". A successful project needs to answer both questions. The two questions, however, are contradicted in difficulty. When the goal is ambiguous, normally the path to the goal is blurry. When the path is clear and confident, normally the goal is plain. For the goal of "making a cup of espresso?", for instance, most people can have a clear precise path immediately. On the other hand, "Building a spacecraft to Jupyter" is an ambiguous goal. But most people have no idea about how to achieve it. 

Can we build a spacecraft by purely thinking about the "spacecraft"? No. The spacecraft is built based on the development of diverse areas, such as material, computer, engine. 

The story above revises the path to hard problems, that is "Search/develop diverse areas (directions). Then a clear path may appear upon them. Otherwise, continuing search more."

The rule above is also the key idea of the proposed Rich Feature Construction (RFC) method. 



## required environments

* wilds==2.0.0
* einops=0.4.1
* python=3.6.13
* pytorch=1.10.2
* torch-geometric=2.0.3
* torch-scatter=2.0.9
* torch-sparse=0.6.12
* torchvision=0.11.3
* tqdm=4.62.3
* transformers=4.17.0


## Optimization difficulties of OOD methods (ColoredMNIST)
OOD methods are sensitive to the network initialization. We test nine OOD methods, [IRMv1](https://arxiv.org/abs/1907.02893), [VREx](https://arxiv.org/abs/2003.00688), [FISH](https://arxiv.org/abs/2104.09937), [SD](https://arxiv.org/abs/2011.09468), [IGA](https://arxiv.org/abs/2008.01883), [LfF](https://arxiv.org/abs/2007.02561), [RSC](https://arxiv.org/abs/2007.02454), [CLOvE](https://arxiv.org/abs/2102.10395), [fishr](https://arxiv.org/abs/2109.02934), on the ColoredMNIST benchmark. Fig1 shows the OOD performance with different ERM pretrain-epochs. None of the nine OOD methods can work with a random initialzation. 

<p align="center">
  <image src='figures/anneal_nll_full.png'/>
</p>

<p align="center">
  Fig1: Test performance of nine penalized OoD methods as
a function of the number of epochs used to pre-train the neural
network with ERM. The final OoD testing performance is very
dependent on choosing the right number of pretraining epochs,
illustrating the challenges of these optimization problems.
</p>
 
To reproduce the results, run: 

`bash  script/coloredmnist/coloredmnist_anneal.sh`


## generalization difficulties of OOD methods (ColoredMNIST)

Starting from a 'perfect' initialization where the model only uses the robust feature (OOD performance is maximized), what is going on if we continue training these OOD methods? Will they maintain the robustness? or decay to a spurious/singular solution? Fig2 (top) gives the latter answer. 

<p align="center">
<image src='figures/long_train_vstack.png'/>
</p>

<p align="center">
  Fig2: Test performance of OoD methods as a function of training epochs. 
  Top: Six OoD methods are trained from a ‘perfect’ initialization where only the robust feature is well learned. 
  The blue star indicates the initial test accuracy. 
  Bottom: The OoD methods are trained from the proposed (frozen) RFC representation.
</p>

To reproduce the results (top), run:

`bash  script/coloredmnist/coloredmnist_perfect_initialization_longtrain.sh`


## The proposed RFC on ColoredMNIST
The proposed RFC method creates a **rich** & **simple** representation to solve the optimization-generalization dilemma above. Tab1 shows the comparison of Random initialization (Rand), ERM pretrained initialization (ERM), RFC pretrained initialization (RFC / RFC(cf)). The proposed RFC consistantly boost OOD methods. 

<p align="center">
<image src="figures/coloredmnist.png"/>
</p>
<p align="center">
  Tab1: OoD testing accuracy achieved on the COLORMNIST.
The first six rows of the table show the results achieved by six
OoD methods using respectively random initialization (Rand),
ERM initialization (ERM), RFC initialization (RFC). The last
column, RFC(cf), reports the performance achieved by running
the OoD algorithm on top of the frozen RFC representations. The
seventh row reports the results achieved using ERM under the same
conditions. The last row reminds us of the oracle performance
achieved by a network using data from which the spurious feature
(color) has been removed.
</p>

To reproduce the results, run: 

`bash  script/coloredmnist/coloredmnist_rfc.sh`

## Aiming for the second easiest-to-find feature is not OOD generalization
A line of works seek OOD generalization by discovering the second easiest-to-find features, such as [PI](https://arxiv.org/abs/2105.12628). Here we claim that the second easiest-to-find feature is not the robust solution in general. To showcase the idea, we create a 'InverseColoredMNIST' dataset where the robust feature (digits) is more predictive than the spurious feature (color). 

<p align="center">
<image src="figures/inversecoloredmnist.png"/>
</p>

<p align="center">
Tab2: OoD test accuracy of PI and OOD/ERM methods on COLOREDMNIST and INVERSECOLOREDMNIST. The OOD/ERM
methods are trained on top of a frozen RFC representation.
</p>


To reproduce the results, run:

`bash  script/coloredmnist/inversecoloredmnist_rfc.sh`


## Camelyon17 experiments
<p align="center">
<image src='figures/lambda_valid_test_irm_vrex_clove.png'>
</p>

### ERM baseline

`python examples/run_expt.py --version "1.0" --root_dir ${directory to your data} --log_dir ${resdir} \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed} --save_step 1 \
--weight_decay ${final_wd} `


### IRM/VREX/CLOvE baseline

`python examples/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir ${resdir} --dataset camelyon17 \
--algorithm IRM/VREX/CLOvE \
--model densenet121 --seed ${final_seed} --save_step 1 --irm_penalty_anneal_iters ${final_anneal} \
--irm_lambda ${final_lambda} --kernel_scale ${kernel_scale_for_CLOvE}`





