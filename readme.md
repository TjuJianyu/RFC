# Official code for "Rich Feature Construction for the Optimization-Generalization Dilemma"

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
<p align="center">
  <image src='figures/anneal_nll_full.png'/>
</p>

`bash  script/coloredmnist/coloredmnist_anneal.sh`

## generalization difficulties of OOD methods (ColoredMNIST)
<p align="center">
<image src='figures/long_train_vstack.png'/>
</p>

`bash  script/coloredmnist/coloredmnist_perfect_initialization_longtrain.sh`

## The proposed RFC on ColoredMNIST and InverseColoredMNIST

`bash  script/coloredmnist/coloredmnist_rfc.sh`
`bash  script/coloredmnist/inversecoloredmnist_rfc.sh`


## Camelyon17 experiments
<p align="center">
<image src='figures/lambda_valid_test_irm_vrex_clove.png'>
</p>

`resdir=results/erm_camelyon17_wd0.01_seed${final_seed}`

`python src/run_expt.py --version "1.0" --root_dir data/camelyon17/ --log_dir ${resdir} \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed} --save_step 1 \
--weight_decay 0.01`


`python src/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir results/${method}_camelyon17_anneal${final_anneal}_lambda${final_lambda}_seed${final_seed} \
--dataset camelyon17 --algorithm ${method} --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters ${final_anneal} --irm_lambda ${final_lambda}`


...