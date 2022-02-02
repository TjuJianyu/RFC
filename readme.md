# Official code for "Rich Feature Construction for the Optimization-Generalization Dilemma"

## required environments
...

## Optimization difficulties of OOD methods (ColoredMNIST)
<embed src='figures/anneal_nll_full.pdf' type='application/pdf'>

`bash  script/coloredmnist/coloredmnist_anneal.sh`

## generalization difficulties of OOD methods (ColoredMNIST)
`bash  script/coloredmnist/coloredmnist_perfect_initialization_longtrain.sh`

## The proposed RFC on ColoredMNIST and InverseColoredMNIST

`bash  script/coloredmnist/coloredmnist_rfc.sh`
`bash  script/coloredmnist/inversecoloredmnist_rfc.sh`


## Camelyon17 experiments

`resdir=results/erm_camelyon17_wd0.01_seed${final_seed}`

`python src/run_expt.py --version "1.0" --root_dir data/camelyon17/ --log_dir ${resdir} \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed} --save_step 1 \
--weight_decay 0.01`


`python src/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir results/${method}_camelyon17_anneal${final_anneal}_lambda${final_lambda}_seed${final_seed} \
--dataset camelyon17 --algorithm ${method} --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters ${final_anneal} --irm_lambda ${final_lambda}`


...