#!/bin/bash
#SBATCH --job-name=irmcamelyon17
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --account=cds
#SBATCH --array=0-224
#SBATCH --chdir=/home/jz3786/wilds
#SBATCH --output=/home/jz3786/wilds/logs/ermcamelyon17%x-%j.out

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate py36;

i=0;
                                 

for seed in 0 1 2 3 4;
	do
	for anneal in 0 500 1000 5000 10000;
		do 
		for lambda in  0 0.5 1 5 10 50 100 500 1000;
			do 
				seeds[$i]=$seed;
				anneals[$i]=$anneal;
				lambdas[$i]=$lambda;
				i=$(($i+1));
			done
		done				
	done  

final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_anneal=${anneals[$SLURM_ARRAY_TASK_ID]}
final_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

python examples/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir results/irm_camelyon17_anneal${final_anneal}_lambda${final_lambda}_seed${final_seed} \
--dataset camelyon17 --algorithm IRM --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters ${final_anneal} --irm_lambda ${final_lambda}



