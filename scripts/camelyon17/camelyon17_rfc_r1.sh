#!/bin/bash
#SBATCH --job-name=rfcr1camelyon17
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --account=cds
#SBATCH --array=0-4
#SBATCH --chdir=/home/jz3786/wilds
#SBATCH --output=/home/jz3786/wilds/logs/%x-%j.out

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate py36;

i=0;

for mode in iid;
	do 
	for seed in 0 1 2 3 4; 
		do
			for weight_decay in 0.01; 
			do 	
				wd[$i]=$weight_decay;
				seeds[$i]=$seed;
				eval_mode[$i]=$mode;
				i=$(($i+1));
			done 				
		done                                     
	done


final_mode=${eval_mode[$SLURM_ARRAY_TASK_ID]}
final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_wd=${wd[$SLURM_ARRAY_TASK_ID]}

python examples/analysis_earlystop.py --result_dir results/erm_camelyon17_seed${final_seed}/


python examples/run_expt_rfc_eval.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir results/erm_camelyon17_seed${final_seed} --eval_only \
--eval_epoch `cat results/erm_camelyon17_seed${final_seed}/eval_epochs_${final_mode}.txt` \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed}



resdir=results/rfc_camelyon17_eval${final_mode}_wd${final_wd}_seed${final_seed}/
mkdir $resdir


python examples/generate_groups.py --dataset camelyon17 \
--pred_dirs results/erm_camelyon17_seed${final_seed}/ \
--pred_epochs `cat results/erm_camelyon17_seed${final_seed}/eval_epochs_${final_mode}.txt` \
--pred_seed ${final_seed}  --result_dir ${resdir}  --result_dir ${resdir}


python examples/run_expt_rfc.py --version "1.0" --root_dir data/camelyon17/ --save_step 1 \
--rfc_groups_dir ${resdir} --log_dir ${resdir} --weight_decay ${final_wd} \
--dataset camelyon17 --algorithm groupDRO --model densenet121 --seed ${final_seed}


