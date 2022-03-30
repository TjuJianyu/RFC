#!/bin/bash
#SBATCH --job-name=ermcamelyon17
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --array=0-4
#SBATCH --chdir=
#SBATCH --output=

conda activate py36;

i=0;


for seed in  0 1 2 3 4; 
do
	for wd in 0.01;
	do 
		seeds[$i]=$seed;
		wds[$i]=$wd;
		i=$(($i+1));		
	done 		
done                                     


final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_wd=${wds[$SLURM_ARRAY_TASK_ID]}


resdir=results/erm_camelyon17_wd${final_wd}_seed${final_seed}
python examples/run_expt.py --version "1.0" --root_dir data/camelyon17/ --log_dir ${resdir} \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed} --save_step 1 \
--weight_decay ${final_wd}




