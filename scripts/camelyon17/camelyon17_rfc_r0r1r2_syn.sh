#!/bin/bash
#SBATCH --job-name=camelyon17_rfc_r0r1r2_syneven
#SBATCH --time=5:00:00
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

for seed in 0 1 2 3 4;
	do
		for method in old;
		do 
			for weight_decay in 0.000001;
			do 	
				for thr in 0;
				do 
					methods[$i]=$method;
					wd[$i]=$weight_decay;
					seeds[$i]=$seed;
					thrs[$i]=$thr;
					i=$(($i+1));
				done 

			done 	
		done			
	done                                     


final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_wd=${wd[$SLURM_ARRAY_TASK_ID]}
final_thr=${thrs[$SLURM_ARRAY_TASK_ID]}
final_method=${methods[$SLURM_ARRAY_TASK_ID]}



final_mode=iid

mark=3round_r0r1r2_even

source1_dir=results/erm_camelyon17_wd0.01_seed${final_seed}
source2_dir=results/rfc_camelyon17_eval${final_mode}_wd0.01_seed${final_seed}
source3_dir=results/rfc_camelyon17_2round_rfc1erm_rfc20.01_eval${final_mode}_wd0.01_seed${final_seed}

 
#eval (iid/ood) 
for source_dir in ${source1_dir} ${source2_dir} ${source2_dir};
do
	echo ${source_dir}
	python examples/analysis_earlystop.py --result_dir ${source_dir}/	

	python examples/run_expt_rfc_eval.py --version "1.0" --root_dir data/camelyon17/ \
	--log_dir ${source_dir} --eval_only \
	--eval_epoch `cat ${source_dir}/eval_epochs_${final_mode}.txt` \
	--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed}

# done
# #resdir=results/debug
resdir=results/syn_camelyon17_${mark}_eval${final_mode}_wd${final_wd}_seed${final_seed}/
mkdir $resdir


python src/generate_groups.py --dataset camelyon17 \
--pred_dirs ${source1_dir} ${source2_dir} ${source3_dir} \
--pred_epochs `cat ${source1_dir}/eval_epochs_${final_mode}.txt` `cat ${source2_dir}/eval_epochs_${final_mode}.txt` `cat ${source3_dir}/eval_epochs_${final_mode}.txt` \
--pred_seed ${final_seed}  --result_dir ${resdir}  


python src/run_expt_synthesis.py --version "1.0" --root_dir data/camelyon17/ --save_step 1 \
--rfc_groups_dir ${resdir} --log_dir ${resdir} --weight_decay ${final_wd} --n_epochs 20 \
--dataset camelyon17 --algorithm mtERM --model densenet121 --seed ${final_seed} \
--distinct_groups True --uniform_over_groups True --n_groups_per_batch 2 --train_loader group \
--loss_function rfcmt_cross_entropy --algo_log_metric rfcmtaccuracy --process_outputs_function None \
--syn_sample False --syn_threshold ${final_thr} --group_method ${final_method}
# 
# 
# 
# 
