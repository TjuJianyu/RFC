#!/bin/bash
#SBATCH --job-name=camelyon17_rfc_r0r1_syneven_vrex
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --account=cds
#SBATCH --array=0-79
#SBATCH --chdir=/home/jz3786/wilds
#SBATCH --output=/home/jz3786/wilds/logs/%x-%j.out

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate py36;

i=0;



for syn_wd in 0.000001;
do 
	for seed in 0 1 2 3 4;
		do
		for lambda in 0.5 1 5 10 50 100 500 1000 0.25 0.75 2.5 7.5 25 75 250 750; 
			do 
				seeds[$i]=$seed;
				lambdas[$i]=$lambda;
				syn_wds[$i]=$syn_wd;
				i=$(($i+1));
			done
		done				
done 




final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}
final_synwd=${syn_wds[$SLURM_ARRAY_TASK_ID]}
final_randclf=False

mark=syn${final_synwd}_rfcr0r1_iid


resdir=results/vrexl2${final_synwd}_camelyon17_${mark}_lambda${final_lambda}_seed${final_seed} 
syndir=results/syn_camelyon17_2round_rfc1erm_rfc20.01_even_evaliid_wd${final_synwd}_seed${final_seed}

mkdir $resdir

python src/analysis_earlystop.py --result_dir ${syndir}


python src/run_expt_synthesis_convert_model.py  \
--version "1.0" --root_dir data/camelyon17/ --save_step 1  \
--rfc_groups_dir  ${syndir} \
--log_dir ${syndir} \
--weight_decay 0.01 --dataset camelyon17 --algorithm mtERM --model densenet121 \
--seed ${final_seed} --loss_function rfcmt_cross_entropy  --algo_log_metric rfcmtaccuracy \
--process_outputs_function None --eval_only --save_model_dir ${resdir} \
--eval_epoch `cat ${syndir}/eval_epochs_iid.txt` --rand_classifier ${final_randclf}


echo "converted model"

python src/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir ${resdir} --weight_decay ${final_synwd} \
--dataset camelyon17 --algorithm VREX --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters 0 --irm_lambda ${final_lambda} --resume --freeze_featurizer False 





