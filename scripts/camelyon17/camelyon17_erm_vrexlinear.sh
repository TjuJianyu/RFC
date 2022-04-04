#!/bin/bash
#SBATCH --job-name=camelyon17_erm_vrexlinear
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --array=0-44
#SBATCH --chdir=/home/jz3786/wilds
#SBATCH --output=/home/jz3786/wilds/logs/17%x-%j.out

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate py36;

i=0;


for seed in 5 6 7 8 9;
	do
	 
	for lambda in 0 0.5 1 5 10 50 100 500 1000;  #0 0.5 1 5 10 50 100;
		do 
			seeds[$i]=$seed;
			anneals[$i]=$anneal;
			lambdas[$i]=$lambda;
			i=$(($i+1));
		done
	done				


final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_anneal=${anneals[$SLURM_ARRAY_TASK_ID]}
final_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

# final_seed=0
# final_lambda=1

final_mode=iid

mark=erm

resdir=results/vrexlinear_camelyon17_${mark}_lambda${final_lambda}_seed${final_seed} 
syndir=results/erm_camelyon17_seed${final_seed}
mkdir $resdir

python examples/analysis_earlystop.py --result_dir ${syndir}


python examples/run_expt_rfc_eval.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir ${syndir} --eval_only \
--eval_epoch `cat ${syndir}/eval_epochs_${final_mode}.txt` \
--dataset camelyon17 --algorithm ERM --model densenet121 --seed ${final_seed}


python examples/generate_groups.py --dataset camelyon17 \
--pred_dirs ${syndir} \
--pred_epochs `cat ${syndir}/eval_epochs_${final_mode}.txt` \
--pred_seed ${final_seed}  --result_dir ${syndir} 


python examples/run_expt_synthesis_convert_model.py  \
--version "1.0" --root_dir data/camelyon17/ --save_step 1  \
--rfc_groups_dir  ${syndir} \
--log_dir ${syndir} \
--weight_decay 0.01 --dataset camelyon17 --algorithm mtERM --model densenet121 \
--seed ${final_seed} --loss_function rfcmt_cross_entropy  --algo_log_metric rfcmtaccuracy \
--process_outputs_function None --eval_only --save_model_dir ${resdir} \
--eval_epoch `cat ${syndir}/eval_epochs_iid.txt`


echo "converted model"

python examples/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir ${resdir} \
--dataset camelyon17 --algorithm VREX --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters 0 --irm_lambda ${final_lambda} --resume --freeze_featurizer True 




