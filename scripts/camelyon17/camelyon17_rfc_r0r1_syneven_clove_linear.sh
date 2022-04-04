#!/bin/bash
#SBATCH --job-name=camelyon17_rfc_r0r1_syneven_clovelinear
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --account=cds
#SBATCH --array=0-85
#SBATCH --chdir=/home/jz3786/wilds
#SBATCH --output=/home/jz3786/wilds/logs/%x-%j.out

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate py36;

i=0;

 
for syn_wd in 0.000001;
do 
	for seed in 0 1 2 3 4;
		do
		for kernel_scale in 0.1;
		do 
			for lambda in 0 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 0.00075 0.0025 0.0075 0.025 0.075 0.25 0.75;

		
				do 
					seeds[$i]=$seed;
					lambdas[$i]=$lambda;
					syn_wds[$i]=$syn_wd;
					kernel_scales[$i]=$kernel_scale;
					i=$(($i+1));
				done
			done
		done 				
done 



final_seed=${seeds[$SLURM_ARRAY_TASK_ID]}
final_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}
final_synwd=${syn_wds[$SLURM_ARRAY_TASK_ID]}
final_ks=${kernel_scales[$SLURM_ARRAY_TASK_ID]}
final_randclf=False

mark=syn${final_synwd}_rfcr0r1_iid

resdir=results/clovelinear_ks${final_ks}_camelyon17_${mark}_lambda${final_lambda}_seed${final_seed} 
syndir=results/syn_camelyon17_2round_rfc1erm_rfc20.01_even_evaliid_wd${final_synwd}_seed${final_seed}
mkdir $resdir

python examples/analysis_earlystop.py --result_dir ${syndir}


python examples/run_expt_synthesis_convert_model.py  \
--version "1.0" --root_dir data/camelyon17/ --save_step 1  \
--rfc_groups_dir  ${syndir} \
--log_dir ${syndir} \
--weight_decay 0.01 --dataset camelyon17 --algorithm mtERM --model densenet121 \
--seed ${final_seed} --loss_function rfcmt_cross_entropy  --algo_log_metric rfcmtaccuracy \
--process_outputs_function None --eval_only --save_model_dir ${resdir} \
--eval_epoch `cat ${syndir}/eval_epochs_iid.txt` --rand_classifier ${final_randclf}

echo "converted model"

python examples/run_expt.py --version "1.0" --root_dir data/camelyon17/ \
--log_dir ${resdir} \
--dataset camelyon17 --algorithm CLOvE --model densenet121 --seed ${final_seed} \
--save_step 1 --irm_penalty_anneal_iters 0 --irm_lambda ${final_lambda} --resume --freeze_featurizer True --kernel_scale ${final_ks}






