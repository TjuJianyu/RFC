
round=1

data=coloredmnist025;
ermstep=50;
rfcstep=500;
synstep=500;
l2=0.0011;
lr=0.0005;

ermresdir=results/${data}_erm${ermstep}_l2${l2}_lr${lr}
rfcresdir=${ermresdir}_rfc${rfcstep}
synr0r1resdir=${rfcresdir}_synr0r1${synstep}

# RFC round0 (erm)
python coloredmnist/run_exp.py  --methods erm  --verbose True --dataset ${data} \
--steps ${ermstep} --save_dir ${ermresdir}  --penalty_anneal_iters 0 \
--hidden_dim 390 --l2_regularizer_weight ${l2} \
--lr ${lr} --n_restarts ${round} 

# RFC round1 
python  coloredmnist/run_exp.py  --methods dro  --verbose True --steps ${rfcstep} --penalty_anneal_iters 0 \
--penalty_weight 0 --dataset ${data}  \
--save_dir ${rfcresdir} --group_dirs ${ermresdir} --l2_regularizer_weight ${l2} \
--lr ${lr} --n_restarts ${round} 

# RFC synthesis phase
python  coloredmnist/run_exp_syn.py   --verbose True --steps ${synstep} \
--save_dir ${synr0r1resdir} \
--dataset ${data} \
--group_dirs ${ermresdir} ${rfcresdir} \
--penalty_anneal_iters 0 \
--penalty_weight 0 \
--l2_regularizer_weight ${l2} \
--lr ${lr} --n_restarts ${round} 

for freeze_featurizer in True False;
do 
	
	for method in irm clove vrex iga fishr sd;
	do
		if [ ${method} == irm ]
		then 
			methodstep=3001
		else 
			methodstep=1001
		fi 

		if [ ${method} == sd ]
		then 
			penalty_weight_array=(10 50 100 500 1000)
		else
			penalty_weight_array=(1000 5000 10000 50000 100000)
		fi

		for i in 0 1 2 3 4;
		do 
			penalty_weight=${penalty_weight_array[$i]}
			
			linearresdir=${synr0r1resdir}_${method}_penalty${penalty_weight}_freeze${freeze_featurizer}
			python  coloredmnist/run_exp.py  --methods ${method}  --verbose True --steps ${methodstep} \
			--save_dir ${linearresdir} --load_model_dir ${synr0r1resdir} \
			--penalty_anneal_iters 0 \
			--dataset ${data} \
			--penalty_weight ${penalty_weight} \
			--l2_regularizer_weight ${l2} \
			--lr ${lr} \
			--freeze_featurizer ${freeze_featurizer} \
			--eval_steps 1 --n_restarts ${round} 
		done 
	done 


	for method in erm;
	do 
		for l2 in  0.0001 0.0005 0.001 0.005 0.01;
		do
			methodstep=1001
			linearresdir=${synr0r1resdir}_${method}_l2${l2}_freeze${freeze_featurizer}
			python  coloredmnist/run_exp.py  --methods ${method}  --verbose True --steps ${methodstep} \
			--save_dir ${linearresdir} --load_model_dir ${synr0r1resdir} \
			--penalty_anneal_iters 0 \
			--dataset ${data} \
			--l2_regularizer_weight ${l2} \
			--lr ${lr} \
			--freeze_featurizer ${freeze_featurizer} \
			--eval_steps 1 --n_restarts ${round} 
		done 
	done 


done 


