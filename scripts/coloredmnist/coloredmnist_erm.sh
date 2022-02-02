
for data in coloredmnist025;
do 
	methodstep=1001
	anneal=0
	penalty_weight=0
	for method in  erm oracle; 
	do
		for l2 in 0.0001 0.0005 0.001 0.005 0.01;
		do 	
			resdir=${data}_${anneal}_${method}_${penalty_weight}
			final_data=${data}

			if [ $method == oracle ]
			then 
				final_data=${data}gray
			fi 
			#echo ${final_data}
			python coloredmnist/run_exp.py  --methods erm  --verbose True --steps ${methodstep} \
			--save_dir ${resdir} \
			--penalty_anneal_iters ${anneal} \
			--dataset ${data} \
			--penalty_weight ${penalty_weight} \
			--l2_regularizer_weight ${l2} \
			--lr 0.0005 \
			--freeze_featurizer False --anneal_val 0  --eval_steps 1 \
			--norun True 
		done 
	done 
done 

