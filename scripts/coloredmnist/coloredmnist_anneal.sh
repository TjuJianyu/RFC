

for data in coloredmnist025;
do

	if [ $data == coloredmnist025 ]
	then 
		l2=0.0011
	fi 

	for anneal in 0 50 100 150 200 250;
	do 
		for method in irm vrex iga fishr clove lff sd gm rsc; 
		do
			penalty_weight_array=(1000 5000 10000 50000 100000)
			anneal_val=1
			methodstep=1001

			if [ $method == lff ]
			then 
				penalty_weight_array=(0.1 0.2 0.3 0.4 0.5)
				anneal_val=0

			elif [ $method == sd ]
			then
				penalty_weight_array=(10 50 100 500 100)
	

			elif [ $method == gm ]
			then 
				penalty_weight_array=(0.0001 0.0005 0.001 0.005 0.01)
				anneal_val=0

			elif [ $method == rsc ]
			then
				penalty_weight_array=(0.95 0.97 0.98 0.99 1)
				rsc_f_array=(0.94525 0.96515 0.9751  0.98505 0.995)
				rsc_b_array=(0.931 0.9506 0.9604 0.9702 0.98 )
			
			fi 

			for i in 0 1 2 3 4;
			do 
				penalty_weight=${penalty_weight_array[$i]}
				
				resdir=results/${data}_${anneal}_${method}_${penalty_weight}
				
	
				if [ $method == rsc ]
				then 
					rsc_f=${rsc_f_array[$i]}
					rsc_b=${rsc_b_array[$i]}

					python coloredmnist/run_exp.py  --methods ${method}  --verbose True --steps ${methodstep} \
					--save_dir ${resdir} \
					--penalty_anneal_iters ${anneal} \
					--dataset ${data} \
					--penalty_weight ${penalty_weight} \
					--l2_regularizer_weight ${l2} \
					--lr 0.0005 --hidden_dim 390 \
					--freeze_featurizer False \
					--eval_steps 1 \
					--rsc_f ${rsc_f} --rsc_b ${rsc_b} --anneal_val ${anneal_val} \
					--norun False 
				else 
					python coloredmnist/run_exp.py  --methods ${method}  --verbose True --steps ${methodstep} \
					--save_dir ${resdir} \
					--penalty_anneal_iters ${anneal} \
					--dataset ${data} \
					--penalty_weight ${penalty_weight} \
					--l2_regularizer_weight ${l2} \
					--lr 0.0005 --hidden_dim 390 \
					--freeze_featurizer False \
					--eval_steps 1 --anneal_val ${anneal_val} \
					--norun False 
				fi 
			done 
		done 
	done 
done 

		


