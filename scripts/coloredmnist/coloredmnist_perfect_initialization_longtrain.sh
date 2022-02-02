
python coloredmnist/run_exp.py  --methods erm  --verbose True \
--dataset coloredmnist025gray --steps 500 \
--save_dir results/coloredmnist025gray  \
--penalty_anneal_iters 0 --hidden_dim 390 \
--l2_regularizer_weight 0.0011  --steps 80


for method in irm vrex iga fishr clove;
do 
	python coloredmnist/run_exp.py  --methods ${method}  --verbose True \
	--dataset coloredmnist025 \
	--load_model_dir results/coloredmnist025gray/ \
	--lr 0.0005 --save_dir results/coloredmnist025_perfectinit_${method}_100000 \
	--penalty_anneal_iters 0 --hidden_dim 390 --l2_regularizer_weight 0.0011  \
	--steps 10001 --penalty_weight 100000 
done 

for method in sd;
do 
	python coloredmnist/run_exp.py  --methods erm  --verbose True \
	--dataset coloredmnist025 \
	--load_model_dir results/coloredmnist025gray/ \
	--lr 0.0005 --save_dir results/coloredmnist025_perfectinit_${method}_1000 \
	--penalty_anneal_iters 0 --hidden_dim 390 --l2_regularizer_weight 0.0011  \
	--steps 10001 --penalty_weight 1000 
done 
