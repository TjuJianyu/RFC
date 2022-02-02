from rich_rep_ood import main as rich_main
import sys 
import time
import numpy as np  
class FLAGS(object):
	"""docstring for FLAGS"""
	def __init__(self):
		super(FLAGS, self).__init__()
		self.verbose = True 
		self.n_restarts = 10
		self.dataset = 'coloredmnist01'
		self.hidden_dim = 390
		
		self.l2_regularizer_weight = 0.0011
		self.rich_lr = 0.0005 
		self.lr = self.rich_lr 
		self.steps = 301
		self.lossf = 'nll'
		self.featuredistangle_reinit = True
		self.penalty_anneal_iters = 0 
		self.penalty_weight = 90000.0
		self.anneal_val = 1 
		self.rep_init = 'none'
		self.methods = 'irmv2'
		self.lr_s2_decay = 500
		# self.bias=False
		# self.n_top_layers = 2

		self.bias=True
		self.n_top_layers = 1
		#irmv2
		self.learner_match_order = 'order2'
		self.n_matching_points = 3 
		##irmv2 first order
		self.inner_steps = 4
		self.inner_lr = 0.015
		
		#rsc
		self.rsc_f = 0.99 
		self.rsc_b = 0.97


methods  = sys.argv[1]
step_factor = 1



t = time.time()



if methods == 'irmv2':
	# irmv2 
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'irmv2'
			flags.rep_init = 'none'
			flags.penalty_anneal_iters = penalty_anneal_iters 
			flags.penalty_weight = penalty_weight
			flags.lr = flags.rich_lr
			flags.steps = 701 
			flags.dataset = 'coloredmnist01'
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == '1stirmv2':
	# irmv2 first order
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'irmv2'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.learner_match_order = 'order1'
			flags.penalty_anneal_iters = penalty_anneal_iters 
			flags.penalty_weight = penalty_weight
			flags.lr = flags.rich_lr
			flags.steps = 701 
			
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'sd':
	#sd
	p = penalty_weight_center = 100
	for penalty_anneal_iters in [  250,0, 50, 100,150, 200,]:
		for penalty_weight in [ p, p*5, p*10,p//10, p//2,]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'sd'
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.anneal_val = 1
			flags.penalty_weight = penalty_weight
			flags.rep_init = 'none'
			flags.lr = flags.rich_lr 
			flags.dataset = 'coloredmnist01'
			flags.steps = 701
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'vrex':

	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			# penalty_anneal_iters = 50 
			t = time.time()
			flags = FLAGS()
			flags.methods = 'vrex'
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.rep_init = 'none'
			flags.lr = flags.rich_lr 
			flags.dataset = 'coloredmnist01'
			flags.steps = 701
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'rsc':
	p = 1
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p*0.95, p*0.97, p*0.98, p*0.99, p]:
			t = time.time()
			flags = FLAGS()
			flags.rsc_f = 0.995 * penalty_weight
			flags.rsc_b = 0.98 * penalty_weight

			flags.methods = 'rsc'
			flags.lr = flags.rich_lr 
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.steps  = penalty_anneal_iters + 25 
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%.6f.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'iga':
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
	#for penalty_anneal_iters in [200, 250]:
	
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'iga'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.lr = flags.rich_lr
			flags.steps  = 701
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'irmv1':
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			# penalty_weight = 10000
			# penalty_anneal_iters = 150
			t = time.time()
			flags = FLAGS()
			flags.methods = 'irmv1'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.lr = flags.rich_lr 
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.steps = 701
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'gm':
	p = penalty_weight_center = 0.001
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p/10, p/2, p, p*5, p*10]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'gm'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.lr = flags.rich_lr 
			flags.anneal_val = 0
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.steps = 701 
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%.6f.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'lff':
	#for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
	for penalty_anneal_iters in [200, 250]:
	
		for penalty_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'lff'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.lr = flags.rich_lr 
			flags.anneal_val = 0
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.steps = 701 
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%.1f.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'erm':
	for l2_regularizer_weight in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
		t = time.time()
		flags = FLAGS()
		flags.l2_regularizer_weight = l2_regularizer_weight
		flags.methods = 'erm'
		flags.rep_init = 'none'
		flags.dataset = 'coloredmnist01'
		flags.lr = flags.rich_lr 
		flags.anneal_val = 0
		flags.penalty_anneal_iters = 0
		flags.penalty_weight = 0
		flags.steps = 701 
		results, logs = rich_main(flags)
		np.save('log/final/coloredmnist01_nll_%s_%d_%.6f.npy' %  \
			(methods, 0, l2_regularizer_weight), logs)
		print(time.time() - t)

elif methods == 'oracle':
	for l2_regularizer_weight in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
		t = time.time()
		flags = FLAGS()
		flags.l2_regularizer_weight = l2_regularizer_weight
		flags.methods = 'erm'
		flags.rep_init = 'none'
		flags.dataset = 'coloredmnist01gray'
		flags.lr = flags.rich_lr 
		flags.anneal_val = 0
		flags.penalty_anneal_iters = 0
		flags.penalty_weight = 0
		flags.steps = 701 
		results, logs = rich_main(flags)
		np.save('log/final/coloredmnist01_nll_%s_%d_%.6f.npy' %  \
			(methods, 0, l2_regularizer_weight), logs)
		print(time.time() - t)

elif methods == 'fishr':
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:
			t = time.time()
			flags = FLAGS()
			flags.methods = 'fishr'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.lr = flags.rich_lr 
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.steps = 701
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)

elif methods == 'clove':
	p = penalty_weight_center = 10000
	for penalty_anneal_iters in [0, 50, 100, 150, 200, 250]:
		for penalty_weight in [p//10, p//2, p, p*5, p*10]:

			t = time.time()
			flags = FLAGS()
			flags.methods = 'clove'
			flags.rep_init = 'none'
			flags.dataset = 'coloredmnist01'
			flags.lr = flags.rich_lr 
			flags.penalty_anneal_iters = penalty_anneal_iters
			flags.penalty_weight = penalty_weight
			flags.steps = 1001
			results, logs = rich_main(flags)
			np.save('log/final/coloredmnist01_nll_%s_%d_%d.npy' %  \
				(methods, penalty_anneal_iters, penalty_weight), logs)
			print(time.time() - t)
# if methods == 'irmv2':
# 	# irmv2 
# 	flags = FLAGS()
# 	flags.methods = 'irmv2'
# 	flags.lr = flags.rich_lr / 20
# 	flags.steps = 301 * step_factor
# 	rich_main(flags)

# elif methods == '1stirmv2':
# 	# irmv2 first order
# 	flags = FLAGS()
# 	flags.methods = 'irmv2'
# 	flags.learner_match_order = 'order1'
# 	flags.lr = flags.rich_lr / 10
# 	flags.steps = 301 * step_factor
# 	rich_main(flags)
# elif methods == 'sd':
# 	#sd
# 	flags = FLAGS()
# 	flags.methods = 'sd'
# 	flags.penalty_weight = 100
# 	flags.lr = flags.rich_lr/500
# 	flags.steps = 2001 * step_factor
# 	rich_main(flags)
# elif methods == 'vrex':
# 	flags = FLAGS()
# 	flags.methods = 'vrex'
# 	flags.lr = flags.rich_lr / 20
# 	flags.penalty_weight = 20000
# 	flags.steps = 501 * step_factor
# 	rich_main(flags)
# elif methods == 'rsc':
# 	flags = FLAGS()
# 	flags.methods = 'rsc'
# 	flags.lr = flags.rich_lr / 20
# 	flags.steps  = 501 * step_factor
# 	rich_main(flags)
# elif methods == 'iga':
# 	flags = FLAGS()
# 	flags.methods = 'iga'
# 	flags.penalty_weight = 10000
# 	flags.lr = flags.rich_lr / 80
# 	flags.steps  = 1601 * step_factor
# 	rich_main(flags)

# elif methods == 'irmv1':
# 	flags = FLAGS()
# 	flags.methods = 'irmv1'
# 	flags.lr = flags.rich_lr / 40
# 	flags.steps = 501 * step_factor
# 	rich_main(flags)

# elif methods == 'gm':
# 	flags = FLAGS()
# 	flags.methods = 'gm'
# 	flags.lr = flags.rich_lr / 80
# 	flags.steps = 301 * step_factor
# 	rich_main(flags)

# 	pass








