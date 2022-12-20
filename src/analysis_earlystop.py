import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import os
import time
import argparse
import pandas as pd

import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir',type=str)
parser.add_argument('--epochs',type=int,default=999999)

config = parser.parse_args()


result  = []
for name in ['train', 'val', 'id_val', 'test']:
	tmp = pd.read_csv(os.path.join(config.result_dir, '%s_eval.csv' % name ))['acc_avg'].values
	result.append(tmp)
train, val, idval, test= result 

validx = np.argmax(val[:config.epochs])
idvalidx = np.argmax(idval[:config.epochs])

res = pd.DataFrame([[validx,idvalidx]], columns=['validx','idvalidx'])

res.to_csv(os.path.join(config.result_dir, 'eval_epochs.csv'))

with open(os.path.join(config.result_dir, 'eval_epochs_iid.txt'), 'w') as f:
	f.write("%d" % idvalidx)
f.close()

with open(os.path.join(config.result_dir, 'eval_epochs_ood.txt'), 'w') as f:
	f.write("%d" % validx)
f.close()


print('validx %d, idvalidx %d' % (validx, idvalidx))
print('val test %.3f, idval test %.3f' % (test[validx], test[idvalidx]))
print('val idval %.3f, idval idval %.3f' % (idval[validx], idval[idvalidx]))


