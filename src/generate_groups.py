import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from utils import ParseKwargs, parse_bool
import numpy as np 
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('--pred_dirs', nargs='*',type=str)
parser.add_argument('--pred_epochs', nargs='*',type=int)
parser.add_argument('--pred_seed', type=int)
#parser.add_argument('--eval_mode', default=None, type=str)
parser.add_argument('--result_dir', type=str)


# Dataset
parser.add_argument('--root_dir', default='data/camelyon17', 
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
parser.add_argument('--split_scheme',  default='official', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, tries to download the dataset if it does not exist in root_dir.')
parser.add_argument('--version', default="1.0", type=str, help='WILDS labeled dataset version number.')



config = parser.parse_args()




full_dataset = Camelyon17Dataset(
    version=config.version, 
    root_dir=config.root_dir, 
    download=config.download, 
    split_scheme=config.split_scheme, 
    **config.dataset_kwargs)

split_array = full_dataset.split_array
split_dict = full_dataset.split_dict
y = full_dataset.y_array.numpy().astype(int)

print(np.unique(split_array))
print(split_array)
print(len(split_array))
# if config.eval_mode == 'iid':
# 	config.pred_epochs = []
# 	for val in config.pred_dirs:
# 		tmp = pd.read_csv(os.path.join(val, 'eval_epochs.csv'))['idvalidx'].values[0]
# 		config.pred_epochs.append(tmp)
# elif config.eval_mode == 'ood':
# 	config.pred_epochs = []
# 	for val in config.pred_dirs:
# 		tmp = pd.read_csv(os.path.join(val, 'eval_epochs.csv'))['validx'].values[0]
# 		config.pred_epochs.append(tmp)

assert len(config.pred_dirs) == len(config.pred_epochs)

results  = np.zeros((len(split_array), len(config.pred_dirs))) - 1 
labeldistributions=None
print(config.pred_dirs, config.pred_epochs)

for i in range(len(config.pred_dirs)):
	for split_name in ['train','id_val','test','val']:
		res = pd.read_csv(os.path.join(config.pred_dirs[i], 'camelyon17_split:%s_seed:%d_epoch:%d_pred.csv' \
			% (split_name, config.pred_seed, config.pred_epochs[i])), header=None)
		res = res[0].values
		
		logits = pd.read_csv(os.path.join(config.pred_dirs[i], 'camelyon17_split:%s_seed:%d_epoch:%d_logits.csv' \
			% (split_name, config.pred_seed, config.pred_epochs[i])), header=None).values
		if labeldistributions is None:
			labeldistributions = np.zeros((len(split_array), len(config.pred_dirs)*logits.shape[1])) - 1 
		
		prob = torch.nn.functional.softmax(torch.FloatTensor(logits),dim=1).numpy()
		labeldistributions[split_array == split_dict[split_name], i*logits.shape[1]:(i+1)*logits.shape[1]] = prob
		results[split_array == split_dict[split_name],i] = res.flatten()

results = results.astype(int)
assert (results== -1).sum() == 0
assert (labeldistributions == -1).sum() == 0

pd.DataFrame(results).to_csv(os.path.join(config.result_dir, 'pseudolabels.csv'), header=None, index=None)
pd.DataFrame(labeldistributions).to_csv(os.path.join(config.result_dir, 'pseudolabel_dists.csv'), header=None, index=None)
ngroups = 0
for i in range(results.shape[1]):
	results[:,i] = (results[:,i] ^ y) + ngroups
	assert len(np.unique(results[:,i])) == 2
	ngroups += 2
print(results.mean(axis=0))

pd.DataFrame(results).to_csv(os.path.join(config.result_dir, 'groups.csv'), header=None, index=None)





