import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
import copy
import os 
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from collections import OrderedDict

#from rich_representation import feature_disentangle, rich_representation_by_sampling , RFC

from mydatasets import coloredmnist,cow_camel
from models import MLP, TopMLP
from utils import pretty_print, correct_pred,GeneralizedCELoss, EMA, mean_weight, mean_nll, mean_mse, mean_accuracy,validation

from train import get_train_func



def main(flags):
    if flags.verbose:
        print('Flags:')
        for k,v in sorted(vars(flags).items()):
            print("\t{}: {}".format(k, v))
    if flags.save_dir is not None and not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    final_train_accs = []
    final_train_losses = []
    final_test_accs = []
    final_test_losses = []
    logs = []


    for restart in range(flags.n_restarts):

        if flags.verbose:
            print("Restart", restart)
        

        ### loss function binary_cross_entropy 
        input_dim = 2 * 14 * 14
        if flags.methods in ['rsc', 'lff']:
            n_targets = 2
            lossf = F.cross_entropy
            int_target = True 
        else:
            n_targets = 1 
            lossf = mean_nll 
            int_target = False  


        ### load datasets 
        if flags.dataset == 'coloredmnist025':
            envs, test_envs = coloredmnist(0.25, 0.1, 0.2,seed=restart, int_target = int_target)
        elif flags.dataset == 'coloredmnist025gray':
            envs, test_envs = coloredmnist(0.25, 0.5, 0.5,seed=restart, int_target = int_target)
        elif flags.dataset == 'coloredmnist01':
            envs, test_envs = coloredmnist(0.1, 0.2, 0.25,seed=restart, int_target = int_target)
        elif flags.dataset == 'coloredmnist01gray':
            envs, test_envs = coloredmnist(0.1, 0.5, 0.5, seed=restart, int_target = int_target)
        else:
            raise NotImplementedError

        n_groups = 2
        mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
        topmlp = TopMLP(hidden_dim = flags.hidden_dim, n_top_layers=1, \
            n_targets=n_targets*n_groups, fishr= flags.methods=='fishr').cuda()


        if flags.load_model_dir is not None and os.path.exists(flags.load_model_dir):
            device = torch.device("cuda")
            state = torch.load(os.path.join(flags.load_model_dir,'mlp%d.pth' % restart), map_location=device)
            mlp.load_state_dict(state)
            state = torch.load(os.path.join(flags.load_model_dir,'topmlp_multitask%d.pth' % restart), map_location=device)
            topmlp.load_state_dict(state)
            print("Load model from %s" % flags.load_model_dir)
        
        n_groups = 2 
        for env in envs + test_envs:
            logits = topmlp(mlp(env['images']))

            for i in range(n_groups):
                acc = mean_accuracy(logits[:,i:i+1], env['labels'])

                print(acc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Colored MNIST & CowCamel')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='coloredmnist025')
    parser.add_argument('--hidden_dim', type=int, default=390)
    #parser.add_argument('--n_top_layers', type=int, default=1)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.0011)
    #parser.add_argument('--rich_lr', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--steps', type=int, default=501)
    #parser.add_argument('--lossf', type=str, default='nll')
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--anneal_val', type=float, default=1)
    #parser.add_argument('--rep_init', type=str, default='rich_rep')
    #parser.add_argument('--train_rate', type=float, default=0.99)
    parser.add_argument('--methods', type=str, default='erm')
    parser.add_argument('--lr_s2_decay', type=float, default=500)
    #parser.add_argument('--bias', type=bool, default=False)
    #parser.add_argument('--featuredistangle_reinit', type=bool, default=False)
    
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--group_dirs', type=str, nargs='*',default={})
    
    # #IRMV2
    # parser.add_argument('--inner_steps', type=int, default=6)
    # parser.add_argument('--inner_lr', type=float, default=0.015)
    # parser.add_argument('--learner_match_order', type=str, default='order2')
    # parser.add_argument('--n_matching_points', type=int, default=3)

    #RSC

    parser.add_argument('--rsc_f', type=float, default=0.99)
    parser.add_argument('--rsc_b', type=float, default=0.97)

    #parser.add_argument('--grayscale_model', action='store_true')
    
    #parser.add_argument('--penalty', type=str, default='irm')
    #parser.add_argument('--label_noise_rate', type=float, default=0.25)
    #parser.add_argument('--trenv1', type=float, default=0.1)
    #parser.add_argument('--trenv2', type=float, default=0.2)
    #parser.add_argument('--rep_size', type=int, default=1)
    
    #parser.add_argument('--freeze_layers', type=int, default=2)
    
    #parser.add_argument('--eps', type=float, default=0.0000001)
    #parser.add_argument('--inner_steps', type=int, default=1)
    #parser.add_argument('--learner_match_order', type=str, default='order1')
    #parser.add_argument('--erm_coin_match', type=str, default='False')
    #parser.add_argument('--n_matching_points', type=int, default=3)
    
    #parser.add_argument('--inner_lr', type=float, default=0.001)
    
    flags = parser.parse_args()

    main(flags)

# python -u inv_by_worse_data_sample_vrex.py   --hidden_dim=390   --l2_regularizer_weight=0.00110794568   --lr=0.0004898536566546834   --penalty_anneal_iters=0   --penalty_weight=10000 --steps=1001 --verbose True --lossf mse --anneal_val=1 --label_noise_rate=0.25 --trenv1=0.1 --trenv2=0.2 --inner_lr=0.015 --inner_steps=6 --learner_match_order=order2 --n_matching_points=3


# python -u rich_rep_ood.py   --hidden_dim=390   --l2_regularizer_weight=0.00110794568   --lr=0.0005   --verbose True --lossf mse --anneal_val=1 --inner_steps=6 --learner_match_order=order2 --n_matching_points=3 --penalty_weight=90000 --penalty_anneal_iters=0 --steps=301 --methods irmv2 --dataset coloredmnist025 --rich_lr 0.0005 --lr 0.000025

# python -u rich_rep_ood.py   --hidden_dim=390   --l2_regularizer_weight=0.00110794568   --lr=0.0005   --verbose True --lossf mse --anneal_val=1 --inner_steps=4 --inner_lr 0.015 --learner_match_order=order1 --n_matching_points=3 --penalty_weight=90000 --penalty_anneal_iters=0 --steps=301 --methods irmv2 --dataset coloredmnist025 --rich_lr 0.0005 --lr 0.00005
#
#
#
#