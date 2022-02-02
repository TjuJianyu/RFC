import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
import copy

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from collections import OrderedDict

from rich_representation import feature_disentangle, rich_representation_by_sampling , RFC

from mydatasets import coloredmnist,cow_camel
from models import MLP, TopMLP
from utils import pretty_print, correct_pred,GeneralizedCELoss, EMA, mean_weight, mean_nll, mean_mse, mean_accuracy,validation


def IGA_penalty(envs_logits, envs_y, scale, lossf):
    
    grads = []
    grad_mean = 0
    for i in range(len(envs_logits)):

        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = [val.view(-1) for val in autograd.grad(loss, scale, create_graph=True)]
        grad0 = torch.cat(grad0)
        grads.append(grad0)
        grad_mean += grad0 / len(envs_logits)

    grad_mean  = grad_mean.detach()

    train_penalty = 0 
    for i in range(len(grads)):
        train_penalty += torch.sum((grads[i] - grad_mean)**2) 

    return train_penalty 

def IRM_penalty(envs_logits, envs_y, scale, lossf):

    #scale = torch.tensor([1.]*flags.rep_size).cuda().requires_grad_()
    #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
    train_penalty = 0 
    for i in range(len(envs_logits)):
        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = autograd.grad(loss, [scale], create_graph=True)[0]
        train_penalty += torch.sum(grad0**2)

    train_penalty /= len(envs_logits)

    return train_penalty

def GM_penalty(envs_logits, envs_y, scale, lossf):
    #@lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
    
    grads = []
    grad_mean = 0
    for i in range(len(envs_logits)):

        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = [val.view(-1) for val in autograd.grad(loss, scale, create_graph=True)]
        grad0 = torch.cat(grad0)
        grads.append(grad0)

    train_penalty  = 0 
    for i in range(len(grads)-1):
        for j in range(i+1,len(grads)):
            train_penalty += -torch.sum(grads[i]*grads[j])
    
    return train_penalty


def main(flags):
    if flags.verbose:
        print('Flags:')
        for k,v in sorted(vars(flags).items()):
            print("\t{}: {}".format(k, v))

    # parameters setting
    if flags.dataset[: len('coloredmnist025')] == 'coloredmnist025':
        input_dim = 2 * 14 * 14
        # feature_disentangle_steps = [5,495]
        # rich_rep_steps = 401
        if flags.lossf == 'mse':
            feature_disentangle_steps = [5,501]
            rich_rep_steps = 251

            RFC_steps = [5,501,251]
        else:
            #feature_disentangle_steps = [5,301]
            #rich_rep_steps = 251

            feature_disentangle_steps = [5,501]
            rich_rep_steps = 351
            RFC_steps = [5,501,351]
            # feature_disentangle_steps = [10,701]
            # rich_rep_steps = 551
        # feature_disentangle_steps = [10,301]
        # rich_rep_steps = 251
    elif flags.dataset[: len('coloredmnist01')] == 'coloredmnist01':
        input_dim = 2 * 14 * 14
        if flags.lossf == 'mse':
            raise NotImplementedError
        else:
            feature_disentangle_steps = [50,401]
            rich_rep_steps = 201




    final_train_accs = []
    final_train_losses = []
    final_test_accs = []
    final_test_losses = []
    logs = []


    for restart in range(flags.n_restarts):

        if flags.verbose:
            print("Restart", restart)
            #pretty_print('step', 'tr_cross_loss', 'train acc', 'train penalty', 'test loss','test acc', 'lossa', 'lossb')

        ### loss function binary_cross_entropy or mse 
        if flags.methods in ['rsc', 'lff']:
            n_targets = 2
            #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
            lossf = F.cross_entropy
            int_target = True 
        else:
            n_targets = 1 
            lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
            int_target = False 
        
        ### load datasets 
        

        if flags.dataset == 'coloredmnist025':
            envs, test_envs = coloredmnist(0.25, 0.1, 0.2, int_target = int_target)
        elif flags.dataset == 'coloredmnist025gray':
            envs, test_envs = coloredmnist(0.25, 0.5, 0.5, int_target = int_target)
        elif flags.dataset == 'coloredmnist01':
            envs, test_envs = coloredmnist(0.1, 0.2, 0.25, int_target = int_target)
        elif flags.dataset == 'coloredmnist01gray':
            envs, test_envs = coloredmnist(0.1, 0.5, 0.5, int_target = int_target)
        else:
            raise NotImplementedError
        
            

        def get_topmlp_func():
                return TopMLP(hidden_dim = flags.hidden_dim, \
                 n_top_layers=flags.n_top_layers, n_targets=n_targets,\
                 fishr= flags.methods=='fishr').cuda()
        

        def get_model_optimizer(mode = 0, n_targets=1):
            if mode == 0:

                _mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
                _topmlp =get_topmlp_func()
                #_model = torch.nn.Sequential(_mlp, _topmlp).cuda()

                optimizer =  optim.Adam([var for var in _mlp.parameters()]+[var for var in _topmlp.parameters()], \
                    weight_decay=flags.l2_regularizer_weight, lr=flags.rich_lr)
                return _mlp, _topmlp, optimizer
            
            elif mode == 1:
                _mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
                _topmlp = TopMLP(hidden_dim = flags.hidden_dim,  n_top_layers=1, n_targets=n_targets).cuda()
        
                #_model = torch.nn.Sequential(_mlp, _topmlp).cuda()

                optimizer =  optim.Adam([var for var in _mlp.parameters()]+[var for var in _topmlp.parameters()], \
                    weight_decay=flags.l2_regularizer_weight, lr=flags.rich_lr)
                return _mlp, _topmlp, optimizer


        ### representation initialization 
        if flags.rep_init == 'rich_rep':
            x = torch.cat([envs[i]['images'] for i in range(len(envs))])
            y = torch.cat([envs[i]['labels'] for i in range(len(envs))])
            #weight_decay = flags.l2_regularizer_weight
            
            mlp, topmlp = RFC(x,y,RFC_steps, get_model_optimizer, lossf, correct_pred, tol=500, verbose=True,train_rate=0.9, eval_steps=50)
    

            # _model, optimizer = get_model_optimizer()
            # model_optimizer_generator = get_model_optimizer 


            # A,B,_ = feature_disentangle(x,y, feature_disentangle_steps, _model, lossf, optimizer, \
            #     correct_pred, tol = 500, verbose=flags.verbose, model_optimizer_generator = model_optimizer_generator)
            
            # mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
            # topmlp = get_topmlp_func()
            # model = torch.nn.Sequential(mlp, topmlp)
            # optimizer = optim.Adam(model.parameters(), \
            #     weight_decay=flags.l2_regularizer_weight, lr=flags.rich_lr)

            # model = rich_representation_by_sampling(x,y,A,B, rich_rep_steps, model, lossf, optimizer,tol = 500,verbose=flags.verbose)
            
        else:
            mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
            topmlp = get_topmlp_func()
            #topmlp = TopMLP(hidden_dim = flags.hidden_dim,  n_top_layers=flags.n_top_layers, n_targets=n_targets, ).cuda()
        
            #model = torch.nn.Sequential(mlp, topmlp)



        print(mlp, topmlp)
        
        def rsc_train(mlp, topmlp,      steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):
            

            optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
                [var for var in topmlp.parameters()],  lr=lr) 
            drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
            drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
            num_classes = 2
            logs = []
            for step in range(steps):
                # inputs
                all_x = torch.cat([envs[i]['images'] for i in range(len(envs))])
                all_y = torch.cat([envs[i]['labels'] for i in range(len(envs))])



                # one-hot labels
                all_o = torch.nn.functional.one_hot(all_y, num_classes)
                # features
                all_f = mlp(all_x)
                # predictions
                all_p = topmlp(all_f)

                if step < penalty_anneal_iters:
                    loss = F.cross_entropy(all_p, all_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # elif step == penalty_anneal_iters:
                #     # reset optimizer
                #     optimizer = optim.Adam([var for var in mlp.parameters()] + \
                #         [var for var in topmlp.parameters()],  lr=lr / 2) 
                else:
                    # Equation (1): compute gradients with respect to representation
                    all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

                    # Equation (2): compute top-gradient-percentile mask
                    percentiles = np.percentile(all_g.cpu(), drop_f, axis=1)
                    percentiles = torch.Tensor(percentiles)
                    percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
                    mask_f = all_g.lt(percentiles.cuda()).float()

                    # Equation (3): mute top-gradient-percentile activations
                    all_f_muted = all_f * mask_f

                    # Equation (4): compute muted predictions
                    all_p_muted = topmlp(all_f_muted)

                    # Section 3.3: Batch Percentage
                    all_s = F.softmax(all_p, dim=1)
                    all_s_muted = F.softmax(all_p_muted, dim=1)
                    changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
                    percentile = np.percentile(changes.detach().cpu(), drop_b)
                    mask_b = changes.lt(percentile).float().view(-1, 1)
                    mask = torch.logical_or(mask_f, mask_b).float()

                    # Equations (3) and (4) again, this time mutting over examples
                    all_p_muted_again = topmlp(all_f * mask)

                    # Equation (5): update
                    loss = F.cross_entropy(all_p_muted_again, all_y)
                    #print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                
                if step % 1 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        np.int32(0),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
            #python -u rich_rep_irmv2.py   --hidden_dim=390   --l2_regularizer_weight=0.00110794568   --lr=0.0001  --verbose True --penalty_anneal_iters=100 --steps=501 --methods rsc --dataset coloredmnist025 --rep_init other
        
        def vrex_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True ):
            logs = []
            optimizer = optim.Adam([var for var in mlp.parameters()] + \
                [var for var in topmlp.parameters()],  lr=lr) 

            for step in range(steps):

                train_penalty = 0
                erm_losses = []
                for env in envs:
                    logits = topmlp(mlp(env['images']))
                    #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
                    env['nll'] = lossf(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                    erm_losses.append(env['nll'])

                erm_losses = torch.stack(erm_losses)
                #erm_loss = erm_loss.mean()
                #erm_loss = erm_losses.mean()
                train_penalty = erm_losses.var()
                erm_loss = erm_losses.sum() 

                #train_penalty = (envs[0]['nll'] - envs[1]['nll'])**2
                #erm_loss = (envs[0]['nll'] + envs[1]['nll'])

                loss = erm_loss.clone()

                weight_norm = 0
                for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)
                loss += l2_regularizer_weight * weight_norm

                penalty_weight = (penalty_term_weight if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
  
        def iga_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):

            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            for step in range(steps):
                train_penalty = 0
                envs_logits = []
                envs_y = []
                erm_loss = 0
                for env in envs:
                    logits = topmlp(mlp(env['images']))
                    #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
                    env['nll'] = lossf(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                    envs_logits.append(logits)
                    envs_y.append(env['labels'])
                    erm_loss += env['nll']

                train_penalty = IGA_penalty(envs_logits, envs_y, [var for var in mlp.parameters()] + [var for var in topmlp.parameters()], lossf)
            
                

                loss = erm_loss.clone()


                weight_norm = 0
                for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)
                loss += l2_regularizer_weight * weight_norm


                penalty_weight = (penalty_term_weight 
                        if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
        def sd_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={'lr_s2_decay':500}):
            optimizer_lower = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            for step in range(steps):

                train_penalty = 0
                erm_loss = 0
                for env in envs:
                    logits = topmlp(mlp(env['images']))
                    lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
                    env['nll'] = lossf(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                
                    train_penalty += (logits**2).mean() 
                    erm_loss += env['nll']
            

                loss = erm_loss.clone()


                weight_norm = 0
                for w in [var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)

                loss += l2_regularizer_weight * weight_norm


                penalty_weight = (penalty_term_weight 
                        if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight

                if penalty_anneal_iters > 0 and step >= penalty_anneal_iters:
                    # using anneal, so decay lr
                    loss /= hparams['lr_s2_decay']
                    #print(hparams['lr_s2_decay'])
                    #loss /= 500
                optimizer_lower.zero_grad()
                loss.backward()
                optimizer_lower.step()
                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
        def irm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):
            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            for step in range(steps):


                envs_logits = []
                envs_y = []
                erm_loss = 0
                scale = torch.tensor([1.])[0].cuda().requires_grad_() 
                for env in envs:
                    logits = topmlp(mlp(env['images'])) * scale
                    #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
                    env['nll'] = lossf(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                    envs_logits.append(logits)
                    envs_y.append(env['labels'])
                    erm_loss += env['nll']
                 
                train_penalty = IRM_penalty(envs_logits, envs_y,scale, lossf)

                loss = erm_loss.clone()


                weight_norm = 0
                for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)

                loss += l2_regularizer_weight * weight_norm


                penalty_weight = (penalty_term_weight 
                        if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
        def clove_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):
            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            #optimizer = optim.Adagrad([var for var in mlp.parameters()] \
            #        +[var for var in topmlp.parameters()], lr=lr)
            logs = []

            def mmce_penalty(logits, y, kernel='laplacian'):

                c = ~((logits.flatten() > 0) ^ (y.flatten()>0.5))
                c = c.detach().float()

                preds = F.sigmoid(logits).flatten()

                y_hat = (preds < 0.5).detach().bool()
                #print(preds)
                confidence = torch.ones(len(y_hat)).cuda()
                confidence[y_hat] = 1-preds[y_hat]
                confidence[~y_hat] = preds[~y_hat]

                k = (-(confidence.view(-1,1)-confidence).abs() / 0.4).exp()

                
                conf_diff = (c - confidence).view(-1,1)  * (c -confidence) 

                res = conf_diff * k

                return res.sum() / (len(logits)**2)

            pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

            batch_size = 512
            for step in range(steps):
                length = min(len(envs[0]['labels']), len(envs[1]['labels']))

                idx0 = np.arange(length)
                np.random.shuffle(idx0)
                idx1 = np.arange(length)
                np.random.shuffle(idx1)
                idx = [idx0, idx1]

                for i in range(length // batch_size):

                    train_penalty = 0
                    train_nll = 0
                    train_acc = 0
                    for j, env in enumerate(envs[0:2]):
                        x, y = env['images'], env['labels']
                        x_batch, y_batch = x[idx[j][i*batch_size:(i+1)*batch_size]], y[idx[j][i*batch_size:(i+1)*batch_size]]
                        logits = topmlp(mlp(x_batch))
                        nll = mean_nll(logits, y_batch)
                        acc = mean_accuracy(logits, y_batch)
                        mmce = mmce_penalty(logits, y_batch)
                        train_penalty += mmce
                        train_nll += nll 
                        train_acc += acc 

                #train_nll /=2
                train_acc /=2
                #train_penalty /= 2 


                weight_norm = torch.tensor(0.).cuda()
                for w in mlp.parameters():
                    weight_norm += w.norm().pow(2)

                loss = train_nll.clone()
                penalty_weight = (penalty_term_weight 
                            if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight

                optimizer.zero_grad()
                

                loss.backward()
                optimizer.step()
                

                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
                
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
           

        def fishr_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):

            def compute_grads_variance(features, labels, classifier):
                logits = classifier(features)
                loss = bce_extended(logits, labels)
                
                with backpack(BatchGrad()):
                    loss.backward(
                        inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
                    )

                dict_grads = OrderedDict(
                    [
                        (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                        for name, weights in classifier.named_parameters()
                    ]
                )
                dict_grads_variance = {}
                for name, _grads in dict_grads.items():
                    grads = _grads * labels.size(0)  # multiply by batch size
                    env_mean = grads.mean(dim=0, keepdim=True)
    
                    dict_grads_variance[name] = (grads).pow(2).mean(dim=0)

                return dict_grads_variance

            def l2_between_grads_variance(cov_1, cov_2):
                assert len(cov_1) == len(cov_2)
                cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
                cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
                return (
                    torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
                    torch.cat(tuple([t.view(-1) for t in cov_2_values]))
                ).pow(2).sum()
            
            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            
            bce_extended = extend(nn.BCEWithLogitsLoss())
            for step in range(steps):
                for edx, env in enumerate(envs):
                    #features, logits = topmlp(mlp(env['images']))
                    features = mlp(env['images'])
                    logits = topmlp(features)
                    env['nll'] = mean_nll(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                    #env['irm'] = compute_irm_penalty(logits, env['labels'])
                    if edx in [0, 1]:
                        # True when the dataset is in training
                        optimizer.zero_grad()
                        env["grads_variance"] = compute_grads_variance(features, env['labels'], topmlp)

                train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).sum()
                train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

                weight_norm = torch.tensor(0.).cuda()
                for w in mlp.parameters():
                    weight_norm += w.norm().pow(2)

                loss = train_nll.clone()
                loss += flags.l2_regularizer_weight * weight_norm

                dict_grads_variance_averaged = OrderedDict(
                    [
                        (
                            name,
                            torch.stack([envs[0]["grads_variance"][name], envs[1]["grads_variance"][name]],
                                        dim=0).mean(dim=0)
                        ) for name in envs[0]["grads_variance"]
                    ]
                )
                fishr_penalty = (
                    l2_between_grads_variance(envs[0]["grads_variance"], dict_grads_variance_averaged) +
                    l2_between_grads_variance(envs[1]["grads_variance"], dict_grads_variance_averaged)
                )
                train_penalty = fishr_penalty

                
                penalty_weight = (penalty_term_weight 
                            if step >= penalty_anneal_iters else anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
                
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
           


        def gm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):

            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            for step in range(steps):

                train_penalty = 0
                envs_logits = []
                envs_y = []
                for env in envs:
                    logits = topmlp(mlp(env['images']))
                    #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
                    env['nll'] = lossf(logits, env['labels'])
                    env['acc'] = mean_accuracy(logits, env['labels'])
                    envs_logits.append(logits)
                    envs_y.append(env['labels'])

                    
                
                train_penalty = GM_penalty(envs_logits, envs_y, [var for var in mlp.parameters()] + [var for var in topmlp.parameters()], lossf)
        
                erm_loss = (envs[0]['nll'] + envs[1]['nll'])
                

                loss = erm_loss.clone()


                weight_norm = 0
                for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)

                loss += flags.l2_regularizer_weight * weight_norm


                penalty_weight = (flags.penalty_weight 
                        if step >= flags.penalty_anneal_iters else flags.anneal_val)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
        def lff_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):
            
            x = torch.cat([envs[i]['images'] for i in range(len(envs))])
            y = torch.cat([envs[i]['labels'] for i in range(len(envs))])

            y = y.long().flatten()
            logs = []
            if penalty_anneal_iters > 0:
                optimizer = torch.optim.Adam([var for var in mlp.parameters()] \
                    + [var for var in topmlp.parameters()],
                    lr=lr, weight_decay=l2_regularizer_weight,)
                
                for step  in range(penalty_anneal_iters):
                    logits = topmlp(mlp(x))
                    loss = F.cross_entropy(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    train_penalty = torch.tensor([0]).cuda()[0]

                    if step % 5 == 0:
                        train_loss, train_acc, test_worst_loss, test_worst_acc = \
                        validation(topmlp, mlp, envs, test_envs, lossf)
                        log = [np.int32(step), train_loss, train_acc,\
                            train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                        logs.append(log)
                        if verbose:
                            pretty_print(*log)
       

            _mlp = copy.deepcopy(mlp)
            _topmlp = copy.deepcopy(topmlp)
            
            model_b = torch.nn.Sequential(_mlp, _topmlp)
      
            model_d = torch.nn.Sequential(mlp, topmlp)

            optimizer_b = torch.optim.Adam(
                model_b.parameters(),
                lr=lr / 100,
                weight_decay=l2_regularizer_weight,
            )
            optimizer_d = torch.optim.Adam(
                model_d.parameters(),
                lr=lr / 100,
                weight_decay= l2_regularizer_weight,
            )
            lossf = nn.CrossEntropyLoss(reduction='mean')
            criterion = nn.CrossEntropyLoss(reduction='none')
            bias_criterion = GeneralizedCELoss(q = penalty_term_weight)
            
            sample_loss_ema_b = EMA(y.cpu().numpy(), alpha=0.7)
            sample_loss_ema_d = EMA(y.cpu().numpy(), alpha=0.7)

            index = np.arange(len(y))
            for step in range(penalty_anneal_iters, steps):
                
                logit_b = model_b(x)
                logit_d = model_d(x)

                loss_b = criterion(logit_b, y).cpu().detach()
                loss_d = criterion(logit_d, y).cpu().detach()
                
                sample_loss_ema_b.update(loss_b,index)
                sample_loss_ema_d.update(loss_d,index)
                
                loss_b = sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = sample_loss_ema_d.parameter[index].clone().detach()

                # mnist target has one class, so I can do in this way.
                label_cpu = y.cpu()
                num_classes = 2
                for c in range(num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = sample_loss_ema_b.max_loss(c)
                    max_loss_d = sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d

                loss_weight = loss_b / (loss_b + loss_d + 1e-8)

                loss_b_update = bias_criterion(logit_b, y)
                loss_d_update = criterion(logit_d, y) * loss_weight.cuda()
                #print(loss_weight)
                loss = loss_b_update.mean() + loss_d_update.mean()
                #loss =loss_b_update.mean() +  criterion(logit_d, y).mean() 
                
                optimizer_b.zero_grad()
                optimizer_d.zero_grad()
                loss.backward()
                optimizer_b.step()
                optimizer_d.step()
            
                train_penalty = torch.tensor([0]).cuda()[0]

                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        
        def erm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
            penalty_anneal_iters, penalty_term_weight, anneal_val, \
            lr,l2_regularizer_weight, verbose=True,hparams={}):
            x = torch.cat([envs[i]['images'] for i in range(len(envs))])
            y = torch.cat([envs[i]['labels'] for i in range(len(envs))])

            optimizer = optim.Adam([var for var in mlp.parameters()] \
                    +[var for var in topmlp.parameters()], lr=lr)
            logs = []
            for step  in range(steps):
                logits = topmlp(mlp(x))
                loss = lossf(logits, y)
                weight_norm = 0
                for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                    weight_norm += w.norm().pow(2)

                loss += l2_regularizer_weight * weight_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_penalty = torch.tensor([0]).cuda()[0]

                if step % 5 == 0:
                    train_loss, train_acc, test_worst_loss, test_worst_acc = \
                    validation(topmlp, mlp, envs, test_envs, lossf)
                    log = [np.int32(step), train_loss, train_acc,\
                        train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                    logs.append(log)
                    if verbose:
                        pretty_print(*log)
            return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs
        

        if flags.verbose:
            pretty_print('step', 'tr_cross_loss', 'train acc', 'train penalty', 'test loss','test acc', 'lossa', 'lossb')

        params = [mlp, topmlp, flags.steps, envs, test_envs,lossf,\
            flags.penalty_anneal_iters, flags.penalty_weight, \
            flags.anneal_val, flags.lr, \
            flags.l2_regularizer_weight, flags.verbose]

        if flags.methods == 'vrex':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = vrex_train(*params)
          
        elif flags.methods == 'rsc':
            #hparams = {'rsc_f_drop_factor' : 0.99, 'rsc_b_drop_factor':0.97}
            hparams = {'rsc_f_drop_factor' : flags.rsc_f, 'rsc_b_drop_factor': flags.rsc_b}

            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = rsc_train(*params, hparams)
        
        elif flags.methods == 'irmv1':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = irm_train(*params)
        
        elif flags.methods == 'iga':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = iga_train(*params)
        
        elif flags.methods == 'sd':
            hparams = {'lr_s2_decay': flags.lr_s2_decay}
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = sd_train(*params, hparams)
        
        elif flags.methods == 'gm':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = gm_train(*params)
        
        elif flags.methods == 'lff':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = lff_train(*params)
        elif flags.methods == 'erm':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = erm_train(*params)
        elif flags.methods == 'fishr':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = fishr_train(*params)
        elif flags.methods == 'clove':
            (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = clove_train(*params)
           
        logs.extend(per_logs)
        final_train_accs.append(train_acc)
        final_train_losses.append(train_loss)
        final_test_accs.append(test_worst_acc)
        final_test_losses.append(test_worst_loss)

        if flags.verbose:
            print('Final train acc (mean/std across restarts so far):')
            print(np.mean(final_train_accs), np.std(final_train_accs))
            print('Final train loss (mean/std across restarts so far):')
            print(np.mean(final_train_losses), np.std(final_train_losses))
            print('Final worest test acc (mean/std across restarts so far):')
            print(np.mean(final_test_accs), np.std(final_test_accs))
            print('Final worest test loss (mean/std across restarts so far):')
            print(np.mean(final_test_losses), np.std(final_test_losses))

        results = [np.mean(final_train_accs), np.std(final_train_accs), 
                                np.mean(final_train_losses), np.std(final_train_losses), 
                                np.mean(final_test_accs), np.std(final_test_accs), 
                                np.mean(final_test_losses), np.std(final_test_losses), 
                                ]
    logs = np.array(logs)
    return results, logs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Colored MNIST & CowCamel')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='coloredmnist025')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--n_top_layers', type=int, default=2)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.0005)
    parser.add_argument('--rich_lr', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--lossf', type=str, default='nll')
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--anneal_val', type=float, default=1)
    parser.add_argument('--rep_init', type=str, default='rich_rep')
    parser.add_argument('--train_rate', type=float, default=0.99)
    parser.add_argument('--methods', type=str, default='irmv2')
    parser.add_argument('--lr_s2_decay', type=float, default=500)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--featuredistangle_reinit', type=bool, default=False)
    
    #RSC
    parser.add_argument('--rsc_f', type=float, default=0.99)
    parser.add_argument('--rsc_b', type=float, default=0.97)


    flags = parser.parse_args()

    main(flags)

