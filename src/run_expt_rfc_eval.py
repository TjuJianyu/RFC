import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

try:
    import wandb
except Exception as e:
    pass

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions,evaluate_logits
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

# Necessary for large images of GlobalWheat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    
    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--freeze_featurizer', default=False, type=parse_bool)
    
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)
    
    # amd gpu config
    os.environ['MIOPEN_USER_DB_PATH']=config.log_dir
    

    # For the GlobalWheat detection dataset,
    # we need to change the multiprocessing strategy or there will be
    # too many open file descriptors.
    if config.dataset == 'globalwheat':
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)


    train_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)
    eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

    # Configure unlabeled datasets
    unlabeled_dataset = None
    if config.unlabeled_split is not None:
        raise NotImplementedError
    else:
        train_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=config.groupby_fields
        )

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)
    if unlabeled_dataset is not None:
        log_group_data({"unlabeled": unlabeled_dataset}, log_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
        unlabeled_dataset=unlabeled_dataset,
    )

    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.eval_only:
        raise NotImplementedError
    else:
        if config.eval_epoch is None:
            eval_model_path = model_prefix + 'epoch:best_model.pth'
        else:
            eval_model_path = model_prefix +  f'epoch:{config.eval_epoch}_model.pth'
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True

        # evaluate(
        #     algorithm=algorithm,
        #     datasets=datasets,
        #     epoch=epoch,
        #     general_logger=logger,
        #     config=config,
        #     is_best=is_best, skip_train=False)
        evaluate_logits(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best, skip_train=False)
        

    if config.use_wandb:
        wandb.finish()
    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    main()
