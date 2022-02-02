

import numpy as np
import torch
from torchvision import datasets
import math
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate


from misc import split_dataset,make_weights_for_balanced_classes,seed_hash
from fast_data_loader import InfiniteDataLoader, FastDataLoader


def coloredmnist(label_noise_rate, trenv1, trenv2, int_target=False):
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments
    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(label_noise_rate, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        
        if int_target:

            return {
                'images': (images.float() / 255.).cuda(), 
                'labels': labels[:, None].long().flatten().cuda()
            }
        else:
            print('non int label')
            return {
                'images': (images.float() / 255.).cuda(), 
                'labels': labels[:, None].cuda()
            }
             

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], trenv1),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], trenv2)]
    
    # init 3 test environments [0.1, 0.5, 0.9] 
    test_envs = [    
        #make_environment(mnist_val[0], mnist_val[1], 0.225),
        #make_environment(mnist_val[0], mnist_val[1], 0.225),
        make_environment(mnist_val[0], mnist_val[1], 0.9),
        make_environment(mnist_val[0], mnist_val[1], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.5),
    ]
    print(torch.rand(4))
    print(np.random.randint(10)) 


    return envs, test_envs



class Example2:
    """
    Cows and camels
    """

    def __init__(self, dim_inv, dim_spu, n_envs, envs = { 'E0': {"p": 0.90, "s": 0.5},
        'E1': {"p": 0.90, "s": 0.5},"E2": {"p": 0.90, "s": 0.5}}):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        #self.envs = {}
        self.envs = envs


        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

    def sample(self, n=1000, env="E0", split="train"):


        p = self.envs[env]["p"]
        s = self.envs[env]["s"]

        w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        print(self.avg_fg[i])
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()

        

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p)
        idx = torch_bernoulli(0.05, len(outputs))

        inputs[idx,:self.dim_inv] = -inputs[idx,:self.dim_inv]

        # def torch_xor(a, b):
        #     return (a-b).abs() 
        # outputs = torch_xor(outputs.flatten(), torch_bernoulli(0.05, len(outputs))).reshape(-1,1)
        return inputs, outputs
# class Example2:
#     """
#     Cows and camels
#     """

#     def __init__(self, dim_inv, dim_spu, n_envs, envs = { 'E0': {"p": 0.90, "s": 0.5},
#         'E1': {"p": 0.90, "s": 0.5},"E2": {"p": 0.90, "s": 0.5}}):
#         self.scramble = torch.eye(dim_inv + dim_spu)
#         self.dim_inv = dim_inv
#         self.dim_spu = dim_spu
#         self.dim = dim_inv + dim_spu

#         self.task = "classification"
#         #self.envs = {}
#         self.envs = envs


#         print("Environments variables:", self.envs)

#         # foreground is 100x noisier than background
#         self.snr_fg = 1e-2
#         self.snr_bg = 1

#         # foreground (fg) denotes animal (cow / camel)
#         cow = torch.ones(1, self.dim_inv)
#         self.avg_fg = torch.cat((cow, cow, -cow, -cow))

#         # background (bg) denotes context (grass / sand)
#         grass = torch.ones(1, self.dim_spu)
#         self.avg_bg = torch.cat((grass, -grass, -grass, grass))

#     def sample(self, n=1000, env="E0", split="train"):
#         p = self.envs[env]["p"]
#         s = self.envs[env]["s"]
#         w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
#         i = torch.multinomial(w, n, True)
#         x = torch.cat((
#             (torch.randn(n, self.dim_inv) /
#                 math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
#             (torch.randn(n, self.dim_spu) /
#                 math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

#         if split == "test":
#             x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

#         inputs = x @ self.scramble
#         print(self.avg_fg[i])
#         outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()
#         return inputs, outputs
class Example2s(Example2):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)
        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))

# def cow_camel(int_target=False):


#     def gene_data(size,dim_inv,label_noise_rate):
#         size = 10000
#         dim_inv = 100
#         label_noise_rate = 0.1
#         def torch_bernoulli(p, size):
#             return (torch.rand(size) < p).float()
#         def torch_xor(a, b):
#             return (a-b).abs() 
#         labels = torch_bernoulli(0.5,size)
#         noise_idx = torch_bernoulli(label_noise_rate, len(labels))
        
#         noise_labels = torch_xor(labels, noise_idx)


#         x = torch.randn(size, dim_inv) / math.sqrt(10) + (noise_labels.view(-1,1) * 2  - 1)
#         b = torch_xor(torch.ones(size), noise_idx)
#         b = torch_xor(b, torch_bernoulli(0.05, size))

#         b = b.reshape(-1,1)
#         #print(labels)
#         #print(x)
#         x = torch.cat((x, b), dim=1)
#         print(x)
#         print(labels)
#         return x, labels.view(-1,1)
#     x0, y0 = gene_data(10000, 10,0.1)
#     print(x0.size(), y0.size())
#     envs=[
#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         ]
#     x0, y0 = gene_data(1000, 10,0.1)
#     test_envs=[
#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         ]   
#     return envs, test_envs


def cow_camel(int_target = False, n_examples=18000):

    data = Example2(5,95,3,envs = { 'E0': {"p": 0.95, "s": 0.}, 'E1': {"p": 0.95, "s": 0.},"E2": {"p": 0.95, "s": 0}})
    #data = Example2(10,90,3)
    
    #python rich_rep_ood.py --verbose True --dataset cowcamel --hidden_dim 64 --lossf nll --rep_init riep  --methods erm --steps 6001 --lr 0.00005  --rich_lr 0.00005 --l2_regularizer_weight 0.00
    #x0,y0 = data.sample(12000 + 1200 +1466, env='E0',split="train")
    #x0,y0 = data.sample((20000 - 2000), env='E0',split="train")
    x0,y0 = data.sample(n_examples, env='E0',split="train")
    print(x0)
    print(len(x0))
    envs=[
        {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
        ]

    x1,y1 = data.sample(5000,env='E0',split="train")
    test_envs=[
        {'images': x1.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y1.cuda() },
        ]

    # x1,y1 = data.sample(1000,env='E0',split="test")
    # test_envs=[
    #     {'images': x1.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y1.cuda() },
    #     ]
    return envs, test_envs

def cow_camels(int_target = False, n_examples=18000):

    data = Example2s(5,95,3)
    #data = Example2(10,90,3)
   
    #python rich_rep_ood.py --verbose True --dataset cowcamel --hidden_dim 64 --lossf nll --rep_init riep  --methods erm --steps 6001 --lr 0.00005  --rich_lr 0.00005 --l2_regularizer_weight 0.00
    #x0,y0 = data.sample(12000 + 1200 +1466, env='E0',split="train")
    #x0,y0 = data.sample((20000 - 2000), env='E0',split="train")
    x0,y0 = data.sample(n_examples, env='E0',split="train")
    
    print(len(x0))
    envs=[
        {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
        ]

    x1,y1 = data.sample(1000,env='E0',split="train")
    test_envs=[
        {'images': x1.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y1.cuda() },
        ]

    # x1,y1 = data.sample(1000,env='E0',split="test")
    # test_envs=[
    #     {'images': x1.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y1.cuda() },
    #     ]
    return envs, test_envs    
# def cow_camels(int_target = False):
#     data = Example2(5,95,3)

#     x0,y0 = data.sample(20000,env='E0',split="train")
#     print(x0,y0)
#     envs=[
#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         ]

#     x0,y0 = data.sample(1000,env='E0',split="train")
#     test_envs=[
#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         ]
#     return envs, test_envs


# def cow_camels(int_target = False):
#     data = Example2s(5,95,3)
#     x0,y0 = data.sample(5000,env='E0',split="train")
#     x1,y1 = data.sample(5000,env='E1',split="train")
#     x2,y2 = data.sample(5000,env='E2',split="train")
    
#     envs=[

#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         {'images': x1.cuda(), 'labels':y1.long().flatten().cuda() if int_target else y1.cuda()},
#         {'images': x2.cuda(), 'labels':y2.long().flatten().cuda() if int_target else y2.cuda()}
#     ]

#     x0,y0 = data.sample(5000,env='E0',split="test")
#     x1,y1 = data.sample(5000,env='E1',split="test")
#     x2,y2 = data.sample(5000,env='E2',split="test")
    
#     test_envs=[
#         {'images': x0.cuda(), 'labels':y0.long().flatten().cuda() if int_target else y0.cuda() },
#         {'images': x1.cuda(), 'labels':y1.long().flatten().cuda() if int_target else y1.cuda()},
#         {'images': x2.cuda(), 'labels':y2.long().flatten().cuda() if int_target else y2.cuda()}
#     ]
#     return envs, test_envs
     


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST14025",
    "ColoredMNIST14010",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)



class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root,):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            #if augment and (i not in test_envs):
            #    env_transform = augment_transform
            #else:
            env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir,)


def get_pacs(data_dir,trial_seed,holdout_fraction=0.2,class_balanced=False):

    dataset = PACS(data_dir)


    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):

        out, in_ = split_dataset(env,
            int(len(env)*holdout_fraction),
            seed_hash(trial_seed, env_i))


        if class_balanced:
            in_weights = make_weights_for_balanced_classes(in_)
            out_weights = make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights, = None, None,
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]


    # eval_loaders = [FastDataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=dataset.N_WORKERS)
    #     for env, _ in (in_splits + out_splits)]

    return  in_splits, out_splits, dataset.num_classes


# Final train acc (mean/std across restarts so far):
# 0.9677 0.002318635
# Final train loss (mean/std across restarts so far):
# 0.09310232 0.0047965683
# Final worest test acc (mean/std across restarts so far):
# 0.9403 0.0065122964
# Final worest test loss (mean/std across restarts so far):
# 0.13243343 0.011757278
# Final train acc (mean/std across restarts so far):
# 0.98973334 0.0026614275
# Final train loss (mean/std across restarts so far):
# 0.04375518 0.0066196364
# Final worest test acc (mean/std across restarts so far):
# 0.9383 0.010237666
# Final worest test loss (mean/std across restarts so far):
# 0.14117496 0.02083630






# 110%
# #  Final train acc (mean/std across restarts so far):
# 0.9905001 0.0028557864
# Final train loss (mean/std across restarts so far):
# 0.03617125 0.0035661852
# Final worest test acc (mean/std across restarts so far):
# 0.93520004 0.004853856
# Final worest test loss (mean/std across restarts so far):
# 0.10103923 0.007200872

# 120%
#Final train acc (mean/std across restarts so far):
# 0.987412 0.003351059
# Final train loss (mean/std across restarts so far):
# 0.039729778 0.0033610212
# Final worest test acc (mean/std across restarts so far):
# 0.9348 0.005564168
# Final worest test loss (mean/std across restarts so far):
# 0.09960848 0.005073187

# 150%
# Final train acc (mean/std across restarts so far):
# 0.9802481 0.0066377516
# Final train loss (mean/std across restarts so far):
# 0.046342798 0.0044505643
# Final worest test acc (mean/std across restarts so far):
# 0.94150007 0.0062169214
# Final worest test loss (mean/std across restarts so far):
# 0.092574194 0.0059805214
# 150% again
# Final train acc (mean/std across restarts so far):
# 0.9773777 0.0038570361
# Final train loss (mean/std across restarts so far):
# 0.04794962 0.0029606314
# Final worest test acc (mean/std across restarts so far):
# 0.9416 0.006086051
# Final worest test loss (mean/std across restarts so far):
# 0.09282162 0.0070087826


# 200%
# Final train acc (mean/std across restarts so far):
# 0.9625527 0.0041393214
# Final train loss (mean/std across restarts so far):
# 0.05833249 0.0024709834
# Final worest test acc (mean/std across restarts so far):
# 0.9413001 0.0070576235
# Final worest test loss (mean/std across restarts so far):
# 0.09103755 0.009425646

# 200% again
# Final train acc (mean/std across restarts so far):
# 0.9615 0.004066371
# Final train loss (mean/std across restarts so far):
# 0.058461715 0.0029921476
# Final worest test acc (mean/std across restarts so far):
# 0.94399995 0.01094532
# Final worest test loss (mean/std across restarts so far):
# 0.086051665 0.012780004


# 300%
# Final train acc (mean/std across restarts so far):
# 0.9532315 0.0019425267
# Final train loss (mean/std across restarts so far):
# 0.066988304 0.0019486233
# Final worest test acc (mean/std across restarts so far):
# 0.94990003 0.00771298
# Final worest test loss (mean/std across restarts so far):
# 0.08005561 0.006507563
