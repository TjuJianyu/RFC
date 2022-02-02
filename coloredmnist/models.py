import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torchvision
from backpack import backpack, extend
from backpack.extensions import BatchGrad

# Define and instantiate the model
class Linear(nn.Module):
    def __init__(self, hidden_dim=1, input_dim=2*14*14):
        super(Linear, self).__init__()

        self.input_dim = input_dim

        lin1 = nn.Linear(self.input_dim, hidden_dim)
        
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)

        self._main = lin1 
    def forward(self,input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_dim=390, input_dim=2*14*14):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim

        
        # if n_low_layers == 1:
        #     lin1 = nn.Linear(self.input_dim, hidden_dim)
        #     nn.init.xavier_uniform_(lin1.weight)
        #     nn.init.zeros_(lin1.bias)
        #     self._main = nn.Sequential(lin1, nn.ReLU(True))
            
        # elif n_low_layers == 2:
        lin1 = nn.Linear(self.input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.zeros_(lin2.bias)

        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))
        
    def forward(self, input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class MLP3(nn.Module):
    def __init__(self, hidden_dim=390, input_dim=2*14*14):
        super(MLP3, self).__init__()
        
        self.input_dim = input_dim

        lin1 = nn.Linear(self.input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 10)

        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.zeros_(lin2.bias)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.zeros_(lin3.bias)

        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True),lin3,nn.ReLU(True))
        
    def forward(self, input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class TopMLP(nn.Module):
    def __init__(self, hidden_dim=390, n_top_layers=2, n_targets=1, fishr=False):

        super(TopMLP, self).__init__()

        if fishr:
            self.lin1 = lin1 = extend(nn.Linear(hidden_dim,n_targets))
        else:
            self.lin1 = lin1 = nn.Linear(hidden_dim,n_targets)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)
        self._main = nn.Sequential(lin1)
        self.weights = [lin1.weight, lin1.bias]
        # if n_top_layers == 1:
        # self.lin1 = lin1 = nn.Linear(hidden_dim,n_targets,bias=bias)
            
            
        #     if bias:
        #         nn.init.xavier_uniform_(lin1.weight)
        #         nn.init.zeros_(lin1.bias)
        #         self.weights = [lin1.weight, lin1.bias]  
        #     else:
        #         nn.init.xavier_uniform_(lin1.weight)
        #         self.weights = [lin1.weight,]



        #     self._main = nn.Sequential(lin1)
        # else:
        #     rep2_size = 128
        #     self.lin1 = lin1 = nn.Linear(hidden_dim, rep2_size,bias=bias)
        #     self.lin2 = lin2 = nn.Linear(rep2_size, n_targets,bias=bias)

        #     if bias:
        #         nn.init.xavier_uniform_(lin1.weight)
        #         nn.init.xavier_uniform_(lin2.weight)
        #         nn.init.zeros_(lin1.bias)
        #         nn.init.zeros_(lin2.bias)
        #         self.weights = [lin1.weight, lin1.bias, lin2.weight, lin2.bias]
        #     else:
        #         nn.init.xavier_uniform_(lin1.weight)
        #         nn.init.xavier_uniform_(lin2.weight)
        #         self.weights = [lin1.weight, lin2.weight]

        #     self._main = nn.Sequential(lin1, nn.ReLU(True), lin2)
    
    def forward(self,input):
        out = self._main(input)
        return out 

# class MLP_model(nn.Module):
#     def __init__(self, hidden_dim=390, input_dim=2*14*14, n_targets=1):
#         super(MLP_model, self).__init__()
        
#         # NETWORK ARCHITECTURE
#         self.input_dim = input_dim
#         lin1 = nn.Linear(self.input_dim, hidden_dim)
#         lin2 = nn.Linear(hidden_dim, hidden_dim)
#         lin3 = nn.Linear(hidden_dim,n_targets)

#         # WEIGHTS INITIALIZATION    
#         nn.init.xavier_uniform_(lin1.weight)
#         nn.init.zeros_(lin1.bias)
#         nn.init.xavier_uniform_(lin2.weight)
#         nn.init.zeros_(lin2.bias)
#         nn.init.xavier_uniform_(lin3.weight)
#         nn.init.zeros_(lin3.bias)


#         self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

#     def forward(self, input):
#         out = input.view(input.shape[0], self.input_dim)
#         out = self._main(out)
#         return out




# # Define and instantiate the model
# class iga_MLP(nn.Module):
#     def __init__(self, hidden_dim=390, input_dim=2*14*14, n_low_layers = 2):
#         super(MLP, self).__init__()
        
#         self.input_dim = input_dim

#         if n_low_layers == 1:
#             lin1 = nn.Linear(self.input_dim, hidden_dim)
#             nn.init.xavier_uniform_(lin1.weight)
#             nn.init.zeros_(lin1.bias)
#             self._main = nn.Sequential(lin1, nn.ReLU(True))
            
#         elif n_low_layers == 2:
#             lin1 = nn.Linear(self.input_dim, hidden_dim)
#             lin2 = nn.Linear(hidden_dim, hidden_dim)
#             nn.init.xavier_uniform_(lin1.weight)
#             nn.init.zeros_(lin1.bias)
#             nn.init.xavier_uniform_(lin2.weight)
#             nn.init.zeros_(lin2.bias)
#             b1 = nn.BatchNorm1d(hidden_dim, affine=False)
#             b2 = nn.BatchNorm1d(hidden_dim, affine=False)
#             print("fix batchnorm")
#             self._main = nn.Sequential(lin1, nn.ELU(True), b1, lin2, nn.ELU(True),b2)

#     def forward(self, input):
#         out = input.view(input.shape[0], self.input_dim)
#         out = self._main(out)
#         return out


# class iga_TopMLP(nn.Module):
#     def __init__(self, hidden_dim=390, n_top_layers=2, n_targets=1, bias=True):
#         super(TopMLP, self).__init__()
#         if n_top_layers == 1:
#             self.lin1 = lin1 = nn.Linear(hidden_dim,n_targets,bias=False)
#             nn.init.xavier_uniform_(lin1.weight)
#             self.weights = [lin1.weight,]
#             self._main = nn.Sequential(lin1)
#         else:
#             rep2_size = hidden_dim
#             self.lin1 = lin1 = nn.Linear(hidden_dim, rep2_size,bias=True)
#             self.lin2 = lin2 = nn.Linear(rep2_size, n_targets,bias=False)

#             nn.init.xavier_uniform_(lin1.weight)
#             nn.init.xavier_uniform_(lin2.weight)

#             self.weights = [lin1.weight, lin2.weight]
#             b1 = nn.BatchNorm1d(hidden_dim, affine=False)

#             self._main = nn.Sequential(lin1, nn.ELU(True), b1, lin2)
    
#     def forward(self,input):
#         out = self._main(input)
#         return out 








# # from https://github.com/facebookresearch/DomainBed/tree/master/domainbed
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# # from https://github.com/facebookresearch/DomainBed/tree/master/domainbed
# class ResNet(torch.nn.Module):
#     """ResNet with the softmax chopped off and the batchnorm frozen"""
#     def __init__(self, dropout=0):
#         super(ResNet, self).__init__()

#         self.network = torchvision.models.resnet50(pretrained=True)
#         self.n_outputs = 2048

#         # if hparams['resnet18']:
#         #     self.network = torchvision.models.resnet18(pretrained=True)
#         #     self.n_outputs = 512
#         # else:
#         #     self.network = torchvision.models.resnet50(pretrained=True)
#         #     self.n_outputs = 2048

#         # self.network = remove_batch_norm_from_resnet(self.network)

#         # # adapt number of channels
#         # nc = input_shape[0]
#         # if nc != 3:
#         #     tmp = self.network.conv1.weight.data.clone()

#         #     self.network.conv1 = nn.Conv2d(
#         #         nc, 64, kernel_size=(7, 7),
#         #         stride=(2, 2), padding=(3, 3), bias=False)

#         #     for i in range(nc):
#         #         self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

#         # save memory
#         del self.network.fc
#         self.network.fc = Identity()

#         self.freeze_bn()
#         #self.hparams = hparams
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """Encode x into a feature vector of size n_outputs."""
#         return self.dropout(self.network(x))

#     def train(self, mode=True):
#         """
#         Override the default train() to freeze the BN parameters
#         """
#         super().train(mode)
#         self.freeze_bn()

#     def freeze_bn(self):
#         for m in self.network.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

