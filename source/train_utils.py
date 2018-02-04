from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets, transforms

norm_dic = {   'cifar100':([0.49137, 0.48235, 0.44667],[0.24706, 0.24353, 0.26157]),
               'mnist': ([0.1307], [0.3081]),
               'cifar10':([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])}

def getDataset(dataset,sample_size=0,train=True,b_size=64,shuff=True,**kwargs):
    if sample_size> 0:
        org_batch_size = b_size
        b_size = sample_size

    t_fun = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(*norm_dic[dataset]),
                       lambda t: t.double()
                   ])

    ds = torch.utils.data.DataLoader(datasets.__dict__[dataset.upper()]('./data_'+dataset, train=train, 
                                            download=True,transform=t_fun),
                                            batch_size=b_size, shuffle=shuff,**kwargs)
    if sample_size> 0:
        for small_x,small_y in ds:
            break
        ds=torch.utils.data.DataLoader(zip(small_x,small_y),batch_size=org_batch_size, shuffle=shuff,**kwargs)
    
    return ds


def cycle_loader(dataloader):
    while 1:
        for d in dataloader:
            yield d
