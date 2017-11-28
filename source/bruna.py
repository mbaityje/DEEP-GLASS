from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Trainer, modified from the MNIST example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--load', type=str, default='nil', metavar='N',
                    help='load data and train.')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--periods', type=int, default=10, metavar='N',
                    help='number of periods to train (default: 10)')
parser.add_argument('--lr', type=str, default='0.01', metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                    help='pick one: mnist, cifar10, cifar100')
parser.add_argument('--model', type=str, default='convTest', metavar='N',
                    help='Model to be load')
parser.add_argument('--out', type=str, default='pretrained/', metavar='N',
                    help='Path to be saved')
parser.add_argument('--steps_per_period', type=int, default=1000, metavar='N',
                    help='Redefining period')
parser.add_argument('--data-size', type=int, default=0, metavar='N',
                    help='Default -1: original dataset size, otherwise dataset is downsampled')
parser.add_argument('--save-every', type=int, default=10, metavar='N',
                    help='1: means saved at every period, 3:means saved every three period. No matter what happens it is saved at the end again.')
parser.add_argument('--test-freq', type=int, default=-1, metavar='N',
                    help='0:means calculated per period 1: means calculate at every batch-step, 3:means test every three step. No matter what happens it is tested at the end again.')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='In some networks we can specify the number of hidden nodes through this option.')
parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                    help='Positive: Coefficient of the L2 regularization. Negative: -coefficient of the Bruna-lie regularization. Zero: no regularization.')


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


## New Arguments ##
#The size of the image is needed
if args.dataset=='cifar100' or args.dataset=='cifar10':
    input_size=3*32*32
elif args.dataset=='mnist':
    input_size=28*28
else:
    print("Wrong args.dataset: ",args.dataset); sys.exit()
#Size of the hidden layer    
hidden_size=args.hidden_size
assert(hidden_size>0)
#Regularization parameters
if args.weight_decay>0:
    weight_decay=args.weight_decay
    bruna_decay=0
else:
    weight_decay=0
    bruna_decay=-args.weight_decay

#Bruna's Network
class bruna(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(bruna, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(-1,input_size))
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
#Load Bruna's network
model=bruna(input_size, hidden_size, 1)
#The MSE loss, used by Bruna
def bruna_loss(y, y_predicted):
    return (y-y_predicted.double()).pow(2).sum()/args.batch_size
#This is the same as in Bruna's paper
def L1_regularization(mod):
    return torch.norm(mod.fc2.weight,1) + torch.norm(mod.fc2.bias,1)
#In Bruna's paper, they put a bound on the L2 norm of the single row - this is different.
def L2_regularization(mod):
    return torch.norm(mod.fc1.weight,2)+torch.norm(mod.fc1.bias,2)


## Data Loading
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
from train_utils import getDataset,cycle_loader

train_loader = getDataset(args.dataset,sample_size=args.data_size,b_size=args.batch_size,**kwargs)
circ_train_loader = cycle_loader(train_loader)
test_loader = getDataset(args.dataset,train=False,b_size=args.test_batch_size,**kwargs)

if args.cuda:
    model.cuda()

args.lr = float(args.lr)

def updateOptimizer(old,fun,model,period,batch_idx,*f_args):
    new_lr = fun(model,*f_args)
    if new_lr and 0<new_lr<1:
        loss_hist['lr'].append((period,batch_idx,new_lr))
        old = optim.SGD(model.parameters(), lr=new_lr, momentum=args.momentum, weight_decay=weight_decay)
    return old

loss_hist = {'train':[],'test':[],'lr':[]} ##losses before steps
def logPerformance(model,period,batch_idx):
    loss_tuple = test(period,test_loader,print_c=True)
    loss_hist['test'].append((period,batch_idx,loss_tuple))
    loss_tuple = test(period,train_loader,print_c=True,label='Train')
    loss_hist['train'].append((period,batch_idx,loss_tuple))

def train(period,n_step = 1000,lr=0.1):
    model.train()
    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    test_at = set(range(0,n_step,args.test_freq))
    for batch_idx, (data, target) in enumerate(circ_train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = bruna_loss(output, target)+bruna_decay*(L1_regularization(model)+L2_regularization(model))
        loss.backward()
        if  batch_idx in test_at:
            logPerformance(model,period,batch_idx)
        optimizer.step()
        if batch_idx % args.print_interval == 0:
            print('Train Period: {} [{}/{} ({:.0f}%)]\tLoss: {: .6f}'.format(
                period, batch_idx * len(data), n_step * len(data),
                100. * batch_idx / n_step, loss.data[0])) 
        if batch_idx==n_step:
            break
    logPerformance(model,period,-1) ##loss after period 

def test(period,data_loader,print_c=False,label='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += (bruna_loss(output, target)+bruna_decay*(L1_regularization(model)+L2_regularization(model))).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(data_loader) # loss function already averages over batch size
    if print_c: print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(label,
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return (test_loss, correct, len(data_loader.dataset))

base_path = args.out+'_'.join([args.dataset,str(args.data_size),str(args.batch_size),args.model])
save_at = set(range(0,args.periods+1,args.save_every))
save_at.add(args.periods)
if args.data_size != 0:
    torch.save(list(train_loader.dataset),base_path+'.data')
for period in range(0, args.periods + 1):
    if period != 0:
        train(period,n_step=args.steps_per_period)
    if period in save_at:
        out = model.state_dict()
        for k,v in out.items():
            out[k]=v.cpu()
        torch.save(out,base_path+'_%03d.pyT'%period)
torch.save(args,base_path+'.args') 
torch.save(loss_hist,base_path+'.hist') 


