from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import numpy as np
import matplotlib.pyplot as plt
import sys

if 2==len(sys.argv):
    filename=sys.argv[1]
else:
    print("This script converts a .hist (output of neural net program) file into a .data file (a txt table).\nLaunch as\n",sys.argv[0]," file.hist\nOutput is on stdout, so it generally should be redirected")
    print("sys.argv:",len(sys.argv))
    sys.exit()

print("#filename:",filename)

loss_hist=torch.load(filename)
#loss_hist=torch.load('../output/bruna_m50/B64/sam0/cifar10_0_64_bruna.hist')
#loss_hist=torch.load('../output/bruna_m50/B64/sam0/cifar10_0_64_bruna_00000-00002.hist')
#Structure of loss_hist dictionary:
#loss_hist=('test':[],'train':[],'lr':[])
#loss_hist['test']=(period,batch_idx,loss_tuple)
#loss_tuple=(loss, correct, total)
test=loss_hist['test']
train=loss_hist['train']

train_period=[elem[0] for elem in train]
train_batchidx=[elem[1] for elem in train]
train_loss=[elem[2][0] for elem in train]
train_acc=[float(elem[2][1])/float(elem[2][2]) for elem in train]

test_period=[elem[0] for elem in test]
test_batchidx=[elem[1] for elem in test]
test_loss=[elem[2][0] for elem in test]
test_acc=[float(elem[2][1])/float(elem[2][2]) for elem in test]

print("#1)train_period 2)test_period 3)train_acc 4)test_acc 5)train_loss 6)test_loss")
for i in range(len(train_period)):
    print('%d %d %.14g %.14g %.14g %.14g' % (train_period[i], test_period[i],train_acc[i], test_acc[i],train_loss[i], test_loss[i]))

sys.exit()

#The following is a plot of the data
plt.figure(1)
plt.ylabel('Loss')
plt.ylabel('time')
plt.semilogx(train_period, train_acc,  label='$\mathcal{A}$ train', linestyle='-', color='red', linewidth='2.0')
plt.semilogx(test_period, test_acc,  label='$\mathcal{A}$ test', linestyle=':', color='red', linewidth='2.0')
plt.semilogx(train_period, train_loss,  label='$\mathcal{L}$ train', linestyle='-', color='blue', linewidth='2.0')
plt.semilogx(test_period, test_loss,  label='$\mathcal{L}$ test', linestyle=':', color='blue', linewidth='2.0')
plt.legend(loc='center left')
plt.show()

