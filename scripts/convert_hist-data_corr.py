from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import numpy as np
import matplotlib.pyplot as plt
import sys

if 3==len(sys.argv):
    filename=sys.argv[1]
    steps_per_period=int(sys.argv[2])
else:
    print("This script converts a .hist (output of neural net program) file into a .data file (a txt table).\n")
    print("Launch as\n",sys.argv[0]," <file.hist> <steps_per_period>\nOutput is on stdout, so it generally should be redirected")
    print("sys.argv:",len(sys.argv))
    sys.exit()

print("#filename:",filename)
print("#steps_per_period:",filename)

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
train_times=[steps_per_period*(train_period[i]-1)+train_batchidx[i] for i in range(len(train_period))]

test_period=[elem[0] for elem in test]
test_batchidx=[elem[1] for elem in test]
test_loss=[elem[2][0] for elem in test]
test_acc=[float(elem[2][1])/float(elem[2][2]) for elem in test]
test_times=[steps_per_period*(test_period[i]-1)+test_batchidx[i] for i in range(len(test_period))]

assert((np.array(test_times)-np.array(train_times)).sum()==0) #Check that they are the same

print("#1)time 2)train_acc 3)test_acc 4)train_loss 5)test_loss")
for i in range(len(train_period)):
    print('%d %.14g %.14g %.14g %.14g' % (train_times[i],train_acc[i], test_acc[i],train_loss[i], test_loss[i]))

sys.exit()

