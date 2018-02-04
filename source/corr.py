#
# Questo e` uno spinoff da bruna10.py. Ecco le modifiche (+ se e` implementata, - se sto per farlo):
# + Vengono letti nuovi parametri relativi ai tempi
# + Vengono generate liste di tempi
# + L'inizializzazione dei pesi viene fatta esplicitamente
# + Viene calcolata la distribuzione dei pesi
# + Viene calcolata la funzione di correlazione
# + Il programma ora puo` correre con qualsiasi delle reti, non solo bruna10
#
# Note: La loss function viene salvata alla fine del run. Se si
# riprende il run, viene salvata in un file separato, alla fine del
# nuovo run. Questo va bene fintantoche i runs non vengono
# interrotti. Una cosa da fare sarebbe salvarla in un formato
# 'appendable' (testo per esempio), e salvarla a ogni time step, o
# almeno ogni volta che viene salvato il backup .pyT.
#
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import argparse
from models.generic_models import loadNet
import numpy as np #for the generation of the time lists and subdataset
from operator import mul
import collections #probabilmente non necessario


print("Pytorch version is ",torch.__version__)

#####################
# Training settings #
#####################
torch.set_default_tensor_type('torch.DoubleTensor')
parser = argparse.ArgumentParser(description='PyTorch Trainer, modified from the MNIST example')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 100)')
parser.add_argument('--load', type=str, default='nil',
                    help='load data and train.')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='test B',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--periods', type=int, default=10,
                    help='number of periods to train (default: 10)')
parser.add_argument('--steps_per_period', type=int, default=100, metavar='spp',
                    help='Redefining period (default=100)')
parser.add_argument('--lr', type=str, default='0.01', metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-interval', type=int, default=50,
                    help='how many batches to wait before logging training status. default: 50. 0: do not log.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='pick one: mnist, cifar10, cifar100')
parser.add_argument('--model', type=str, default='bruna10',
                    help='Model to be loaded: bruna10 (default), convTest, and many more...')
parser.add_argument('--out', type=str, default='./pretrained/',
                    help='Path to be saved (default: ./pretrained/)')
parser.add_argument('--data-size', type=int, default=0,
                    help='Default 0: original dataset size, otherwise dataset is downsampled')
parser.add_argument('--save-every', type=int, default=10,
                    help='1: means saved at every period, 3:means saved every three period. No matter what happens it is saved at the end again.')
#parser.add_argument('--test-freq', type=int, default=0,
#                    help='Default is 0:means that the loss function is measured in a logarithmic succession. If it is not zero, we go back to the usual linear spacing, where -1:means calculated per period 1: means calculate at every batch-step, 3:means test every three step. If test-freq!=0, no matter what happens it is tested at the end again.')
parser.add_argument('--hidden_size', type=int, default=10, metavar='m',
                    help='In some networks we can specify the number of hidden nodes through this option.')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD',
                    help='Positive: Coefficient of the L2 regularization. Negative: -coefficient of the Bruna-like regularization. Zero: no regularization.')
parser.add_argument('--t0', type=int, default=1, metavar='t0',
                    help='initial t time for C(tw,tw+t) (default: 1)')
parser.add_argument('--tw0', type=int, default=1, metavar='tw0',
                    help='initial tw time for C(tw,tw+t) (default: 1)')
parser.add_argument('--tbar0', type=int, default=1, metavar='tbar0',
                    help='initial t time for Loss(tbar) (default: 1)')
parser.add_argument('--nt', type=int, default=10, metavar='nt',
                    help='number of times t (default: 10)')
parser.add_argument('--ntw', type=int, default=4, metavar='ntw',
                    help='number of times tw (default: 4)')
parser.add_argument('--ntbar', type=int, default=10, metavar='ntbar',
                    help='number of times tbar (default: 10). If ntbar==0, the loss is never computed.')
parser.add_argument('--distr', type=str, default='default',
                    help='distribution of weights. pytorch default (default), uniform, uniform1, uniform01, zero, ones, normal')
parser.add_argument('--losstxt', type=bool, default=False,
                    help='If True, saves loss and accuracy on a txt file every time it is calculated (default is False).')
parser.add_argument('--grad', type=int, default=0, choices=[0,1,2],
                    help='0: do not calculate the gradient of the loss (default). 1: Calculate the gradient of the loss at every tw. 2: Calculate it at every tw+t.')


##################
# Cuda and Seeds #
##################
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

######################
# Output information #
######################
base_path = args.out+args.dataset
save_at = set(range(0,args.periods+1,max(1,args.save_every)))
save_at.add(args.periods)
if args.data_size != 0:
    torch.save(list(train_loader.dataset),base_path+'.data')

#Loss function in txt format
if True==args.losstxt:
	losstxt_name=base_path+'_loss.txt'
	f = open(losstxt_name, 'w')
	f.write("#1)time 2)train_acc 3)test_acc 4)train_loss 5)test_loss\n")
	f.close()

#Gradient of the loss
if 0<args.grad:
	gradtxt_name=base_path+'_gradloss.txt'
	f = open(gradtxt_name, 'w')
	f.write('#1)itw 2)it 3)tw 4)t 5)C(tw,tw+t) 6)D(tw,tw+t) 7)Y=D/C^2 8)loss 9)var(loss) 10)gradloss 11)var(grad)\n')
	f.close()

##################
# Checks on args #
##################
#Hidden size
if args.model=='bruna10':
    assert(args.hidden_size>0)

#Regularization parameters
if args.weight_decay>=0:
	weight_decay=args.weight_decay
	bruna_decay=0
else:
	weight_decay=0
	bruna_decay=-args.weight_decay
	if args.model != 'bruna10':
		print("A negative weight decay is only accepted if the model is bruna10.")
		sys.exit()

####################
# Data and Network #
####################
#Choose model
if args.dataset=='cifar100':
	import models.cifar100 as my_models
	model = getattr(my_models, args.model)
elif args.dataset=='cifar10':
	import models.cifar10 as my_models
	model = getattr(my_models, args.model)
elif args.dataset=='mnist':
	import models.mnist as my_models
	model = getattr(my_models, args.model)
else:
	print("Unknown dataset: ",args.dataset); 
	sys.exit()

#Load network
if args.load == 'nil':
	if args.model=='bruna10':
		model = model(hidden_size=args.hidden_size)
	else:
		model = model()
	iniPeriod=0
else:
	model=loadNet(args.load,model)
	from re import search
	iniPeriod=1+int(search('_',args.model,'_(.+?).pyT',args.load).group(1))


#################################
# Initialization of the weights #
#################################

# Function that yields all the optimizable parameters of the network (the network's size)
def getNumParam(mymodel):
	tot=0
	for item in list(mymodel.parameters()):
		n_layer=item.numel()
		tot+=n_layer
	return tot

num_params=getNumParam(model)
inv_num_params=1.0/num_params
print("num_params=",num_params)

# Function that initializes all the weights according to the chosen distribution
def weightsInit(mymodel, distr='uniform'):
	for m in mymodel.modules():
		if isinstance(m, nn.Linear) \
		or isinstance(m, nn.Conv2d) \
		or isinstance(m, nn.Conv3d) \
		or isinstance(m, nn.BatchNorm1d) \
		or isinstance(m, nn.BatchNorm2d) \
		or isinstance(m, nn.BatchNorm3d):
			if distr == 'zero':
				m.weight.data.fill_(0.0)
				m.bias.data.fill_(0.0)
			elif distr == 'ones':
				m.weight.data.fill_(1.0)
				m.bias.data.fill_(1.0)
			elif distr == 'normal':
				m.weight.data.normal_(0.0, 1.0)
				m.bias.data.normal_(0.0, 1.0)
			elif distr == 'uniform':
				m.weight.data.uniform_(-0.02, 0.02)
				m.bias.data.uniform_(-0.02, 0.02)
			elif distr == 'uniform1':
				m.weight.data.uniform_(-1.0, 1.0)
				m.bias.data.uniform_(-1.0, 1.0)
			elif distr == 'uniform01':
				m.weight.data.uniform_(0.0, 1.0)
				m.bias.data.uniform_(0.0, 1.0)
			else:
				print("Unrecognised weight distribution")
				sys.exit()
if 'default' != args.distr:
	weightsInit(model, distr=args.distr)

# Function that returns a flattened list of all the weights
def getWeights(mymodel):

	for i,item in enumerate(model.parameters()):
		if 0==i:
			full_list=item.data.view(item.data.numel())
		else:
			full_list=torch.cat( (full_list, item.data.view(item.data.numel()) ) )
	return full_list

##########################
# Custom Regularizations #
##########################
#This is the same as in Bruna's paper
def L1_regularizationBruna(mod):
	return torch.norm(mod.fc2.weight,1) + torch.norm(mod.fc2.bias,1)

#In Bruna's paper, they put a bound on the L2 norm of the single row - this is different.
def L2_regularizationBruna(mod):
	return torch.norm(mod.fc1.weight,2)+torch.norm(mod.fc1.bias,2)

#################
# Loss Function #
#################

# A Mean Square Error loss to mimic Bruna's paper
def bruna_loss10(output, target):
	num_classes=len(output[0]) #output is a B x num_classes matrix. len(output)=B, len(output[0])=num_classes
	this_batch_size=target.numel() #the last batch may be shorter
	temp=output[0].pow(2).sum()+1-2*output[0][target[0].data[0]]    
	for i in range(1,this_batch_size):
		temp+=output[i].pow(2).sum()+1-2*output[i][target[i].data[0]]
	myloss=temp/(this_batch_size*num_classes)
	if bruna_decay>0:#This is because bruna_decay is often chosen to be zero
		myloss+=bruna_decay*(L1_regularizationBruna(model)+L2_regularizationBruna(model))
	return myloss

def loss_function(output, target):
	if args.model=='bruna10':
		myloss=bruna_loss10(output, target)
	else:
		myloss=F.nll_loss(output, target)
	return myloss



################################
# Generation of the time lists #
################################
def ListaLogaritmica(x0,xn,n,ints=False,addzero=False):
	assert(xn>x0)
	assert(x0>0)
	n=np.int64(n)
	y0=np.log(x0)
	yn=np.log(xn)
	delta=np.float64(yn-y0)/(n-1)
	listax=np.exp([y0+i*delta for i in range(n)])
	if ints:
		listax=np.unique(np.round(listax,0).astype(int))
	if addzero:
		listax=np.insert(listax,0,0)
	return listax

#Parameters that are not argparsed
total_time=np.int64(args.steps_per_period*args.periods) #Total number of batches, which is our time unit
tbarn=total_time
tn=np.int64(0.5*total_time);
twn=np.int64(0.5*total_time);

listatw=ListaLogaritmica(args.tw0,twn,args.ntw,ints=True,addzero=True)
listat=ListaLogaritmica(args.t0,tn,args.nt,ints=True,addzero=True)
listatbar=set(ListaLogaritmica(args.tbar0,tbarn,args.ntbar,ints=True,addzero=True)) if args.ntbar>0 else []
#The list of the tprimes is a little harder
listatprime=[]; which_itwit=[]; howmany_tprime=[]
itprime=0
for itw in range(len(listatw)):
	for it in range(len(listat)):
		value=listat[it]+listatw[itw]
		if value in listatprime:
			itprime_old=listatprime.index(value)
			which_itwit[itprime_old].append([itw,it])
			howmany_tprime[itprime_old]+=1
		else:
			listatprime.append(value)
			which_itwit.append([[itw,it]])
			howmany_tprime.append(1)
			itprime+=1
listatprime=listatprime
print("listatw = ",listatw)
print("listat = ",listat)
print("listatbar = ",listatbar)
print("listatprime = ",listatprime)

#############################################################
# Definition of Correlation Functions and Other Observables #
#############################################################
#Histogram of the weights
nbins=20
histw_evol_x=torch.Tensor(args.ntw+1,nbins+1) #Histogram of all weights. The +1 is because I include time zero.
histw_evol_y=torch.Tensor(args.ntw+1,nbins)
#1-time quantities
#Weights at time tw
w_evol=torch.Tensor(args.ntw+1,num_params)
losstw=torch.Tensor(args.ntw+1)
varlosstw=torch.Tensor(args.ntw+1)
grad2tw=torch.Tensor(args.ntw+1)
vargradtw=torch.Tensor(args.ntw+1)
#2-time quantities
#Correlation function
corrw=torch.Tensor(args.ntw+1,args.nt+1)      #Correlation: \sum [w(t)-w(t')]^2
dorrw=torch.Tensor(args.ntw+1,args.nt+1)      #Correlation: \sum [w(t)-w(t')]^4


################
# Data Loading #
################
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
def logPerformance(model, period, batch_idx, n_step):
	loss_tuple = test(period,test_loader,print_c=True)
	loss_hist['test'].append((period,batch_idx,loss_tuple))
	test_loss=loss_tuple[0]
	test_acc=float(loss_tuple[1])/loss_tuple[2]
	loss_tuple = test(period,train_loader,print_c=True,label='Train')
	loss_hist['train'].append((period,batch_idx,loss_tuple))
	if args.losstxt == True:
		absolute_batch_idx=batch_idx+(period-1)*n_step
		train_acc=float(loss_tuple[1])/loss_tuple[2]
		train_loss=loss_tuple[0]
		f = open(losstxt_name, 'a')
		f.write(str(absolute_batch_idx)+" "+str(train_acc)+" "+str(test_acc)+" "+str(train_loss)+" "+str(test_loss)+"\n")
		f.close()


def train(period, n_step = 1000, lr=args.lr):
	model.train()
	optimizer=optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=weight_decay)
	for batch_idx, (data, target) in enumerate(circ_train_loader):
		absolute_batch_idx=batch_idx+(period-1)*n_step #The -1 is because periods start from 1
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_function(output, target)
		loss.backward()

		if absolute_batch_idx in listatbar: #Loss function and accuracy
			print("abs_batch_idx:",absolute_batch_idx," batch_idx:",batch_idx,"epoch:",absolute_batch_idx/len(train_loader)," period:",period-1)
			logPerformance(model,period,batch_idx, n_step)

		if absolute_batch_idx in listatw: #Save states w, measure p(w), measure gradient
			itw=np.where(listatw==absolute_batch_idx)[0][0]
			w=getWeights(model)
			w_evol[itw]=w.clone() #We need this for C(tw,t'=t+tw)
			histw=np.histogram(w.numpy(),bins=nbins,normed=False,weights=None)
			histw_evol_x[itw]=torch.from_numpy(np.array(histw[1]))
			histw_evol_y[itw]=torch.from_numpy(np.array(histw[0]))
			if 0<args.grad:
				losstw[itw], varlosstw[itw], grad2tw[itw], vargradtw[itw] = measureLossGradient(train_loader, optimizer)

		if absolute_batch_idx in listatprime: #Measure correlation functions
			itprime=list(listatprime).index(absolute_batch_idx)
			w=getWeights(model)
			for icomb in range(howmany_tprime[itprime]):
				[itw,it]=which_itwit[itprime][icomb]
				assert(listatw[itw]+listat[it]==absolute_batch_idx)
				square_corrw=torch.pow(w-w_evol[itw],2).sum()
				fourth_corrw=torch.pow(w-w_evol[itw],4).sum()
				corrw[itw][it]=square_corrw
				dorrw[itw][it]=fourth_corrw
				assert(square_corrw>=0)
				assert(fourth_corrw>=0)
				if args.grad>0:
					if args.grad==1:
						ml,vl,mg2,vg = losstw[itw], varlosstw[itw], grad2tw[itw], vargradtw[itw]
					elif args.grad==2:
						if icomb==0:
							ml,vl,mg2,vg = losstw[itw], varlosstw[itw], grad2tw[itw], vargradtw[itw] if listat[it]==0 else measureLossGradient(train_loader, optimizer)
				fgrad = open(gradtxt_name, 'a')
				fgrad.write(str(itw)+' '+str(it)+' '+str(listatw[itw])+' '+str(listat[it])+' '+
					str(inv_num_params*corrw[itw][it])+' '+str(inv_num_params*dorrw[itw][it])+' '+ str(dorrw.numpy()[itw][it]/(corrw.numpy()[itw][it]*corrw.numpy()[itw][it])) +' '+\
					str(ml)+' '+str(vl)+' '+str(mg2)+' '+str(vg)+"\n")
				fgrad.close()
		optimizer.step()

		if args.print_interval and batch_idx % args.print_interval == 0:
			print('Train Period: {} [{}/{} ({:.0f}%)]\tLoss: {: .6f}'.format(
				period, batch_idx * len(data), n_step * len(data),
				100. * batch_idx / n_step, loss.data[0])) 

		if batch_idx==n_step-1:
			break

def test(period,data_loader,print_c=False,label='Test '):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in data_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += loss_function(output, target)
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss.data[0]
	test_loss /= len(data_loader) # loss function already averages over batch size
	if print_c: print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(label,
		test_loss, correct, len(data_loader.dataset),
		100. * correct / len(data_loader.dataset)))
	return (test_loss, correct, len(data_loader.dataset))

def measureLossGradient(data_loader, optimizer): 
	""" 
    This function measures the loss, its variance, the gradient of the loss, and its variance.
    It is written so that it will be easy to generalize to per-layer measurements. 
    """
	model.eval()

	#Initialize to zero
	optimizer.zero_grad()
	meanGrad = collections.OrderedDict([(key, np.zeros(value.size())) for key, value in model.state_dict().items()])
	meanGrad2 = collections.OrderedDict([(key, 0) for key, value in model.state_dict().items()])
	varGrad = collections.OrderedDict([(key, 0) for key, value in model.state_dict().items()])
	meanLoss=0
	meanLoss2=0

	#Loop over the data
	numbatches=0
	for data, target in data_loader:
		if np.random.ranf()>.2: #Use only 20% of the data
			continue
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, requires_grad=True), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_function(output, target)
		meanLoss+=loss.data
		meanLoss2+=loss.data*loss.data
		loss.backward()

		state = model.state_dict(keep_vars=True) #keep_vars requires pytorch 0.3 or newer
		gradient = collections.OrderedDict([(key, value.grad.data.cpu().numpy().copy() if ('weight' in key or 'bias' in key) else np.zeros(value.size())	)  for key, value in state.items()])

		for key in state:
			if ('weight' in key or 'bias' in key): #Exclude keys that don't have grad, such as running
				meanGrad[key]+=gradient[key]
				meanGrad2[key] += (gradient[key]*gradient[key]).sum()
		numbatches+=1

	meanLoss/=numbatches
	meanLoss2/=numbatches

	#Now join different layers
	totMeanGrad2 = 0  # <grad>^2 of all the layers
	totVarGrad = 0    # var(grad) of all layers
	for key in state:
		if ('weight' in key or 'bias' in key):
			meanGrad[key]/=len(data_loader)
			meanGrad2[key]/=len(data_loader)
			D1=(meanGrad[key]*meanGrad[key]).sum()
			varGrad[key]=meanGrad2[key]-D1
			totMeanGrad2+=D1
			totVarGrad+=varGrad[key]
	totMeanGrad2/=len(meanGrad2)

	varLoss=meanLoss2-meanLoss*meanLoss
	return meanLoss[0],varLoss[0],totMeanGrad2,totVarGrad




#####################
# Train the network #
#####################
for period in range(iniPeriod, args.periods + 1):
	if period != 0: #So that initial state is saved
		train(period,n_step=args.steps_per_period)
	if period in save_at:
		out = model.state_dict()
		for k,v in out.items():
			out[k]=v.cpu()
		torch.save(out,base_path+'_%05d.pyT'%period)
torch.save(args,base_path+'.args')
torch.save(loss_hist,base_path+"_{0}-{1}.hist".format("%05d"%iniPeriod,"%05d"%period))


###################
#Some more saving #
###################
#Save P(w)
histfile=open(base_path+'_histw.txt','ab')
for itw in range(len(listatw)):
	delta=histw_evol_x[itw][1]-histw_evol_x[itw][0]
	xcenters=histw_evol_x[itw][0:nbins].numpy()+0.5*delta
	ycenters=histw_evol_y[itw].numpy()
	normalized_ycenters=ycenters/(num_params*delta)
	# plt.plot(xcenters, normalized_ycenters, linewidth='3.0', label='t='+str(listatw[itw]))
	header='1)itw 2)tw 3)w 4)h(w) 5)p(w)' if itw==0 else ''
	np.savetxt(histfile, np.stack(([itw for i in range(len(xcenters))],[listatw[itw] for i in range(len(xcenters))],xcenters,ycenters,normalized_ycenters),axis=1), fmt='%.14g', delimiter=' ', newline='\n', header=header, footer='', comments='# ')
# plt.legend(loc='lower center')
# plt.show()
histfile.close()
#save C(tw,t')
f1=open(base_path+'_C.txt', 'w+')
f1.write('#1)itw 2)it 3)tw 4)t 5)C(tw,tw+t) 6)D(tw,tw+t) 7)Y=D/C^2\n')
for itprime in range(len(listatprime)):
	for icomb in range(howmany_tprime[itprime]):
		[itw,it]=which_itwit[itprime][icomb]
		f1.write(str(itw)+' '+str(it)+' '+str(listatw[itw])+' '+str(listat[it])+' '+str(inv_num_params*corrw[itw][it])+' '+str(inv_num_params*dorrw[itw][it])+' '+ str(dorrw.numpy()[itw][it]/(corrw.numpy()[itw][it]*corrw.numpy()[itw][it])) +'\n')
f1.close()

