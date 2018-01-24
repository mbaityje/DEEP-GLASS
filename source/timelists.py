#!/usr/bin/python
#
# Training program that generates the three lists of times that we need
# for the simulations
# - listat
# - listatw
# - listatbar
# 
from __future__ import print_function
import numpy as np

##############################
# Input:
##############################
#User input
total_time=500000
tw0=1; t0=1; tbar0=1
ntw=50; nt=100; ntbar=100

#Derived input
tbarn=total_time
tn=np.int64(0.5*total_time);
twn=np.int64(0.5*total_time);


##############################
# Definition of functions
##############################

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


##############################
# Creation of the lists
##############################
listatw=ListaLogaritmica(tw0,twn,ntw,ints=True,addzero=True)
listat=ListaLogaritmica(t0,tn,nt,ints=True,addzero=True)
listatbar=ListaLogaritmica(tbar0,tbarn,ntbar,ints=True,addzero=True)
#
#The list of the tprimes is a little harder
#
#listatprime=[listat[it]+listatw[itw] for it in range(len(listat)) for itw in range(len(listatw))]

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
        
        

print(listatw)
print(listat)
print(listatbar)
