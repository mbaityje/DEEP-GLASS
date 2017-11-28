#!/bin/bash
# Script to launch bruna network
# 

#DATASET
readonly dataset='cifar10'

#DIRECTORIES
scriptDIR=$PWD/
myROOT=$PWD/../
exeDIR=$myROOT/source/
outROOT=$myROOT/output/ #Parent directory for all the output
dataDIR=$myROOT/data_$dataset/

#NETWORK
#prog=bruna.py #Shitty network, exactly like Bruna's
prog=bruna10.py #Decent network

#HYPERPARAMETERS
weight_decay=0.01 #Positive: L2, Negative: Bruna, Zero: None
LEARNING_RATES="0.001"
HIDDEN_SIZES="10"

#Length of simulation (keep these quantities multiples of 10)
TOTLENGTH=`echo 3*10^7|bc` #Total number of processed images
BATCH_SIZES="100"
periods=1000  # number of periods
save_every=250 # every how many periods we save





cd $myROOT
for LR in $LEARNING_RATES
do
    for BS in $BATCH_SIZES
    do
	echo " BS = $BS"
	spp=`echo "$TOTLENGTH/($BS*$periods)"|bc`
	echo "Steps per period: $spp"
	
	for HS in $HIDDEN_SIZES
	do
	    for ISAM in 0 
	    do
	    model='bruna'                        # Actually it is hard-coded in bruna.py, so it is useless when sent as a parameter to the program
	    modelname=${model}_m$HS              # For the directory
	    outDIR=$outROOT/${modelname}/B${BS}/sam$ISAM/
	    mkdir -p $outDIR

	    lastSave=`ls -rt $outDIR/$dataset*.pyT|tail -1`
	    echo "LAST SAVE: $lastSave"
	    startFrom=${lastSave:-'nil'}
	    echo "START FROM $startFrom"
	    #startFrom="$outDIR/cifar10_0_64_bruna_000014.pyT"
	    #startFrom='nil'

	    
	    echo "python $prog --dataset=$dataset --steps_per_period=$spp --periods=$periods --batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$weight_decay"
	    python $exeDIR/$prog --dataset=$dataset --steps_per_period=$spp --periods=$periods --batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$weight_decay --load=$startFrom
	    
	    done
	done
    done
done
cd $scriptDIR
