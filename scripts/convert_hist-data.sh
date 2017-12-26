#!/bin/bash

for network in bruna10
do
    for m in 30 100 300 1000 3000 10000 30000
    do
	for B in 100
	do
	    for lr in 0.1 0.01 0.001
	    do
		for wd in 0 0.01
		do
		    for isam in 0
		    do
			DIRECTORIO=../output/$network/m$m/B$B/LR$lr/WD$wd/sam$isam
			FICHERO=$DIRECTORIO/cifar10_0_100_bruna10_00000-01000.hist
			OUTPUT=$DIRECTORIO/cifar10_0_100_bruna10_00000-01000.data
			python convert_hist-data.py $FICHERO > $OUTPUT
		    done
		done
	    done
	done
    done
done
