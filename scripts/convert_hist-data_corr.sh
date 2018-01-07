#!/bin/bash

TOTLENGTH_NORM=`echo 10^6|bc`
periods=1000
INIT_DISTROS="uniform uniform1 normal"


for network in conv1020relu #bruna10
do
    for m in 10 100 1000 10000
    do
	for B in 1000 100 10
	do
	    for LR in 0.01 1.0 0.5 0.2 0.1 0.05 0.02 0.01
	    do
		TOTLENGTH=`echo $TOTLENGTH_NORM/$LR|bc`
		spp=`echo "$TOTLENGTH/($B*$periods)"|bc`
		echo spp=$spp
		for wd in 0
		do
		    for momentum in 0.5
		    do
			for INIT_DISTR in $INIT_DISTROS
			do
			    for isam in 1 0 999
			    do
				DIRECTORIO=../output/$network/m$m/B$B/LR$LR/WD$wd/MOM$momentum/$INIT_DISTR/sam$isam
				FICHERO=$DIRECTORIO/cifar10_0_${B}_${network}_00000-01000.hist
				OUTPUT=$DIRECTORIO/cifar10_0_${B}_${network}_00000-01000.data
				python convert_hist-data_corr.py $FICHERO $spp > $OUTPUT
			    done
			done
		    done
		done
	    done
	done
    done
done

exit
