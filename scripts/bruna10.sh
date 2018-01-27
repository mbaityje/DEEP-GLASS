#!/bin/bash
# Script to launch bruna network
# 

#--SYSTEM RECOGNITION--
SYSTEM="kondo"
#SYSTEM="PennPuter"
if [ $SYSTEM == "kondo" ];then 
    if ! [[ `hostname` == *"kondo"* ]];then #the string "kondo" must be contained in hostname
	echo "ATTENZIONE: System declared as $SYSTEM, but it actually is $(hostname)"
    fi
    readonly QUEUE="glass" 
fi
if [ $SYSTEM == "PennPuter" ];then 
    if ! [ `hostname` == "PennPuter" ];then
	echo "ATTENZIONE: System declared as $SYSTEM, but it actually is $(hostname)"
    fi
fi

#--DATASET--
readonly dataset='cifar10'

#--DIRECTORIES--
scriptDIR=$PWD/
myROOT=$PWD/../
exeDIR=$myROOT/source
outROOT=$myROOT/output #Parent directory for all the output
dataDIR=$myROOT/data_$dataset
logDIR=$myROOT/logs

#--NETWORK--
model='bruna10'
prog=$model.py

#--HYPERPARAMETERS--
LEARNING_RATES="0.001 0.01 0.1"
HIDDEN_SIZES="30 100 300 1000 3000 10000 30000"
WEIGHT_DECAYS="0 0.01" #Positive: L2, Negative: Bruna, Zero: None
SAMPLES="0 1"

#Length of simulation (keep these quantities multiples (or powers) of 10)
TOTLENGTH=`echo 3*10^7|bc` #Total number of processed images
BATCH_SIZES="100"
periods=1000     # number of periods
save_every=`echo $periods/4|bc`  # we save 4 times during the run
USE_BACKUPS="no"


#--CYCLE OVER PARAMETERS--
cd $myROOT
for LR in $LEARNING_RATES
do
    for BS in $BATCH_SIZES
    do
	spp=`echo "$TOTLENGTH/($BS*$periods)"|bc`

	for HS in $HIDDEN_SIZES
	do
	    for WD in $WEIGHT_DECAYS
	    do
		for ISAM in $SAMPLES
		do
		    echo "LR=$LR, BS=$BS, m=$HS, WD=$WD, ISAM=$ISAM"

		#--Output directory--
		outDIR=$outROOT/${model}/m$HS/B${BS}/LR$LR/WD$WD/sam$ISAM/
		mkdir -p $outDIR

		#--Backups--
		if [ $USE_BACKUPS == "yes" ]; then
		    lastSave=`ls -rt $dataDIR/*.pyT|tail -1`
		    echo "LAST SAVE: $lastSave"
	            startFrom=${lastSave:-'nil'}
		elif [ $USE_BACKUPS == "no" ]; then
		    startFrom='nil' #I never want to start from the backup
		else
		    echo "USE_BACKUPS = $USE_BACKUPS   not recognized"
		    exit
		fi

		#--Launch in interactive (PennPuter) or via queues (kondo)
		if [ $SYSTEM == "PennPuter" ]
		then
	    	    echo "python $prog --dataset=$dataset --steps_per_period=$spp --periods=$periods --batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$WD"
		    time (python $exeDIR/$prog --dataset=$dataset --steps_per_period=$spp --periods=$periods --batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$WD --load=$startFrom) 2>&1
		elif [ $SYSTEM == "kondo" ]
		then
		    nombre=${model}m${HS}lr${LR}bs${BS}s${ISAM}
		    echo "Process name: $nombre"
		    echo "qsub -N $nombre -v dataset=$dataset,spp=$spp,periods=$periods,BS=$BS,HS=$HS,outDIR=$outDIR,save_every=$save_every,LR=$LR,model=$model,WD=$WD,startFrom=$startFrom -j oe -o $LOGS/ipercubov2.2.$nombre.$$.txt bruna10.qsub"
		    qsub -N $nombre -q $QUEUE -v dataset=$dataset,spp=$spp,periods=$periods,BS=$BS,HS=$HS,outDIR=$outDIR,save_every=$save_every,LR=$LR,model=$model,WD=$WD,startFrom=$startFrom -j oe -o $logDIR/$nombre.$$.txt $scriptDIR/bruna10.qsub
		fi
	    
	    done
	    done
	done
    done
done
cd $scriptDIR
