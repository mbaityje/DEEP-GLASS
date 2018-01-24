#!/bin/bash
# Script to launch bruna network
# 

echo "Python Modules:"
module load python/2.7.5
module list
python --version

#export PYTHONPATH=$PYTHONPATH:/home/mbaityje/miniconda2/pkgs/pytorch-0.2.0-py27h44c09ab_2cu75/lib/python2.7/site-packages/:
#export PATH=/home/mbaityje/miniconda2/bin:$PATH

#--SYSTEM RECOGNITION--
myusername=`whoami`
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

#--DIRECTORIES--
scriptDIR=$PWD/
myROOT=$PWD/../
exeDIR=$myROOT/source
outROOT=$myROOT/output #Parent directory for all the output
dataDIR=$myROOT/data_$dataset
logDIR=$myROOT/logs

#--NETWORK--
prog=corr.py

#--HYPERPARAMETERS--
#readonly dataset='cifar100'; MODELS="conv1020relu alex6464"; HIDDEN_SIZES="0"; LEARNING_RATES="0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5"
#readonly dataset='cifar10'; MODELS="conv1020relu alex6464"; HIDDEN_SIZES="0"; LEARNING_RATES="0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5"
readonly dataset='cifar100'; MODELS="bruna10"; HIDDEN_SIZES="10 100 1000 10000"; LEARNING_RATES="0.01 0.02 0.05 0.1 0.2 0.5"
#readonly dataset='cifar10'; MODELS="bruna10"; HIDDEN_SIZES="10 100 1000 10000"; LEARNING_RATES="0.01 0.02 0.05 0.1 0.2 0.5"
#readonly dataset='mnist'; MODELS="bruna10"; HIDDEN_SIZES="10 100 1000 10000"; LEARNING_RATES="0.01 0.02 0.05 0.1 0.2 0.5"
#readonly dataset='cifar10'; MODELS="resnet18 resnet34"; HIDDEN_SIZES="0"; LEARNING_RATES="0.001 0.005 0.01 0.05 0.1"
#readonly dataset='cifar100'; MODELS="resnet18 resnet34"; HIDDEN_SIZES="0"; LEARNING_RATES="0.001 0.005 0.01 0.05 0.1"


BATCH_SIZES="10 100 1000"
WEIGHT_DECAYS="0" #Positive: L2, Negative: Bruna, Zero: None
INIT_DISTROS="default"
MOMENTA="0.5"
SAMPLES="0"
NTW=50; NT=100; NTBAR=100
TW0=10; T0=1; TBAR0=1

#Length of simulation
TOTLENGTH_NORM=`echo 10^6|bc` #This is the relevant time, LR*TOTLENGTH, which we want equal for all LR
USE_BACKUPS="no"


#--CYCLE OVER PARAMETERS--
cd $myROOT
for model in $MODELS
do
    for LR in $LEARNING_RATES
    do
        #Length of simulation (keep these quantities multiples (or better powers) of 10)
	TOTLENGTH=`echo $TOTLENGTH_NORM/$LR|bc` #Total number of processed images
	periods=1000     # number of periods
	save_every=`echo $periods/4|bc`  # we save 4 times during the run
	
	for BS in $BATCH_SIZES
	do
	    spp=`echo "$TOTLENGTH/($BS*$periods)"|bc`
	    if [ $spp -lt 1 ];then echo "spp=$spp is not positive"; exit; fi
	    
	    for HS in $HIDDEN_SIZES
	    do
		for WD in $WEIGHT_DECAYS
		do
		    for MOMENTUM in $MOMENTA
		    do
			for INIT_DISTR in $INIT_DISTROS
			do
			    for ISAM in $SAMPLES
			    do
				echo "$model LR=$LR, BS=$BS, m=$HS, WD=$WD, ISAM=$ISAM, INIT_DISTR=$INIT_DISTR"
				
			        #--Output directory--
				outDIR=$outROOT/${model}/m$HS/B${BS}/LR$LR/WD$WD/MOM$MOMENTUM/$INIT_DISTR/sam$ISAM/
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
				
			        #Take the seed from urandom device
				SEED=`od -vAn -N4 -tu4 < /dev/urandom | sed 's/[[:space:]]//g'`
				

				#If output already exists, skip this sample
				if [ -e $outDIR/${dataset}_C.txt ]; then
				    echo "Output already exists in this directory. We skip it."
				    continue  
				fi

			        #--Launch in interactive (PennPuter) or via queues (kondo)
				if [ $SYSTEM == "PennPuter" ]
				then
	    			    echo "python $exeDIR/$prog --dataset=$dataset --seed=$SEED --steps_per_period=$spp --periods=$periods --batch-size=$BS --test-batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$WD --load=$startFrom --momentum=$MOMENTUM --t0=$T0 --tw0=$TW0 --tbar0=$TBAR0 --nt=$NT --ntw=$NTW --ntbar=$NTBAR  --distr=$INIT_DISTR --losstxt=True"
				    time (python $exeDIR/$prog --dataset=$dataset --seed=$SEED --steps_per_period=$spp --periods=$periods --batch-size=$BS --test-batch-size=$BS --hidden_size=$HS --out=$outDIR --save-every=$save_every --lr=$LR --model=$model --weight_decay=$WD --load=$startFrom --momentum=$MOMENTUM --t0=$T0 --tw0=$TW0 --tbar0=$TBAR0 --nt=$NT --ntw=$NTW --ntbar=$NTBAR --distr=$INIT_DISTR --losstxt=True) 2>&1
				elif [ $SYSTEM == "kondo" ]
				then
				    nombre=${model}m${HS}lr${LR}bs${BS}s${ISAM}${INIT_DISTR}
				    echo "Process name: $nombre"
				    if [ -z "`qstat -f | grep $nombre`" ];
				    then
					echo "qsub -N $nombre -q $QUEUE -v dataset=$dataset,SEED=$SEED,spp=$spp,periods=$periods,BS=$BS,HS=$HS,outDIR=$outDIR,save_every=$save_every,LR=$LR,model=$model,WD=$WD,MOMENTUM=$MOMENTUM,startFrom=$startFrom,NT=$NT,NTW=$NTW,NTBAR=$NTBAR,T0=$T0,TW0=$TW0,TBAR0=$TBAR0,INIT_DISTR=$INIT_DISTR -j oe -o $logDIR/$nombre.$$.txt $scriptDIR/corr.qsub"
					qsub -N $nombre -q $QUEUE -v dataset=$dataset,SEED=$SEED,spp=$spp,periods=$periods,BS=$BS,HS=$HS,outDIR=$outDIR,save_every=$save_every,LR=$LR,model=$model,WD=$WD,MOMENTUM=$MOMENTUM,startFrom=$startFrom,NT=$NT,NTW=$NTW,NTBAR=$NTBAR,T0=$T0,TW0=$TW0,TBAR0=$TBAR0,INIT_DISTR=$INIT_DISTR -j oe -o $logDIR/$nombre.$$.txt $scriptDIR/corr.qsub
				    fi
				fi
			    done    
			done
		    done
		done
	    done
	done
    done
done
cd $scriptDIR
