#!/bin/bash 
#PBS -M rajo@fotonik.dtu.dk 
#PBS -m a
#PBS -A fotonano
#PBS -N NN
#PBS -q fotonano
#PBS -l walltime=6:00:00 
#PBS -l mem=16gb
#PBS -t 1-120
#PBS -o logs/out.log
#PBS -e logs/error.log

module swap python/2.7.12_ucs4
module load numpy/1.11.2-python-2.7.12-openblas-0.2.15_ucs4
module load scipy/scipy-0.18.1-python-2.7.12_ucs4
module load matplotlib/matplotlib-2.0.2-python-2.7.12-ucs4
source /appl/tensorflow/1.1cpu/bin/activate

cd $PBS_O_WORKDIR 
pwd > logs/NN-log-$PBS_ARRAYID.txt
printenv | grep PBS >> logs/NN-log-$PBS_ARRAYID.txt
/appl/glibc/2.17/lib/ld-linux-x86-64.so.2  --library-path /appl/glibc/2.17/lib/:/appl/gcc/4.8.5/lib64/:/lib64:/usr/lib64:$LD_LIBRARY_PATH $(which python) trainNFTreceiver_single.py -n $PBS_ARRAYID >> logs/NN-log-$PBS_ARRAYID.txt
