#!/bin/bash
#PBS -N EZlarge2pars
##PBS -t 1-10
##PBS -M sseelig@uni-potsdam.de
#PBS -m a
#PBS -j oe
#PBS -l ncpus=35
#PBS -l nodes=1:ppn=35
#PBS -l walltime=96:00:00
#PBS -l mem=2g

LOGFILE=$PBS_O_WORKDIR/${PBS_JOBID}.log
NUMPROCS=`wc -l < $PBS_NODEFILE` 

export OMP_NUM_THREADS=$NUMPROCS

python3 -c 'import sys; print(sys.path)'

time python3 EZ_DREAM_large_Alpha1Lambda.py
