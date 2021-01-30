#!/bin/bash
#PBS -N Train_{classifier}_{feature}
#PBS -l nodes=1:ppn=20
#PBS -l walltime=88888:00:00
#PBS -q adf
#PBS -j oe
#PBS -m ae
#PBS -M your@email.com

which python

cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`

python classifier_{classifier}.py ${feature}

