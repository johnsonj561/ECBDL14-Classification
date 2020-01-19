#!/bin/bash

#SBATCH -J ecbdl14_cv_kk2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -n 3   # 3 tasks
#SBATCH -N 1   # 1 node
#SBATCH -c 5   # 5 cores per task
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../env/bin/activate

srun -n 1 python ../cross-validator.py \
  debug=true \
  epochs=5 \
  hidden_layers=32|32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &

srun -n 1 python ../cross-validator.py \
  debug=true \
  epochs=5 \
  hidden_layers=64|64 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &
 
srun -n 1 python ../cross-validator.py \
  debug=true \
  epochs=5 \
  hidden_layers=128|64|32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &
