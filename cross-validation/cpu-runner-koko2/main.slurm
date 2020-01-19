#!/bin/bash

#SBATCH -J ecbdl14_cv_kk2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../env/bin/activate

srun python ../cross-validator.py \
  debug=true \
  epochs=5 \
  hidden_layers=32|32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true \
 
