#!/bin/bash

#SBATCH -J ecbdl14_cv
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -n 3
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../env/bin/activate

python ../cross-validator.py \
  debug=true \
  epochs=50 \
  hidden_layers=32+32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &

python ../cross-validator.py \
  debug=true \
  epochs=50 \
  hidden_layers=64+64 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &

python ../cross-validator.py \
  debug=true \
  epochs=50 \
  hidden_layers=128+128 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  batchnorm=true &

wait
