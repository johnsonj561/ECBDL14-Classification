#!/bin/bash

#SBATCH -J ecbdl14_6
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../../env/bin/activate

srun -n1 python ../../cross-validator.py \
  debug=false \
  epochs=250 \
  hidden_layers=128+128+64+32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.2 \
  batchnorm=true &

srun -n1 python ../../cross-validator.py \
  debug=false \
  epochs=250 \
  hidden_layers=128+128+64+32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.2 \
  batchnorm=true &

srun -n1 python ../../cross-validator.py \
  debug=false \
  epochs=250 \
  hidden_layers=128+128+128+64+32 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.2 &

srun -n1 python ../../cross-validator.py \
  debug=false \
  epochs=250 \
  hidden_layers=128+128+128+64 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.2 \
  batchnorm=true &

wait
