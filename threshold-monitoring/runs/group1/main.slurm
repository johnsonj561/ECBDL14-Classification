#!/bin/bash

#SBATCH -J thresh_mon
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=3
#SBATCH -w node087
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../../env/bin/activate

srun -n1 python ../../threshold-monitor.py \
  debug=true \
  epochs=20 \
  hidden_layers=64+64 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  runs=2 \
  batchnorm=true &

srun -n1 python ../../threshold-monitor.py \
  debug=true \
  epochs=20 \
  hidden_layers=128+128 \
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  runs=2 \
  batchnorm=true &

srun -n1 python ../../threshold-monitor.py \
  debug=true \
  epochs=20 \
  hidden_layers=32+32+32\
  learn_rate=0.001 \
  batch_size=128 \
  dropout_rate=0.5 \
  runs=2 \
  batchnorm=true &

wait
