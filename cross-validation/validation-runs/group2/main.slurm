#!/bin/bash

#SBATCH -J ecbdl14_cv2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

module load anaconda/anaconda3/5.0.1

source activate jjohn273-tf2

#python3 ../../cross-validator.py \
#  debug=false \
#  epochs=250 \
#  hidden_layers=32+32 \
#  learn_rate=0.001 \
#  batch_size=256 \
#  dropout_rate=0.5 \
#  batchnorm=true

#python3 ../../cross-validator.py \
#  debug=false \
#  epochs=250 \
#  hidden_layers=128+128 \
#  learn_rate=0.001 \
#  batch_size=256 \
#  dropout_rate=0.5 \
#  batchnorm=true

python3 ../../cross-validator.py \
  debug=false \
  epochs=250 \
  hidden_layers=128+128+64 \
  learn_rate=0.001 \
  batch_size=256 \
  dropout_rate=0.5 \
  batchnorm=true
