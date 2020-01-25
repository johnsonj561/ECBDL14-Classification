#!/bin/bash

#SBATCH -J ecbdl14_t1
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

module load anaconda/anaconda3/5.0.1

source activate jjohn273-tf2

python3 ../trainer.py \
  epochs=2 \
  runs=2 \
  positive_fraction=1 \
  negative_fraction=0.1

