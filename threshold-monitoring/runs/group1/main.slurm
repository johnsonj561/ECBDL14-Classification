#!/bin/bash

#SBATCH -J thresh_mon
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node087
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../../env/bin/activate

# 0.5%
srun python3 ../../threshold-monitor.py \
  epochs=50 \
  runs=5 \
  positive_fraction=0.2296 \
  negative_fraction=1.0 \
  positive_ratio=0.5% \
  delta_thresh=0.001

