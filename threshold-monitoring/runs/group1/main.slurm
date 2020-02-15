#!/bin/bash

#SBATCH -J thresh_mon
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node087
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../../env/bin/activate

srun python ../../threshold-monitor.py \
  debug=true \
  epochs=20 \
  runs=3 \
  positive_fraction=1 \
  negative_fraction=0.03282 \
  positive_ratio=40% 

