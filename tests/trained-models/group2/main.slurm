#!/bin/bash

#SBATCH -J ecbdl14_test_g2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node088
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source /home/jjohn273/git/ECBDL14-Classification/env/bin/activate
 
# 10%
srun python3 ../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.1969 \
  positive_ratio=10% &
  
# 20%
srun python3 ../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.087531 \
  positive_ratio=20% &
  
# 30%
srun python3 ../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.05106 \
  positive_ratio=30% &
  
# 40%
srun python3 ../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.03282 \
  positive_ratio=40% &

wait
