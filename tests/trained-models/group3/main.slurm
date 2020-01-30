#!/bin/bash

#SBATCH -J ecbdl14_test_g3
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node089
#SBATCH --cpus-per-task=19
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source /home/jjohn273/git/ECBDL14-Classification/env/bin/activate

# 50%
srun python3 ../../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.021882 \
  positive_ratio=50% &

# 60%
srun python3 ../../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.0145885 \
  positive_ratio=60% &

# 70%
srun python3 ../../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.009378 \
  positive_ratio=70% &

# 80%
srun python3 ../../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.0054707 \
  positive_ratio=80% &

# 90%
srun python3 ../../trainer.py \
  epochs=50 \
  runs=30 \
  positive_fraction=1 \
  negative_fraction=0.0024314 \
  positive_ratio=90% &

wait
