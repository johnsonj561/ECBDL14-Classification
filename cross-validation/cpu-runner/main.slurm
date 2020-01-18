#!/bin/bash

#SBATCH -J ecbdl14_cv_1
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source ../../env/bin/activate

srun python ../cross-validator.py
