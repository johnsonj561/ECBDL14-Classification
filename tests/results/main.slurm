#!/bin/bash

#SBATCH -J ecbdl14_eval
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source /home/jjohn273/git/ECBDL14-Classification/env/bin/activate

srun python3 ../evaluator.py 
