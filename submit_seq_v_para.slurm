#!/bin/bash

## Presets
#SBATCH --job-name=syn_param

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate
python sequential_vs_parallel_inh_param_sweep.py
deactivate

## Exit
exit 0
