#!/bin/bash

## Presets
#SBATCH --job-name=syn_param

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=3G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate
for i in {0..5};
do python run_prop_speed.py --env local --title seed_$i --rng_seed $i --hetero_comp_mech none --stdp_type w_minus --cond no_repl_no_syn --w_ee 0.8e-3 --w_ei 3.5e-5 --w_ie 2e-5 --alpha_5 6;
done
deactivate

## Exit
exit 0
