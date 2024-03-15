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
python run.py --env cluster --title reduced_ie_uva_fast_TITLE --rng_seed SEED --hetero_comp_mech COMP_MECH --stdp_type w_minus --dropout_per DROP_SEV  --dropout_iter 3000 --cond CONDITION --w_ee W_E_E --w_ei W_E_I --w_ie W_I_E --silent_fraction PER_SILENT --alpha_5 A5
zip -r -qq robustness/TITLE.zip robustness/TITLE
deactivate

## Exit
exit 0
