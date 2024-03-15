#!/bin/bash

# caffeinate python3 run.py --title summed_bound --rng_seed $3 --hetero_comp_mech firing_rate --stdp_type w_minus --dropout_per $1 --dropout_iter $2 --cond no_repl_no_syn --w_ee 1.2e-3 --w_ei 7e-5 --w_ie 4e-5 --silent_fraction $4 --alpha_5 0.3
caffeinate python run_prop_speed.py --env local --title rectified_STDP --rng_seed $1 --hetero_comp_mech none --stdp_type w_minus --cond no_repl_no_syn --w_ee 0.8e-3 --w_ei 3.5e-5 --w_ie 2e-5 --alpha_5 6