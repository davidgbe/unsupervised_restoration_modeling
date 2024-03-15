from copy import deepcopy as copy
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys
import scipy.io as sio
from scipy.sparse import csc_matrix, csr_matrix, kron
from functools import partial
import argparse
import time
import tracemalloc

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

import resource

matplotlib.use('agg')

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--env', metavar='E', type=str)
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--rng_seed', metavar='r', type=int, nargs=1)
parser.add_argument('--dropout_per', metavar='d', type=float, nargs=1)
parser.add_argument('--dropout_iter', metavar='di', type=int, nargs=1)
parser.add_argument('--cond', metavar='r', type=str, nargs=1)
parser.add_argument('--w_ee', metavar='ee', type=float, nargs=1)
parser.add_argument('--w_ei', metavar='ei', type=float, nargs=1)
parser.add_argument('--w_ie', metavar='ie', type=float, nargs=1)
parser.add_argument('--alpha_5', metavar='a5', type=float, nargs=1)
parser.add_argument('--silent_fraction', metavar='sf', type=float, nargs=1)
parser.add_argument('--hetero_comp_mech', metavar='H', type=str, nargs=1)
parser.add_argument('--stdp_type', metavar='S', type=str, nargs=1)
parser.add_argument('--load_run', metavar='L', type=str, nargs=2)
args = parser.parse_args()

print(args)

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=0.25e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.07,  # membrane leak potential (V)
    V_TH_E=-.043,  # membrane spike threshold (V)
    T_R_E=1e-3,  # refractory period (s)
    E_R_E=-0.065, # reset voltage (V)
    
    # Inhibitory membrane
    C_M_I=1e-6,
    G_L_I=.4e-3, 
    E_L_I=-.053,
    V_TH_I=-.043,
    T_R_I=1e-3, #0.25e-3,
    E_R_I=-.053, # reset voltage (V)
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.09, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC_OLD=200,
    N_UVA=0,
    N_INH=50,
    M=20,
    
    # Input params
    DRIVING_HZ=1, # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=10,
    PROJECTION_NUM=10,
    INPUT_STD=1e-3,
    BURST_T=1.5e-3,
    INPUT_DELAY=10e-3,
    
    # OTHER INPUTS
    SGM_N_EXC=1e-10, 
    SGM_N_INH=1e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input

    # Connection probabilities
    CON_PROB_R=0.,
    E_E_LOCAL_CON_PROB=0.8,
    E_I_CON_PROB=0.075 / (1 - 0.8 * args.silent_fraction[0]),
    I_E_CON_PROB=0.5,

    # Weights
    W_E_I_R=args.w_ei[0],
    W_I_E_R=args.w_ie[0],
    W_A=0,
    W_E_E_R=args.w_ee[0],
    W_E_R_MIN=1e-8,
    W_E_E_R_MAX=1e-3,
    W_E_I_R_MAX=2 * args.w_ei[0],
    SUPER_SYNAPSE_SIZE=1.5e-3,

    # Dropout params
    DROPOUT_MIN_IDX=0,
    DROPOUT_MAX_IDX=0, # set elsewhere
    DROPOUT_ITER=args.dropout_iter[0],
    DROPOUT_SEV=args.dropout_per[0],
    RANDOM_SYN_ADD_ITERS_EE=[i for i in range(args.dropout_iter[0] + 1, args.dropout_iter[0] + 251)],
    RANDOM_SYN_ADD_ITERS_OTHER=[i for i in range(args.dropout_iter[0] + 1, 3001)],

    # Synaptic plasticity params
    TAU_STDP_TRIP=40e-3,
    TAU_STDP_PAIR_PLUS=16.8e-3,
    TAU_STDP_PAIR_MINUS=33.7e-3,

    A_PAIR_PLUS=0,
    A_PAIR_MINUS=-2 * 0.3,
    A_TRIP_PLUS=5 * 0.3,
    A_TRIP_MINUS=0,

    ETA=0.0005,
    ALPHA_1=1,
    ALPHA_2=0,
    ALPHA_3=5,
    ALPHA_4=-50,
    ALPHA_5=args.alpha_5[0],

    HETERO_COMP_MECH=args.hetero_comp_mech[0],
    STDP_TYPE=args.stdp_type[0],

    SETPOINT_MEASUREMENT_PERIOD=(1100, 1200),
)

print(M.HETERO_COMP_MECH)
print(args.cond[0])

S = Generic(RNG_SEED=args.rng_seed[0], DT=0.1e-3, T=115e-3, EPOCHS=7000)
np.random.seed(S.RNG_SEED)

M.SUMMED_W_E_E_R_MAX = M.W_E_E_R
M.W_U_E = 0.26 * 0.004
if not args.cond[0].startswith('no_repl'):
    M.N_EXC_NEW = int(M.N_EXC_OLD * M.DROPOUT_SEV)
else:
    M.N_EXC_NEW = 0
M.N_EXC = M.N_EXC_OLD + M.N_EXC_NEW
M.DROPOUT_MAX_IDX = M.N_EXC

## SMLN

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)

def compute_secreted_levels(spks_for_e_cells, exc_locs, m, surviving_cell_mask=None, target_locs=None):
    curr_firing_rates = np.sum(spks_for_e_cells > 0, axis=0)

    if surviving_cell_mask is not None:
        curr_firing_rates[~surviving_cell_mask] = 0

    activity_metric = partial(gaussian_metric, w=curr_firing_rates, v=0.3)

    target_locs = exc_locs if target_locs is None else target_locs
    return compute_aggregate_dist_metric(target_locs, exc_locs, activity_metric)


def gen_continuous_network(size, m):
    w = m.W_E_E_R / m.PROJECTION_NUM

    active_cell_mask = np.random.rand(size) > args.silent_fraction[0]
    cont_idx_steps = np.random.rand(size) * 2
    cont_idx = np.array([np.sum(cont_idx_steps[:i]) for i in range(cont_idx_steps.shape[0])])

    active_inactive_pairings = np.outer(active_cell_mask, active_cell_mask).astype(bool)
    cont_idx_dists = cont_idx.reshape(cont_idx.shape[0], 1) - cont_idx.reshape(1, cont_idx.shape[0])

    def gen_local_ee_connectivity(dist, cutoff):
        connected = np.logical_and(dist >= 0, dist < cutoff)
        connected = np.logical_and(connected, np.random.rand(*dist.shape) < m.E_E_LOCAL_CON_PROB)
        return np.where(connected, 1., 0) # np.exp(-dist/tau), 0)

    inactive_weights = np.concatenate([exp_if_under_val(0.075, (1, size), 0.5 * r * w) for r in np.random.rand(size)], axis=0)

    cont_dist_cutoff = 20 #25

    sequence_weights = np.where(active_inactive_pairings, w / (1 - args.silent_fraction[0]) * gen_local_ee_connectivity(cont_idx_dists, cont_dist_cutoff), inactive_weights)
    sequence_delays = np.abs(cont_idx_dists)
    sequence_delays = np.where(sequence_delays < cont_dist_cutoff, sequence_delays, cont_dist_cutoff * np.random.rand(size, size))

    np.fill_diagonal(sequence_delays, 0)

    weights = np.zeros((m.N_EXC, m.N_EXC))
    weights[:size, :size] = sequence_weights

    delays = np.zeros((m.N_EXC, m.N_EXC))
    delays[:size, :size] = sequence_delays

    all_active_inactive_pairings = np.zeros((m.N_EXC, m.N_EXC)).astype(bool)
    all_active_inactive_pairings[:size, :size] = active_inactive_pairings

    undefined_delays = ~all_active_inactive_pairings

    delays[undefined_delays] = cont_dist_cutoff * np.random.rand(np.count_nonzero(undefined_delays))
    delays = delays / np.mean(delays[weights > m.W_E_R_MIN])

    return weights, delays, np.concatenate([active_cell_mask, np.zeros(m.N_EXC - size)]).astype(bool)

### RUN_TEST function
def run(m, output_dir_name, dropout={'E': 0, 'I': 0}, w_r_e=None, w_r_i=None):

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)
    
    w_u_proj = np.diag(np.ones(m.N_DRIVING_CELLS)) * m.W_U_E * 0.5
    w_u_uva = np.diag(np.ones(m.N_EXC_OLD - m.N_DRIVING_CELLS)) * m.W_U_E * 0.15 # initially 0.25

    w_u_e = np.zeros([m.N_EXC_OLD, m.N_EXC_OLD])
    w_u_e[:m.N_DRIVING_CELLS, :m.N_DRIVING_CELLS] += w_u_proj
    w_u_e[m.N_DRIVING_CELLS:m.N_EXC_OLD, m.N_DRIVING_CELLS:m.N_EXC_OLD] += w_u_uva

    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e ],
            [ np.zeros([m.N_EXC_NEW + m.N_INH, m.N_EXC_OLD]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_INH, m.N_EXC_OLD)),

        'A': np.zeros((m.N_EXC + m.N_INH, m.N_EXC_OLD)),
    }

    if w_r_e is None:
        w_e_e_r, ee_delays, active_cell_mask = gen_continuous_network(m.N_EXC_OLD, m)
        ee_delays = 3e-3 * ee_delays
        np.fill_diagonal(w_e_e_r, 0.)

        e_i_r = gaussian_if_under_val(m.E_I_CON_PROB, (m.N_INH, m.N_EXC), m.W_E_I_R, 0)

        e_i_r[:, m.N_EXC_OLD:] = 0
        e_i_r[:, ~active_cell_mask] = 0.1 * e_i_r[:, ~active_cell_mask]
        # e_i_r[:, m.N_EXC_OLD - m.PROJECTION_NUM:m.N_EXC_OLD] = gaussian_if_under_val(0.1, (m.N_INH, m.PROJECTION_NUM), m.W_E_I_R, 0)

        w_r_e = np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ e_i_r,  np.zeros((m.N_INH, m.N_INH)) ],
        ])

    if w_r_i is None:

        i_e_r = gaussian_if_under_val(m.I_E_CON_PROB, (m.N_EXC, m.N_INH), m.W_I_E_R, 0)

        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC + m.N_UVA)), i_e_r],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_UVA + m.N_INH)) ],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ]),
    }

    ee_connectivity = np.where(w_r_e[:m.N_EXC, :m.N_EXC] > 0, 1, 0)
    ei_connectivity = np.where(w_r_e[m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] > 0, 1, 0)

    delay_bins, delay_freqs = bin_occurrences(ee_delays.flatten(), bin_size=0.05e-3)

    scale = 1
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(10 * scale, 10 * scale), tight_layout=True)
    axs = [fig.add_subplot(gs[0])]
    axs[0].plot(delay_bins[1:], delay_freqs[1:])
    fig.savefig(os.path.join(output_dir, 'exc_delay_distribution.png'))
    plt.close(fig)

    # (ee_delays / S.DT).astype(int)

    pairwise_spk_delays = np.block([
        [(ee_delays / S.DT).astype(int), np.ones((m.N_EXC, m.N_UVA)), int(0.5e-3 / S.DT) * np.ones((m.N_EXC, m.N_INH))],
        [int(0.5e-3 / S.DT) * np.ones((m.N_INH + m.N_UVA, m.N_EXC + m.N_INH + m.N_UVA))],
    ]).astype(int)

    # turn pairwise delays into list of cells one cell is synapsed to with some delay tau
   
    def make_delay_map(w_r):
        delay_map = {}
        summed_w_r_abs = np.sum(np.stack([np.abs(w_r[syn]) for syn in w_r.keys()]), axis=0)
        for i in range(pairwise_spk_delays.shape[1]):
            cons = summed_w_r_abs[:, i].nonzero()[0]
            delay_map[i] = (pairwise_spk_delays[cons, i], cons)
        return delay_map

    delay_map = make_delay_map(w_r)


    # create spatial structure
    exc_locs = sample_sphere(m.N_EXC)

    # spatial_dists = compute_aggregate_dist_metric(exc_locs, exc_locs, lambda x: x)
    # connected_pairwise_spk_delays = spatial_dists[ee_connectivity.nonzero()].flatten()

    # delay_bins, delay_freqs = bin_occurrences(connected_pairwise_spk_delays, bin_size=0.01)

    # scale = 1
    # gs = gridspec.GridSpec(1, 1)
    # fig = plt.figure(figsize=(10 * scale, 10 * scale), tight_layout=True)
    # axs = [fig.add_subplot(gs[0])]

    # axs[0].plot(delay_bins, delay_freqs)

    # fig.savefig('./spatial_dist_distribution.png')


    def create_prop(prop_exc, prop_inh):
        return cc([prop_exc * np.ones(m.N_EXC + m.N_UVA), prop_inh * np.ones(m.N_INH)])

    c_m = create_prop(m.C_M_E, m.C_M_I)
    g_l = create_prop(m.G_L_E, m.G_L_I)
    e_l = create_prop(m.E_L_E, m.E_L_I)
    v_th = create_prop(m.V_TH_E, m.V_TH_I)
    e_r = create_prop(m.E_R_E, m.E_R_I)
    t_r = create_prop(m.T_R_E, m.T_R_I)

    e_cell_fr_setpoints = np.ones(m.N_EXC) * 4
    target_secreted_levels = np.zeros((m.N_EXC, m.SETPOINT_MEASUREMENT_PERIOD[1] - m.SETPOINT_MEASUREMENT_PERIOD[0]))

    sampled_e_cell_rasters = []
    e_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_EXC).astype(int))
    sampled_i_cell_rasters = []
    i_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_INH + m.N_EXC).astype(int))

    w_r_copy = copy(w_r)

    # tracemalloc.start()
    # snapshot = None
    # last_snapshot = tracemalloc.take_snapshot()

    surviving_cell_mask = None
    ei_initial_summed_inputs = np.sum(w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC], axis=1)

    initial_first_spike_times = None

    batched_data_to_save = []

    n_dropout_iters = 5
    p_dropout_for_i_e = 1 - np.power((1 - dropout['E']), 1/n_dropout_iters)

    for i_e in range(S.EPOCHS):

        progress = f'{i_e / S.EPOCHS * 100}'
        progress = progress[: progress.find('.') + 2]
        print(f'{progress}% finished')

        start = time.time()

        # if i_e == 10:
        #     w_r_copy['I'][100, :] = 0

        if i_e >= m.DROPOUT_ITER and i_e < m.DROPOUT_ITER + n_dropout_iters:
            w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC_OLD], surviving_cell_mask_new = dropout_on_mat(w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC_OLD], p_dropout_for_i_e)
            surviving_cell_mask_new = np.concatenate([surviving_cell_mask_new, np.ones(m.N_EXC_NEW)])
            surviving_cell_mask_new = surviving_cell_mask_new.astype(bool)
            if surviving_cell_mask is not None:
                surviving_cell_mask = np.logical_and(surviving_cell_mask, surviving_cell_mask_new)
            else:
                surviving_cell_mask = surviving_cell_mask_new
            # print(surviving_cell_mask)
            ee_connectivity = np.where(w_r_copy['E'][:m.N_EXC, :m.N_EXC] > 0, 1, 0)
            ei_connectivity = np.where(w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] > 0, 1, 0)

        # growth_prob = 0.0005
        # if not args.cond[0].startswith('no_repl_no_syn'):
        #     if i_e in m.RANDOM_SYN_ADD_ITERS_EE:
        #         new_synapses = exp_if_under_val(0.00022, (m.N_EXC, m.N_EXC), 0.4 * m.W_E_E_R / M.PROJECTION_NUM)
        #         new_synapses[~surviving_cell_mask, :] = 0
        #         new_synapses[:, ~surviving_cell_mask] = 0
        #         np.fill_diagonal(new_synapses, 0)
        #         w_r_copy['E'][:m.N_EXC, :m.N_EXC] += new_synapses
        #         ee_connectivity = np.where(w_r_copy['E'][:(m.N_EXC), :(m.N_EXC + m.N_UVA)] > 0, 1, 0)

        # if i_e in m.RANDOM_SYN_ADD_ITERS_OTHER:
        #     new_ei_synapses = gaussian_if_under_val(0.4 * growth_prob, (m.N_INH, m.N_EXC), m.W_E_I_R, 0)
        #     new_ei_synapses[:, ~surviving_cell_mask] = 0
        #     new_ei_synapses[np.sum(w_r_copy['E'][(m.N_EXC + m.N_UVA):, :m.N_EXC], axis=1) >= ei_initial_summed_inputs, :] = 0
        #     w_r_copy['E'][(m.N_EXC + m.N_UVA):, :m.N_EXC] += new_ei_synapses

            # new_ie_synapses = gaussian_if_under_val(10 * growth_prob, (m.N_EXC_NEW, m.N_INH), m.W_I_E_R, 0)
            # new_ie_synapses[w_r_copy['I'][m.N_EXC_OLD:m.N_EXC, (m.N_EXC + m.N_UVA):] > 0] = 0
            # w_r_copy['I'][m.N_EXC_OLD:m.N_EXC, (m.N_EXC + m.N_UVA):] += new_ie_synapses

        # if i_e in m.RANDOM_SYN_ADD_ITERS_EE or i_e in m.RANDOM_SYN_ADD_ITERS_OTHER:
        #     delay_map = make_delay_map(w_r_copy)

        t = np.arange(0, S.T, S.DT)

        ## external currents
        i_ext = np.concatenate([m.SGM_N_EXC/S.DT * np.random.randn(len(t), m.N_EXC), m.SGM_N_INH/S.DT * np.random.randn(len(t), m.N_INH)], axis=1) + m.I_EXT_B

        ## inp spks
        spks_u_base = np.zeros((len(t), m.N_EXC_OLD), dtype=int)

        # trigger inputs
        activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
        activation_times[0, :] = 1

        spks_u = copy(spks_u_base)
        spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
        burst_t = np.arange(0, 5 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

        trip_spk_hist = [[] for n_e in range(m.N_EXC + m.N_INH)]

        for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
            input_noise_t = np.array(np.random.normal(scale=m.INPUT_STD / S.DT), dtype=int)
            try:
                spks_u[burst_t + t_idx + input_noise_t + int(m.INPUT_DELAY / S.DT), driving_cell_idx] = 1
            except IndexError as e:
                pass

        def make_poisson_input(dur=0.1, offset=0.005):
            x = np.zeros(len(t))
            x[int(offset/S.DT):int(offset/S.DT) + int(dur/S.DT)] = np.random.poisson(lam=50 * S.DT, size=int(dur/S.DT)) # initially 10
            return x

        spks_u[:, m.N_DRIVING_CELLS:m.N_EXC_OLD] = np.stack([make_poisson_input(offset=m.INPUT_DELAY) for i in range(m.N_EXC_OLD - m.N_DRIVING_CELLS)]).T

        ntwk = LIFNtwkG(
            c_m=c_m,
            g_l=g_l,
            e_l=e_l,
            v_th=v_th,
            v_r=e_r,
            t_r=t_r,
            e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
            t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
            stdp_t_s={'TAU_STDP_PAIR_PLUS': M.TAU_STDP_PAIR_PLUS, 'TAU_STDP_PAIR_MINUS': M.TAU_STDP_PAIR_MINUS, 'TAU_STDP_TRIP_PLUS': M.TAU_STDP_TRIP, 'TAU_STDP_TRIP_MINUS': M.TAU_STDP_TRIP},
            stdp_coefs={'A_PAIR_PLUS': M.A_PAIR_PLUS, 'A_PAIR_MINUS': M.A_PAIR_MINUS, 'A_TRIP_PLUS': M.A_TRIP_PLUS, 'A_TRIP_MINUS': M.A_TRIP_MINUS},
            w_r=w_r_copy,
            w_u=w_u,
            pairwise_spk_delays=pairwise_spk_delays,
            delay_map=delay_map,
        )

        clamp = Generic(v={0: e_l}, spk={})

        # run smln
        rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext, spks_u=spks_u)

        scale = 0.8
        gs = gridspec.GridSpec(14, 1)
        fig = plt.figure(figsize=(9 * scale, 35 * scale), tight_layout=True)
        axs = [
            fig.add_subplot(gs[:2]),
            fig.add_subplot(gs[2]),
            fig.add_subplot(gs[3]),
            fig.add_subplot(gs[4]),
            fig.add_subplot(gs[5]),
            fig.add_subplot(gs[6:8]),
            fig.add_subplot(gs[8:10]),
            fig.add_subplot(gs[10:12]),
            fig.add_subplot(gs[12:]),
        ]

        w_e_e_r_copy = w_r_copy['E'][:m.N_EXC, :m.N_EXC]
        if surviving_cell_mask is not None:
            w_e_e_r_copy = w_e_e_r_copy[surviving_cell_mask, :]

        # 0.05 * np.mean(w_e_e_r_copy.sum(axis=1)
        summed_w_bins, summed_w_counts = bin_occurrences(w_e_e_r_copy.sum(axis=1), bin_size=1e-4, max_val=0.004)
        axs[3].plot(summed_w_bins, summed_w_counts)
        axs[3].set_xlabel('Normalized summed synapatic weight')
        axs[3].set_ylabel('Counts')

        incoming_con_counts = np.count_nonzero(w_e_e_r_copy, axis=1)
        incoming_con_bins, incoming_con_freqs = bin_occurrences(incoming_con_counts, bin_size=1)
        axs[4].plot(incoming_con_bins, incoming_con_freqs)
        axs[4].set_xlabel('Number of incoming synapses per cell')
        axs[4].set_ylabel('Counts')

        cmap = cm.viridis.copy()
        cmap.set_under(color='white')

        min_ee_weight = w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)].min()
        graph_weight_matrix(w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)], 'w_e_e_r\n', ax=axs[5],
            v_min=min_ee_weight, v_max=m.W_E_E_R_MAX, cmap=cmap)
        graph_weight_matrix(w_r_copy['E'][m.N_EXC:, :m.N_EXC], 'w_e_i_r\n', ax=axs[6], v_max=m.W_E_I_R_MAX, cmap=cmap)

        spks_for_e_cells = rsp.spks[:, :m.N_EXC]

        print('uninhibited_activity', np.count_nonzero(spks_for_e_cells[:, 100]))
        spks_for_i_cells = rsp.spks[:, (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH)]

        spks_received_for_e_cells = rsp.spks_received[:, :m.N_EXC, :m.N_EXC]
        spks_received_for_i_cells = rsp.spks_received[:, m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC]

        if surviving_cell_mask is not None:
            print('surviving_count:', np.count_nonzero(surviving_cell_mask))
            spk_bins, freqs = bin_occurrences(spks_for_e_cells[:, surviving_cell_mask].sum(axis=0), max_val=800, bin_size=1)
            print('TOTAL_ACTIVITY:', spks_for_e_cells[:, surviving_cell_mask].sum())
        else:
            spk_bins, freqs = bin_occurrences(spks_for_e_cells.sum(axis=0), max_val=800, bin_size=1)
            print('TOTAL_ACTIVITY:', spks_for_e_cells.sum())

        axs[1].bar(spk_bins, freqs, alpha=0.5)
        axs[1].set_xlabel('Spks per neuron')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlim(-0.5, 30.5)
        axs[1].set_ylim(0, 100)

        raster = np.stack([rsp.spks_t, rsp.spks_c])
        exc_raster = raster[:, raster[1, :] < m.N_EXC]
        inh_raster = raster[:, raster[1, :] >= (m.N_EXC + m.N_UVA)]

        spk_bins_i, freqs_i = bin_occurrences(spks_for_i_cells.sum(axis=0), max_val=800, bin_size=1)

        axs[2].bar(spk_bins_i, freqs_i, color='black', alpha=0.5, zorder=-1)
        axs[2].set_xlim(-0.5, 100)

        axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)
        axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :] - m.N_UVA, s=1, c='red', zorder=0, alpha=1)

        axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
        axs[0].set_xlim(0, S.T * 1000)
        axs[0].set_ylabel('Cell Index')
        axs[0].set_xlabel('Time (ms)')

        for i in range(len(axs)):
            set_font_size(axs[i], 14)

        first_spk_times = process_single_activation(exc_raster, m)

        if i_e == 0:
            initial_first_spike_times = first_spk_times

        if i_e >= m.SETPOINT_MEASUREMENT_PERIOD[0] and i_e < m.SETPOINT_MEASUREMENT_PERIOD[1]:
            target_secreted_levels[:, i_e - m.SETPOINT_MEASUREMENT_PERIOD[0]] = compute_secreted_levels(spks_for_e_cells, exc_locs, m)

        if i_e == m.SETPOINT_MEASUREMENT_PERIOD[1]:
            target_secreted_levels = np.mean(target_secreted_levels, axis=1)
            if not args.cond[0].startswith('no_repl'):
                target_secreted_levels[M.N_EXC_OLD:M.N_EXC] = np.mean(target_secreted_levels[:M.N_EXC_OLD])

        if i_e > 0:
            exc_ee_weights = w_r_copy['E'][:m.N_EXC, :m.N_EXC]
            exc_ei_weights = w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC]

            # Firing rate homeostasis
            e_diffs = np.maximum(np.sum(spks_for_e_cells > 0, axis=0) - 3, 0)
            e_diffs_squared = np.power(e_diffs, 2)

            fr_update_e = e_diffs_squared.reshape(e_diffs_squared.shape[0], 1) * np.ones((m.N_EXC, m.N_EXC)).astype(float)
            firing_rate_homeo_depression = m.ALPHA_4 * fr_update_e

            firing_rate_homeo_potentiation = m.ALPHA_3 if m.HETERO_COMP_MECH.startswith('firing_rate') else 0
            w_r_copy['E'][:m.N_EXC, :m.N_EXC] += (m.ETA * (firing_rate_homeo_potentiation + firing_rate_homeo_depression) * exc_ee_weights)

            # E-->E STDP
            ee_update_plus = rsp.pair_update_plus[:m.N_EXC, :m.N_EXC] + rsp.trip_update_plus[:m.N_EXC, :m.N_EXC]
            ee_update_minus = rsp.pair_update_minus[:m.N_EXC, :m.N_EXC] + rsp.trip_update_minus[:m.N_EXC, :m.N_EXC]

            print('nonzero ee_connectivity count:', np.count_nonzero(ee_connectivity))
            w_r_copy['E'][:m.N_EXC, :m.N_EXC] += m.ETA * ((m.W_E_E_R_MAX * ee_connectivity - exc_ee_weights) * ee_update_plus + exc_ee_weights * ee_update_minus)
            
            # E-->I STDP
            ei_update_plus = rsp.pair_update_plus[m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] + rsp.trip_update_plus[m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC]
            ei_update_minus = 0 * rsp.pair_update_minus[m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] + rsp.trip_update_minus[m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC]

            w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] += 5 * m.ETA * ((m.W_E_I_R_MAX * ei_connectivity - exc_ei_weights) * ei_update_plus + exc_ei_weights * ei_update_minus)

            # HETEROSYNAPTIC COMPETITION RULES

            if m.HETERO_COMP_MECH.startswith('secreted_regulation'):

                if i_e > m.SETPOINT_MEASUREMENT_PERIOD[1]:
                    if i_e >= m.DROPOUT_ITER:
                        secreted_levels = compute_secreted_levels(spks_for_e_cells, exc_locs, m, surviving_cell_mask=surviving_cell_mask)
                    else:
                        secreted_levels = compute_secreted_levels(spks_for_e_cells, exc_locs, m)

                    n_steps = int(2/1e-2)
                    x, y, z = np.meshgrid(np.linspace(-1, 1, n_steps), np.linspace(-1, 1, n_steps), [0])
                    square_coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

                    if i_e >= m.DROPOUT_ITER:
                        square_levels = compute_secreted_levels(spks_for_e_cells, exc_locs, m, target_locs=square_coords, surviving_cell_mask=surviving_cell_mask)
                    else:
                        square_levels = compute_secreted_levels(spks_for_e_cells, exc_locs, m, target_locs=square_coords)
                    graph_weight_matrix(square_levels.reshape(n_steps, n_steps), '', ax=axs[8], cmap='viridis')

                    secreted_diffs = target_secreted_levels - secreted_levels

                    print(secreted_diffs)

                    print('lacking activity count:', np.count_nonzero(secreted_diffs > 0))

                    def sigmoid_tranform(x):
                        return (np.exp(x) - 1) / (np.exp(x) + 1)

                    sigmoid_transform_e_diffs = sigmoid_tranform(secreted_diffs / 10)

                    w = m.W_E_E_R / m.PROJECTION_NUM

                    # if i_e >= m.DROPOUT_ITER and i_e < m.DROPOUT_ITER + 100:
                    #     for l_syn in range(50):
                    #         new_synapses_ee = np.where(np.random.rand(m.N_EXC, m.N_EXC) < 0.0002, 3 * w, 0)
                    #         new_synapses_ee[secreted_diffs <= 0, :] = 0
                    #         if surviving_cell_mask is not None:
                    #             # new_synapses_ee[~surviving_cell_mask, :] = 0
                    #             new_synapses_ee[:, ~surviving_cell_mask] = 0
                    #             new_synapses_ee[:, spks_for_e_cells.sum(axis=0) <= 0] = 0
                    #             # new_synapses_ee[spks_for_e_cells.sum(axis=0) >= 4, :] = 0
                    #         np.fill_diagonal(new_synapses_ee, 0)
                    #         w_r_copy['E'][:m.N_EXC, :m.N_EXC] += new_synapses_ee
                    #         ee_connectivity = np.where(np.logical_or(ee_connectivity.astype(bool), new_synapses_ee > 0), 1, 0)

                    new_synapses_ee = 0.01 * w * sigmoid_transform_e_diffs.reshape((len(sigmoid_transform_e_diffs), 1)) * ee_connectivity
                    if surviving_cell_mask is not None:
                        new_synapses_ee[:, ~surviving_cell_mask] = 0
                        # new_synapses_ee[:, spks_for_e_cells.sum(axis=0) <= 0] = 0
                    np.fill_diagonal(new_synapses_ee, 0)
                    w_r_copy['E'][:m.N_EXC, :m.N_EXC] += new_synapses_ee
                    ee_connectivity = np.where(np.logical_or(ee_connectivity.astype(bool), new_synapses_ee > 0), 1, 0)

                    # new_synapses_ei = exp_if_under_val(0.002, (m.N_INH, m.N_EXC), m.W_E_I_R)
                    # new_synapses_ei[:, secreted_diffs <= 0] = 0
                    # if surviving_cell_mask is not None:
                    #     new_synapses_ei[:, ~surviving_cell_mask] = 0
                    # w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] += new_synapses_ei
                    # ei_connectivity = np.where(np.logical_or(ei_connectivity, new_synapses_ei > 0), 1, 0)

                    # w_update = sigmoid_transform_e_diffs.reshape(sigmoid_transform_e_diffs.shape[0], 1) * np.ones((m.N_EXC, m.N_EXC + m.N_UVA)).astype(float)
                    # w_r_copy['E'][:m.N_EXC, :m.N_EXC] += (m.ETA * m.ALPHA_5 * w_update * exc_ee_weights)

            w_r_copy['E'][:m.N_EXC, :m.N_EXC][np.logical_and((w_r_copy['E'][:m.N_EXC, :m.N_EXC] < m.W_E_R_MIN), ee_connectivity.astype(bool))] = m.W_E_R_MIN
            w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC][np.logical_and((w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] < m.W_E_R_MIN), ei_connectivity.astype(bool))] = m.W_E_R_MIN

            w_r_copy['E'][:m.N_EXC, :m.N_EXC][w_r_copy['E'][:m.N_EXC, m.N_EXC] > m.W_E_E_R_MAX] = m.W_E_E_R_MAX
            w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC][w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] > m.W_E_I_R_MAX] = m.W_E_I_R_MAX

            # output weight bound
            i_cell_summed_inputs = w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC].sum(axis=1)
            rescaling = np.where(i_cell_summed_inputs  > ei_initial_summed_inputs, ei_initial_summed_inputs / i_cell_summed_inputs, 1.)
            w_r_copy['E'][m.N_EXC:(m.N_EXC + m.N_INH), :m.N_EXC] *= rescaling.reshape(rescaling.shape[0], 1)

            # print('ei_mean_stdp', np.mean(m.ETA * m.BETA * stdp_burst_pair_e_i))
            # w_r_copy['I'][:(m.N_EXC + m.N_SILENT), (m.N_EXC + m.N_SILENT):] += 1e-4 * m.ETA * m.BETA * stdp_burst_pair_e_i
            # w_r_copy['I'][w_r_copy['I'] < 0] = 0
            # w_r_copy['I'][w_r_copy['I'] > m.W_I_E_R_MAX] = m.W_I_E_R_MAX

        if i_e % 1 == 0:
            base_data_to_save = {
                'w_e_e': m.W_E_E_R,
                'w_e_i': m.W_E_I_R,
                'w_i_e': m.W_I_E_R,
                'n_exc': m.N_EXC,
                'n_inh': m.N_INH,
                'first_spk_times': first_spk_times,
                'spk_bins': spk_bins,
                'freqs': freqs,
                'exc_raster': exc_raster,
                'inh_raster': inh_raster,
            }

            if i_e % 20 == 0:
                base_data_to_save.update({
                    'w_r_e': copy(rsp.ntwk.w_r['E']),
                    'w_r_i': copy(rsp.ntwk.w_r['I']),
                })


            if i_e >= m.DROPOUT_ITER:
                base_data_to_save['surviving_cell_mask'] = copy(surviving_cell_mask)

            batched_data_to_save.append(base_data_to_save)

            save_freq = 100
            if i_e % save_freq == (save_freq - 1):
                sio.savemat(robustness_output_dir + '/' + f'title_{args.title[0]}_idx_{zero_pad(i_e, 4)}', {'data': batched_data_to_save})
                batched_data_to_save = []

            fig_save_freq = 1 if args.env == 'local' else 1000
            if i_e % fig_save_freq == 0:
                fig.savefig(f'{output_dir}/{zero_pad(i_e, 4)}.png')

        log_file = open(os.path.join(robustness_output_dir, 'log'), 'a+')

        end = time.time()
        secs_per_cycle = f'{end - start}'
        secs_per_cycle = secs_per_cycle[:secs_per_cycle.find('.') + 2]
        print(f'{secs_per_cycle} s')
        print(f'{secs_per_cycle} s', file=log_file)

        usage_in_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / np.power(10, 9)
        print(usage_in_gb)
        print(f'{usage_in_gb}', file=log_file)

        log_file.close()

        plt.close('all')

        # snapshot = tracemalloc.take_snapshot()
        # if last_snapshot is not None:
        #     top_stats = snapshot.compare_to(last_snapshot, 'lineno')
        #     print("[ Top 3 differences ]")
        #     for stat in top_stats[:3]:
        #         print(stat)



def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC + m.N_UVA)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times

def load_previous_run(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded


### Simulation setup
w_r_e = None
w_r_i = None

# Load previous saved weight matrices if applicable
if args.load_run is not None:
    loaded = load_previous_run(os.path.join('./robustness', args.load_run[0]), int(args.load_run[1]))
    w_r_e = np.array(loaded['w_r_e'].todense())
    w_r_i = np.array(loaded['w_r_i'].todense())

# Define the output directory name
output_dir_name = f'{args.title[0]}_{time_stamp(s=True)}:{zero_pad(int(np.random.rand() * 9999), 4)}'
# Specify which populations to drop out and at what probability
dropout = {'E': M.DROPOUT_SEV, 'I': 0}

# Begin the simulation
run(M, output_dir_name=output_dir_name, dropout=dropout, w_r_e=w_r_e, w_r_i=w_r_i)


