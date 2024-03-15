import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as copy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
import pickle
import pandas as pd
from ast import literal_eval
import json
from aux import bin_occurrences

value_range = 256
color_list = [
    (0,   '#ad262f'),
    (0.5,   'black'),
    (1,   'yellow'),
]
activity_colormap = LinearSegmentedColormap.from_list('activity', color_list, N=value_range)

hsv_cmap = matplotlib.colormaps['hsv']
color_indices = np.linspace(0, 1, 100)
np.random.shuffle(color_indices)
rainbow_colors = hsv_cmap(color_indices)

def write_data(write_path, data):
    f = open(write_path, 'wb')
    pickle.dump(data, f)
    f.close()
    
def load_data(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def p_active(w, x, b, g):
    return sigmoid(g * (np.einsum('...jk,...j', w, x) - b))

def step(x, w, b, g, w_r, t=0, gabazine_mask=None):
    if gabazine_mask is not None:
        v_th = b + np.where(~(gabazine_mask.astype(bool)), np.sum(w_r * x[t, :, :]), 0)
    else:
        v_th = b + np.sum(w_r * x[t, :, :])
    
    p = p_active(w, x[t, :, :], v_th, g)
    x[t + 1, 1:, :] =  (np.random.rand(*(p.shape)) < p).astype(int)[:-1]
    x[t + 1, 0, :] = 0
    
def run_activation(w, b, g, w_r, gabazine_mask=None):
    x = np.zeros((w.shape[0], w.shape[0], w.shape[1])).astype(int)
    x[0, 0, :] = 1
    for t in range(w.shape[0] - 1):
        step(x, w, b, g, w_r, t=t, gabazine_mask=gabazine_mask)
    return x, w

def make_w_transform_homeostatic(rate, setpoint):
    def w_transform_homeostatic(w, x):
        x_total_activity = np.sum(x, axis=0)
        w[:-1, ...] += (w[:-1, ...] > 0).astype(float) * np.repeat(rate * (setpoint - x_total_activity[1:, :, np.newaxis]), w.shape[2], axis=2)
        return w
    return w_transform_homeostatic

def make_w_transform_seq(rate, setpoint):
    def w_transform_seq(w, x):
        # x = (t, length, width)
        # w = (length, width, width)
        for i in range(w.shape[0] - 1):
            w_update = rate * (np.einsum('ij,ik->jk', x[:-1, i, :], x[1:, i+1, :]) - np.einsum('ij,ik->jk', x[:-1, i+1, :], x[1:, i, :]))
            w[i, ...] += (w[i, ...] > 0).astype(float) * w_update
        
            setpoint_diffs = setpoint - w[i, ...].sum(axis=0)
            over_setpoint = np.nonzero(setpoint_diffs < 0)[0]
            w[i, :, over_setpoint] += rate * np.repeat(setpoint_diffs[over_setpoint, np.newaxis], w.shape[1], axis=1)
            w[i, ...] = np.maximum(w[i, ...], 0)
        return w
    return w_transform_seq

# def make_w_transform_seq_w(rate, setpoint):
#     def w_transform_seq_w(w, x):
        
#         # x = (t, 50, 3)
#         # w = (50, 3, 3)
        
#         for i in range(len(w) - 1):
            
#             np.einsum('ij,ik->jk', x[:-1, i, :], x[1:, i+1, :]) + np.einsum('ij,ik->jk', x[:-1, i, :], x[1:, i+1, :])
            
#             w_update = rate * w[i] * np.sum(x[:-1, i] * x[1:, i+1] - x[:-1, i+1] * x[1:, i])
#             w[i] += w_update
#         w = np.minimum(w, setpoint)
#         return w
#     return w_transform_seq_w

def make_w_r_transform(rate, setpoint):
    def w_r_transform(w_r, x):
        w_r += rate * x.sum(axis=0)
        w_r = np.minimum(w_r, setpoint)
        return w_r
    return w_r_transform

def make_dropout(p):
    def dropout(w, w_r):
        dropout_idxs = np.nonzero(np.random.rand((w.shape[1] * w.shape[2])) < p)[0]
        
        dropped_w = copy(w).reshape((w.shape[0] * w.shape[1], w.shape[2]))
        dropped_w[dropout_idxs, :] = 0
        dropped_w_r = copy(w_r).reshape((w.shape[0] * w.shape[1],))
        dropped_w_r[dropout_idxs] = 0
        
        return dropped_w.reshape(w.shape), dropped_w_r.reshape(w_r.shape)
    return dropout      

def run_n_activations(w_0, b, g, w_r_0, n, w_transform=None, w_r_transform=None, dropout_iter=1000, dropout_func=None, gabazine_mask=None):
    all_weights = []
    w = copy(w_0)
    w_r = copy(w_r_0)
    sequential_activity = []
    all_activity = []
    sequential_activity_with_blowups = []
    for i in range(n):
        if i == dropout_iter:
            w, w_r = dropout_func(w, w_r)
        x, w = run_activation(w, b, g, w_r, gabazine_mask=gabazine_mask)
        
        x_seq = np.zeros((x.shape[0]))
        s = 0
        while s < len(x_seq) and x[s, s, :].sum() > 0 and (x[s, :s, :].sum(axis=1) == 0).all() and (x[s, s+1:, :].sum(axis=1) == 0).all():
            x_seq[s] = x[s, s, :].sum()
            s += 1
        sequential_activity.append(x_seq)
        all_activity.append(copy(x))
        sequential_activity_with_blowups.append(sequential_activity[-1])
        
        if i > dropout_iter and w_transform is not None:
            w = w_transform(w, x)    
            
        if i > dropout_iter and w_r_transform is not None:
            w_r = w_r_transform(w_r, x) 
            
        all_weights.append(copy(w))
    return np.array(sequential_activity), np.array(all_activity), np.array(sequential_activity_with_blowups), np.array(all_weights)
    
def extract_lengths(X):
    l = np.zeros(X.shape[0]).astype(int)
    x_prod = np.ones(X.shape[0]).astype(int)
    for i in range(X.shape[1]):
        x_for_idx = (X[:, i] > 0).astype(int)
        l += (x_for_idx * x_prod)
        x_prod *= x_for_idx
    l = np.where(X[:, 0] < 0, 0, l)
    return l

def extract_lengths_bulk(X_all):
    lengths_all = []
    for i_X, X in enumerate(X_all):
        lengths_all.append(extract_lengths(X))
    return lengths_all

def extract_first_hitting_times(X_all, benchmark_lens, start=10):
    all_times = []
    for i_X, X in enumerate(X_all):
        times = np.nan * np.ones(len(benchmark_lens))
        counter = 0
        ls = extract_lengths(X)
        for j, l in enumerate(ls[start:]):
            if counter < len(times) and l >= benchmark_lens[counter]:
                while counter < len(times) and l >= benchmark_lens[counter]:
                    times[counter] = j
                    counter += 1
        all_times.append(times)
    return np.array(all_times)

def extract_jumps(X_all):
    all_hitting_times = extract_first_hitting_times(X_all, np.arange(1, 51))
    all_jump_size_counts = []
    
    for hitting_times in all_hitting_times:
        
        last_hitting_time = None
        jump_sizes_count = np.zeros((50,))
        jump_size = 0
        for i, hitting_time in enumerate(shave_front_zeros_except_last(hitting_times)):
            if last_hitting_time is None:
                last_hitting_time = hitting_time
            elif last_hitting_time == hitting_time:
                jump_size += 1
            else:
                jump_sizes_count[jump_size] += 1
                jump_size = 1
            last_hitting_time = hitting_time
        jump_sizes_count[jump_size] += 1
        all_jump_size_counts.append(jump_sizes_count)
    return np.array(all_jump_size_counts)

def extract_len_diffs(X_all, bounds=None):
    all_lens = extract_lengths_bulk(X_all)
    
    if bounds is None:
        bounds = (np.array(all_lens).astype(int).min(), np.array(all_lens).astype(int).max())
    
    all_len_diffs = []
    len_diffs_count = np.zeros((len(all_lens), bounds[1] - bounds[0] + 1)).astype(int)
    
    for i_l, lens in enumerate(all_lens):
        len_diffs = lens[1:] - lens[:-1]
        for l in len_diffs.astype(int):
            len_diffs_count[i_l, l - bounds[0]] += 1

    return len_diffs_count, np.arange(bounds[0], bounds[1] + 1)

def determine_recovered(X_all, n_activations=50, threshold=0.9, n_cells=None):
    recovered_vec = np.zeros((len(X_all),))
    for i_X, X in enumerate(X_all):
        ls = extract_lengths(X)
        recovered = (np.count_nonzero(ls[-n_activations:] == n_cells[i_X]) / n_activations) > threshold
        recovered_vec[i_X] = recovered
    return recovered_vec 

def shave_front_zeros_except_last(arr):
    for i, x in enumerate(arr):
        if x != 0:
            if i == 0:
                return arr
            else:
                return arr[i-1:]
    return np.array([])

def extract_max_length_var(lengths, window_size=10):
    v_max = 0
    i_max = np.nan
    all_v = []
    for i in range(0, len(lengths) - window_size):
        v = np.var(lengths[i:i + window_size])
        all_v.append(v)
        if v > v_max:
            v_max = v
            i_max = i
    return v_max, i_max, np.array(all_v)

if __name__ == '__main__':
	run_name = 'param_sweep_13'

	network_size = 50
	width = 10
	w_0 = 1 * np.ones((network_size, width, width))
	dropout_percentages = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
	learning_rates = [0.05, 0.1, 0.2]
	inh_learning_rate = 0.01
	n_networks = 50
	g = 3
	w_r_0 = 0.5
	b = 1.5

	total_points = len(learning_rates) * len(dropout_percentages) * 2
	print('total points:', total_points)

	df = None
	point_count = 0

	for rate in learning_rates:
		for dropout in dropout_percentages: 
			X_homeo = []
			for i in range(n_networks):
				w_r = w_r_0 * np.ones((network_size, width))
				sa, nsa, sawbu, ws = run_n_activations(w_0, b, g, w_r, 1000, make_w_transform_homeostatic(rate, 1), make_w_r_transform(inh_learning_rate, w_r_0), dropout_iter=10, dropout_func=make_dropout(dropout))
				X_homeo.append(sawbu)
			X_homeo = np.array(X_homeo)

			data = {
				'rule': ['homeostatic'],
				'rate': [rate],
				'w_r': [w_r_0],
				'b' : [b],
				'dropout': [dropout],
				'activations': [list(X_homeo.flatten().astype(int))],
				'activations_shape': [X_homeo.shape],
			}

			if df is None:
				df = pd.DataFrame(data)
				df.to_csv(f'data/{run_name}.csv', index=False)
			else:
				df = pd.DataFrame(data)
				df.to_csv(f'data/{run_name}.csv', index=False, mode='a', header=False)

			print(point_count)
			point_count += 1


			X_stdp = []
			for i in range(n_networks):
				w_r = w_r_0 * np.ones((network_size, width))
				sa, nsa, sawbu, ws = run_n_activations(w_0, b, g, w_r, 1000, make_w_transform_seq(rate, w_0[0, 0, 0] * width), make_w_r_transform(inh_learning_rate, w_r_0), dropout_iter=10, dropout_func=make_dropout(dropout))
				X_stdp.append(sawbu)
			X_stdp = np.array(X_stdp)

			data = {
				'rule': ['stdp'],
				'rate': [rate],
				'w_r': [w_r_0],
				'b' : [b],
				'dropout': [dropout],
				'activations': [list(X_stdp.flatten().astype(int))],
				'activations_shape': [X_stdp.shape],
			}

			if df is None:
				df = pd.DataFrame(data)
				df.to_csv(f'data/{run_name}.csv', index=False)
			else:
				df = pd.DataFrame(data)
				df.to_csv(f'data/{run_name}.csv', index=False, mode='a', header=False)

			print(point_count)
			point_count += 1
