import os
from collections import OrderedDict
import functools


base_path = os.curdir
scripts = [
	'submit_klone.slurm',
]

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		if k == 'TITLE': 
			continue
		s = s.replace(k, v)

	if 'TITLE' in repl_dict:
		s = s.replace('TITLE', repl_dict['TITLE'])

	return s

def format_title(params):
	title = ''
	for k, v in params.items():
		title += ('_' + k + '_' + v)
	return title

def iter_range(r, n):
	if n == 1:
		yield (0, r[0])
	else:
		for i in range(n):
			yield (i, i * (r[1] - r[0]) / (n - 1) + r[0])

def map_to_list(func, l):
    '''
    Maps the list 'l' through the function 'func'
    Parameters
    ----------
    func : function
        Takes a single argument of type of 'l'
    l : list
    '''
    return list(map(func, l))

def reduce_mult(l):
    return functools.reduce(lambda e1, e2: e1 * e2, l, 1)

# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
    domain = map_to_list(lambda a: len(a), arrs)
    coordinate_lists = []
    for i, dim in enumerate(domain):
        coords = []
        mult = 1
        if i != len(domain) - 1:
            mult = reduce_mult(domain[i+1:])
        for e in arrs[i]:
            coords += (mult * [e])
        repeat_factor = reduce_mult(domain[0:i])
        if repeat_factor > 0:
            coords *= repeat_factor
        coordinate_lists.append(coords)
    return coordinate_lists

def pad_zeros(to_pad, length):
	padded = str(to_pad)
	while len(padded) < length:
		padded = '0' + padded
	return padded

batch_size = 1

params = OrderedDict()
params['SEED'] = [str(i) for i in range(7000, 7010)]
params['W_E_E'] = [str(0.4e-3)]
params['W_E_I'] = [str(3.5e-5)] # 3.5e-5 for silent
params['W_I_E'] = [str(4e-5)]
params['DROP_SEV'] = [str(0.5), str(0.6), str(0.7)]
params['COMP_MECH'] = ['secreted_regulation']
params['CONDITION'] = ['no_repl_no_syn']
params['PER_SILENT'] = [str(0.4)]
params['A5'] = [str(5)]

n_seeds = len(params['SEED'])

for key in params.keys():
	if key == 'SEED' or type(params[key][0]) is str:
		continue
	params[key] = [str(v[1]) for v in iter_range(params[key][0], params[key][1])]

all_values = cartesian(*(params.values()))
n_scripts = len(all_values[0])
n_scripts_exec = 0

for src_name in scripts:
	for n in range(0, n_scripts, batch_size):

		name_parts = src_name.split('.')
		dst_name = name_parts[0] + '_' + pad_zeros(n, 4) + '.' + name_parts[1]

		src = open(src_name, 'rt')
		dst = open(dst_name, 'wt')

		for line in src:
			if line.find('python run.py') >= 0:
				for batch_idx in range(batch_size):
					script_num = n + batch_idx
					if script_num >= n_scripts:
						continue

					augmented_params = {}

					for param_idx, v in enumerate(params.keys()):
						augmented_params[v] = all_values[param_idx][script_num]

					augmented_params['TITLE'] = format_title(augmented_params)
					augmented_params['INDEX'] = str(int(script_num / n_seeds))

					line_replaced = replace_all(line, augmented_params)
					dst.write(line_replaced)
			else:
				dst.write(line)
		src.close()
		dst.close()

		os.system('sbatch ./' + dst_name)
		print(dst_name)
