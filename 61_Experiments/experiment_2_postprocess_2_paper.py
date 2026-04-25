import matplotlib.pyplot as plt
import splinepy
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yaml
import numpy as np
import torch
from sympy.printing.pretty.pretty_symbology import line_width
from torch.distributed.tensor.parallel import loss_parallel
from scipy.optimize import curve_fit

from helper_functions import plot_Bspline
from preparation import specify_dirichlet_boundaries_with_single_value
from postprocessing import calculate_cartesian_difference
from neural_network import Pinn
from helper_functions import load_Bspline
from validation import evaluate_validation_samples

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

colors = ['green', 'red', 'blue', 'orange', 'magenta', 'cyan']

"""
Results 2
This file postprocesses the results of the multi instance analysis with presampled data. 
Creates various graphics and saves them to "results_2" folder.
"""
# Enter here the same sample numbers an architecture as in the multi_instance_analysis
number_training_samples = [1, 10, 100, 1000]
architecture = [1, 1]
base_model_results = f"results_1_paper/metrics_{architecture}_r_8.yaml"
compare_model_results = f"results_2_paper/metrics_{architecture}.yaml"

# Load the results produced by multi_instance_analysis.py
with open(base_model_results) as f:
    data_base = yaml.safe_load(f)

loss_base                =   data_base["loss"]
validation_error_base    =   data_base["validation_error"]
prep_time_base           =   data_base["prep_time"]
run_time_base            =   data_base["runtime"]

with open(compare_model_results) as f:
    data_compare = yaml.safe_load(f)

loss_compare                =   data_compare["loss"]
validation_error_compare    =   data_compare["validation_error"]
prep_time_compare           =   data_compare["prep_time"]
run_time_compare            =   data_compare["runtime"]

# Get the min validation error in a list
error_base = [np.min(errors) for _, errors in validation_error_base]
error_compare = [np.min(errors) for _, errors in validation_error_compare]

# Get the number of iterations

# Get the final evaluation error for each model
validation_set = "validation_set_hard"
# Template boundary conditions
dirichlet_boundaries = {}
dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
neumann_boundaries = {}
neumann_boundaries["c"] = {}
neumann_boundaries["d"] = {}

final_error_base_max = []
final_error_base_mean = []
final_error_compare_max = []
final_error_compare_mean = []

# Load the networks
networks = [f"results_1/network_{architecture}_r_8_{i}.pth" for i in number_training_samples]
for network in networks:
    model = torch.load(network, weights_only=False)
    result_pinn = Pinn(model)
    val_error_max, val_error_mean = evaluate_validation_samples(result_pinn, dirichlet_boundaries, neumann_boundaries, validation_set, return_value='both')
    final_error_base_max.append(val_error_max)
    final_error_base_mean.append(val_error_mean)

networks = [f"results_2/network_{architecture}_{i}.pth" for i in number_training_samples]
for network in networks:
    model = torch.load(network, weights_only=False)
    result_pinn = Pinn(model)
    val_error_max, val_error_mean = evaluate_validation_samples(result_pinn, dirichlet_boundaries, neumann_boundaries, validation_set, return_value='both')
    final_error_compare_max.append(val_error_max)
    final_error_compare_mean.append(val_error_mean)

vanilla_preparation_times = [1080/100*i for i in number_training_samples]



# Time and error plot
pos = np.asarray(number_training_samples, dtype=float)
labels = [str(i) for i in number_training_samples]
dx = 0.08
f_prep, f_run, f_err_max, f_err_mean = 10**(-dx*1.5), 10**(-dx*0.5), 10**(dx*0.5), 10**(dx*1.4)
bar_width = 1 * pos * (10**dx-1)

fig, ax = plt.subplots(figsize=(12, 6.5))

bars_1 = ax.bar(pos*f_prep, prep_time_base, bar_width, label='preparation time', color='tab:blue')
bars_5 = ax.bar(pos*f_prep, prep_time_compare, bar_width, label='_nolegend_', facecolor='none', edgecolor='black', linewidth=2)
bars_2 = ax.bar(pos*f_run, run_time_base, bar_width, label='training time', color='lightblue')
bars_6 = ax.bar(pos*f_run, run_time_compare, bar_width, label='_nolegend_', facecolor='none', edgecolor='black', linewidth=2)

ax.set_xticks(pos)
ax.set_xscale('log')
ax.set_xticklabels(labels, fontsize=14)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_ylabel('time [s]', fontsize=14, color='tab:blue')
ax.set_xlabel('number of training samples', fontsize=14)
ax.tick_params(axis='y', colors='tab:blue')
ax.set_yscale('log')


ax2 = ax.twinx()
bars_3 = ax2.bar(pos*f_err_max, final_error_base_max, bar_width, label='max evaluation error', color='tab:orange')
bars_7 = ax2.bar(pos*f_err_max, final_error_compare_max, bar_width, label='_nolegend_', facecolor='none', edgecolor='black', linewidth=2)

bars_4 = ax2.bar(pos*f_err_mean, final_error_base_mean, bar_width, label='mean evaluation error', color='brown')
bars_8 = ax2.bar(pos*f_err_mean, final_error_compare_mean, bar_width, label='_nolegend_', facecolor='none', edgecolor='black', linewidth=2)

ax2.set_ylabel('evaluation error [length units]', fontsize=14, color='orange')
ax2.tick_params(axis='y', colors='orange')
ax2.set_yscale('log')

handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
square_handle = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='supervised')
#leg = ax.legend(handles_1+handles_2+[square_handle], labels_1+labels_2+['supervised'], fontsize=14, loc="upper left", bbox_to_anchor=(0.45, 1.0), borderaxespad=0, frameon=True, framealpha=1.0)
ax.legend(
    handles_1 + handles_2 + [square_handle],
    labels_1 + labels_2 + ['supervised'],
    fontsize=14,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(labels_1 + labels_2 + ['supervised']),
    borderaxespad=0,
    frameon=False,
    framealpha=1.0)
ax.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)
ax2.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)

ax.set_title("Comparison between physics-based and supervised network", fontsize=16)

ax.scatter(pos*f_prep*0.95, vanilla_preparation_times, marker='*', s=250, label='_nolegend')


plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"results_2_paper/error_plot_combined_{architecture}.pdf",
            format="pdf",
            bbox_inches='tight',
            pad_inches=0.05)
