import matplotlib.pyplot as plt
import splinepy
from matplotlib.ticker import ScalarFormatter
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
from python_utils.helper_functions import load_BSpline
from validation import evaluate_validation_samples

colors = ['green', 'red', 'blue', 'orange', 'magenta', 'cyan']

"""
Results 1
This file postprocesses the results of the multi instance analysis with presampled data. 
Creates various graphics and saves them to "results_1" folder.
"""
# Enter here the same sample numbers an architecture as in the multi_instance_analysis
number_training_samples = [1, 10, 100, 1000]
architecture = [1, 1]

c_gismo = "cyan"
c_network = "yellow"
fc = (0.4, 0.4, 0.4)

# Load the results produced by multi_instance_analysis.py
with open(f"results_1/metrics_{architecture}_r_8.yaml") as f:
    data = yaml.safe_load(f)

loss                =   data["loss"]
validation_error    =   data["validation_error"]
prep_time           =   data["prep_time"]
run_time            =   data["runtime"]

# Get the min validation error in a list
error = [np.min(errors) for _, errors in validation_error]

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

final_error_max = []
final_error_mean = []

# Load the networks
networks = [f"results_1/network_{architecture}_r_8_{i}.pth" for i in number_training_samples]
for network in networks:
    model = torch.load(network, weights_only=False)
    result_pinn = Pinn(model)
    val_error_max, val_error_mean = evaluate_validation_samples(result_pinn, dirichlet_boundaries, neumann_boundaries, validation_set, return_value='both')
    final_error_max.append(val_error_max)
    final_error_mean.append(val_error_mean)

# Loss plot
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(len(number_training_samples)):
    ax.plot(np.arange(len(loss[i])) + 1, loss[i], '-', color=colors[i], label=f"loss samples {number_training_samples[i]}")
    ax.text(len(loss[i])+0.5, loss[i][-1], f"{number_training_samples[i]}", fontsize=12)
    ax.plot(validation_error[i][0], validation_error[i][1], '--', color=colors[i], label=f"validation error samples {number_training_samples[i]}")
    ax.text(validation_error[i][0][-1] + 0.5, validation_error[i][1][-1], f"{number_training_samples[i]}", fontsize=12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('iterations', fontsize=14)
ax.set_ylabel('loss function [solid] / validation error [dashed]', fontsize=14)
ax.set_title("Loss and validation error for various numbers of training samples", fontsize=16)
ax.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)
ax.grid(True)
ax.legend(fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"results_1/loss_plot_{architecture}.svg",
            format="svg",
            bbox_inches='tight',
            pad_inches=0.05)


# Time and error plot
pos = np.asarray(number_training_samples, dtype=float)
labels = [str(i) for i in number_training_samples]
dx = 0.08
f_prep, f_run, f_err_max, f_err_mean = 10**(-dx*1.5), 10**(-dx*0.5), 10**(dx*0.5), 10**(dx*1.4)
bar_width = 1 * pos * (10**dx-1)

fig, ax = plt.subplots(figsize=(12, 6))

bars_1 = ax.bar(pos*f_prep, prep_time, bar_width, label='preparation time', color='tab:blue')
bars_2 = ax.bar(pos*f_run, run_time, bar_width, label='training time', color='lightblue')
ax.set_xticks(pos)
ax.set_xscale('log')
ax.set_xticklabels(labels, fontsize=14)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_ylabel('time [s]', fontsize=14, color='tab:blue')
ax.set_xlabel('number of training samples', fontsize=14)
ax.tick_params(axis='y', colors='tab:blue')
ax.set_yscale('log')


ax2 = ax.twinx()
bars_3 = ax2.bar(pos*f_err_max, final_error_max, bar_width, label='max evaluation error', color='tab:orange')
bars_4 = ax2.bar(pos*f_err_mean, final_error_mean, bar_width, label='mean evaluation error', color='brown')
ax2.set_ylabel('evaluation error [length units]', fontsize=14, color='orange')
ax2.tick_params(axis='y', colors='orange')
ax2.set_yscale('log')

#ax.plot(pos*f_prep, prep_time,    '--', color='tab:blue', marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')
#ax.plot(pos*f_run, run_time,      '--', color='lightblue',  marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')
#ax2.plot(pos*f_err, final_error,  '--', color='orange', marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')

handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(handles_1+handles_2, labels_1+labels_2, fontsize=14, loc="upper left", bbox_to_anchor=(0.45, 0.98), borderaxespad=0, frameon=True, framealpha=1.0)
ax.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)
ax2.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)

ax.set_title("Preparation / Training time and evaluation error for different numbers of training samples", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"results_1/error_plot_{architecture}.svg",
            format="svg",
            bbox_inches='tight',
            pad_inches=0.05)



# Plot to visualize deformations with certain sample sizes

# Those are the boundary conditions the samples in "visualization_geometries" got created with
delta_x = [0, -2, -1.5, 0, 1]
delta_y = [3, 1, 3, -0.3, 2]
poisson = [0.4, 0.18, 0.25, 0.35, 0.3]

# Helper function
def displacement_plot(G, S_gismo, S_net, ax, color1='black', color2=c_gismo, color3=c_network):
    plot_Bspline(G, ax, color=color1, smoothness=64, linestyle='--')
    plot_Bspline(S_gismo, ax, color=color2, smoothness=64, linestyle='-')
    plot_Bspline(S_net, ax, color=color3, smoothness=64, linestyle='--')


def create_visu_plot(delta_x, delta_y, poisson, net_architecture, number_training_samples):
    # Template boundary conditions
    dirichlet_boundaries = {}
    dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
    dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
    neumann_boundaries = {}
    neumann_boundaries["c"] = {}
    neumann_boundaries["d"] = {}

    error_total = 0
    fig, ax = plt.subplots(1,5, sharex=True, sharey=True, constrained_layout=False, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(20, 4))
    for i in range(5):
        # Load the model
        model = torch.load(f"results_1/network_{net_architecture}_r_8_{number_training_samples}.pth", weights_only=False)
        result_pinn = Pinn(model)

        # Prepare the dirichlet boundary conditions
        dirichlet = specify_dirichlet_boundaries_with_single_value(dirichlet_boundaries, ['b'], [delta_x[i]], [delta_y[i]])

        # Load the base geometry
        base = f"visualization_geometries/simple_{i+1}.xml"
        G = load_BSpline(base)

        # Calculate the network solution
        u_net = result_pinn.evaluate(G, dirichlet, neumann_boundaries, poisson[i])
        S_net = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u_net.control_points)

        # Load the gismo solution
        u_gismo = load_BSpline(base[0:-4]+"_gismo.xml")
        S_gismo = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u_gismo.control_points)

        # Get the max error
        single_error = np.max(calculate_cartesian_difference(u_gismo, u_net))
        error_total += single_error

        # Make the plot
        displacement_plot(G, S_gismo, S_net, ax[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_aspect('equal')
        ax[i].set_xlim(-3.2, 3.2)
        ax[i].set_ylim(-1.2, 4.2)
        ax[i].set_title(rf'Sample {i+1}' + '\n' + rf'$\Delta x = {delta_x[i]}, \Delta y = {delta_y[i]}, \nu = {poisson[i]}$' + '\n' + rf'$\mathrm{{error_{{max}}}} = {single_error:.3e}$', fontsize=12)
        ax[i].set_facecolor(fc)

    handles = [
        Line2D([0], [0], color='black', linestyle='--', label='undeformed'),
        Line2D([0], [0], color=c_gismo, linestyle='-', label='gismo reference'),
        Line2D([0], [0], color=c_network, linestyle='--', label='neural network'),
    ]
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0), bbox_transform=ax[2].transAxes)
    fig.suptitle(f"Validation samples, network architecture: {net_architecture}, number of training samples: {number_training_samples}", y=0.99, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"results_1/displacement_plot_{net_architecture}_{number_training_samples}.svg",
                format="svg",
                bbox_inches='tight',
                pad_inches=0.05)

    return fig

for n in number_training_samples:
    create_visu_plot(delta_x, delta_y, poisson, architecture, n)

