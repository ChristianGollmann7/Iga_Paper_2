import matplotlib.pyplot as plt
import splinepy
from IPython.core.pylabtools import figsize
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import yaml
import numpy as np
import torch
import matplotlib.ticker as mticker

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

from sympy.printing.pretty.pretty_symbology import line_width
from torch.distributed.tensor.parallel import loss_parallel
from scipy.optimize import curve_fit

from helper_functions import plot_Bspline
from preparation import specify_dirichlet_boundaries_with_single_value
from postprocessing import calculate_cartesian_difference, get_difference_visu_data
from neural_network import Pinn
from helper_functions import load_Bspline
from validation import evaluate_validation_samples

colors = ['green', 'red', 'blue', 'orange', 'magenta', 'cyan']

"""
Results 1
This file postprocesses the results of the multi instance analysis with presampled data. 
Creates various graphics and saves them to "results_1_paper..." folder.
"""

results_folder = "results_2_paper/"
validation_set = "validation_set_hard"


# Enter here the same sample numbers an architecture as in the multi_instance_analysis
number_training_samples = [1, 10, 100, 1000]
architecture = [1, 1]

c_gismo = "orange"
c_network = "blue"
fc = (1.0, 1.0, 1.0)

# Load the results produced by multi_instance_analysis.py
with open(f"{results_folder}metrics_{architecture}.yaml") as f:
    data = yaml.safe_load(f)

loss                =   data["loss"]
validation_error    =   data["validation_error"]
prep_time           =   data["prep_time"]
run_time            =   data["runtime"]

# Get the min validation error in a list
error = [np.min(errors) for _, errors in validation_error]

# Get the number of iterations


# Template boundary conditions, values do not matter
dirichlet_boundaries = {}
dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
neumann_boundaries = {}
neumann_boundaries["c"] = {}
neumann_boundaries["d"] = {}

final_error_max = []
final_error_mean = []

# Load the networks
networks = [f"{results_folder}network_{architecture}_{i}.pth" for i in number_training_samples]
for network in networks:
    model = torch.load(network, weights_only=False)
    result_pinn = Pinn(model)
    val_error_max, val_error_mean = evaluate_validation_samples(result_pinn, dirichlet_boundaries, neumann_boundaries, validation_set, return_value='both')
    final_error_max.append(val_error_max)
    final_error_mean.append(val_error_mean)

# Loss plot
fig, ax = plt.subplots(figsize=(12, 6.5))
for i in range(len(number_training_samples)):
    lab1 = f"Loss Value for {number_training_samples[i]} Training Samples"
    if number_training_samples[i] == 1:
        lab1 = f"Loss Value for {number_training_samples[i]} Training Sample"
    ax.plot(np.arange(len(loss[i])) + 1, loss[i], '-', color=colors[i], label=lab1)
    #ax.text(len(loss[i])+0.5, loss[i][-1], f"{number_training_samples[i]}", fontsize=12)
    lab2 = f"Validation Error for {number_training_samples[i]} Training Samples"
    if number_training_samples[i] == 1:
        lab2 = f"Validation Error for {number_training_samples[i]} Training Sample"
    ax.plot(validation_error[i][0], validation_error[i][1], '--', color=colors[i], label=lab2)
    #ax.text(validation_error[i][0][-1] + 0.5, validation_error[i][1][-1], f"{number_training_samples[i]}", fontsize=12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Epochs', fontsize=14)
ax.set_ylabel('Validation Error (dashed) / Loss Function (solid)', fontsize=14)
#ax.set_title("Loss and validation error for various numbers of training samples", fontsize=16)
ax.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)
ax.grid(True)
ax.legend(fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"{results_folder}loss_plot_{architecture}.pdf",
            format="pdf",
            bbox_inches='tight',
            pad_inches=0.05)


# Time and error plot
pos = np.asarray(number_training_samples, dtype=float)
labels = [str(i) for i in number_training_samples]
dx = 0.08
f_prep, f_run, f_err_max, f_err_mean = 10**(-dx*1.5), 10**(-dx*0.5), 10**(dx*0.5), 10**(dx*1.4)
bar_width = 1 * pos * (10**dx-1)

fig, ax = plt.subplots(figsize=(12, 6.5))

bars_1 = ax.bar(pos*f_prep, prep_time, bar_width, label='Preparation Time', color='tab:blue')
bars_2 = ax.bar(pos*f_run, run_time, bar_width, label='Training Time', color='lightblue')
ax.set_xticks(pos)
ax.set_xscale('log')
ax.set_xticklabels(labels, fontsize=14)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_ylabel('Wall-Clock Time (in s)', fontsize=14, color='tab:blue')
ax.set_xlabel('Number of Training Samples', fontsize=14)
ax.tick_params(axis='y', colors='tab:blue')
ax.set_yscale('log')
ax.xaxis.set_minor_locator(mticker.NullLocator())


ax2 = ax.twinx()
bars_3 = ax2.bar(pos*f_err_max, final_error_max, bar_width, label='Maximum Error', color='tab:orange')
bars_4 = ax2.bar(pos*f_err_mean, final_error_mean, bar_width, label='Mean Error', color='brown')
ax2.set_ylabel('Cartesian Error', fontsize=14, color='orange')
ax2.tick_params(axis='y', colors='orange')
ax2.set_yscale('log')

#ax.plot(pos*f_prep, prep_time,    '--', color='tab:blue', marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')
#ax.plot(pos*f_run, run_time,      '--', color='lightblue',  marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')
#ax2.plot(pos*f_err, final_error,  '--', color='orange', marker='x', markeredgecolor='black', linewidth=1.5, label='_nolegend_')

handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    fontsize=14,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(labels_1 + labels_2),
    borderaxespad=0,
    frameon=False,
    framealpha=1.0)
ax.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)
ax2.tick_params(axis='both', which='both', labelsize=14, length=6, width=1.5)

#ax.set_title("Preparation / Training time and evaluation error for different numbers of training samples", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"{results_folder}error_plot_{architecture}.pdf",
            format="pdf",
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
    plot_Bspline(S_net, ax, color=color3, smoothness=64, linestyle='-')


def error_plot(G, S_gismo, S_net, ax, ax_cb):
    error, poly_data = get_difference_visu_data(G, S_gismo, S_net, relative=False)
    plot_Bspline(G, ax, color='black', smoothness=64)
    ax.add_collection(poly_data)
    ax.set_title(f"Cartesian error\n$\mathrm{{error_{{max}}}} = {np.max(error):.3e}$", fontsize=12)
    fig = ax.figure
    cbar = fig.colorbar(poly_data, cax=ax_cb)
    ax_cb.set_xticks([])
    ax_cb.set_xlabel("")
    cbar_min, cbar_max = poly_data.get_clim()
    cbar.ax.tick_params(labelsize=10, length=4, width=1.2)
    number_ticks = 5
    my_ticks = np.linspace(cbar_min, cbar_max, number_ticks)
    cbar.set_ticks(my_ticks)
    cbar.formatter = mticker.FormatStrFormatter('%.5f')
    cbar.update_ticks()


def error_plot_sub(G, S_gismo, S_net, fig, ax):
    error, poly_data = get_difference_visu_data(G, S_gismo, S_net, relative=False)
    plot_Bspline(G, ax, color='black', smoothness=64)
    ax.add_collection(poly_data)
    cbar = fig.colorbar(poly_data, fraction=0.046, pad=0.04)
    cbar_min, cbar_max = poly_data.get_clim()
    cbar.ax.tick_params(labelsize=10, length=4, width=1.2)
    number_ticks = 5
    my_ticks = np.linspace(cbar_min, cbar_max, number_ticks)
    cbar.set_ticks(my_ticks)
    cbar.formatter = mticker.FormatStrFormatter('%.5f')
    cbar.set_label("Absolute Cartesian Error")


def create_visu_plot(delta_x, delta_y, poisson, net_architecture, number_training_samples):
    # Template boundary conditions
    dirichlet_boundaries = {}
    dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
    dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
    neumann_boundaries = {}
    neumann_boundaries["c"] = {}
    neumann_boundaries["d"] = {}

    error_total = 0
    fig, ax = plt.subplots(5,3, constrained_layout=False, gridspec_kw={'width_ratios': [1, 1, 0.08], 'wspace': 0.2, 'hspace': 0.1}, figsize=(8, 16))
    handles = [
        Line2D([0], [0], color='black', linestyle='--', label='undeformed'),
        Line2D([0], [0], color=c_gismo, linestyle='-', label='Gismo'),
        Line2D([0], [0], color=c_network, linestyle='-', label='supervised network'),
    ]
    for i in range(5):
        # Load the model
        model = torch.load(
            f"{results_folder}network_{architecture}_{number_training_samples}.pth", weights_only=False)
        result_pinn = Pinn(model)

        # Prepare the dirichlet boundary conditions
        dirichlet = specify_dirichlet_boundaries_with_single_value(dirichlet_boundaries, ['b'], [delta_x[i]], [delta_y[i]])

        # Load the base geometry
        base = f"visualization_geometries/simple_{i+1}.xml"
        G = load_Bspline(base)

        # Calculate the network solution
        u_net = result_pinn.evaluate(G, dirichlet, neumann_boundaries, poisson[i])
        S_net = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u_net.control_points)

        u_gismo = load_Bspline(base[0:-4]+"_gismo.xml")
        S_gismo = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u_gismo.control_points)

        # Get the max error
        single_error = np.max(calculate_cartesian_difference(u_gismo, u_net))
        error_total += single_error

        # Make the plot
        displacement_plot(G, S_gismo, S_net, ax[i,0])
        #ax[i,0].set_xticks([])
        #ax[i,0].set_yticks([])
        ax[i,0].set_aspect('equal', adjustable='box')
        ax[i,0].set_xlim(-1, 1)
        ax[i,0].set_ylim(np.min(S_net.control_points[:,1])*1.1, np.max(S_net.control_points[:,1])*1.1)
        ax[i,0].set_title(rf'Sample {i+1}' + '\n' + rf'$\Delta x = {delta_x[i]}, \Delta y = {delta_y[i]}, \nu = {poisson[i]}$', fontsize=12)
        ax[i,0].set_facecolor(fc)

        error_plot(G, S_gismo, S_net, ax[i,1], ax[i, 2])
        ax[i,1].set_aspect('equal', adjustable='box')
        ax[i,1].set_xlim(-1, 1)
        ax[i, 1].set_ylim(np.min(S_net.control_points[:, 1]) * 1.1, np.max([np.max(S_net.control_points[:, 1]), np.max(S_gismo.control_points[:, 1]), np.max(G.control_points[:, 1])]) * 1.1)
        ax[i, 1].tick_params(labelleft=False)

        pos_plot = ax[i, 1].get_position()
        pos_cb = ax[i, 2].get_position()
        ax[i, 2].set_position([pos_cb.x0, pos_plot.y0, pos_cb.width, pos_plot.height])


        """ Small subplots """
        sub_fig, sub_ax = plt.subplots(figsize=(5,5))
        displacement_plot(G, S_gismo, S_net, sub_ax)
        sub_ax.set_aspect('equal', adjustable='box')
        sub_ax.set_xlim(-4, 4)
        sub_ax.set_ylim(-1.2, 4.2)
        sub_ax.set_xlabel(f"$x$-Coordinate")
        sub_ax.set_ylabel(f"$y$-Coordinate")
        #sub_ax.set_ylim(np.min(G.control_points[:, 1]) * 1.1, np.max([np.max(S_net.control_points[:, 1]), np.max(S_gismo.control_points[:, 1]), np.max(G.control_points[:, 1])]) * 1.1)
        sub_fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        sub_fig.subplots_adjust(bottom=0.15)
        # plt.tight_layout(rect=[0, 0, 1, 0.99])
        sub_fig.set_size_inches(5, 5)
        sub_fig.savefig(f"{results_folder}sample_{number_training_samples}/displacement_plot_sample_{i}_dx_{delta_x[i]}_dy_{delta_y[i]}.pdf",
                    format="pdf",
                    bbox_inches='tight',
                    pad_inches=0.05)

        sub_fig, sub_ax = plt.subplots(figsize=(5,5))
        error_plot_sub(G, S_gismo, S_net, sub_fig, sub_ax)
        sub_ax.set_aspect('equal', adjustable='box')
        sub_ax.set_xlim(-2.5, 2.5)
        sub_ax.set_ylim(-1.2, 1.2)
        sub_ax.set_xlabel(f"$x$-Coordinate")
        sub_ax.set_ylabel(f"$y$-Coordinate")
        #sub_ax.set_ylim(np.min(S_net.control_points[:, 1]) * 1.1, np.max([np.max(S_net.control_points[:, 1]), np.max(S_gismo.control_points[:, 1]), np.max(G.control_points[:, 1])]) * 1.1)
        sub_fig.set_size_inches(5, 5)
        sub_fig.savefig(f"{results_folder}sample_{number_training_samples}/error_plot_sample_{i}_dx_{delta_x[i]}_dy_{delta_y[i]}.pdf",
                    format="pdf",
                    bbox_inches='tight',
                    pad_inches=0.05)


    fig.subplots_adjust(bottom=0.05, top=0.95)
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f"network architecture: {net_architecture}, number of training samples: {number_training_samples}", y=0.99, fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{results_folder}displacement_plot_{net_architecture}_{number_training_samples}.pdf",
                format="pdf",
                bbox_inches='tight',
                pad_inches=0.05)

    return fig

for n in number_training_samples:
    create_visu_plot(delta_x, delta_y, poisson, architecture, n)

