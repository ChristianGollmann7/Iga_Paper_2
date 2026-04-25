import copy

from IPython.core.pylabtools import figsize
from torchsummary import summary
from vedo import close
from gustaf import Vertices
import torch
import yaml
import random
import time
import matplotlib.pyplot as plt
import os
import subprocess
from pathlib import Path, PurePosixPath
import splinepy
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

import sys
from python_utils.helper_functions import *
from python_utils.geometry_preparation import *

#torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float64)


def calculate_absolute_difference(S_real, S_net, ref):
    S = splinepy.BSpline(degrees=S_real.degrees, knot_vectors=S_real.knot_vectors, control_points=S_real.control_points)
    for _ in range(ref):
        S.uniform_refine()
    coll_pts = S.greville_abscissae()
    val_real = S_real.evaluate(coll_pts)
    val_net = S_net.evaluate(coll_pts)
    diff = val_real - val_net
    return np.max(np.linalg.norm(diff, axis=1))


def displacement_plot(G, S_gismo, S_net, ax, color_1='black', color_2='cyan', color_3="magenta"):
    handles, labels = ax.get_legend_handles_labels()
    plot_Bspline(G, ax, color=color_1, n=64, linestyle='--')
    plot_Bspline(S_gismo, ax, color=color_2, n=64, linestyle='-')
    plot_Bspline(S_net, ax, color=color_3, n=64, linestyle='--')


def update_dirichlet(dirichlet_boundary, x, y):
    for side in dirichlet_boundary.keys():
        if side == 'b':
            for key in dirichlet_boundary[side].keys():
                dirichlet_boundary[side][key][0] = x
                dirichlet_boundary[side][key][1] = y
    return dirichlet_boundary


def check_validation_samples(delta_x, delta_y, poisson, nodes_factor, hidden_layers, number_samples):
    # Load the model
    model = torch.load(f"operator_network_{nodes_factor}_{hidden_layers}_{number_samples}.pth", weights_only=False)
    mypinn = pinn(model)

    # Load the boundary conditions as template
    with open("boundary_conditions.yaml", "r", encoding="utf-8") as f:
        boundary_conditions = yaml.safe_load(f)
    dirichlet = boundary_conditions["dirichlet"]
    neumann = boundary_conditions["neumann"]

    error_complete = 0
    fig, ax = plt.subplots(1,5, sharex=True, sharey=True, constrained_layout=False, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(20, 4))
    for i in range(5):
        # Load the base geometry
        base = f"input_geometries/simple_{i+1}.xml"
        G = load_BSpline(base)

        # Calculate the network solution
        dirichlet = update_dirichlet(dirichlet, delta_x[i], delta_y[i])
        u_net = mypinn.evaluate(G, dirichlet, neumann, poisson[i])
        S_net = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors,
                                 control_points=G.control_points + u_net.control_points)

        # Load the gismo solution
        u_gismo = load_BSpline(base[0:-4]+"_gismo.xml")
        S_gismo = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors,
                                   control_points=G.control_points + u_gismo.control_points)

        # Calculate absolute mean difference
        error = calculate_absolute_difference(S_gismo, S_net, 4)
        error_complete += error

        # Make the plot
        displacement_plot(G, S_gismo, S_net, ax[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_aspect('equal')
        ax[i].set_xlim(-3.2,  3.2)
        ax[i].set_ylim(-1.2,  4.2)
        ax[i].set_title("", fontsize=14)
        ax[i].set_facecolor((0.2, 0.2, 0.2))
        ax[i].set_title(rf'Sample {i+1}' + '\n' + rf'$\Delta_x = {delta_x[i]}, \Delta_y = {delta_y[i]}, \mu = {poisson[i]}$' + '\n' + rf'$\mathrm{{error_{{max}}}} = {error:.3e}$', fontsize=12)


    handles = [
        Line2D([0], [0], color='black', linestyle='--', label='undeformed'),
        Line2D([0], [0], color='cyan', linestyle='-', label='gismo reference'),
        Line2D([0], [0], color='magenta', linestyle='--', label='neural network')]
    fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        bbox_transform=ax[2].transAxes)
    fig.suptitle(f"Validation Samples, Network parameters: nodes factor {nodes_factor}, hidden layers {hidden_layers}, training samples {number_samples}", y=0.99, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(
        f"displacement_plot_{nodes_factor}_{hidden_layers}_{number_samples}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05
    )
    #plt.show()
    return float(error_complete / 5)


if __name__ == "__main__":
    # User inputs
    delta_x = [0, -2, -1.5, 0, 1]
    delta_y = [3, 1, 3, -0.3, 2]
    poisson = [0.4, 0.18, 0.25, 0.35, 0.3]

    error_complete = check_validation_samples(delta_x, delta_y, poisson, 1, 2, 1000)






