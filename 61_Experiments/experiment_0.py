import splinepy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from single_instance_network import train_single_instance_network
from helper_functions import load_Bspline, plot_Bspline
from postprocessing import calculate_cartesian_difference, get_difference_visu_data, calculate_cartesian_difference_at_positions

geometry = "input_geometries/simple_5.xml"
E = 1
poisson = 0.3
refinements = 15
architecture = [0]
use_gauss = True

# Get the base geometry
G = load_Bspline("input_geometries/simple_5.xml")

# Set the boundary conditions
dirichlet_boundaries = {}
dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
neumann_boundaries = {}
neumann_boundaries["c"] = {}
neumann_boundaries["d"] = {}

# Load the gismo reference solution
u_gismo = load_Bspline("input_geometries/simple_5_gismo_1_2_0.3.xml")
S_gismo = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u_gismo.control_points)

loss_list = []
prep_list = []
run_list = []
error_list = []
for i in range(refinements+1)[1:]:
    #if i == 1 and use_gauss:
    #    continue
    print("\nrefinement: ", i)
    result_pinn, loss, prep_time, runtime = train_single_instance_network(G,
                                                                      E,
                                                                      poisson,
                                                                      dirichlet_boundaries,
                                                                      neumann_boundaries,
                                                                      architecture,
                                                                      refinements=i,
                                                                      max_iterations=100,
                                                                      use_gauss=use_gauss)
    # Calculate the network solution
    u_net = result_pinn.evaluate(G, dirichlet_boundaries, neumann_boundaries, poisson)
    S_net = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors,
                             control_points=G.control_points + u_net.control_points)
    error, poly_data = get_difference_visu_data(G, S_gismo, S_net)
    loss_list.append(loss[-1])
    prep_list.append(prep_time)
    run_list.append(runtime)
    error_list.append(np.max(error))

    # Create error plot
    fig, ax = plt.subplots()
    plot_Bspline(S_net, ax, color='black', smoothness=64)
    ax.add_collection(poly_data)
    cbar = fig.colorbar(poly_data, ax=ax)
    cbar_min, cbar_max = poly_data.get_clim()
    number_ticks = refinements
    my_ticks = np.linspace(cbar_min, cbar_max, number_ticks)
    cbar.set_ticks(my_ticks)
    cbar.ax.tick_params(labelsize=14, length=4, width=1.2)
    cbar.formatter = mticker.FormatStrFormatter('%.5f')
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
    ax.set_aspect('equal')
    fig.set_size_inches(10, 10, forward=True)
    if not use_gauss:
        ax.set_title(f"absolute error\n{i} refinements", fontsize=24)
        fig.savefig(f"results_0/no_gauss/error_plot_{i}.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
    else:
        ax.set_title(f"absolute error\n{i} Gauss points", fontsize=24)
        fig.savefig(f"results_0/gauss/error_plot_{i}.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)


loss_list   = np.array(loss_list)
prep_list   = np.array(prep_list)
run_list    = np.array(run_list)
error_list  = np.array(error_list)

# Create an overview time plot
x = range(refinements+1)[1:]
if use_gauss:
    x = range(refinements + 1)[1:]

fig, ax = plt.subplots(figsize=(10,8))
ax.fill_between(x, 0, prep_list, color='blue', alpha=0.5, label="preparation time")
ax.fill_between(x, prep_list, prep_list+run_list, color='orange', alpha=0.5, label="training time")
ax.plot(x, prep_list, color='blue')
ax.plot(x, prep_list+run_list, color='orange')
ax.legend()
ax.set_ylabel("time [s]", fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
ax.grid(True)
ax.legend(fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.99])

if not use_gauss:
    ax.set_xlabel('Greville refinements', fontsize=14)
    ax.set_title('Preparation and training time for Greville refinements of integration points', fontsize=16)
    fig.savefig(f"results_0/no_gauss/runtime_plot.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
else:
    ax.set_xlabel('one directional Gauss points per knot span', fontsize=14)
    ax.set_title('Preparation and training time for Gauss integration', fontsize=16)
    fig.savefig(f"results_0/gauss/runtime_plot.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)





# Create an overview error plot
fig, ax = plt.subplots(figsize=(10,8))
l1 = ax.plot(x, loss_list, color='green', linestyle='-', marker='s', label='potential energy / loss')
ax.set_ylabel("energy potential / loss [-]", fontsize=14, color="green")
#ax.set_yscale("log")
ax.tick_params(axis="y", colors="green")



ax2 = ax.twinx()
l2 = ax2.plot(x, error_list, color="red", marker="^", label="max error")
ax2.tick_params(axis="y", colors="red")
ax2.set_ylabel("max error [length unit]", fontsize=14, color="red")
ax2.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
ax2.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.tight_layout(rect=[0, 0, 1, 0.99])

if not use_gauss:
    ax.set_xlabel("Greville refinements", fontsize=14)
    ax.set_title("Total potential energy and maximum error for Greville refinements of integration points", fontsize=16)
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc='best')
    fig.savefig(f"results_0/no_gauss/error_plot.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
else:
    ax.set_xlabel("one directional Gauss points per knot span", fontsize=14)
    ax.set_title("Total potential energy and maximum error for Gauss integration", fontsize=16)
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc='center')
    fig.savefig(f"results_0/gauss/error_plot.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)





