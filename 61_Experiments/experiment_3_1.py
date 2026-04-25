import yaml

from multi_instance_network import train_multi_instance_network
from helper_functions import load_Bspline
from postprocessing import calculate_cartesian_difference, get_difference_visu_data

"""
Experiment 3
Train several multi_instance_networks. Training data is recorded and saved to file.
"""

geometry = "input_geometries/simple_5.xml"
E = 1
poisson_range = (0.15, 0.45)
refinements = 8
architecture = [1, 1]
number_training_samples = [1, 10, 100, 1000]
validation_set = "validation_set_simple"
use_generated_samples = "vanilla_training_set_simple"

# Get the base geometry, just for setting things up, not used for training
G = load_Bspline("input_geometries/simple_5.xml")

# Set the boundary conditions
dirichlet_boundaries = {}
dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
neumann_boundaries = {}
neumann_boundaries["c"] = {}
neumann_boundaries["d"] = {}

# Container to hold training data
loss_list = []
validation_error_list = []
prep_list = []
run_list = []

# Start the training processes
for i in number_training_samples:
    saving_name = f"results_3/physics/network_{architecture}_r_{refinements}_{i}"
    my_pinn, loss, validation_error, prep_time, runtime = train_multi_instance_network(G,
                                                                                       E,
                                                                                       poisson_range,
                                                                                       dirichlet_boundaries,
                                                                                       neumann_boundaries,
                                                                                       architecture,
                                                                                       number_training_samples=i,
                                                                                       refinements=refinements,
                                                                                       max_iterations=5000,
                                                                                       save=saving_name,
                                                                                       show_plots=False,
                                                                                       validation_set=validation_set,
                                                                                       use_generated_samples=use_generated_samples)

    loss_list.append(loss)
    validation_error_list.append(validation_error)
    prep_list.append(prep_time)
    run_list.append(runtime)

# Write training metrics to dict and then save them to file
metrics = {
    "loss": loss_list,
    "validation_error": validation_error_list,
    "prep_time": prep_list,
    "runtime": run_list,
}

with open(f"results_3/physics/metrics_r_{refinements}.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(metrics, f)


