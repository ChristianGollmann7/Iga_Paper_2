import random

import yaml
import time
from pathlib import Path
import torch
import splinepy
import numpy as np
from numbers import Real
import matplotlib.pyplot as plt
import copy

from neural_network import Mlp, Pinn
from preparation import prepare_geometry, prepare_input_values, generate_neumann_normals_2D, unpack_coefficients_xy, specify_dirichlet_boundaries_with_single_value
from helper_functions import outline_Bspline, fill_up_basis_tensors, fill_up_gradient_tensors, load_Bspline
from random_geometry import random_geometry
from validation import evaluate_validation_samples, retrieve_validation_samples

# Define the Boundary Condition type
BC = dict[str, dict[int, list[Real]]]

torch.set_default_dtype(torch.float64)

# Set a seed for reproduceability (works only for cpu option, not gpu) - does not work yet
# TODO: implement proper seeding
torch.manual_seed(13)
random.seed(13)

def train_vanilla_network(G: splinepy.BSpline,
                          datafolder: str | Path,
                          dirichlet_boundaries_template: BC,
                          neumann_boundaries_template: BC,
                          architecture: list[int],
                          number_training_samples: int,
                          max_iterations: int=1000,
                          save: str=None,
                          show_plots: bool=False,
                          validation_set: str=None) -> tuple[Pinn, list[Real], tuple[list[int], list[Real]], Real, Real]:
    """
    Trains a generic neural network that can be used for a variety of input geometries and boundary conditions.
    Training samples and boundary conditions and poisson value are generated randomly.

    :param G: any geometry with the same parametrization as the target ones, only used to set up parameters, not used in training
    :param E: Youngs modulus, if only displacement dirichlet conditions are specified, it is not relevant. Set to 1 not to introduce numeric problems
    :param poisson_value: Poisson parameter
    :param dirichlet_boundaries_template: template of Dirichlet boundary conditions, not the actual values used in training
    :param neumann_boundaries_template: same as Dirichlet BC, empty for now
    :param architecture: network architecture
    :param number_training_samples: how many random instances are created to train the network
    :param validation_set: directory to use for validation during training, omitted if set to None

    See the main section at the end for an example usage.

    Returns:
        1. The ready to use result Pinn
        2. A list of losses achieved during training
        3. A list of tuples containing (iteration number, validation error)
        3. The preparation time
        4. The training time
    """

    # Theoretically GPU is supported, but the code is not optimized for that and probably there is not much gain here
    device = "cpu"      # switch to "cuda" when using GPU

    if show_plots:
        # Visualize the geometry for convenience, can be commented out
        G.show()
        # Show outline to make sure side names are specified correctly, can be commented out
        fig, ax = plt.subplots()
        outline_Bspline(G, ax)
        plt.show()

    # Actual training begins now

    # Start the timer to measure preparation time
    start = time.perf_counter()

    # Get basic parameters and indices that depend on the parameter space
    # using gauss quadrature or not has no effect here, however we need to choose something. same holds for refinements
    training_parameters = prepare_geometry(G, dirichlet_boundaries_template, neumann_boundaries_template, dim=2, refinements=1, use_gauss=True)
    number_basis_functions     =    training_parameters["number_basis_functions"]
    dirichlet_values           =    training_parameters["dirichlet_values"]
    indices_global_torch       =    training_parameters["indices_global_torch"]
    indices_dirichlet_torch    =    training_parameters["indices_dirichlet_torch"]

    input_complete                   = []
    dirichlet_values_complete        = []
    solution_complete                = []

    """
    Prepare all input samples
    """
    # retrieve the already done simulations
    samples = retrieve_validation_samples(datafolder, n=number_training_samples)

    for i in range(number_training_samples):
        if i % 10 == 0:
            print(f"preparing sample {i}")

        elem = samples[i]
        G_t = load_Bspline(datafolder + '/' + elem[0])
        delta_x, delta_y, poisson = elem[2]
        dirichlet_boundaries = specify_dirichlet_boundaries_with_single_value(dirichlet_boundaries_template, ['b'], [delta_x], [delta_y])
        input_values, dirichlet_values = prepare_input_values(G_t, dirichlet_boundaries, poisson)
        input_complete.append(input_values)     # input numbers that actually go into the neural net
        dirichlet_values_complete.append(dirichlet_values)      # those values later get set by the strong imposition of Dirichlet conditions
        # Load the gismo solution
        u_t = load_Bspline(datafolder + '/' + elem[1])
        sol_gismo = unpack_coefficients_xy(u_t.control_points)
        solution_complete.append(torch.tensor(sol_gismo, dtype=torch.float64))      # known solution of the problem


    torch_input_complete                =   torch.stack(input_complete, dim=0)
    torch_dirichlet_values_complete     =   torch.stack(dirichlet_values_complete, dim=0)
    torch_solution_complete             =   torch.stack(solution_complete, dim=0)


    """
    Create the neural network
    """
    n_input = len(input_complete[0])
    n_output = number_basis_functions*2 - len(dirichlet_values)
    model = Mlp(n_input, n_output, architecture)

    # Explicitly set the datatype for the tensors
    model                           = model.to(dtype=torch.float64)
    torch_input_complete            = torch_input_complete.to(dtype=torch.float64)
    torch_dirichlet_values_complete = torch_dirichlet_values_complete.to(dtype=torch.float64)
    torch_solution_complete         = torch_solution_complete.to(dtype=torch.float64)

    # Move everything to the graphics card if there is one
    model                           = model.to(device)
    torch_input_complete            = torch_input_complete.to(device)
    torch_dirichlet_values_complete = torch_dirichlet_values_complete.to(device)
    torch_solution_complete         = torch_solution_complete.to(device)

    # Preparation done, stop the timer
    end = time.perf_counter()
    prep_time = end - start

    """
    Training starts here
    """
    # Start the timer for training
    start = time.perf_counter()

    # Define the loss function
    def closure_energy():
        output = model(torch_input_complete)
        number_samples = output.size(0)

        # separate x and y displacements
        F = torch.zeros(number_samples, number_basis_functions*2, device=device)
        F[:, indices_global_torch] = output
        F[:, indices_dirichlet_torch] = torch_dirichlet_values_complete

        # Loss
        diff = F - torch_solution_complete
        LOSS = diff.pow(2).mean()

        return LOSS

    """
    Optimizer
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        tolerance_grad=1e-8,
        tolerance_change=1e-8,
        history_size=20,
        line_search_fn="strong_wolfe"
    )

    # Define how a training step works
    def training_step():
        optimizer.zero_grad()
        loss = closure_energy()
        loss.backward()
        return loss

    """
    Training loop
    """
    # If there is no loss improvement for 3 iterations, training finishes
    best_loss = np.inf
    counter = 0
    # Training also finishes if validation is activated and there is no better validation loss within 10 consecutive validation tries
    best_model = copy.deepcopy(model)
    best_val_value = np.inf
    val_counter = 0

    loss_list = []
    val_iter_list = []
    val_error_list = []
    for i in range(max_iterations):
        loss = optimizer.step(training_step)
        loss_list.append(loss.item())
        if torch.isnan(loss):
            return False
        print(f"Step {i + 1:2d}, loss = {loss.item()}")
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        if counter == 3:
            break

        if i % 3 == 0 and validation_set is not None:
            val_error = float(evaluate_validation_samples(Pinn(model), dirichlet_boundaries_template, neumann_boundaries_template, "validation_set"))
            val_error_list.append(val_error)
            val_iter_list.append(i)
            print(f"validation error: {val_error}")

            if val_error < best_val_value:
                best_val_value = val_error
                best_model = copy.deepcopy(model)
                val_counter = 0
            else:
                val_counter += 1
            if val_counter == 100:
                break


    end = time.perf_counter()
    runtime = end - start

    """
    Model Saving
    """
    if save is not None:
        best_model.cpu()
        torch.save(best_model, save+".pth")

    result_pinn = Pinn(best_model)

    if show_plots:
        # Visualize the result
        u = result_pinn.evaluate(G, dirichlet_boundaries_template, neumann_boundaries_template, 0.3)
        S = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u.control_points)
        S.show()

    print(f"Training finished in {end - start} seconds\n\n")
    return result_pinn, loss_list, (val_iter_list, val_error_list), prep_time, runtime



if __name__ == "__main__":
    dirichlet_boundaries = {}
    dirichlet_boundaries["a"] = {0: [0,0], 7: [0,0], 14: [0,0], 21: [0,0], 28: [0,0], 35: [0,0], 42: [0,0]}
    dirichlet_boundaries["b"] = {6: [1,2], 13: [1,2], 20: [1,2], 27: [1,2], 34: [1,2], 41: [1,2], 48: [1,2]}

    neumann_boundaries = {}
    neumann_boundaries["c"] = {}
    neumann_boundaries["d"] = {}

    # Save these example boundary conditions as template so they can be used later on for other training sessions
    boundary_conditions = {"dirichlet": dirichlet_boundaries, "neumann": neumann_boundaries}
    out_path = Path("boundary_conditions.yaml")

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(boundary_conditions, f)

    # Load a test geometry
    G = load_Bspline("input_geometries/simple_5.xml")

    # Define a network architecture
    architecture = [1, 1]

    my_pinn, loss, validation_error, prep_time, runtime = train_vanilla_network(G,
                                                                                "validation_set",
                                                                                dirichlet_boundaries,
                                                                                neumann_boundaries,
                                                                                architecture,
                                                                                number_training_samples=100,
                                                                                max_iterations=1000,
                                                                                save="vanilla_test_network",
                                                                                show_plots=True,
                                                                                validation_set="validation_set")

    print(f"Training finished => final loss: {loss[-1]:.4f}, prep_time: {prep_time:.4f}, runtime: {runtime:.4f}")
