import yaml
import time
from pathlib import Path
import torch
import splinepy
import numpy as np
from numbers import Real
import matplotlib.pyplot as plt

from neural_network import Mlp, Pinn
from preparation import prepare_geometry, prepare_input_values, generate_neumann_normals_2D
from helper_functions import outline_Bspline, fill_up_basis_tensors, fill_up_gradient_tensors, load_Bspline



# Define the Boundary Condition type
BC = dict[str, dict[int, list[Real]]]

torch.set_default_dtype(torch.float64)

def train_single_instance_network(G: splinepy.BSpline,
                                  E: Real,
                                  poisson_value: Real,
                                  dirichlet_boundaries_template: BC,
                                  neumann_boundaries_template: BC,
                                  architecture: list[int],
                                  refinements: int=0,
                                  max_iterations: int=1000,
                                  save: str=None,
                                  show_plots: bool=False,
                                  use_gauss: bool=True) -> tuple[Pinn, list[Real], Real, Real]:
    """
    Trains a neural network for a single problem instance: 1 geometry, 1 set ob boundary conditions.
    It's not really a general neural network, more an equation solver to test the mathematics and the implementation.
    The generic network follows the exact same structure though.

    See the main section at the end for an example usage.

    Returns:
        1. The ready to use result Pinn
        2. A list of losses achieved during training
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
    training_parameters = prepare_geometry(G, dirichlet_boundaries_template, neumann_boundaries_template, dim=2, refinements=refinements, use_gauss=use_gauss)
    number_basis_functions     =    training_parameters["number_basis_functions"]
    dirichlet_values           =    training_parameters["dirichlet_values"]
    indices_global_torch       =    training_parameters["indices_global_torch"]
    indices_dirichlet_torch    =    training_parameters["indices_dirichlet_torch"]
    collocation_interior       =    training_parameters["collocation_interior"]
    collocation_dirichlet      =    training_parameters["collocation_dirichlet"]
    collocation_neumann        =    training_parameters["collocation_neumann"]
    collocation_total          =    training_parameters["collocation_total"]
    integration_boundaries     =    training_parameters["integration_boundaries"]
    param_surfaces             =    training_parameters["param_surfaces"]

    input_complete                   = []
    dirichlet_values_complete        = []
    normal_values_torch              = []
    torch_coeffs_gradient_x_total    = []
    torch_coeffs_gradient_y_total    = []
    torch_surface_values             = []

    """
    Prepare all input samples (here only 1 input is used, still we work with the full setup for convenience)
    """
    for _ in range(1):
        G_t = G
        dirichlet_boundaries = dirichlet_boundaries_template
        poisson = poisson_value
        input_values, dirichlet_values = prepare_input_values(G_t, dirichlet_boundaries, poisson)
        input_complete.append(input_values)     # input numbers that actually go into the neural net
        dirichlet_values_complete.append(dirichlet_values)      # those values later get set by the strong imposition of Dirichlet conditions

        # Get normal vectors of Neumann boundaries (not used in this setup but for anyone who does this later on)
        # This does not work with gauss integration at the moment
        #if len(collocation_neumann) > 0:
        #    normal_values_x, normal_values_y = generate_neumann_normals_2D(G_t, dirichlet_boundaries_template, neumann_boundaries_template, dim=2, refinements=refinements)
        #    # Create torch values
        #    normal_values_torch.append(torch.stack(
        #        [torch.tensor(normal_values_x, dtype=torch.float64), torch.tensor(normal_values_y, dtype=torch.float64)],
        #        dim=1))

        # For each collocation point, create the coefficients of the basis and derivative functions. Those coefficients are
        # evaluated at respective collocation points.

        mapper = G_t.mapper(reference=G_t)

        """
        Complete domain
        """
        basis_collocation_points_total = G_t.basis_and_support(collocation_total)
        gradient_collocation_points_total = mapper.basis_gradient_and_support(collocation_total)

        # Basis coefficients (not needed in this setup)
        coeffs_basis_total = fill_up_basis_tensors(basis_collocation_points_total, number_basis_functions)

        coeffs_gradient_x_total, coeffs_gradient_y_total = fill_up_gradient_tensors(gradient_collocation_points_total, number_basis_functions)
        torch_coeffs_gradient_x_total.append(torch.from_numpy(coeffs_gradient_x_total).double())
        torch_coeffs_gradient_y_total.append(torch.from_numpy(coeffs_gradient_y_total).double())

        # Jacobian of geometry with respect to parametric coordinates
        J = G_t.jacobian(collocation_total)
        det_J = np.abs(np.linalg.det(J)).reshape(-1)
        surfaces = det_J * param_surfaces.reshape(-1)
        torch_surface_values.append(torch.from_numpy(surfaces).double())

        """
        Neumann sides (not used in this setup, write the code the same as for the total control points)
        """

    torch_input_complete                =   torch.stack(input_complete, dim=0)
    torch_dirichlet_values_complete     =   torch.stack(dirichlet_values_complete, dim=0)
    torch_coeffs_gradient_x_total       =   torch.stack(torch_coeffs_gradient_x_total, dim=0)
    torch_coeffs_gradient_y_total       =   torch.stack(torch_coeffs_gradient_y_total, dim=0)
    torch_surface_values                =   torch.stack(torch_surface_values, dim=0)

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
    torch_coeffs_gradient_x_total   = torch_coeffs_gradient_x_total.to(dtype=torch.float64)
    torch_coeffs_gradient_y_total   = torch_coeffs_gradient_y_total.to(dtype=torch.float64)
    torch_surface_values            = torch_surface_values.to(dtype=torch.float64)

    # Move everything to the graphics card if there is one
    model                           = model.to(device)
    torch_input_complete            = torch_input_complete.to(device)
    torch_dirichlet_values_complete = torch_dirichlet_values_complete.to(device)
    torch_coeffs_gradient_x_total   = torch_coeffs_gradient_x_total.to(device)
    torch_coeffs_gradient_y_total   = torch_coeffs_gradient_y_total.to(device)
    torch_surface_values            = torch_surface_values.to(device)

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
        # Calculate lame parameters, therefore poisson value must always be the last entry
        po = torch_input_complete[:, -1]
        mu_ = E / (2 * (1+po))
        lambda_ = po * E /  ((1+po) * (1-2*po))

        # separate x and y displacements
        F = torch.zeros(number_samples, number_basis_functions*2, device=device)
        F[:, indices_global_torch] = output
        F[:, indices_dirichlet_torch] = torch_dirichlet_values_complete
        F_1, F_2 = torch.chunk(F, 2, dim=1)

        # Initialize the loss
        LOSS = 0

        # Build up the first derivatives
        # Use of einstein summation, the b stands just for the batch number since we're operating on every sample at the same time.
        u1_x = torch.einsum('bk,bmk->bm', F_1, torch_coeffs_gradient_x_total)
        u1_y = torch.einsum('bk,bmk->bm', F_1, torch_coeffs_gradient_y_total)
        u2_x = torch.einsum('bk,bmk->bm', F_2, torch_coeffs_gradient_x_total)
        u2_y = torch.einsum('bk,bmk->bm', F_2, torch_coeffs_gradient_y_total)

        # Calculate the determinant of the deformation gradient
        J_t = 1 + u1_x + u2_y + torch.mul(u1_x, u2_y) - torch.mul(u1_y, u2_x)
        # Apply the softplus function to deal with ln(<0) issue
        J = torch.nn.functional.softplus(J_t, 10.0)

        # Calculate strain energy function Pi
        W = lambda_[:, None] / 2 * torch.square(torch.log(J)) + mu_[:, None] / 2 * (torch.square(1 + u1_x) + torch.square(u1_y) + torch.square(u2_x) + torch.square(1 + u2_y) - 2 - 2 * torch.log(J))
        Pi = torch.sum(W * torch_surface_values)

        # Loss
        LOSS += Pi

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

    loss_list = []
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

    end = time.perf_counter()
    runtime = end - start

    """
    Model Saving
    """
    if save is not None:
        model.cpu()
        torch.save(model, save+".pth")

    result_pinn = Pinn(model)

    if show_plots:
        # Visualize the result
        u = result_pinn.evaluate(G, dirichlet_boundaries_template, neumann_boundaries_template, poisson)
        S = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points+u.control_points)
        S.show()

    return result_pinn, loss_list, prep_time, runtime



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
    architecture = [0]

    my_pinn, loss, prep_time, runtime = train_single_instance_network(G,
                                                                      1,
                                                                      0.3,
                                                                      dirichlet_boundaries,
                                                                      neumann_boundaries,
                                                                      architecture,
                                                                      refinements=4,
                                                                      max_iterations=100,
                                                                      save="test_network",
                                                                      show_plots=True)

    print(f"Training finished => final loss: {loss[-1]:.4f}, prep_time: {prep_time:.4f}, runtime: {runtime:.4f}")
