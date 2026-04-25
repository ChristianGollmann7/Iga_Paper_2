import numpy as np
import splinepy
import torch
from torch import nn
from collections.abc import Sequence
from numbers import Real

from preparation import extract_dirichlet_information, prepare_input_values

# Define the Boundary Condition type
BC = dict[str, dict[int, list[Real]]]

torch.set_default_dtype(torch.float64)

class Mlp(nn.Module):
    def __init__(self,
                 number_inputs: int,
                 number_outputs: int,
                 structure: Sequence[Real],
                 activation: nn.Module=nn.Sigmoid):
        """
        Creates a fully connected feed forward neural network.

        :param number_inputs: number of input nodes
        :param number_outputs: number of output nodes
        :param structure: size of each hidden layer whereas the input size gets multiplied with respective factor, must contain at least one entry
        :param activation: torch activation function

        example call: neural_network(12, 10, [1.0, 0.8, 0.5], nn.Sigmoid)
        """
        super().__init__()
        network_layers_list = []

        # Connect the input layer
        network_layers_list.append(nn.Linear(number_inputs, int(np.maximum(number_inputs*structure[0], 1))))
        network_layers_list.append(activation())

        # Connect the rest of the hidden layers
        for i in range(len(structure)-1):
            network_layers_list.append(nn.Linear(int(np.maximum(number_inputs*structure[i], 1)), int(np.maximum(number_inputs*structure[i+1], 1))))
            network_layers_list.append(activation())

        # Connect the output layer, no activation function
        network_layers_list.append(nn.Linear(int(np.maximum(number_inputs*structure[-1], 1)), number_outputs))

        self.net = nn.Sequential(*network_layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class Pinn:
    def __init__(self, model: Mlp):
        """
        Creates an instance of a physics informed neural network based on Mlp.
        This class internally performs proper operations on the input for more convenience.

        :param model: trained model of type Mlp
        """
        self.model = model
        self.instantiated = False
        self.number_basis_functions = None
        self.indices_global_torch, self.indices_dirichlet_torch = None, None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self,
                 G: splinepy.BSpline,
                 dirichlet_boundaries: BC,
                 neumann_boundaries: BC,
                 poisson: Real) -> splinepy.BSpline:
        """
        Evaluates the network for geometry G and given boundary conditions.
        So far, only Dirichlet BCs are supported.

        :param G: undeformed geometry spline
        :param dirichlet_boundaries: dict of dirichlet boundary conditions, example BCs:
            dirichlet_boundaries = {}
            dirichlet_boundaries["side (a, b, c or d)"] = {control point number: [delta X, delta Y], ...}
            example: dirichlet_boundaries["a"] = {0: [0,0], 7: [0,0], 14: [0,0]}
        :param poisson: poisson value of material

        Returns the displacement BSpline.
        """
        if not self.instantiated:
            self.number_basis_functions = len(G.control_points)
            _, self.indices_global_torch, self.indices_dirichlet_torch = extract_dirichlet_information(G, dirichlet_boundaries, dim=2)
        input_values, prescribed_dirichlet_values = prepare_input_values(G, dirichlet_boundaries, poisson)
        # Run input through model
        output = self.model(input_values)
        # Separate x and y displacements
        F = torch.zeros(self.number_basis_functions * 2, device=self.device)
        F[self.indices_global_torch] = output
        F[self.indices_dirichlet_torch] = prescribed_dirichlet_values
        F_1, F_2 = torch.chunk(F, chunks=2, dim=0)
        F_1_np = F_1.detach().cpu().numpy()
        F_2_np = F_2.detach().cpu().numpy()

        F = np.array([[F_1_np[i], F_2_np[i]] for i in range(len(F_1_np))])
        u = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=F)

        return u
