from validation import evaluate_validation_samples
from neural_network import Pinn
import torch

"""
For each specified network, evaluate the validation samples of given validation folder
"""

if __name__ == "__main__":
    samples = 1000
    networks = [f"multi_instance_results_presampled_simple/network_[1, 1]_{samples}.pth",
                f"multi_instance_results_presampled/network_[1, 1]_{samples}.pth",
                f"multi_instance_results_presampled/network_[1, 1, 1]_{samples}.pth",
                f"vanilla_results_simple/network_[1, 1]_{samples}.pth",
                f"vanilla_results/network_[0.5]_{samples}.pth",
                f"vanilla_results/network_[0.5, 0.5]_{samples}.pth",
                f"vanilla_results/network_[0.5, 0.5, 0.5]_{samples}.pth",
                f"vanilla_results/network_[1]_{samples}.pth",
                f"vanilla_results/network_[1, 1]_{samples}.pth",
                f"vanilla_results/network_[1, 1, 1]_{samples}.pth",
                f"vanilla_results/network_[2]_{samples}.pth",
                f"vanilla_results/network_[2, 2]_{samples}.pth",
                f"vanilla_results/network_[2, 2, 2]_{samples}.pth",
                f"multi_instance_results_presampled/network_[1]_r_2_{samples}.pth",
                f"multi_instance_results_presampled/network_[1]_r_3_{samples}.pth",
                f"multi_instance_results_presampled/network_[1]_r_4_{samples}.pth",
                ]
    validation_set = "validation_set_hard"
    error_list = []

    # Template boundary conditions
    dirichlet_boundaries = {}
    dirichlet_boundaries["a"] = {0: [0, 0], 7: [0, 0], 14: [0, 0], 21: [0, 0], 28: [0, 0], 35: [0, 0], 42: [0, 0]}
    dirichlet_boundaries["b"] = {6: [1, 2], 13: [1, 2], 20: [1, 2], 27: [1, 2], 34: [1, 2], 41: [1, 2], 48: [1, 2]}
    neumann_boundaries = {}
    neumann_boundaries["c"] = {}
    neumann_boundaries["d"] = {}

    for network in networks:
        model = torch.load(network, weights_only=False)
        result_pinn = Pinn(model)
        val_error = evaluate_validation_samples(result_pinn, dirichlet_boundaries, neumann_boundaries, validation_set)
        error_list.append(val_error)
        print(network + f" : {val_error}")


