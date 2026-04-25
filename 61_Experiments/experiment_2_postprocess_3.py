from platform import architecture

import torch
import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})

def load_model(location: str):
    return torch.load(location, map_location="cpu", weights_only=False)


def weights_heatmap(ax: Axes,
                    weights: torch.Tensor,
                    v_min: float,
                    v_max: float,
                    title: str=None,
                    cmap: str="seismic"):
    my_heatmap = ax.imshow(weights.detach().cpu(), aspect='auto', cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title(title, fontsize= 18)
    return my_heatmap


def bias_heatmap(ax: Axes,
                 bias: torch.Tensor,
                 v_min: float,
                 v_max: float,
                 title: str=None,
                 cmap: str="seismic"):
    my_heatmap = ax.imshow(bias.detach().cpu().reshape(-1, 1), aspect='auto', cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_xticks([])
    ax.set_title(title, fontsize= 18)
    return my_heatmap

architecture = [1, 1]

model_physics = load_model(f"results_1/network_{architecture}_r_8_1000.pth")
parameters_physics = model_physics.state_dict()

model_data = load_model(f"results_2/network_{architecture}_1000.pth")
parameters_data = model_data.state_dict()

titles = ["weights between input and 1st hidden layer", "weights between 1st and 2nd hidden layer", "weights between 2nd hidden layer and output"]
layers = [0, 2, 4]
for i in range(3):
    layer = layers[i]

    fig, ax = plt.subplots(3, 2, figsize=(20, 28), gridspec_kw={"width_ratios":[9, 1]}, constrained_layout=True)

    weights_heatmap(ax[0][0], parameters_physics[f"net.{layer}.weight"], v_min=-1, v_max=1, title=titles[i]+" \nphysics-based network")
    my_heatmap = bias_heatmap(ax[0][1], parameters_physics[f"net.{layer}.bias"], v_min=-1, v_max=1, title="bias vector")
    fig.colorbar(my_heatmap, ax=ax[0, 1], location="right", pad=0.5, fraction=0.4)
    #ax[0,0].set_xticks([])

    weights_heatmap(ax[1][0], parameters_data[f"net.{layer}.weight"], v_min=-1, v_max=1, title=titles[i]+" \nsupervised network")
    my_heatmap = bias_heatmap(ax[1][1], parameters_data[f"net.{layer}.bias"], v_min=-1, v_max=1, title="bias vector")
    fig.colorbar(my_heatmap, ax=ax[1, 1], location="right", pad=0.5, fraction=0.4)
    #ax[1,0].set_xticks([])

    cmap = 'viridis'
    weights_heatmap(ax[2][0], np.abs(parameters_physics[f"net.{layer}.weight"] - parameters_data[f"net.{layer}.weight"]), v_min=0, v_max=1, title=titles[i]+" \nabs( physics - supervised )", cmap=cmap)
    my_heatmap = bias_heatmap(ax[2][1], np.abs(parameters_physics[f"net.{layer}.bias"] - parameters_data[f"net.{layer}.bias"]), v_min=0, v_max=1, title="difference\n bias vector", cmap=cmap)
    fig.colorbar(my_heatmap, ax=ax[2, 1], location="right", pad=0.5, fraction=0.4)

    plt.savefig(f"results_2/heatmap_{architecture}_{i}.svg")



pass