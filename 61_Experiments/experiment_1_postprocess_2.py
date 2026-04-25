from platform import architecture

import torch
import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14})

def load_model(location: str):
    return torch.load(location, map_location="cpu", weights_only=False)


def weights_heatmap(ax: Axes,
                    weights: torch.Tensor,
                    v_min: float,
                    v_max: float,
                    title: str=None):
    my_heatmap = ax.imshow(weights.detach().cpu(), aspect='auto', cmap='seismic', vmin=v_min, vmax=v_max)
    ax.set_title(title, fontsize= 18)
    return my_heatmap


def bias_heatmap(ax: Axes,
                 bias: torch.Tensor,
                 v_min: float,
                 v_max: float,
                 title: str=None):
    my_heatmap = ax.imshow(bias.detach().cpu().reshape(-1, 1), aspect='auto', cmap='seismic', vmin=v_min, vmax=v_max)
    ax.set_xticks([])
    ax.set_title(title, fontsize= 18)
    return my_heatmap

architecture = [1, 1]

model_physics = load_model(f"results_1/network_{architecture}_r_8_1000.pth")
parameters_physics = model_physics.state_dict()


titles = ["weights between input and 1st hidden layer", "weights between 1st and 2nd hidden layer", "weights between 2nd hidden layer and output"]
layers = [0, 2, 4]



fig, ax = plt.subplots(3, 2, figsize=(20, 28), gridspec_kw={"width_ratios":[9, 1]}, constrained_layout=True)

i = 0
layer = layers[i]
weights_heatmap(ax[i][0], parameters_physics[f"net.{layer}.weight"], v_min=-1, v_max=1, title=titles[i])
my_heatmap = bias_heatmap(ax[i][1], parameters_physics[f"net.{layer}.bias"], v_min=-1, v_max=1, title="bias vector")
fig.colorbar(my_heatmap, ax=ax[0, 1], location="right", pad=0.5, fraction=0.4)
#ax[0,0].set_xticks([])

i = 1
layer = layers[i]
weights_heatmap(ax[i][0], parameters_physics[f"net.{layer}.weight"], v_min=-1, v_max=1, title=titles[i])
my_heatmap = bias_heatmap(ax[i][1], parameters_physics[f"net.{layer}.bias"], v_min=-1, v_max=1, title="bias vector")
fig.colorbar(my_heatmap, ax=ax[i, 1], location="right", pad=0.5, fraction=0.4)
#ax[0,0].set_xticks([])

i = 2
layer = layers[i]
weights_heatmap(ax[i][0], parameters_physics[f"net.{layer}.weight"], v_min=-1, v_max=1, title=titles[i])
my_heatmap = bias_heatmap(ax[i][1], parameters_physics[f"net.{layer}.bias"], v_min=-1, v_max=1, title="bias vector")
fig.colorbar(my_heatmap, ax=ax[i, 1], location="right", pad=0.5, fraction=0.4)
#ax[0,0].set_xticks([])

plt.savefig(f"results_1/heatmap_{architecture}.svg")



pass