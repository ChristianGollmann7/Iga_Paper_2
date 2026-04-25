import matplotlib.pyplot as plt
from helper_functions import load_Bspline, plot_Bspline
from preparation import create_gauss_points


geometry = "input_geometries/simple_5.xml"

# Get the base geometry
G = load_Bspline("input_geometries/simple_5.xml")

fig, ax = plt.subplots(1,3, figsize=(15,5))
densities = [2, 3, 4]
colors = ["blue", "orange", "green"]
for i in range(3):
    # Get the gauss points
    pts, _ = create_gauss_points(G, densities[i])
    pts = G.evaluate(pts)


    plot_Bspline(G, ax[i])
    ax[i].scatter(pts[:, 0], pts[:, 1], c=colors[i], s=20, edgecolors=colors[i])
    ax[i].set_title(f'Gauss points, {densities[i]} per knot span', fontsize=16)

    ax[i].tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
    ax[i].set_aspect("equal", adjustable='box')
    ax[i].set_ylim(-1.2, 1.2)

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"results_0/gauss_points.png", dpi=600, bbox_inches='tight', pad_inches=0.05)