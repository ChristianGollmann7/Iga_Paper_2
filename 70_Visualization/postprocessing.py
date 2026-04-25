import matplotlib
import numpy as np
import splinepy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from numpy.typing import NDArray
from sympy.physics.quantum.circuitplot import Line2D
from numbers import Real

from helper_functions import load_Bspline
from python_utils.geometry_preparation import plot_Bspline


def calculate_cartesian_difference(S1: splinepy.BSpline,
                                   S2: splinepy.BSpline,
                                   coll_pts: NDArray=None) -> NDArray[float]:
    """
    Calculate the cartesian difference of two splines evaluated at coll_pts

    Returns differences in a numpy array.
    """
    diff = S1.control_points - S2.control_points
    return np.linalg.norm(diff, axis=1)


def calculate_cartesian_difference_at_positions(S1: splinepy.BSpline,
                                   S2: splinepy.BSpline,
                                   coll_pts: NDArray=None) -> NDArray[float]:
    """
    Calculate the cartesian difference of two splines evaluated at coll_pts

    Returns differences in a numpy array.
    """
    sample_points = coll_pts
    val_1 = S1.evaluate(sample_points)
    val_2 = S2.evaluate(sample_points)
    diff = val_2 - val_1
    return np.linalg.norm(diff, axis=1)


def get_difference_visu_data(base: splinepy.BSpline,
                             S_real: splinepy.BSpline,
                             S_net: splinepy.BSpline,
                             resolution: int=32) -> tuple[NDArray[Real], PolyCollection]:
    """
    Creates the visualization data for the difference between two splines.

    :param base: undeformed base spline
    :param S_real: reference solution
    :param S_net: predicted spline solution, this spline will be plotted as outline
    :param resolution: resolution of the visualization

    Returns the cartesian difference and a PolyCollection object that can be plotted.
    """
    # Base geometry gets sampled into a mesh
    my_mesh = base.extract.faces([resolution, resolution])
    # Determine Vertices and connectivity objects
    vertices = my_mesh.vertices
    connectivity = np.asarray(my_mesh.faces)
    # Get parameteric coordinates of the vertices
    coords = base.proximities(vertices)
    # Calculate difference values at vertices
    difference = calculate_cartesian_difference_at_positions(S_real, S_net, coords)

    # Deal with deformed mesh now
    deformed_mesh = S_net.extract.faces([resolution, resolution])
    deformed_vertices = deformed_mesh.vertices
    deformed_faces = connectivity

    # Get the value of a face by averaging over corner point values
    face_values = difference[deformed_faces].mean(axis=1)
    # Get coordinates of faces
    face_coordinates = [deformed_vertices[i] for i in deformed_faces]

    # Generate the poly data
    poly_data = PolyCollection(face_coordinates, array=face_values, cmap='viridis', linewidths=0)

    return difference, poly_data


def displacement_plot(base: splinepy.BSpline,
                             S: splinepy.BSpline,
                             ax: matplotlib.axes.Axes,
                             color_base: str='black',
                             color_S: str='green'):
    """
    Creates a simple displacement plot, shows undeformed and deformed spline.

    :param base: undeformed base spline.
    :param S: predicted spline solution.
    """
    handles, labels = ax.get_legend_handles_labels()
    plot_Bspline(base, ax, color=color_base, n=64, linestyle='--')
    handles.append(Line2D([0], [0], color=color_base, linestyle='--'))
    labels.append('undeformed')
    plot_Bspline(S, ax, color=color_S, n=64, linestyle='-')
    handles.append(Line2D([0], [0], color=color_S, linestyle='-'))
    labels.append('deformed')
    ax.legend(handles, labels, fontsize=14)


