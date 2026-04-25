import numpy as np
import splinepy
import xmltodict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
from splinepy.helpme.extract import control_points
from sympy.printing.pretty.pretty_symbology import line_width

torch.set_default_dtype(torch.float64)

def load_Bspline(xml_file: str) -> splinepy.BSpline:
    """
    This function loads a BSpline from an xml definition according to gismo style.
    """
    # Read the file
    with open(xml_file, "r") as f:
        data = xmltodict.parse(f.read())
    degrees = []
    knot_vectors = []
    control_points = []

    # Extract the knot vectors
    knot_data = data['xml']['Geometry']['Basis']['Basis']
    for knot_element in knot_data:
        # Extract the degree
        degrees.append(int(knot_element['KnotVector']['@degree']))
        # Extract the knot values
        knot_vectors.append([float(i) for i in knot_element['KnotVector']['#text'].split()])

    # Extract the control points
    dimension = int(data['xml']['Geometry']['coefs']['@geoDim'])
    coeffs = data['xml']['Geometry']['coefs']['#text'].split()
    control_points = [coeffs[i:i+dimension] for i in range(0, len(coeffs), dimension)]

    return splinepy.BSpline(degrees=degrees, knot_vectors=knot_vectors, control_points=control_points)



def plot_Bspline(G: splinepy.BSpline,
                 ax: matplotlib.axes.Axes,
                 smoothness: int=50,
                 color: str='black',
                 linestyle: str='-'):
    """
    This function plots a Bspline object into existing axes object.
    """
    def draw_iso_lines(knots_1, knots_2, mode=1):
        """
        Short helper function to plot iso lines in Bspline object.
        """
        for i in knots_1:
            sample_spacing = np.linspace(knots_2[0], knots_2[-1], smoothness)
            if mode == 1:
                eval_points = [[i, s] for s in sample_spacing]
            else:
                eval_points = [[s, i] for s in sample_spacing]
            values = np.array(G.evaluate(eval_points))
            ax.plot(values[:, 0], values[:, 1], color=color, linestyle=linestyle, lw=1)

    J, K = G.knot_vectors[0], G.knot_vectors[1]
    j_knots = np.unique(J)
    k_knots = np.unique(K)

    # Plot the iso lines
    draw_iso_lines(j_knots, k_knots, 1)
    draw_iso_lines(k_knots, j_knots, 2)




def outline_Bspline(G: splinepy.BSpline,
                    ax: matplotlib.axes.Axes,
                    number_points: int=50):
    """
    Fancy convenience function to outline a Bspline object, show its parametric corners and side notation.
    """
    middle = number_points // 2
    k1 = G.knot_vectors[0]
    k2 = G.knot_vectors[1]

    # Plot side a
    ls = np.linspace(min(k2), max(k2), number_points)
    side = np.column_stack((np.full_like(ls, min(k1)), ls))
    points = np.array(G.evaluate(side))
    ax.plot(points[:, 0], points[:, 1], lw=1)
    ax.text(points[:, 0][middle], points[:, 1][middle], "a",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Plot side b
    ls = np.linspace(min(k2), max(k2), number_points)
    side = np.column_stack((np.full_like(ls, max(k1)), ls))
    points = np.array(G.evaluate(side))
    ax.plot(points[:, 0], points[:, 1], lw=1)
    ax.text(points[:, 0][middle], points[:, 1][middle], "b",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Plot side c
    ls = np.linspace(min(k1), max(k1), number_points)
    side = np.column_stack((ls, np.full_like(ls, min(k2))))
    points = np.array(G.evaluate(side))
    ax.plot(points[:, 0], points[:, 1], lw=1)
    ax.text(points[:, 0][middle], points[:, 1][middle], "c",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Plot side d
    ls = np.linspace(min(k1), max(k1), number_points)
    side = np.column_stack((ls, np.full_like(ls, max(k2))))
    points = np.array(G.evaluate(side))
    ax.plot(points[:, 0], points[:, 1], lw=1)
    ax.text(points[:, 0][middle], points[:, 1][middle], "d",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Add corner point text
    X = G.evaluate([[min(k1), min(k2)]])
    ax.text(X[:, 0], X[:, 1], f"({min(k1)}, {min(k2)})",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Add corner point text
    X = G.evaluate([[max(k1), min(k2)]])
    ax.text(X[:, 0], X[:, 1], f"({max(k1)}, {min(k2)})",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Add corner point text
    X = G.evaluate([[min(k1), max(k2)]])
    ax.text(X[:, 0], X[:, 1], f"({min(k1)}, {max(k2)})",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])

    # Add corner point text
    X = G.evaluate([[max(k1), max(k2)]])
    ax.text(X[:, 0], X[:, 1], f"({max(k1)}, {max(k2)})",
            va="center", ha="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=6, foreground=ax.get_facecolor())])



def fill_up_basis_tensors(my_input, number_basis_functions: int) -> np.ndarray:
    """
    Splinepy mapper returns basis functions evaluated at the passed collocation points.
    However, only non zero results are returned. We need a result that also contains zeros.
    """
    result = []
    values = my_input[0]
    indices = my_input[1]
    for i in range(len(values)):
        temp = np.zeros(number_basis_functions)
        temp[indices[i]] = values[i]
        result.append(temp)
    return np.array(result)


def fill_up_gradient_tensors(my_input, number_basis_functions: int) -> tuple[np.ndarray, np.ndarray]:
    result_x = []
    result_y = []
    values = my_input[0]
    indices = my_input[1]
    for i in range(len(values)):
        temp_x = np.zeros(number_basis_functions)
        temp_y = np.zeros(number_basis_functions)
        temp_x[indices[i]] = values[i][:,0]
        temp_y[indices[i]] = values[i][:,1]
        result_x.append(temp_x)
        result_y.append(temp_y)
    return np.array(result_x), np.array(result_y)


def fill_up_hessian_tensors(my_input, number_basis_functions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result_xx = []
    result_xy = []
    result_yx = []
    result_yy = []
    values = my_input[0]
    indices = my_input[1]
    for i in range(len(values)):
        temp_xx = np.zeros(number_basis_functions)
        temp_xy = np.zeros(number_basis_functions)
        temp_yx = np.zeros(number_basis_functions)
        temp_yy = np.zeros(number_basis_functions)
        temp_xx[indices[i]] = [K[0][0] for K in values[i]]
        temp_xy[indices[i]] = [K[0][1] for K in values[i]]
        temp_yx[indices[i]] = [K[1][0] for K in values[i]]
        temp_yy[indices[i]] = [K[1][1] for K in values[i]]
        result_xx.append(temp_xx)
        result_xy.append(temp_xy)
        result_yx.append(temp_yx)
        result_yy.append(temp_yy)
    return np.array(result_xx), np.array(result_xy), np.array(result_yx), np.array(result_yy)


