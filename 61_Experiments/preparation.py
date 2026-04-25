import copy
from numpy.typing import NDArray
import numpy as np
import splinepy
import torch
from torch import Tensor
from numbers import Real
from collections.abc import Sequence

# Define the Boundary Condition type
BC = dict[str, dict[int, list[Real]]]

def extract_dirichlet_information(G: splinepy.BSpline,
                                  dirichlet_boundaries: BC,
                                  dim: int) -> tuple[list[Real], Tensor, Tensor]:
    """
    For 2 dimensional solution variable, dim = 2
    Create Dirichlet input from the passed Dirichlet conditions. We have a global indices vector that has an entry for
    each control point, total length is number of control points n * 2. Imagine it as
    [0      for x0,
     1      for x1,
     2      for x2, ...
     n + 0  for y0,
     n + 1  for y1,
     n + 2  for y2, ...]
     We build a similar vector for all known Dirichlet values. Every index that gets written into the Dirichlet vector
     is removed from the global vector.

     For 1 dimensional solution variable just think of as no y values
    """
    number_basis_functions = len(G.control_points)
    # Get the global indices
    indices_global = list(range(0, number_basis_functions * dim))
    # Count how many control points are specified in the Dirichlet Conditions
    counter = 0
    for element in dirichlet_boundaries.values():
        counter += len(element)
    # Create empty lists to hold Dirichlet indices and values
    indices_dirichlet = [None] * counter * dim
    dirichlet_values = [None] * counter * dim
    # Populate both lists
    i = 0
    for element in dirichlet_boundaries.values():
        for key in element.keys():
            indices_dirichlet[i], dirichlet_values[i] = key, element[key][0]    # x value
            if dim == 2:
                indices_dirichlet[i + counter], dirichlet_values[i + counter] = key + number_basis_functions, element[key][1]   # y value
            i += 1
    # Remove Dirichlet indices from the global index list
    indices_global = [j for j in indices_global if j not in indices_dirichlet]

    # Turn indices into torch tensors
    indices_global_torch = torch.tensor(indices_global)
    indices_dirichlet_torch = torch.tensor(indices_dirichlet)

    return dirichlet_values, indices_global_torch, indices_dirichlet_torch


def prepare_input_values(G: splinepy.BSpline,
                         dirichlet_boundaries: BC,
                         poisson: Real) -> tuple[Tensor, Tensor]:
    """
    Based on geometry G, the boundary conditions and the poisson value, create the torch input tensor that goes into the network.
    """
    dirichlet_values, indices_global_torch, indices_dirichlet_torch = extract_dirichlet_information(G, dirichlet_boundaries, dim=2)
    # Get the control points of G
    input_values = unpack_coefficients_xy(G.control_points)
    # Append the dirichlet values
    input_values = input_values + dirichlet_values
    # Append the poisson value
    input_values.append(poisson)
    # Insert into the complete input tensor
    input_values = torch.tensor(input_values, dtype=torch.float64)
    dirichlet_values = torch.tensor(dirichlet_values, dtype=torch.float64)

    return input_values, dirichlet_values


def specify_dirichlet_boundaries_with_single_value(dirichlet_boundaries_template: BC,
                                                   sides: Sequence[str],
                                                   delta_x: Sequence[Real],
                                                   delta_y: Sequence[Real]) -> BC:
    """
    Set a value for Dirichlet boundary conditions. Supports only setting the same value for all x-displacements per side.
    Same for y displacements.

    :param dirichlet_boundaries_template: Dirichlet boundary conditions template
    :param sides: sides for which to specify the values, for example sides=['b', 'c']
    :param delta_x: x-displacement values, one value per side, for example delta_x=[3.2, 1.0]
    :param delta_y: y-displacement values, same as above
    """
    result = copy.deepcopy(dirichlet_boundaries_template)
    for i in range(len(sides)):
        side = sides[i]
        for index in dirichlet_boundaries_template[side].keys():
            result[side][index][0] = delta_x[i]
            result[side][index][1] = delta_y[i]
    return result




def unpack_coefficients_xy(coeffs: list[list]) -> list:
    """
    This function reorders a list of lists into one consecutive list:
    [ [a,b], [c,d], [e,f], ... ] becomes [a, c, e, ..., b, d, f, ...]
    """
    x, y = [], []
    for pair in coeffs:
        x.append(pair[0])
        y.append(pair[1])
    return x + y



def extract_knot_values(G):
    """
    Extract the knot values of the geometry. Conceptually we assume knot values start with 0 and end with 1, however
    they can be any monotonically increasing sequence, that's why we work with min and max here to extract start and
    end points.
    """
    k1, k2 = G.knot_vectors
    k1_min, k1_max = min(k1), max(k1)
    k2_min, k2_max = min(k2), max(k2)
    return k1_min, k1_max, k2_min, k2_max



def create_collocation_points(G: splinepy.BSpline,
                              dirichlet_boundaries: BC,
                              neumann_boundaries: BC,
                              refinements: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate the Greville collocation points. Note that collocation points are parametric and within the range of knot values.
    We split the collocation points into four distinct groups: interior points, Dirichlet points, Neumann points and the union of these three.
    First we remove all Dirichlet points, then the Neumann points and what remains are the interior points.
    If a Neumann and Dirichlet side meet in a shared corner point, respective corner point counts only to Dirichlet collocation points.
    """
    # Get the min and max knot values
    k1_min, k1_max, k2_min, k2_max = extract_knot_values(G)

    # Generate all collocation points
    S = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points)
    if refinements >= 0:
        for _ in range(refinements):
            S.uniform_refine()
    collocation_points = S.greville_abscissae()


    # Define an empty container for Dirichlet points
    collocation_dirichlet = np.empty((0, 2))
    # Define an empty container for Neumann points
    collocation_neumann = np.empty((0, 2))
    # Interior collocation points start as copy of the complete coll points, Dirichlet and Neumann get removed from it
    collocation_interior = copy.deepcopy(collocation_points)
    # The complete collocation points (used in the energy method) are just a copy of the normal Greville points
    collocation_total = copy.deepcopy(collocation_points)

    # Filter out Dirichlet points
    if "a" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 0], k1_min, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_dirichlet = np.concatenate([collocation_dirichlet, addition])
    if "b" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 0], k1_max, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_dirichlet = np.concatenate([collocation_dirichlet, addition])
    if "c" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 1], k2_min, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_dirichlet = np.concatenate([collocation_dirichlet, addition])
    if "d" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 1], k2_max, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_dirichlet = np.concatenate([collocation_dirichlet, addition])

    # Filter out Neumann points
    if "a" in neumann_boundaries:
        mask = np.isclose(collocation_interior[:, 0], k1_min, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_neumann = np.concatenate([collocation_neumann, addition])
    if "b" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 0], k1_max, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_neumann = np.concatenate([collocation_neumann, addition])
    if "c" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 1], k2_min, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_neumann = np.concatenate([collocation_neumann, addition])
    if "d" in dirichlet_boundaries:
        mask = np.isclose(collocation_interior[:, 1], k2_max, rtol=1e-7, atol=1e-7)
        addition = collocation_interior[mask]
        collocation_interior = collocation_interior[~mask]
        collocation_neumann = np.concatenate([collocation_neumann, addition])

    return collocation_interior, collocation_dirichlet, collocation_neumann, collocation_total



def get_parametric_rectangle_boundaries(G: splinepy.BSpline,
                                        coll_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets the boundaries of parametric rectangles around collocation points. This function is intended to be used on Greville points
    that stem from a geometry G according to the normal ordering of control points.
    """
    knots_1 = np.unique(coll_pts[:, 0])
    knots_2 = np.unique(coll_pts[:, 1])
    len_1, len_2 = len(knots_1), len(knots_2)
    min_1, max_1 = np.min(G.knot_vectors[0]), np.max(G.knot_vectors[0])
    min_2, max_2 = np.min(G.knot_vectors[1]), np.max(G.knot_vectors[1])
    boundaries = []

    for j in range(len_2):
        if j == 0:
            c = min_2
            d = (knots_2[0] + knots_2[1]) / 2
        elif j == len_2 - 1:
            c = (knots_2[-1] + knots_2[-2]) / 2
            d = max_2
        else:
            c = (knots_2[j-1] + knots_2[j]) / 2
            d = (knots_2[j+1] + knots_2[j]) / 2
        for i in range(len_1):
            if i == 0:
                a = min_1
                b = (knots_1[0] + knots_1[1]) / 2
            elif i == len_1-1:
                a = (knots_1[-1] + knots_1[-2]) / 2
                b = max_1
            else:
                a = (knots_1[i-1] + knots_1[i]) / 2
                b = (knots_1[i+1] + knots_1[i]) / 2
            boundaries.append([a, b, c, d])

    surfaces = []
    for b in boundaries:
        surfaces.append( (b[1]-b[0])*(b[3]-b[2]))

    return np.array(boundaries), np.array(surfaces)





def prepare_geometry(G: splinepy.BSpline,
                     dirichlet_boundaries: BC,
                     neumann_boundaries: BC,
                     dim: int,
                     refinements: int=0,
                     use_gauss: bool=True) -> dict:
    """
    Creates all the parametric data a neural network needs to know to determine its structure and training setup.
    If use_gauss is set to True, Gauss points and weights are returned as collocation points. In that case, there are no
    proper points on the boundary, this still needs to be implemented.
    """
    number_basis_functions = len(G.control_points)
    k1_min, k1_max, k2_min, k2_max = extract_knot_values(G)
    dirichlet_values, indices_global_torch, indices_dirichlet_torch = extract_dirichlet_information(G, dirichlet_boundaries, dim=dim)
    if use_gauss:
        # if gauss quadrature is used, collocation_total become the collocation points and param_surfaces are the gauss weights
        collocation_total, param_surfaces = create_gauss_points(G, refinements)
        collocation_interior = None
        collocation_dirichlet = None
        collocation_neumann = None
        integration_boundaries = None
    else:
        collocation_interior, collocation_dirichlet, collocation_neumann, collocation_total = create_collocation_points(G, dirichlet_boundaries, neumann_boundaries, refinements=refinements)
        integration_boundaries, param_surfaces = get_parametric_rectangle_boundaries(G, collocation_total)


    return {"number_basis_functions":   number_basis_functions,
            "k1_min":                   k1_min,
            "k1_max":                   k1_max,
            "k2_min":                   k2_min,
            "k2_max":                   k2_max,
            "dirichlet_values":         dirichlet_values,
            "indices_global_torch":     indices_global_torch,
            "indices_dirichlet_torch":  indices_dirichlet_torch,
            "collocation_interior":     collocation_interior,
            "collocation_dirichlet":    collocation_dirichlet,
            "collocation_neumann":      collocation_neumann,
            "collocation_total":        collocation_total,
            "integration_boundaries":   integration_boundaries,
            "param_surfaces":           param_surfaces}



def generate_neumann_normals_2D(G: splinepy.BSpline,
                                dirichlet_boundaries: BC,
                                neumann_boundaries: BC,
                                dim: int,
                                refinements: int) -> tuple[NDArray, NDArray]:
    training_parameters = prepare_geometry(G, dirichlet_boundaries, neumann_boundaries, dim=dim, refinements=refinements)

    k1_min = training_parameters["k1_min"]
    k1_max = training_parameters["k1_max"]
    k2_min = training_parameters["k2_min"]
    k2_max = training_parameters["k2_max"]
    collocation_neumann = training_parameters["collocation_neumann"]

    if len(collocation_neumann) == 0:
        return None, None

    # Extract G's boundaries as spline curves
    c_a, c_b, c_c, c_d = G.extract.boundaries()
    """
    Boundaries are defined like this for this setup
       (1,1) - b - (1,0)
         |           |
         d           c
         |           |
       (0,1) - a - (0,0)
    """
    tangent_values = np.empty((0, 2))

    cp_neumann_a = collocation_neumann[collocation_neumann[:, 0] == k1_min]
    if len(cp_neumann_a) > 0:
        cp_neumann_a = [[i[0]] for i in cp_neumann_a]
        tangent_a = c_a.derivative(cp_neumann_a, [1])
        tangent_values = np.concatenate((tangent_values, tangent_a))

    cp_neumann_b = collocation_neumann[collocation_neumann[:, 0] == k1_max]
    if len(cp_neumann_b) > 0:
        cp_neumann_b = [[i[0]] for i in cp_neumann_b]
        tangent_b = c_b.derivative(cp_neumann_b, [1])
        tangent_values = np.concatenate((tangent_values, tangent_b))

    cp_neumann_c = collocation_neumann[collocation_neumann[:, 1] == k2_min]
    if len(cp_neumann_c) > 0:
        cp_neumann_c = [[i[0]] for i in cp_neumann_c]
        tangent_c = c_c.derivative(cp_neumann_c, [1])
        tangent_values = np.concatenate((tangent_values, tangent_c))

    cp_neumann_d = collocation_neumann[collocation_neumann[:, 1] == k2_max]
    if len(cp_neumann_d) > 0:
        cp_neumann_d = [[i[0]] for i in cp_neumann_d]
        tangent_d = c_d.derivative(cp_neumann_d, [1])
        tangent_values = np.concatenate((tangent_values, tangent_d))

    # Turn tangent vectors into normal vectors by 90° rotation
    normal_values = np.array([[i[1], -i[0]] / np.sqrt(i[0]**2 + i[1]**2) for i in tangent_values])

    # Split normal values into x and y components
    normal_values_x = normal_values[:, 0]
    normal_values_y = normal_values[:, 1]

    # Check orientation of normal vector and make it outward pointing
    for i in range(len(collocation_neumann)):
        P = G.evaluate([collocation_neumann[i]])
        X = P[0][0]
        Y = P[0][1]
        X1 = X + normal_values_x[i] * 0.05      # this approach might not work well for highly concave boundaries
        Y1 = Y + normal_values_y[i] * 0.05
        P = np.asarray([[X1, Y1]])
        (_, _, _, tolerance, *_) = G.proximities(P, return_verbose=True)    # if P lies within G, tolerance will be very low, close to machine precision
        if tolerance[0] < 1e-8:
            normal_values_x[i] *= -1
            normal_values_y[i] *= -1

    return normal_values_x, normal_values_y


def create_gauss_points(G: splinepy.BSpline,
                        n: int=4):
    """
    Creates the Gauss points and weights for each parametric knot span rectangle so Gauss quadrature can happen over G.
    Returns a vector of gauss points pairs and a vector of corresponding weights.
    """
    xi, w = np.polynomial.legendre.leggauss(n)

    def tensor_product_gauss(knot_spans):
        pts_all = []
        weights_all = []
        for (a, b) in knot_spans:
            diff = b-a
            u_gauss = 0.5 * (a+b) + 0.5 * diff * xi
            w_gauss = 0.5 * diff * w
            pts_all.append(u_gauss)
            weights_all.append(w_gauss)

        pts_all = np.concatenate(pts_all)
        weights_all = np.concatenate(weights_all)

        pts = []
        weights = []
        for j in range(pts_all.shape[0]):
            for i in range(pts_all.shape[0]):
                pts.append([pts_all[i], pts_all[j]])
                weights.append([weights_all[i] * weights_all[j]])
        return np.asarray(pts, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    knots = np.asarray(G.knot_vectors[0], dtype=float).ravel()
    knots = np.unique(knots)
    spans = [(a, b) for a, b in zip(knots[:-1], knots[1:])]

    return tensor_product_gauss(spans)


