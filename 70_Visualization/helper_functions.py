import numpy as np
import splinepy
import xmltodict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import meshio
from splinepy.helpme.extract import control_points
from sympy.printing.pretty.pretty_symbology import line_width
from torch.distributions import von_mises

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



def calculate_metrics(G, u, collocation_points, lambda_, mu_):
    # Create the laplacian and normal extraction of basis functions and map everything to physical space
    mapper = G.mapper(reference=G)
    basis_collocation_points = G.basis_and_support(collocation_points)
    gradient_collocation_points = mapper.basis_gradient_and_support(collocation_points)
    hessian_collocation_points = mapper.basis_hessian_and_support(collocation_points)

    # Create torch coefficients for the left hand side of collocation problem
    n = len(G.control_points)
    coeffs_basis = fill_up_basis_tensors(basis_collocation_points, n)
    torch_coeffs_basis = torch.from_numpy(coeffs_basis).float()

    coeffs_gradient_x, coeffs_gradient_y = fill_up_gradient_tensors(gradient_collocation_points, n)
    torch_coeffs_gradient_x = torch.from_numpy(coeffs_gradient_x).float()
    torch_coeffs_gradient_y = torch.from_numpy(coeffs_gradient_y).float()

    coeffs_hessian_xx, coeffs_hessian_xy, coeffs_hessian_yx, coeffs_hessian_yy = fill_up_hessian_tensors(
        hessian_collocation_points, n)
    torch_coeffs_hessian_xx = torch.from_numpy(coeffs_hessian_xx).float()
    torch_coeffs_hessian_xy = torch.from_numpy(coeffs_hessian_xy).float()
    torch_coeffs_hessian_yx = torch.from_numpy(coeffs_hessian_yx).float()
    torch_coeffs_hessian_yy = torch.from_numpy(coeffs_hessian_yy).float()

    F_1 = np.array([i[0] for i in u.control_points])
    F_2 = np.array([i[1] for i in u.control_points])
    F_1 = torch.from_numpy(F_1).float()
    F_2 = torch.from_numpy(F_2).float()

    # Build up first derivatives
    u1_x = torch_coeffs_gradient_x.matmul(F_1)
    u1_y = torch_coeffs_gradient_y.matmul(F_1)
    u2_x = torch_coeffs_gradient_x.matmul(F_2)
    u2_y = torch_coeffs_gradient_y.matmul(F_2)

    # Build up second derivatives
    u1_xx = torch_coeffs_hessian_xx.matmul(F_1)
    u1_xy = torch_coeffs_hessian_xy.matmul(F_1)
    u1_yx = torch_coeffs_hessian_yx.matmul(F_1)
    u1_yy = torch_coeffs_hessian_yy.matmul(F_1)
    u2_xx = torch_coeffs_hessian_xx.matmul(F_2)
    u2_xy = torch_coeffs_hessian_xy.matmul(F_2)
    u2_yx = torch_coeffs_hessian_yx.matmul(F_2)
    u2_yy = torch_coeffs_hessian_yy.matmul(F_2)

    # Calculate J
    J = 1+ u1_x + u2_y + torch.mul(u1_x, u2_y) - torch.mul(u1_y, u2_x)

    # Calculate strain energy density function W
    W = lambda_/2 * torch.square(torch.log(J)) + mu_/2 * (torch.square(1+u1_x)+torch.square(u1_y)+torch.square(u2_x))

    # Calculate derivatives of J
    J_x = u1_xx + u2_yx + u1_xx*u2_y + u1_x*u2_yx - u1_yx*u2_x - u1_y*u2_xx
    J_y = u1_xy + u2_yy + u1_xy*u2_y + u1_x*u2_yy - u1_yy*u2_x - u1_y*u2_xy

    # Calculate A
    A = torch.div(lambda_*torch.log(J) - mu_, J)

    # Calculate derivatives of A
    A_x = torch.div((lambda_ + mu_ - lambda_*torch.log(J))*J_x, torch.square(J))
    A_y = torch.div((lambda_ + mu_ - lambda_*torch.log(J))*J_y, torch.square(J))

    # Calculate components of first Piola Kirchhoff stress tensor
    P11 = mu_*(1 + u1_x) + A*(1 + u2_y)
    P12 = mu_*u1_y - A*u2_x
    P21 = mu_*u2_x - A*u1_y
    P22 = mu_*(1 + u2_y) + A*(1 + u1_x)

    # Calculate P stress scalar
    M = torch.sqrt(torch.square(P11) + torch.square(P22) + 3*torch.square(P12) - P11*P22)

    # Calculate derivatives of first Piola Kirchhoff stress tensor
    P11_x = mu_*u1_xx + A_x + A_x*u2_y + A*u2_yx
    P12_y = mu_*u1_yy - A_y*u2_x - A*u2_xy
    P21_x = mu_*u2_xx - A_x*u1_y - A*u1_yx
    P22_y = mu_*u2_yy + A_y + A_y*u1_x + A*u1_xy

    # Calculate Cauchy stress tensor
    F_t = torch.stack([
        torch.stack([1+u1_x, u2_x], dim=-1),
        torch.stack([u1_y, 1+u2_y], dim=-1)
    ], dim=-2)

    P = torch.stack([
        torch.stack([P11, P12], dim=-1),
        torch.stack([P21, P22], dim=-1)
    ], dim=-2)

    cauchy_stress = (P @ F_t) / J[:, None, None]
    cs11 = cauchy_stress[:, 0, 0]
    cs22 = cauchy_stress[:, 1, 1]
    cs12 = cauchy_stress[:, 0, 1]
    cs21 = cauchy_stress[:, 1, 0]
    von_mises = torch.sqrt(cs11*cs11 - cs11*cs22 + cs22*cs22 + 3*cs12*cs21)

    # Calculate divergence of P
    Div_P1 = P11_x + P12_y
    Div_P2 = P21_x + P22_y
    Div_P = torch.sqrt(Div_P1.pow(2) + Div_P2.pow(2))

    # Transform torch tensors back to numpy arrays
    Div_P1 = Div_P1.numpy()
    Div_P2 = Div_P2.numpy()
    Div_P = Div_P.numpy()
    P11 = P11.numpy()
    P12 = P12.numpy()
    P21 = P21.numpy()
    P22 = P22.numpy()
    W = W.numpy()
    von_mises = von_mises.numpy()

    return Div_P1, Div_P2, Div_P, P11, P12, P21, P22, M, W, von_mises




def export_to_paraview(G, u, lambda_, mu_, name=None):
    # Sample G into a mesh
    mymesh = G.extract.faces([32, 32])
    # Vertice coordinates and connectivity
    vertices = mymesh.vertices
    vertices = np.round(vertices, 6)
    connectivity = np.asarray(mymesh.faces)
    # Get parametric coordinates of the vertices
    coords = G.proximities(vertices)
    # Evaluate the displacement at those coordinates (displacement still in 2D)
    displ_2D = u.evaluate(coords)
    # Add a third component to be able to use Paraview's warp by vector feature
    zeros = np.zeros((displ_2D.shape[0],), dtype=displ_2D.dtype)
    displ_3D = np.column_stack([displ_2D, zeros])
    # Calculate more metrics
    Div_P1, Div_P2, Div_P, P11, P12, P21, P22, M, W, von_mises = calculate_metrics(G, u, coords, lambda_, mu_)
    # Create final mesh object and write to vtu file
    cells = [("quad", connectivity)]
    final_geometry = meshio.Mesh(points=vertices, cells=cells,
                                 point_data={
                                     "displacement": displ_3D,
                                     "Div_P1": Div_P1,
                                     "Div_P2": Div_P2,
                                     "Div_P": Div_P,
                                     "P11": P11,
                                     "P12": P12,
                                     "P21": P21,
                                     "P22": P22,
                                     "P": M,
                                     "W": W,
                                     "von_Mises": von_mises})
    if name is not None:
        meshio.write(name+".vtu", final_geometry)
    return final_geometry


