import numpy as np
import splinepy
import random
import copy
from numbers import Real
from pathlib import Path
import re

from random_geometry import random_geometry
from reference_solution import calculate_gismo_solution
from neural_network import Pinn
from preparation import specify_dirichlet_boundaries_with_single_value
from helper_functions import load_Bspline
from postprocessing import calculate_cartesian_difference

# Define the Boundary Condition type
BC = dict[str, dict[int, list[Real]]]



def generate_random_instance_with_solution(name: str,
                                           x_range: tuple[float, float] | tuple[float, float, float, float] =(-3.5, 3.5),
                                           y_range: tuple[float, float] | tuple[float, float, float, float] =(-1.5, 5),
                                           poisson_range: tuple[float, float] | tuple[float, float, float, float]=(0.15, 0.45),
                                           write_to_file: bool=True):
    """
    Generates a random geometry with a random set of Dirichlet conditions and random poisson value.
    Saves the instance to file.
    Performs a gismo reference solution of the instance and saves result Spline with extension "_gismo".

    :param template_bc: template boundary conditions
    :param name: path + name of instance
    """
    def sample_outside(s: tuple[float, float, float, float]) -> Real:
        """
        Draws a sample from a double range, excluding the middle range.
        """
        a = np.abs(s[0] - s[1])
        b = np.abs(s[2] - s[3])
        which = np.random.choice([0, 1], p=[a/(a+b), 1-a/(a+b)])
        if which == 0:
            return random.uniform(*(s[0], s[1]))
        else:
            return random.uniform(*(s[2], s[3]))

    # Generate random geometry
    G = random_geometry()
    splinepy.io.gismo.export(name, G)

    # Generate random delta_x, delta_y and poisson value
    if len(x_range)==2:
        delta_x = random.uniform(*x_range)
    else:
        delta_x = sample_outside(x_range)
    if len(y_range)==2:
        delta_y = random.uniform(*y_range)
    else:
        delta_y = sample_outside(y_range)
    if len(poisson_range) == 2:
        poisson = random.uniform(*poisson_range)
    else:
        poisson = sample_outside(poisson_range)

    # Calculate Gismo solution
    successful, u_gismo = calculate_gismo_solution(name, delta_x, delta_y, poisson, write_to_file=write_to_file)
    return successful, G, u_gismo


def evaluate_validation_samples(validation_pinn: Pinn,
                                dirichlet_boundary_template: BC,
                                neumann_boundary_template: BC,
                                validation_folder: str | Path,
                                return_value: str='max') -> Real | tuple[Real, Real]:
    """
    Goes through the validation set and evaluates them. Base geometry from validation set is put through the validation_pinn
    together with respective boundary conditions which are retrieved from the name of the gismo solution.

    Returns the average of the max error of each validation sample.
    """
    max_error_list = []
    mean_error_list = []

    # Get all the samples from the validation set
    instances = retrieve_validation_samples(validation_folder)
    for sample in instances:
        instance = sample[0]
        solution = sample[1]
        delta_x, delta_y, poisson = sample[2][0], sample[2][1], sample[2][2]
        dirichlet = specify_dirichlet_boundaries_with_single_value(dirichlet_boundary_template, ['b'], [delta_x], [delta_y])
        # Load base spline
        G = load_Bspline(validation_folder + "/" + instance)
        # Predict network solution
        u_net = validation_pinn.evaluate(G, dirichlet, neumann_boundary_template, poisson)
        S_net = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points + u_net.control_points)
        # Get the gismo solution
        u_gismo = load_Bspline(validation_folder + "/" + solution)
        S_gismo = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points + u_gismo.control_points)
        # Calculate the max error
        error = calculate_cartesian_difference(S_gismo, S_net)
        max_error_list.append(np.max(error))
        mean_error_list.append(np.mean(error))

    # Return average
    if return_value == 'max':
        return np.mean(max_error_list)
    elif return_value == 'mean':
        return np.mean(mean_error_list)
    elif return_value == 'both':
        return np.mean(max_error_list), np.mean(mean_error_list)




def retrieve_validation_samples(folder: str | Path,
                                n: int=99999999) -> list[tuple[str, str, list[Real]]]:
    """
    Retrieves validation instances from a folder. Folder must contain only pairs of base geometries and gismo solutions.

    Returns a list of (base name, solution_name, [delta_x, delta_y, poisson])
    """
    folder = Path(folder)
    results = []

    # Regex search pattern
    regex_pattern = re.compile(r'^(?P<base>val_\d{1,3})_gismo_(?P<numbers>.+)\.xml$')

    counter = 0
    for file in folder.iterdir():
        compare = regex_pattern.match(file.name)
        if not compare:
            continue
        base_name = compare.group("base")
        conditions = compare.group("numbers")

        # Extract the numbers
        conditions_strings = conditions.split("_")
        conditions_numbers = [float(f) for f in conditions_strings]

        base = base_name + ".xml"
        gismo_name = base_name + f"_gismo_{conditions_numbers[0]}_{conditions_numbers[1]}_{conditions_numbers[2]}.xml"

        results.append((base, gismo_name, conditions_numbers))
        counter += 1
        if counter >= n:
            break

    return results



if __name__ == "__main__":
    """
    Generate a bunch of random validation instances together with their respective gismo solution.
    """
    number_instances = 100

    i = 0
    while i < number_instances:
        name = f"validation_set/val_{i}.xml"
        successful = generate_random_instance_with_solution(name)
        if successful:
            print(f"\nSuccessfully generated {name}")
            i += 1
        else:
            print(f"\nFailed to generate {name}")
