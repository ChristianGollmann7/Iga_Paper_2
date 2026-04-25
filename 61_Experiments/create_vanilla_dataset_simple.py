import time
from pathlib import Path

from validation import generate_random_instance_with_solution

if __name__ == "__main__":
    """
    Generate a bunch of random simple training instances together with their respective gismo solution.
    Simple means the x, y and poisson range are small.
    """
    number_instances = 1000

    # Path of result (needed to delete files)
    base_dir = Path(__file__).resolve().parent

    start = time.perf_counter()

    i = 0
    while i < number_instances:
        name = f"vanilla_training_set_simple/val_{i}.xml"
        successful, G, u_gismo = generate_random_instance_with_solution(name,
                                                                        x_range=(-1.5, 1.5),
                                                                        y_range=(-0.5, 2.0),
                                                                        poisson_range=(0.25, 0.3),
                                                                        write_to_file=True)
        if successful:
            print(f"\nSuccessfully generated {name}")
            i += 1
        else:
            print(f"\nFailed to generate {name}")
            # Delete the created base geometry
            output_path = base_dir / name
            output_path.unlink(missing_ok=True)

    end = time.perf_counter()

    print(f"\nTotal time: {end - start} seconds")


    """
    Simple validation samples
    """
    number_instances = 100

    # Path of result (needed to delete files)
    base_dir = Path(__file__).resolve().parent

    start = time.perf_counter()

    i = 0
    while i < number_instances:
        name = f"validation_set_simple/val_{i}.xml"
        successful, G, u_gismo = generate_random_instance_with_solution(name,
                                                                        x_range=(-1.5, 1.5),
                                                                        y_range=(-0.5, 2.0),
                                                                        poisson_range=(0.25, 0.3),
                                                                        write_to_file=True)
        if successful:
            print(f"\nSuccessfully generated {name}")
            i += 1
        else:
            print(f"\nFailed to generate {name}")
            # Delete the created base geometry
            output_path = base_dir / name
            output_path.unlink(missing_ok=True)

    end = time.perf_counter()

    print(f"\nTotal time: {end - start} seconds")