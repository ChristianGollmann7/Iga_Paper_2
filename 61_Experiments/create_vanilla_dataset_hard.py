import time
from pathlib import Path

from validation import generate_random_instance_with_solution

if __name__ == "__main__":
    """
    Generate a bunch of random hard validation instances together with their respective gismo solution.
    Hard means the x, y and poisson range are large and corner points are tested.
    """

    number_instances = 100

    # Path of result (needed to delete files)
    base_dir = Path(__file__).resolve().parent

    start = time.perf_counter()

    i = 0
    while i < number_instances:
        name = f"validation_set_hard/val_{i}.xml"
        successful, G, u_gismo = generate_random_instance_with_solution(name,
                                                                        x_range=(-3.5, -1.5, 1.5, 3.5),
                                                                        y_range=(-1.5, -0.5, 2.0, 5),
                                                                        poisson_range=(0.15, 0.25, 0.3, 0.45),
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