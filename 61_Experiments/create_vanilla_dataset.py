import time
from pathlib import Path

from validation import generate_random_instance_with_solution

if __name__ == "__main__":
    """
    Generate a bunch of random training instances together with their respective gismo solution. 100 samples take 1080 seconds to generate
    """
    number_instances = 100

    # Path of result (needed to delete files)
    base_dir = Path(__file__).resolve().parent

    start = time.perf_counter()

    i = 0
    while i < number_instances:
        name = f"timing_set/val_{i}.xml"
        successful, G, u_gismo = generate_random_instance_with_solution(name, write_to_file=True)
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