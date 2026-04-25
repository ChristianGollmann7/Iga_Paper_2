import torch
import yaml
import subprocess
from pathlib import Path, PurePosixPath
import sys
import splinepy
import numpy as np
from numbers import Real


from helper_functions import load_Bspline

torch.set_default_dtype(torch.float64)


def calculate_gismo_solution(input_file: str,
                             delta_x: Real,
                             delta_y: Real,
                             poisson: Real,
                             write_to_file: bool=True) -> tuple[bool, splinepy.BSpline | None]:
    """
    Creates a reference solution using Gismo.
    Returns the solution as Bspline object. Saves this solution to disk under "displacement_gismo.xml".
    For this to work, the files Experiment_4_nonlinear (binary) and libgismo.so.25.10 (compiled library) must be there.
    File paths inside this function must be adapted manually accordingly.
    """
    def run_gismo_binary():
        """
        Calls the compiled Gismo binary executable. Differentiates between Linux and Windows platform.
        Adapt file paths accordingly.
        The Windows version uses WSL, so this setup is needed with Ubuntu.
        """
        if sys.platform.startswith('linux'):
            folder_path = "/home/chg/Programming/IGN_neoHook/60_Experiment_1/"
            executable = f"{folder_path}/Experiment_4_nonlinear"
            geometry = f"{folder_path}/" + input_file

            # Run the binary
            result = subprocess.run([executable, "-f", geometry, "-x", str(delta_x), "-y", str(delta_y), "-p", str(poisson),
                            "-A", "2", "-B", "2", "-E", "1", "-P", folder_path, "-R", "0.1"],
                                    check=True, text=True, capture_output=True, cwd=folder_path)
        else:
            folder_path = PurePosixPath("/mnt/c/Users/ChristianGollmann/Programming/IGN_neoHook/60_Experiment_1")
            executable = folder_path / "Experiment_4_nonlinear"
            geometry = folder_path / input_file
            output_path = (str(folder_path) + "/")

            cmd = (f"{executable} -f {geometry} -x {delta_x} -y {delta_y} -p {poisson} -A 2 -B 2 -E 1 -P {output_path} -R 0.1")

            # Run the binary
            result = subprocess.run(["wsl", "--cd", str(folder_path), "bash", "-lc",  cmd], check=True, text=True, capture_output=True)


        cmd_output = result.stdout + result.stderr
        if "Invalid configuration: J < 0" in cmd_output:
            print(f"Gismo solution did not converge")
            return False
        else:
            return True

    success = run_gismo_binary()
    if success:
        u = load_Bspline("displacement_gismo.xml")
        # Export the gismo solution
        if write_to_file:
            splinepy.io.gismo.export(input_file[0:-4]+f"_gismo_{delta_x}_{delta_y}_{poisson}.xml", u)
        return True, u
    else:
        return False, None


if __name__ == "__main__":
    input_file = "input_geometries/simple_5.xml"
    delta_x, delta_y = 1, 2
    poisson = 0.45

    success, u = calculate_gismo_solution(input_file, 1, 2, poisson)