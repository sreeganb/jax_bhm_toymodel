import sys
from pathlib import Path

# Make repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the converter - note: IMP is optional
try:
    from io_utils.h5py_to_rmf3 import convert_simple_mcmc_to_rmf3, IMP_AVAILABLE
except ImportError:
    IMP_AVAILABLE = False

color_map = {
    "A": (0.2, 0.6, 1.0),  # blue
    "B": (0.9, 0.4, 0.2),  # orange
    "C": (0.3, 0.8, 0.4),  # green
}

# input the h5 file name and the output rmf3 file name from terminal as user arguments

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/convert_traj.py <input.h5> <output.rmf3>")
        sys.exit(1)

    if not IMP_AVAILABLE:
        print("ERROR: IMP is not installed. RMF3 conversion requires IMP.")
        print("On Linux: conda install -c salilab imp")
        print("On Mac: See https://integrativemodeling.org/download.html")
        sys.exit(1)

    input_h5 = sys.argv[1]
    output_rmf3 = sys.argv[2]

    convert_simple_mcmc_to_rmf3(input_h5, output_rmf3, color_map=color_map)