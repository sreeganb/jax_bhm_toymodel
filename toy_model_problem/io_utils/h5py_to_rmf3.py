"""
Convert HDF5 state/trajectory files to RMF3 for visualization.

This script handles both single-frame files (from save_state) and
multi-frame trajectory files (from save_trajectory).

Command-line usage:
  # Convert a trajectory
  python h5py_to_rmf3.py mcmc_trajectory.h5 trajectory.rmf3

  # Convert a single state
  python h5py_to_rmf3.py initial_state.h5 initial.rmf3

  # Inspect an HDF5 file's structure
  python h5py_to_rmf3.py --inspect mcmc_trajectory.h5
"""
import h5py
import numpy as np
import sys
from pathlib import Path

# IMP imports are optional. The script will load, but conversion will fail
# if IMP is not installed, with a helpful error message.
try:
    import IMP
    import IMP.core
    import IMP.atom
    import IMP.algebra
    import IMP.rmf
    import RMF
    IMP_AVAILABLE = True
except ImportError:
    IMP_AVAILABLE = False
    IMP, RMF = None, None


def convert_to_rmf3(
    hdf5_file: str,
    rmf3_file: str,
    color_map: dict = None,
):
    """
    Convert an HDF5 state or trajectory file to RMF3 format.

    Reads HDF5 files created by `save_state` or `save_trajectory`.

    Args:
        hdf5_file: Path to the input HDF5 file.
        rmf3_file: Path for the output RMF3 file.
        color_map: Optional dictionary mapping type names to RGB tuples.
                   Example: {'ProteinA': (0.2, 0.6, 1.0), 'ProteinB': (0.9, 0.4, 0.2)}
    """
    if not IMP_AVAILABLE:
        raise ImportError(
            "IMP is not installed, which is required for RMF3 conversion.\n"
            "On Linux: conda install -c salilab imp\n"
            "On Mac: IMP must be installed from source or via specific package managers.\n"
            "See: https://integrativemodeling.org/download.html"
        )

    print(f"Converting '{hdf5_file}' to '{rmf3_file}'...")

    with h5py.File(hdf5_file, 'r') as f:
        # Load required datasets
        coords = f['coordinates'][:]
        radii = f['radius'][:]
        
        # Handle both single frame (2D) and trajectory (3D)
        if coords.ndim == 2:
            coords = coords[None, ...]  # Add a frame dimension
        
        n_frames, n_particles, _ = coords.shape

        # Load optional data
        log_probs = f['log_probabilities'][:] if 'log_probabilities' in f else None
        particle_types = f['particle_types'][:] if 'particle_types' in f else None
        
        type_names_attr = f.attrs.get('type_names', None)
        type_names = tuple(type_names_attr.split(',')) if type_names_attr else None

    # --- Create IMP Model and Hierarchy ---
    model = IMP.Model()
    root_h = IMP.atom.Hierarchy.setup_particle(IMP.Particle(model, "root"))

    # Define default colors
    def get_default_color(type_index: int) -> IMP.display.Color:
        palette = [
            (0.2, 0.6, 1.0),  # Blue
            (0.9, 0.4, 0.2),  # Orange
            (0.3, 0.8, 0.4),  # Green
            (0.8, 0.6, 0.2),  # Yellow-Orange
            (0.6, 0.4, 0.8),  # Purple
        ]
        r, g, b = palette[type_index % len(palette)]
        return IMP.display.Color(r, g, b)

    # Create particles
    particles = []
    for i in range(n_particles):
        p = IMP.Particle(model)
        
        type_id = int(particle_types[i]) if particle_types is not None else 0
        type_name = type_names[type_id] if type_names else f"type_{type_id}"
        
        p.set_name(f"{type_name}_{i}")

        # Setup geometry and display properties
        xyzr = IMP.core.XYZR.setup_particle(p)
        xyzr.set_coordinates(IMP.algebra.Vector3D(*coords[0, i]))
        xyzr.set_radius(float(radii[i]))
        xyzr.set_coordinates_are_optimized(True)
        
        mass = float(radii[i]) ** 3
        IMP.atom.Mass.setup_particle(p, mass)
                
        # Set color
        color = None
        if color_map and type_name in color_map:
            r, g, b = color_map[type_name]
            color = IMP.display.Color(float(r), float(g), float(b))
        else:
            color = get_default_color(type_id)
        
        IMP.display.Colored.setup_particle(p, color)
        
        # Add to hierarchy
        h = IMP.atom.Hierarchy.setup_particle(p)
        root_h.add_child(h)
        particles.append(p)

    # --- Write to RMF3 File ---
    rmf = RMF.create_rmf_file(rmf3_file)
    rmf.set_description(f"Trajectory from {Path(hdf5_file).name}")
    IMP.rmf.add_hierarchy(rmf, root_h)
    IMP.rmf.add_restraints(rmf, [])

    print(f"Writing {n_frames} frames...")
    for frame_idx in range(n_frames):
        if frame_idx > 0 and frame_idx % 100 == 0:
            print(f"  ...frame {frame_idx}/{n_frames}")

        # Update coordinates for all particles in the current frame
        for i, p in enumerate(particles):
            IMP.core.XYZR(p).set_coordinates(IMP.algebra.Vector3D(*coords[frame_idx, i]))
        
        model.update()
        frame_name = f"frame {frame_idx}"
        if log_probs is not None:
            frame_name += f" (logp={log_probs[frame_idx]:.2f})"
        
        IMP.rmf.save_frame(rmf, frame_name)

    del rmf  # Close the file
    print(f"\n✅ Conversion complete. Saved to: {rmf3_file}")


def inspect_hdf5(hdf5_file: str):
    """Prints a summary of the contents of an HDF5 file."""
    print(f"\n--- Inspecting: {hdf5_file} ---")
    if not Path(hdf5_file).exists():
        print("File not found.")
        return

    with h5py.File(hdf5_file, 'r') as f:
        print("\n[Attributes]")
        for key, val in f.attrs.items():
            print(f"  - {key}: {val}")
        
        print("\n[Datasets]")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dset = f[key]
                print(f"  - {key}: shape={dset.shape}, dtype={dset.dtype}")
            else:
                print(f"  - {key}/ (Group)")
        
        if 'log_probabilities' in f:
            log_probs = f['log_probabilities'][:]
            print("\n[Log Probabilities Summary]")
            print(f"  - Min: {np.min(log_probs):.3f}")
            print(f"  - Max: {np.max(log_probs):.3f}")
            print(f"  - Mean: {np.mean(log_probs):.3f}")
    print("--- End of Inspection ---\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert:  python h5py_to_rmf3.py <input.h5> [output.rmf3]")
        print("  Inspect:  python h5py_to_rmf3.py --inspect <input.h5>")
        sys.exit(1)
    
    if sys.argv[1] == "--inspect":
        if len(sys.argv) < 3:
            print("Usage: python h5py_to_rmf3.py --inspect <input.h5>")
            sys.exit(1)
        inspect_hdf5(sys.argv[2])
    else:
        input_h5 = sys.argv[1]
        output_rmf = sys.argv[2] if len(sys.argv) > 2 else Path(input_h5).with_suffix('.rmf3').name
        
        # Define a color map for your proteins
        protein_color_map = {
            'ProteinA': (0.2, 0.6, 1.0),  # Blue
            'ProteinB': (0.9, 0.4, 0.2),  # Orange
            'ProteinC': (0.3, 0.8, 0.4),  # Green
        }
        
        try:
            convert_to_rmf3(input_h5, output_rmf, color_map=protein_color_map)
        except ImportError as e:
            print(f"\nERROR: Could not convert file.")
            print(e)
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("You can inspect the HDF5 file structure with:")
            print(f"  python h5py_to_rmf3.py --inspect {input_h5}")
            sys.exit(1)