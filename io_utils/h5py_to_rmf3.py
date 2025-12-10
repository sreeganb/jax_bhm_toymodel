import h5py
import numpy as np
import IMP
import IMP.core
import IMP.atom
import IMP.algebra
import IMP.rmf
import RMF
import sys


def convert_simple_mcmc_to_rmf3(hdf5_file, rmf3_file, radius=1.0, color=None):
    """
    Convert simple 2-particle MCMC HDF5 to RMF3 format.
    
    Args:
        hdf5_file: Path to HDF5 file with structure:
            - coordinates: (n_samples, n_particles, 3)
            - log_probabilities: (n_samples,)
        rmf3_file: Output RMF3 file path
        radius: Particle radius (default: 1.0)
        color: IMP.display.Color (default: blue)
    """
    if color is None:
        color = IMP.display.Color(0.2, 0.6, 1.0)  # Blue
    
    # Read HDF5
    with h5py.File(hdf5_file, 'r') as f:
        coords = f['coordinates'][:]  # Shape: (n_samples, n_particles, 3)
        log_probs = f['log_probabilities'][:]
        n_frames, n_particles, _ = coords.shape
        
        print(f"{'='*70}")
        print(f"Converting MCMC trajectory to RMF3")
        print(f"{'='*70}")
        print(f"Input: {hdf5_file}")
        print(f"Output: {rmf3_file}")
        print(f"Frames: {n_frames}")
        print(f"Particles: {n_particles}")
        print(f"{'='*70}\n")
    
    # Create IMP Model
    model = IMP.Model()
    
    # Create root hierarchy
    p_root = IMP.Particle(model)
    root_h = IMP.atom.Hierarchy.setup_particle(p_root)
    p_root.set_name("root")
    
    # Create particles
    particles = []
    for i in range(n_particles):
        p = IMP.Particle(model)
        p.set_name(f"particle_{i}")
        
        # Setup XYZR
        xyzr = IMP.core.XYZR.setup_particle(p)
        coord = coords[0, i]  # Initial coordinates
        xyzr.set_coordinates(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]))
        xyzr.set_radius(radius)
        xyzr.set_coordinates_are_optimized(True)
        
        # Setup mass and color
        IMP.atom.Mass.setup_particle(p, 1.0)
        IMP.display.Colored.setup_particle(p, color)
        
        # Add to hierarchy
        h = IMP.atom.Hierarchy.setup_particle(p)
        root_h.add_child(h)
        particles.append(p)
    
    # Create RMF file
    rmf = RMF.create_rmf_file(rmf3_file)
    rmf.set_description(f"MCMC trajectory: {n_frames} frames, {n_particles} particles")
    
    # Add hierarchy
    IMP.rmf.add_hierarchy(rmf, root_h)
    IMP.rmf.add_restraints(rmf, [])
    
    # Save frames
    print("Writing frames...")
    for frame_idx in range(n_frames):
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames}")
        
        # Update coordinates
        for i, p in enumerate(particles):
            coord = coords[frame_idx, i]
            xyzr = IMP.core.XYZR(p)
            xyzr.set_coordinates(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]))
        
        model.update()
        IMP.rmf.save_frame(rmf, f"frame_{frame_idx}")
    
    rmf.close()
    del rmf
    
    print(f"\n{'='*70}")
    print(f"Conversion complete!")
    print(f"{'='*70}")
    print(f"Saved: {rmf3_file}")
    print(f"\nVisualize with:")
    print(f"  chimerax {rmf3_file}")
    print(f"{'='*70}")


def inspect_hdf5(hdf5_file):
    """Quick inspection of HDF5 file."""
    print(f"{'='*70}")
    print(f"Inspecting: {hdf5_file}")
    print(f"{'='*70}\n")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("Attributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        print("\nDatasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  {key}: {f[key].shape} {f[key].dtype}")
            else:
                print(f"  {key}/  (group)")
        
        if 'log_probabilities' in f:
            log_probs = f['log_probabilities'][:]
            print(f"\nLog Probabilities:")
            print(f"  min: {np.min(log_probs):.2f}")
            print(f"  max: {np.max(log_probs):.2f}")
            print(f"  mean: {np.mean(log_probs):.2f}")
        
        if 'coordinates' in f:
            coords = f['coordinates'][:]
            print(f"\nCoordinates:")
            print(f"  shape: {coords.shape}")
            print(f"  mean distance: {np.mean(np.linalg.norm(coords[:, 1] - coords[:, 0], axis=1)):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert: python convert_simple_mcmc_to_rmf3.py <input.h5> [output.rmf3]")
        print("  Inspect: python convert_simple_mcmc_to_rmf3.py --inspect <input.h5>")
        sys.exit(1)
    
    if sys.argv[1] == "--inspect":
        if len(sys.argv) < 3:
            print("Usage: python convert_simple_mcmc_to_rmf3.py --inspect <input.h5>")
            sys.exit(1)
        inspect_hdf5(sys.argv[2])
    else:
        input_h5 = sys.argv[1]
        output_rmf = sys.argv[2] if len(sys.argv) > 2 else input_h5.replace('.h5', '.rmf3')
        convert_simple_mcmc_to_rmf3(input_h5, output_rmf)