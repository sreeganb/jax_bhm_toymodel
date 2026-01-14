import h5py
import numpy as np
import IMP
import IMP.core
import IMP.atom
import IMP.algebra
import IMP.rmf
import RMF
import sys
import json


def convert_simple_mcmc_to_rmf3(
    hdf5_file,
    rmf3_file,
    radius: float = 1.0,
    color=None,
    color_map=None,
):
    """
    Convert an HDF5 trajectory to RMF3, supporting per-particle radii and colors.

    Expected HDF5 layout (as written by save_as_h5py or save_particle_trajectory):
      - coordinates: (n_frames, n_particles, 3)  OR (n_particles, 3) single frame
      - log_probabilities: (n_frames,)  [optional]
      - radii: (n_particles,)           [optional]
      - particle_types: (n_particles,)  [optional]
      - attrs.type_names: JSON mapping type_id -> name  [optional]

    Args:
        hdf5_file: input HDF5
        rmf3_file: output RMF3
        radius: fallback scalar radius if /radii missing
        color: fallback IMP.display.Color if no types or color_map provided
        color_map: optional dict mapping type name or type id to RGB tuple (0-1 floats)
    """

    if color is None:
        color = IMP.display.Color(0.2, 0.6, 1.0)  # default blue

    with h5py.File(hdf5_file, 'r') as f:
        coords = f['coordinates'][:]
        if coords.ndim == 2:
            coords = coords[None, ...]  # promote single frame
        n_frames, n_particles, _ = coords.shape

        log_probs = f['log_probabilities'][:] if 'log_probabilities' in f else None
        radii = f['radii'][:] if 'radii' in f else np.full((n_particles,), radius, dtype=float)

        particle_types = f['particle_types'][:] if 'particle_types' in f else None
        type_names_attr = f.attrs.get('type_names', None)
        type_names = None
        if type_names_attr is not None:
            try:
                type_names = {int(k): v for k, v in json.loads(type_names_attr).items()}
            except Exception:
                type_names = None

    # Build color lookup per particle
    if particle_types is not None:
        def default_palette(i: int) -> IMP.display.Color:
            palette = [
                (0.2, 0.6, 1.0),
                (0.9, 0.4, 0.2),
                (0.3, 0.8, 0.4),
                (0.8, 0.6, 0.2),
                (0.6, 0.4, 0.8),
                (0.2, 0.8, 0.8),
            ]
            r, g, b = palette[i % len(palette)]
            return IMP.display.Color(r, g, b)

        def color_for_type(tid: int) -> IMP.display.Color:
            # color_map can be keyed by id or name
            if color_map is not None:
                if isinstance(color_map, dict):
                    if tid in color_map:
                        r, g, b = color_map[tid]
                        return IMP.display.Color(float(r), float(g), float(b))
                    if type_names and tid in type_names and type_names[tid] in color_map:
                        r, g, b = color_map[type_names[tid]]
                        return IMP.display.Color(float(r), float(g), float(b))
            return default_palette(tid)

        particle_colors = [color_for_type(int(t)) for t in particle_types]
    else:
        particle_colors = [color for _ in range(n_particles)]

    # Create IMP model/hierarchy
    model = IMP.Model()
    p_root = IMP.Particle(model)
    root_h = IMP.atom.Hierarchy.setup_particle(p_root)
    p_root.set_name("root")

    particles = []
    for i in range(n_particles):
        p = IMP.Particle(model)
        # Name with type if available
        if particle_types is not None and type_names is not None:
            tname = type_names.get(int(particle_types[i]), f"type_{int(particle_types[i])}")
            pname = f"{tname}_{i}"
        elif particle_types is not None:
            pname = f"type{int(particle_types[i])}_{i}"
        else:
            pname = f"particle_{i}"
        p.set_name(pname)

        xyzr = IMP.core.XYZR.setup_particle(p)
        coord0 = coords[0, i]
        xyzr.set_coordinates(IMP.algebra.Vector3D(coord0[0], coord0[1], coord0[2]))
        xyzr.set_radius(float(radii[i]))
        xyzr.set_coordinates_are_optimized(True)

        IMP.atom.Mass.setup_particle(p, 1.0)
        IMP.display.Colored.setup_particle(p, particle_colors[i])

        h = IMP.atom.Hierarchy.setup_particle(p)
        root_h.add_child(h)
        particles.append(p)

    rmf = RMF.create_rmf_file(rmf3_file)
    desc = f"Trajectory: {n_frames} frames, {n_particles} particles"
    if log_probs is not None:
        desc += f", logp range [{np.min(log_probs):.2f}, {np.max(log_probs):.2f}]"
    rmf.set_description(desc)

    IMP.rmf.add_hierarchy(rmf, root_h)
    IMP.rmf.add_restraints(rmf, [])

    print("Writing frames...")
    for frame_idx in range(n_frames):
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames}")

        for i, p in enumerate(particles):
            coord = coords[frame_idx, i]
            xyzr = IMP.core.XYZR(p)
            xyzr.set_coordinates(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]))

        model.update()
        IMP.rmf.save_frame(rmf, f"frame_{frame_idx}")

    rmf.close()
    del rmf

    print(f"\n{'='*70}")
    print("Conversion complete!")
    print(f"Saved: {rmf3_file}")
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