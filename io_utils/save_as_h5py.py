#------------------------------------------------------------------------------
# HDF5 I/O utilities for particle states and trajectories
# Part of the Validation stage in Bayesian inference pipeline
#------------------------------------------------------------------------------

import numpy as np
import jax 
from jax import numpy as jnp
from jax import random
import h5py
from datetime import datetime
from typing import Dict, Optional, List, Union, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from representation import ParticleState


def save_smc_to_hdf5(particles: np.ndarray, log_probs: np.ndarray, filename: str, method: str = "Tempered SMC") -> None:
    """Save SMC population (flattened or (n,3) coordinates) to HDF5.

    particles: (n_samples, n_particles*3) or (n_samples, n_particles, 3)
    log_probs: (n_samples,)
    """
    particles_np = np.asarray(particles)
    log_probs_np = np.asarray(log_probs)

    if particles_np.ndim == 2:
        n_samples = particles_np.shape[0]
        n_particles = particles_np.shape[1] // 3
        coords = particles_np.reshape(n_samples, n_particles, 3)
    elif particles_np.ndim == 3 and particles_np.shape[2] == 3:
        coords = particles_np
        n_samples, n_particles, _ = coords.shape
    else:
        raise ValueError("particles must be (n_samples, n_particles*3) or (n_samples, n_particles, 3)")

    if log_probs_np.shape[0] != n_samples:
        raise ValueError("log_probs length must match n_samples")

    best_idx = int(np.argmax(log_probs_np))

    with h5py.File(filename, 'w') as f:
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['n_particles'] = n_particles
        f.attrs['method'] = method

        f.create_dataset('coordinates', data=coords, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs_np, compression='gzip')

        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = best_idx
        best_grp.attrs['log_probability'] = float(log_probs_np[best_idx])
        best_grp.create_dataset('coordinates', data=coords[best_idx])

    print(f"Saved to: {filename}")
    print(f"  Population size: {n_samples}")
    print(f"  Best log_prob: {float(np.max(log_probs_np)):.2f}")


def save_state(filename: str, state: Dict) -> None:
    """Save a generic state dictionary to HDF5."""
    with h5py.File(filename, 'w') as f:
        for key, value in state.items():
            f.create_dataset(key, data=np.array(value))
    print(f"State saved to {filename}")


def save_trajectory(filename: str, trajectory: Dict) -> None:
    """Save a trajectory dictionary to HDF5."""
    with h5py.File(filename, 'w') as f:
        for key, value in trajectory.items():
            f.create_dataset(key, data=np.array(value))
    print(f"Trajectory saved to {filename}")


def save_particle_state(
    filename: str,
    state: 'ParticleState',
    metadata: Optional[Dict] = None
) -> None:
    """
    Save a ParticleState to HDF5 with full attribute preservation.
    
    Stores all particle attributes (positions, radii, types, copy_numbers)
    in a standardized format that can be fully reconstructed.
    
    Args:
        filename: Output HDF5 file path
        state: ParticleState object to save
        metadata: Optional dict of additional metadata to store as attributes
        
    File structure:
        /coordinates        (N, 3) particle positions
        /radii              (N,) particle radii
        /particle_types     (N,) integer type labels
        /copy_numbers       (N,) copy numbers
        /type_names         JSON string mapping type_id -> name
        attrs:
            creation_date, n_particles, + custom metadata
    """
    with h5py.File(filename, 'w') as f:
        # File-level metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_particles'] = state.n_particles
        f.attrs['format_version'] = '1.0'
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
        
        # Core datasets
        f.create_dataset('coordinates', data=np.array(state.positions), compression='gzip')
        f.create_dataset('radii', data=np.array(state.radii), compression='gzip')
        f.create_dataset('particle_types', data=np.array(state.particle_types), compression='gzip')
        f.create_dataset('copy_numbers', data=np.array(state.copy_numbers), compression='gzip')
        
        # Type name mapping (stored as JSON string)
        type_names_str = json.dumps({str(k): v for k, v in state.type_names.items()})
        f.attrs['type_names'] = type_names_str
        
        # Optional particle names
        if state.names is not None:
            dt = h5py.special_dtype(vlen=str)
            names_ds = f.create_dataset('particle_names', (len(state.names),), dtype=dt)
            for i, name in enumerate(state.names):
                names_ds[i] = name
    
    print(f"ParticleState saved to {filename}")
    print(f"  Particles: {state.n_particles}")
    print(f"  Types: {list(state.type_names.values())}")


def load_particle_state(filename: str) -> 'ParticleState':
    """
    Load a ParticleState from HDF5.
    
    Args:
        filename: Input HDF5 file path
        
    Returns:
        Reconstructed ParticleState object
    """
    from representation import ParticleState
    
    with h5py.File(filename, 'r') as f:
        positions = jnp.array(f['coordinates'][:])
        radii = jnp.array(f['radii'][:])
        particle_types = jnp.array(f['particle_types'][:])
        copy_numbers = jnp.array(f['copy_numbers'][:])
        
        # Parse type names from JSON
        type_names_str = f.attrs.get('type_names', '{}')
        type_names = {int(k): v for k, v in json.loads(type_names_str).items()}
        
        # Optional particle names
        names = None
        if 'particle_names' in f:
            names = [f['particle_names'][i].decode() if isinstance(f['particle_names'][i], bytes) 
                    else f['particle_names'][i] for i in range(len(f['particle_names']))]
    
    return ParticleState(
        positions=positions,
        radii=radii,
        particle_types=particle_types,
        copy_numbers=copy_numbers,
        names=names,
        type_names=type_names
    )


def save_particle_trajectory(
    filename: str,
    states: List['ParticleState'],
    log_probs: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save a trajectory of ParticleStates to HDF5.
    
    All frames must have the same particle count and attributes.
    Only positions vary across frames.
    
    Args:
        filename: Output HDF5 file path
        states: List of ParticleState objects (one per frame)
        log_probs: Optional (n_frames,) array of log probabilities
        metadata: Optional dict of additional metadata
        
    File structure:
        /coordinates        (n_frames, N, 3) positions per frame
        /radii              (N,) particle radii (shared)
        /particle_types     (N,) type labels (shared)
        /copy_numbers       (N,) copy numbers (shared)
        /log_probabilities  (n_frames,) if provided
        /best_configuration group with best frame data
    """
    if not states:
        raise ValueError("states list cannot be empty")
    
    n_frames = len(states)
    reference = states[0]
    n_particles = reference.n_particles
    
    # Stack all positions
    all_positions = np.stack([np.array(s.positions) for s in states], axis=0)
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_frames'] = n_frames
        f.attrs['n_particles'] = n_particles
        f.attrs['format_version'] = '1.0'
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
        
        # Trajectory data
        f.create_dataset('coordinates', data=all_positions, compression='gzip')
        f.create_dataset('radii', data=np.array(reference.radii), compression='gzip')
        f.create_dataset('particle_types', data=np.array(reference.particle_types), compression='gzip')
        f.create_dataset('copy_numbers', data=np.array(reference.copy_numbers), compression='gzip')
        
        # Type names
        type_names_str = json.dumps({str(k): v for k, v in reference.type_names.items()})
        f.attrs['type_names'] = type_names_str
        
        # Optional particle names
        if reference.names is not None:
            dt = h5py.special_dtype(vlen=str)
            names_ds = f.create_dataset('particle_names', (len(reference.names),), dtype=dt)
            for i, name in enumerate(reference.names):
                names_ds[i] = name
        
        # Log probabilities
        if log_probs is not None:
            f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
            
            # Best configuration
            best_idx = int(np.argmax(log_probs))
            best_grp = f.create_group('best_configuration')
            best_grp.attrs['frame_index'] = best_idx
            best_grp.attrs['log_probability'] = float(log_probs[best_idx])
            best_grp.create_dataset('coordinates', data=all_positions[best_idx])
    
    print(f"Trajectory saved to {filename}")
    print(f"  Frames: {n_frames}")
    print(f"  Particles: {n_particles}")
    if log_probs is not None:
        print(f"  Best log_prob: {np.max(log_probs):.2f} (frame {np.argmax(log_probs)})")


def load_particle_trajectory(filename: str) -> tuple:
    """
    Load a trajectory from HDF5.
    
    Args:
        filename: Input HDF5 file path
        
    Returns:
        Tuple of (list of ParticleState, log_probs array or None)
    """
    from representation import ParticleState
    
    with h5py.File(filename, 'r') as f:
        all_positions = f['coordinates'][:]
        radii = jnp.array(f['radii'][:])
        particle_types = jnp.array(f['particle_types'][:])
        copy_numbers = jnp.array(f['copy_numbers'][:])
        
        type_names_str = f.attrs.get('type_names', '{}')
        type_names = {int(k): v for k, v in json.loads(type_names_str).items()}
        
        names = None
        if 'particle_names' in f:
            names = [f['particle_names'][i].decode() if isinstance(f['particle_names'][i], bytes)
                    else f['particle_names'][i] for i in range(len(f['particle_names']))]
        
        log_probs = None
        if 'log_probabilities' in f:
            log_probs = f['log_probabilities'][:]
    
    # Reconstruct states
    states = []
    for i in range(all_positions.shape[0]):
        states.append(ParticleState(
            positions=jnp.array(all_positions[i]),
            radii=radii,
            particle_types=particle_types,
            copy_numbers=copy_numbers,
            names=names,
            type_names=type_names
        ))
    
    return states, log_probs