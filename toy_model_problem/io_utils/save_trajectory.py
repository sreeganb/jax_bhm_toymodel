"""
Save and load MCMC trajectories (multiple State frames) to/from HDF5.
"""
import h5py
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from representation.state import State


def save_trajectory(states, filename, log_probs=None, metadata=None):
    """
    Save trajectory of States to HDF5.
    
    Args:
        states: List of State objects, one per frame
        filename: Output .h5 file path
        log_probs: Optional array of log probabilities, shape (n_frames,)
        metadata: Optional dict of metadata
    """
    n_frames = len(states)
    n_particles = states[0].n_particles
    
    # Validate all states have same structure
    for i, state in enumerate(states):
        if state.n_particles != n_particles:
            raise ValueError(f"Frame {i} has {state.n_particles} particles, expected {n_particles}")
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_frames'] = n_frames
        f.attrs['n_particles'] = n_particles
        f.attrs['n_types'] = states[0].n_types
        
        if states[0].type_names is not None:
            f.attrs['type_names'] = ','.join(states[0].type_names)
        
        # Stack coordinates into (n_frames, n_particles, 3)
        coords_stack = np.stack([np.array(s.coordinates) for s in states], axis=0)
        
        # Save with compression and chunking for efficient access
        f.create_dataset(
            'coordinates',
            data=coords_stack,
            compression='gzip',
            chunks=(1, n_particles, 3)  # Chunk by frame for sequential access
        )
        
        # Radius and copy_number are constant across frames
        f.create_dataset('radius', data=np.array(states[0].radius), compression='gzip')
        f.create_dataset('copy_number', data=np.array(states[0].copy_number))
        
        # --- FIX: Explicitly create and save particle_types ---
        if states[0].particle_types is None:
            # Generate from copy_number if not present
            type_indices = []
            for type_idx, count in enumerate(states[0].copy_number):
                type_indices.extend([type_idx] * int(count))
            particle_types_data = np.array(type_indices)
        else:
            particle_types_data = np.array(states[0].particle_types)
        
        f.create_dataset('particle_types', data=particle_types_data)
        # --- END FIX ---
                
        if states[0].rigid_body_ids is not None:
            f.create_dataset('rigid_body_ids', data=np.array(states[0].rigid_body_ids))
        
        # Log probabilities (if provided)
        if log_probs is not None:
            f.create_dataset('log_probabilities', data=np.array(log_probs), compression='gzip')
        
        # Additional metadata
        if metadata:
            meta_grp = f.create_group('metadata')
            for key, val in metadata.items():
                try:
                    meta_grp.attrs[key] = val
                except TypeError:
                    meta_grp.attrs[key] = str(val)
        
        # Save best frame (highest log_prob)
        if log_probs is not None:
            best_idx = int(np.argmax(log_probs))
            best_grp = f.create_group('best_frame')
            best_grp.attrs['frame_index'] = best_idx
            best_grp.attrs['log_probability'] = float(log_probs[best_idx])
            best_grp.create_dataset('coordinates', data=coords_stack[best_idx])
    
    print(f"Saved trajectory: {filename}")
    print(f"  Frames: {n_frames}")
    print(f"  Particles: {n_particles}")
    if log_probs is not None:
        print(f"  Best log_prob: {np.max(log_probs):.3f} (frame {np.argmax(log_probs)})")


def load_trajectory(filename, frame_indices=None):
    """
    Load trajectory from HDF5.
    
    Args:
        filename: Input .h5 file path
        frame_indices: Optional list/array of frame indices to load.
                      If None, loads all frames.
    
    Returns:
        states: List of State objects
        log_probs: Array of log probabilities (or None)
        metadata: Dict of metadata
    """
    with h5py.File(filename, 'r') as f:
        n_frames = f.attrs['n_frames']
        
        # Load metadata
        type_names = None
        if 'type_names' in f.attrs:
            type_names = tuple(f.attrs['type_names'].split(','))
        
        metadata = {}
        if 'metadata' in f:
            for key in f['metadata'].attrs:
                metadata[key] = f['metadata'].attrs[key]
        
        # Determine which frames to load
        if frame_indices is None:
            frame_indices = range(n_frames)
        else:
            frame_indices = np.asarray(frame_indices)
        
        # Load static arrays (same for all frames)
        radius = jnp.array(f['radius'][:])
        copy_number = jnp.array(f['copy_number'][:])
        particle_types = jnp.array(f['particle_types'][:]) if 'particle_types' in f else None
        rigid_body_ids = jnp.array(f['rigid_body_ids'][:]) if 'rigid_body_ids' in f else None
        
        # Load coordinates for selected frames
        coords_data = f['coordinates']
        states = []
        
        for idx in frame_indices:
            coords = jnp.array(coords_data[idx])
            state = State(
                coordinates=coords,
                radius=radius,
                copy_number=copy_number,
                particle_types=particle_types,
                rigid_body_ids=rigid_body_ids,
                type_names=type_names,
            )
            states.append(state)
        
        # Load log probabilities
        log_probs = None
        if 'log_probabilities' in f:
            if frame_indices is None:
                log_probs = jnp.array(f['log_probabilities'][:])
            else:
                log_probs = jnp.array(f['log_probabilities'][:][frame_indices])
    
    print(f"Loaded trajectory: {filename}")
    print(f"  Loaded {len(states)} frames")
    
    return states, log_probs, metadata


def load_best_frame(filename):
    """
    Load only the best frame from trajectory.
    
    Returns:
        state: State object for best frame
        log_prob: Log probability of best frame
    """
    with h5py.File(filename, 'r') as f:
        if 'best_frame' not in f:
            raise ValueError("No best_frame group found in file")
        
        best_idx = f['best_frame'].attrs['frame_index']
        log_prob = f['best_frame'].attrs['log_probability']
        
        # Load static data
        radius = jnp.array(f['radius'][:])
        copy_number = jnp.array(f['copy_number'][:])
        particle_types = jnp.array(f['particle_types'][:]) if 'particle_types' in f else None
        rigid_body_ids = jnp.array(f['rigid_body_ids'][:]) if 'rigid_body_ids' in f else None
        
        type_names = None
        if 'type_names' in f.attrs:
            type_names = tuple(f.attrs['type_names'].split(','))
        
        # Load best coordinates
        coords = jnp.array(f['best_frame']['coordinates'][:])
        
        state = State(
            coordinates=coords,
            radius=radius,
            copy_number=copy_number,
            particle_types=particle_types,
            rigid_body_ids=rigid_body_ids,
            type_names=type_names,
        )
    
    print(f"Loaded best frame: log_prob={log_prob:.3f}")
    return state, log_prob


def get_trajectory_info(filename):
    """Get summary information about a trajectory file."""
    with h5py.File(filename, 'r') as f:
        info = {
            'n_frames': f.attrs['n_frames'],
            'n_particles': f.attrs['n_particles'],
            'n_types': f.attrs['n_types'],
            'creation_date': f.attrs.get('creation_date', 'unknown'),
        }
        
        if 'type_names' in f.attrs:
            info['type_names'] = f.attrs['type_names'].split(',')
        
        if 'log_probabilities' in f:
            log_probs = f['log_probabilities'][:]
            info['best_log_prob'] = float(np.max(log_probs))
            info['mean_log_prob'] = float(np.mean(log_probs))
        
        if 'metadata' in f:
            info['metadata'] = dict(f['metadata'].attrs)
        
        # File size
        info['file_size_mb'] = Path(filename).stat().st_size / 1024 / 1024
    
    return info