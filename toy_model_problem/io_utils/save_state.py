"""
Save and load State objects to/from HDF5 files.
"""
import h5py
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from representation.state import State


def save_state(state, filename, metadata=None):
    """
    Save State to HDF5 file.
    
    Args:
        state: State object to save
        filename: Output .h5 file path
        metadata: Optional dict of additional metadata to store
    """
    with h5py.File(filename, 'w') as f:
        # File metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_particles'] = state.n_particles
        f.attrs['n_types'] = state.n_types
        
        if state.type_names is not None:
            # Store as comma-separated string
            f.attrs['type_names'] = ','.join(state.type_names)
        
        # Store arrays
        f.create_dataset('coordinates', data=np.array(state.coordinates), compression='gzip')
        f.create_dataset('radius', data=np.array(state.radius), compression='gzip')
        f.create_dataset('copy_number', data=np.array(state.copy_number), compression='gzip')
        
        if state.particle_types is None:
            # Generate from copy_number if not present
            type_indices = []
            for type_idx, count in enumerate(state.copy_number):
                type_indices.extend([type_idx] * int(count))
            particle_types_data = np.array(type_indices)
        else:
            particle_types_data = np.array(state.particle_types)
            
        f.create_dataset('particle_types', data=particle_types_data)
                
        if state.rigid_body_ids is not None:
            f.create_dataset('rigid_body_ids', data=np.array(state.rigid_body_ids), compression='gzip')
        
        # Optional metadata
        if metadata:
            meta_grp = f.create_group('metadata')
            for key, val in metadata.items():
                meta_grp.attrs[key] = val
    
    print(f"Saved State to: {filename}")


def load_state(filename):
    """
    Load State from HDF5 file.
    
    Args:
        filename: Input .h5 file path
    
    Returns:
        State object
    """
    with h5py.File(filename, 'r') as f:
        coords = jnp.array(f['coordinates'][:])
        radius = jnp.array(f['radius'][:])
        copy_number = jnp.array(f['copy_number'][:])
        
        # Optional fields
        particle_types = jnp.array(f['particle_types'][:]) if 'particle_types' in f else None
        rigid_body_ids = jnp.array(f['rigid_body_ids'][:]) if 'rigid_body_ids' in f else None
        
        type_names = None
        if 'type_names' in f.attrs:
            type_names = tuple(f.attrs['type_names'].split(','))
        
        state = State(
            coordinates=coords,
            radius=radius,
            copy_number=copy_number,
            particle_types=particle_types,
            rigid_body_ids=rigid_body_ids,
            type_names=type_names,
        )
    
    print(f"Loaded State from: {filename}")
    return state


if __name__ == "__main__":
    # Example usage
    import jax
    
    print("Creating example State...")
    key = jax.random.PRNGKey(42)
    coords = jax.random.uniform(key, (10, 3)) * 100
    radius = jnp.array([1.0] * 10)
    copy_number = jnp.array([10])
    
    state = State(coords, radius, copy_number, type_names=('A',))
    
    # Save
    save_state(state, "test_state.h5", metadata={"method": "random_init"})
    
    # Load
    loaded_state = load_state("test_state.h5")
    
    print(f"\nOriginal: {state}")
    print(f"Loaded: {loaded_state}")
    print(f"Coordinates match: {jnp.allclose(state.coordinates, loaded_state.coordinates)}")