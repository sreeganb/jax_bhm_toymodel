#------------------------------------------------------------------------------
# Coordinate generation and ParticleState factory utilities
# Part of the Representation stage in Bayesian inference pipeline
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Union, Optional, List

from . import ParticleState


def get_ideal_coords() -> dict[str, jnp.ndarray]:
    array_A = np.array([
        [63., 0., 0.],
        [44.55, 44.55, 0.],
        [0., 63., 0.],
        [-44.55, 44.55, 0.],
        [-63., 0., 0.],
        [-44.55, -44.55, 0.],
        [-0., -63., 0.],
        [44.55, -44.55, 0.]
    ])
    array_B = np.array([
        [63., 0., -38.5],
        [44.55, 44.55, -38.5],
        [0., 63., -38.5],
        [-44.55, 44.55, -38.5],
        [-63., 0., -38.5],
        [-44.55, -44.55, -38.5],
        [-0., -63., -38.5],
        [44.55, -44.55, -38.5]
    ])
    array_C = np.array([
        [47.00, 0.00, -68.50],
        [79.00, 0.00, -68.50],
        [55.86, 55.86, -68.50],
        [33.23, 33.23, -68.50],
        [0.00, 47.00, -68.50],
        [0.00, 79.00, -68.50],
        [-55.86, 55.86, -68.50],
        [-33.23, 33.23, -68.50],
        [-47.00, 0.00, -68.50],
        [-79.00, 0.00, -68.50],
        [-55.86, -55.86, -68.50],
        [-33.23, -33.23, -68.50],
        [0.00, -47.00, -68.50],
        [0.00, -79.00, -68.50],
        [55.86, -55.86, -68.50],
        [33.23, -33.23, -68.50],
    ])
    return {
        'A': jnp.array(array_A),  # slight offset to avoid zero distances
        'B': jnp.array(array_B),
        'C': jnp.array(array_C)
    }

# Generalized function to generate random coords for arbitrary N_A, N_B, N_C
def generate_random_coords(
    key: jax.Array,
    n_dict: dict[str, int],
    minval: float = -200.0,
    maxval: float = 200.0
) -> dict[str, jnp.ndarray]:
    """
    Docstring for generate_random_coords
    
    :param key: JAX random key
    :type key: jax.Array
    :param n_dict: Dictionary specifying number of particles per type
    :type n_dict: dict[str, int]
    :param minval: Minimum coordinate value
    :type minval: float
    :param maxval: Maximum coordinate value
    :type maxval: float
    :return: Dictionary of coordinates per particle type
    :rtype: dict[str, jnp.ndarray]
    """
    coords = {}
    subkeys = jax.random.split(key, len(n_dict))
    for i, (label, n) in enumerate(n_dict.items()):
        coords[label] = jax.random.uniform(subkeys[i], shape=(n, 3), minval=minval, maxval=maxval)
    return coords


#------------------------------------------------------------------------------
# ParticleState Factory Functions
#------------------------------------------------------------------------------

def create_particle_state(
    coords: Dict[str, jnp.ndarray],
    radii: Union[Dict[str, float], float] = 1.0,
    copy_numbers: Union[Dict[str, int], int] = 1,
    validate: bool = True
) -> ParticleState:
    """
    Factory function to create a validated ParticleState from type dictionaries.
    
    This is the recommended way to initialize particle systems, as it:
    - Validates that all radii are positive
    - Ensures consistent indexing across types
    - Provides clear error messages for invalid inputs
    
    Args:
        coords: Dict mapping type names to coordinate arrays, each (N_type, 3)
        radii: Either a dict mapping type names to radii, or a single float
        copy_numbers: Either a dict mapping type names to copy numbers, or single int
        validate: Whether to run validation checks (default: True)
        
    Returns:
        ParticleState with concatenated coordinates and per-particle attributes
        
    Raises:
        ValueError: If validation fails (e.g., negative radii, empty coords)
        
    Example:
        >>> coords = {'A': jnp.array([[0,0,0]]), 'B': jnp.array([[5,0,0]])}
        >>> state = create_particle_state(coords, radii={'A': 2.0, 'B': 3.0})
    """
    if validate:
        _validate_inputs(coords, radii)
    
    return ParticleState.from_type_dict(coords, radii, copy_numbers)


def create_random_particle_state(
    key: jax.Array,
    n_dict: Dict[str, int],
    radii: Union[Dict[str, float], float] = 1.0,
    copy_numbers: Union[Dict[str, int], int] = 1,
    box_size: float = 100.0,
    center: bool = True
) -> ParticleState:
    """
    Create a ParticleState with random initial positions.
    
    Useful for initializing MCMC/SMC samplers with randomized starting points.
    
    Args:
        key: JAX random key
        n_dict: Dictionary specifying number of particles per type {'A': 8, 'B': 4}
        radii: Per-type radii dict or single float
        copy_numbers: Per-type copy numbers or single int
        box_size: Size of cubic box for random positions
        center: If True, center box at origin; if False, box starts at origin
        
    Returns:
        ParticleState with random positions
        
    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> state = create_random_particle_state(key, {'A': 10, 'B': 5}, radii=2.0)
    """
    if center:
        minval, maxval = -box_size / 2, box_size / 2
    else:
        minval, maxval = 0.0, box_size
        
    coords = generate_random_coords(key, n_dict, minval=minval, maxval=maxval)
    return ParticleState.from_type_dict(coords, radii, copy_numbers)


def create_from_ideal_coords(
    radii: Union[Dict[str, float], float] = 1.0,
    copy_numbers: Union[Dict[str, int], int] = 1
) -> ParticleState:
    """
    Create ParticleState from the ideal ground truth coordinates.
    
    Uses the pre-defined ideal coordinates for A (8 particles), 
    B (8 particles), and C (16 particles) arranged in a symmetric pattern.
    
    Args:
        radii: Per-type radii dict or single float
        copy_numbers: Per-type copy numbers or single int
        
    Returns:
        ParticleState with 32 particles total
    """
    coords = get_ideal_coords()
    return ParticleState.from_type_dict(coords, radii, copy_numbers)


def _validate_inputs(
    coords: Dict[str, jnp.ndarray],
    radii: Union[Dict[str, float], float]
) -> None:
    """Validate coordinate and radius inputs."""
    if not coords:
        raise ValueError("coords dictionary cannot be empty")
    
    for type_name, type_coords in coords.items():
        if type_coords.ndim != 2 or type_coords.shape[1] != 3:
            raise ValueError(
                f"Coordinates for type '{type_name}' must have shape (N, 3), "
                f"got {type_coords.shape}"
            )
        if type_coords.shape[0] == 0:
            raise ValueError(f"Coordinates for type '{type_name}' cannot be empty")
    
    if isinstance(radii, dict):
        for type_name, r in radii.items():
            if r <= 0:
                raise ValueError(
                    f"Radius for type '{type_name}' must be positive, got {r}"
                )
            if type_name not in coords:
                raise ValueError(
                    f"Radius specified for unknown type '{type_name}'. "
                    f"Known types: {list(coords.keys())}"
                )
    elif radii <= 0:
        raise ValueError(f"Radius must be positive, got {radii}")


def get_pairwise_indices(
    state: ParticleState,
    type_pairs: Optional[List[tuple]] = None
) -> jnp.ndarray:
    """
    Generate pairwise indices for restraints between particles.
    
    Args:
        state: ParticleState to generate indices for
        type_pairs: Optional list of (type_i, type_j) tuples to restrict pairs.
                   If None, generates all pairs.
                   
    Returns:
        Array of shape (M, 2) with particle index pairs
        
    Example:
        >>> indices = get_pairwise_indices(state, type_pairs=[('A', 'B')])
    """
    n = state.n_particles
    
    if type_pairs is None:
        # All pairs
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                indices.append([i, j])
        return jnp.array(indices) if indices else jnp.zeros((0, 2), dtype=jnp.int32)
    
    # Filtered pairs by type
    type_name_to_id = {v: k for k, v in state.type_names.items()}
    indices = []
    
    for type_i_name, type_j_name in type_pairs:
        type_i = type_name_to_id.get(type_i_name)
        type_j = type_name_to_id.get(type_j_name)
        
        if type_i is None or type_j is None:
            continue
            
        mask_i = state.particle_types == type_i
        mask_j = state.particle_types == type_j
        
        idx_i = jnp.where(mask_i)[0]
        idx_j = jnp.where(mask_j)[0]
        
        for i in idx_i:
            for j in idx_j:
                if i < j:
                    indices.append([int(i), int(j)])
                elif i > j:
                    indices.append([int(j), int(i)])
    
    # Remove duplicates
    if indices:
        indices = list(set(tuple(x) for x in indices))
        indices = sorted(indices)
        return jnp.array(indices)
    return jnp.zeros((0, 2), dtype=jnp.int32)

