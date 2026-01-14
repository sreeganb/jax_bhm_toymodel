#-------------------------------------------------------------------------
# Scoring function for the excluded volume using JAX, JIT compilation 
# Does not use neighbor list, which means this is going to do all vs all 
#-------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from representation import ParticleState


def jax_excluded_volume(
    positions: Union[jnp.ndarray, 'ParticleState'],
    radii: Optional[jnp.ndarray] = None,
    k: float = 1.0
) -> jnp.ndarray:
    """
    JAX soft sphere excluded volume - ALL PAIRS (no neighbor list).
    
    Computes soft-sphere repulsion: E = sum(0.5 * k * max(0, overlap)^2)
    where overlap = (r_i + r_j) - d_ij
    
    Args:
        positions: Either (N, 3) coordinate array or ParticleState object.
                  If ParticleState, radii are extracted automatically.
        radii: (N,) array of particle radii. Required if positions is array,
               ignored if positions is ParticleState.
        k: Repulsion strength constant
        
    Returns:
        Total excluded volume energy (scalar)
        
    Example:
        >>> # Using arrays (backward compatible)
        >>> E = jax_excluded_volume(positions, radii, k=1.0)
        >>> 
        >>> # Using ParticleState (new API)
        >>> E = jax_excluded_volume(state, k=1.0)
    """
    # Handle ParticleState input
    if hasattr(positions, 'positions'):
        radii = positions.radii
        positions = positions.positions
    elif radii is None:
        raise ValueError("radii must be provided when positions is an array")
    
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    sigma = radii[:, None] + radii[None, :]
    overlap = jax.nn.relu(sigma - dists)
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    return jnp.sum(0.5 * k * (overlap * mask)**2)


def jax_excluded_volume_typed(
    state: 'ParticleState',
    k_matrix: Optional[jnp.ndarray] = None,
    k_default: float = 1.0
) -> jnp.ndarray:
    """
    Type-aware excluded volume with per-type-pair interaction strengths.
    
    Allows different repulsion strengths between different particle types.
    
    Args:
        state: ParticleState object with positions, radii, and particle_types
        k_matrix: (n_types, n_types) matrix of interaction strengths.
                 If None, uses k_default for all pairs.
        k_default: Default interaction strength if k_matrix not provided
        
    Returns:
        Total excluded volume energy (scalar)
        
    Example:
        >>> # Different repulsion for A-A, A-B, B-B interactions
        >>> k_matrix = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        >>> E = jax_excluded_volume_typed(state, k_matrix)
    """
    positions = state.positions
    radii = state.radii
    types = state.particle_types
    
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    sigma = radii[:, None] + radii[None, :]
    overlap = jax.nn.relu(sigma - dists)
    
    # Build per-pair k values from type matrix
    if k_matrix is not None:
        k_values = k_matrix[types[:, None], types[None, :]]
    else:
        k_values = k_default
    
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    return jnp.sum(0.5 * k_values * (overlap * mask)**2)


# JIT compile
jax_ev_jit = jax.jit(jax_excluded_volume)
jax_ev_typed_jit = jax.jit(jax_excluded_volume_typed, static_argnums=(2,))
