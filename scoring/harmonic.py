#-------------------------
# Harmonic score 
#-------------------------
import jax
import jax.numpy as jnp
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from representation import ParticleState


def jax_harmonic_score(
    positions: Union[jnp.ndarray, 'ParticleState'],
    indices: jnp.ndarray,
    d0: float = 2.0,
    k: float = 0.1
) -> jnp.ndarray:
    """
    JAX harmonic pair score - matches IMP exactly.
    
    Computes harmonic restraint energy: E = sum(0.5 * k * (d0 - d)^2)
    
    Args:
        positions: Either (N, 3) coordinate array or ParticleState object
        indices: (M, 2) array of particle index pairs
        d0: Equilibrium distance
        k: Spring constant
        
    Returns:
        Total harmonic energy (scalar)
    """
    # Handle ParticleState input
    if hasattr(positions, 'positions'):
        positions = positions.positions
        
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    return jnp.sum(0.5 * k * (d0 - dists)**2)


def harmonic_upperbound_restraint(
    positions: Union[jnp.ndarray, 'ParticleState'],
    indices: jnp.ndarray,
    d0: float = 2.0,
    k: float = 0.1
) -> jnp.ndarray:
    """
    Harmonic upperbound restraint score.
    
    Only penalizes distances greater than d0 (one-sided harmonic).
    Useful for maximum distance constraints.
    
    Lets say we have a gaussian likelihood:
    1/sqrt(2*pi*sigma^2) * exp(-0.5 * ((d - d0)/sigma)^2)
    
    The sigma parameter can be inferred based on experimental data.
    
    Args:
        positions: Either (N, 3) coordinate array or ParticleState object
        indices: (M, 2) array of particle index pairs
        d0: Maximum allowed distance (equilibrium)
        k: Spring constant
        
    Returns:
        Total upperbound energy (scalar)
    """
    # Handle ParticleState input
    if hasattr(positions, 'positions'):
        positions = positions.positions
        
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    diffs = jnp.maximum(0.0, dists - d0)
    return jnp.sum(0.5 * k * diffs**2)


def harmonic_lowerbound_restraint(
    positions: Union[jnp.ndarray, 'ParticleState'],
    indices: jnp.ndarray,
    d0: float = 2.0,
    k: float = 0.1
) -> jnp.ndarray:
    """
    Harmonic lowerbound restraint score.
    
    Only penalizes distances less than d0 (one-sided harmonic).
    Useful for minimum distance constraints.
    
    Args:
        positions: Either (N, 3) coordinate array or ParticleState object
        indices: (M, 2) array of particle index pairs
        d0: Minimum required distance
        k: Spring constant
        
    Returns:
        Total lowerbound energy (scalar)
    """
    # Handle ParticleState input
    if hasattr(positions, 'positions'):
        positions = positions.positions
        
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    diffs = jnp.maximum(0.0, d0 - dists)
    return jnp.sum(0.5 * k * diffs**2)


# JIT-compiled versions
jax_harmonic_jit = jax.jit(jax_harmonic_score, static_argnums=(2, 3))
harmonic_upper_jit = jax.jit(harmonic_upperbound_restraint, static_argnums=(2, 3))
harmonic_lower_jit = jax.jit(harmonic_lowerbound_restraint, static_argnums=(2, 3))