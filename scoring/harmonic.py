#-------------------------
# Harmonic score 
#-------------------------
import jax
import jax.numpy as jnp

def jax_harmonic_score(positions, indices, d0=2.0, k=0.1):
    """JAX harmonic pair score - matches IMP exactly."""
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    return jnp.sum(0.5 * k * (d0 - dists)**2)

def harmonic_upperbound_restraint(positions, indices, d0=2.0, k=0.1):
    """Harmonic upperbound restraint score. Lets say we have a gaussian likelihood, 
    1/sqrt(2*pi*sigma^2) * exp(-0.5 * ((d - d0)/sigma)^2), lets say that the sigma parameter is 
    not fixed but needs to be inferred based on some experimental data. """
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    diffs = jnp.maximum(0.0, dists - d0)
    return jnp.sum(0.5 * k * diffs**2)