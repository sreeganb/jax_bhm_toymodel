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
