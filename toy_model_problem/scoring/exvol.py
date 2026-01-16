#--------------------------
# Excluded Volume Potentials
#--------------------------
import jax
import jax.numpy as jnp

def jax_excluded_volume(positions, radii, k=1.0):
    """JAX soft sphere excluded volume - ALL PAIRS (no neighbor list)."""
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    sigma = radii[:, None] + radii[None, :]
    overlap = jax.nn.relu(sigma - dists)
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    return jnp.sum(0.5 * k * (overlap * mask)**2)

def harmonic_upper_bound(positions, radii, box_size, k=1.0):
    """Harmonic upper bound potential to keep particles inside box."""
    lower_bound = radii
    upper_bound = box_size - radii
    penalty_lower = jax.nn.relu(lower_bound - positions)
    penalty_upper = jax.nn.relu(positions - upper_bound)
    return jnp.sum(0.5 * k * (penalty_lower**2 + penalty_upper**2))
