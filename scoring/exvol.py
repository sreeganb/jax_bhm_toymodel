#-------------------------------------------------------------------------
# Scoring function for the excluded volume using JAX, JIT compilation 
# Does not use neighbor list, which means this is going to do all vs all 
#-------------------------------------------------------------------------

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

# JIT compile
jax_ev_jit = jax.jit(jax_excluded_volume)
