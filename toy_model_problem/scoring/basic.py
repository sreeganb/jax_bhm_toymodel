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

def jax_pair_score(state):
    """
    Pair scoring function for toy model particles based on distances.
    Compute the distance matrix for all vs all particles, there is a 
    particle type dependent preferred distance. Going to use ideal distances 
    and then make this code work like normal and then include other kind of data generation
    and likelihoods.  
    """
    coords = state.coordinates
    particle_types = state.particle_types
    n_particles = coords.shape[0]
    
    def ideal_distances():
        # for A-A type it is 48.5 angstroms
        # for A-B type it is 38.5 angstroms
        # for B-C type it is 31.0 angstroms

        if state.type_names is not None:
            type_name_to_index = {name: idx for idx, name in enumerate(state.type_names)}
            idx_A = type_name_to_index.get('ProteinA', 0)
            idx_B = type_name_to_index.get('ProteinB', 1)
            idx_C = type_name_to_index.get('ProteinC', 2)
        else:
            idx_A, idx_B, idx_C = 0, 1, 2
