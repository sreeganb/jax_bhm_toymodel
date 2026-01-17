#--------------------------
# Scoring Functions
#--------------------------
import jax
import jax.numpy as jnp
from jax import lax


def jax_excluded_volume(positions, radii, k=1.0):
    """
    JAX soft sphere excluded volume - ALL PAIRS (no neighbor list).
    
    Returns a POSITIVE score when particles overlap (penalty).
    """
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    sigma = radii[:, None] + radii[None, :]
    overlap = jax.nn.relu(sigma - dists)
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    return jnp.sum(0.5 * k * (overlap * mask)**2)


def compute_pair_nll(coords_i, coords_j, d0, sigma, same_type):
    """
    Compute NLL for a single pair type. JAX-traceable.
    
    Args:
        coords_i: (N_i, 3) coordinates of type i
        coords_j: (N_j, 3) coordinates of type j
        d0: target distance
        sigma: gaussian width
        same_type: bool, True if type_i == type_j
    
    Returns:
        pair_nll: scalar
    """
    # Distance matrix (N_i, N_j)
    diff = coords_i[:, None, :] - coords_j[None, :, :]
    dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    
    # NLL matrix
    nll_matrix = (dist_matrix - d0)**2 / (2 * sigma**2) + jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    
    n_i, n_j = coords_i.shape[0], coords_j.shape[0]
    
    # Same type: use upper triangle
    def same_type_nll():
        mask = jnp.triu(jnp.ones((n_i, n_j), dtype=bool), k=1)
        masked_nll = jnp.where(mask, nll_matrix, jnp.inf)
        row_mins = jnp.min(masked_nll, axis=1)
        col_mins = jnp.min(masked_nll, axis=0)
        r_sum = jnp.sum(jnp.where(jnp.isfinite(row_mins), row_mins, 0.0))
        c_sum = jnp.sum(jnp.where(jnp.isfinite(col_mins), col_mins, 0.0))
        return (r_sum + c_sum) / 2.0
    
    # Different types: sum of row-wise + column-wise mins
    def diff_type_nll():
        row_mins = jnp.min(nll_matrix, axis=1)
        col_mins = jnp.min(nll_matrix, axis=0)
        return jnp.sum(row_mins) + jnp.sum(col_mins)
    
    return lax.cond(same_type, same_type_nll, diff_type_nll)


def matrix_pair_score(positions, particle_types, ideal_distances, sigmas, copy_numbers):
    """
    Compute total Negative Log-Likelihood (NLL) for pairwise distance restraints.
    
    This version is fully JAX-traceable (no Python int() calls).
    
    Args:
        positions: (N, 3) array of particle positions
        particle_types: (N,) array of type indices (0, 1, 2, ...) - NOT USED directly
        ideal_distances: Dict mapping pair string (e.g. '00', '01') to target distance
        sigmas: Dict mapping pair string to gaussian width
        copy_numbers: (n_types,) array with count of each particle type [N_A, N_B, N_C]
    
    Returns:
        total_nll: Scalar float (POSITIVE, lower is better).
    """
    # Precompute slices based on copy_numbers
    # For a system with [N_A, N_B, N_C], particles are ordered: all A's, then B's, then C's
    n_A = copy_numbers[0]
    n_B = copy_numbers[1]
    n_C = copy_numbers[2]
    
    # Extract coordinates for each type using static slicing
    coords_0 = positions[:n_A]                    # Type 0 (A)
    coords_1 = positions[n_A:n_A + n_B]           # Type 1 (B)
    coords_2 = positions[n_A + n_B:n_A + n_B + n_C]  # Type 2 (C)
    
    coords_by_type = {0: coords_0, 1: coords_1, 2: coords_2}
    
    total_nll = 0.0
    
    for pair_key, d0 in ideal_distances.items():
        type_i = int(pair_key[0])  # This is fine - it's a Python string, not a traced value
        type_j = int(pair_key[1])
        sigma = sigmas.get(pair_key, 1.0)
        
        coords_i = coords_by_type[type_i]
        coords_j = coords_by_type[type_j]
        
        same_type = (type_i == type_j)
        
        pair_nll = compute_pair_nll(coords_i, coords_j, d0, sigma, same_type)
        total_nll = total_nll + pair_nll
    
    return total_nll


def matrix_pair_score_general(positions, copy_numbers_tuple, ideal_distances, sigmas):
    """
    General version that works with any number of particle types.
    
    Args:
        positions: (N, 3) array of particle positions
        copy_numbers_tuple: tuple of ints (N_0, N_1, N_2, ...) - MUST be Python ints, not arrays
        ideal_distances: Dict mapping pair string (e.g. '00', '01') to target distance
        sigmas: Dict mapping pair string to gaussian width
    
    Returns:
        total_nll: Scalar float
    """
    # Build coordinate slices
    n_types = len(copy_numbers_tuple)
    cumsum = [0]
    for n in copy_numbers_tuple:
        cumsum.append(cumsum[-1] + n)
    
    coords_by_type = {}
    for t in range(n_types):
        coords_by_type[t] = positions[cumsum[t]:cumsum[t+1]]
    
    total_nll = 0.0
    
    for pair_key, d0 in ideal_distances.items():
        type_i = int(pair_key[0])
        type_j = int(pair_key[1])
        sigma = sigmas.get(pair_key, 1.0)
        
        coords_i = coords_by_type[type_i]
        coords_j = coords_by_type[type_j]
        
        same_type = (type_i == type_j)
        pair_nll = compute_pair_nll(coords_i, coords_j, d0, sigma, same_type)
        total_nll = total_nll + pair_nll
    
    return total_nll