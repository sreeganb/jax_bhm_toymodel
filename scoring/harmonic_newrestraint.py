#------------------------------------------------------------------------------
# Harmonic pair restraint with union-of-argmin pairing strategy
# JAX implementation for GPU acceleration
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
from functools import partial


def calculate_pair_scores_matrix(
    pos1: jnp.ndarray,
    pos2: jnp.ndarray,
    target_dist: float,
    sigma: float
) -> jnp.ndarray:
    """
    Calculate pairwise Gaussian negative log-likelihood score matrix.
    
    Score = 0.5 * ((d - target_dist) / sigma)^2 + log(sigma * sqrt(2*pi))
    
    This is the negative log of a Gaussian likelihood:
        P(d | target_dist, sigma) ∝ exp(-0.5 * ((d - target_dist) / sigma)^2)
    
    JAX SPEEDUP:
    - Broadcasting creates (N1, N2, 3) difference array in one operation
    - All N1 * N2 distances computed in parallel
    - Gaussian NLL computed vectorized across entire matrix
    
    Args:
        pos1: (N1, 3) positions of first particle type
        pos2: (N2, 3) positions of second particle type  
        target_dist: equilibrium/target distance
        sigma: standard deviation (uncertainty parameter)
    
    Returns:
        (N1, N2) matrix of Gaussian NLL scores
    """
    # BROADCASTING: pos1[:, None, :] is (N1, 1, 3), pos2[None, :, :] is (1, N2, 3)
    # Result diff is (N1, N2, 3) - all pairwise difference vectors
    # JAX SPEEDUP: Single vectorized operation instead of N1*N2 loops
    diff = pos1[:, None, :] - pos2[None, :, :]
    
    # Compute all N1*N2 distances in parallel
    # JAX SPEEDUP: Vectorized norm computation
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    
    # Gaussian negative log-likelihood (with normalization term)
    # NLL = 0.5 * ((d - d0) / sigma)^2 + log(sigma) + 0.5*log(2*pi)
    # JAX SPEEDUP: All arithmetic fused by XLA compiler
    z = (dists - target_dist) / sigma
    normalization = jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi)
    scores = 0.5 * z**2 + normalization
    
    return scores


def union_of_argmin_same_type(score_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Union-of-argmin pairing for same-type particles.
    
    For same type (e.g., A-A), we only consider upper triangle to avoid
    double counting and self-interactions.
    
    Strategy:
    1. Set diagonal to inf (no self-pairing)
    2. Each particle i finds best partner j > i (row argmin in upper triangle)
    3. Each particle j finds best partner i < j (column argmin in upper triangle)
    4. Take union of these pairs
    
    Args:
        score_matrix: (N, N) symmetric score matrix
    
    Returns:
        (M, 2) array of unique pair indices, or (M,) array of scores
    """
    N = score_matrix.shape[0]
    
    # Set diagonal and lower triangle to inf
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    masked_scores = jnp.where(mask, score_matrix, jnp.inf)
    
    # Row argmins: for each i, find best j > i
    row_argmins = jnp.argmin(masked_scores, axis=1)  # (N,)
    row_mins = jnp.take_along_axis(masked_scores, row_argmins[:, None], axis=1).squeeze(-1)
    row_valid = ~jnp.isinf(row_mins)
    
    # Column argmins: for each j, find best i < j
    col_argmins = jnp.argmin(masked_scores, axis=0)  # (N,)
    col_mins = jnp.take_along_axis(masked_scores, col_argmins[None, :], axis=0).squeeze(0)
    col_valid = ~jnp.isinf(col_mins)
    
    # Build pair indicator matrix: mark (i, j) if selected by row or column argmin
    # Row selections: particle i selected particle row_argmins[i]
    pair_selected = jnp.zeros((N, N), dtype=bool)
    
    # Mark row selections
    row_indices = jnp.arange(N)
    pair_selected = pair_selected.at[row_indices, row_argmins].set(row_valid)
    
    # Mark column selections (ensure canonical order i < j)
    pair_selected = pair_selected.at[col_argmins, jnp.arange(N)].set(
        pair_selected[col_argmins, jnp.arange(N)] | col_valid
    )
    
    # Only keep upper triangle (canonical order)
    pair_selected = pair_selected & mask
    
    # Sum scores for selected pairs
    selected_scores = jnp.where(pair_selected, score_matrix, 0.0)
    return jnp.sum(selected_scores)


def union_of_argmin_different_types(score_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Union-of-argmin pairing for different-type particles.
    
    Strategy:
    1. Each type1 particle finds its best type2 partner (row argmin)
    2. Each type2 particle finds its best type1 partner (column argmin)
    3. Take union of these pairs
    
    Args:
        score_matrix: (N1, N2) score matrix
    
    Returns:
        Total score for unique pairs
    """
    N1, N2 = score_matrix.shape
    
    # Row argmins: each type1 particle finds best type2 partner
    row_argmins = jnp.argmin(score_matrix, axis=1)  # (N1,)
    row_mins = jnp.take_along_axis(score_matrix, row_argmins[:, None], axis=1).squeeze(-1)
    row_valid = ~jnp.isinf(row_mins)
    
    # Column argmins: each type2 particle finds best type1 partner
    col_argmins = jnp.argmin(score_matrix, axis=0)  # (N2,)
    col_mins = jnp.take_along_axis(score_matrix, col_argmins[None, :], axis=0).squeeze(0)
    col_valid = ~jnp.isinf(col_mins)
    
    # Build pair indicator matrix
    pair_selected = jnp.zeros((N1, N2), dtype=bool)
    
    # Mark row selections
    pair_selected = pair_selected.at[jnp.arange(N1), row_argmins].set(row_valid)
    
    # Mark column selections (may overlap with row selections - that's fine, union)
    pair_selected = pair_selected.at[col_argmins, jnp.arange(N2)].set(
        pair_selected[col_argmins, jnp.arange(N2)] | col_valid
    )
    
    # Sum scores for selected pairs
    selected_scores = jnp.where(pair_selected, score_matrix, 0.0)
    return jnp.sum(selected_scores)


def harmonic_pair_score_typed(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    pair_config: Dict[str, Dict],
    sigmas: Dict[str, float],
    type_id_to_name: Dict[int, str],
    pair_weight: float = 1.0,
) -> float:
    """
    Calculate harmonic pair score with union-of-argmin pairing.
    
    This is the main JAX-accelerated scoring function that:
    1. Groups particles by type
    2. For each pair type (e.g., A-A, A-B, B-C), computes score matrix
    3. Applies union-of-argmin pairing strategy
    4. Returns weighted sum of scores
    
    Args:
        positions: (N, 3) all particle positions
        particle_types: (N,) integer type labels
        pair_config: Dict mapping pair keys (e.g., "AB") to {"target_dist": float}
        sigmas: Dict mapping pair keys to sigma values
        type_id_to_name: Dict mapping type_id to type name string
        pair_weight: weight for pairwise scores
    
    Returns:
        Total pairwise score (scalar)
    """
    total_score = 0.0
    name_to_id = {v: k for k, v in type_id_to_name.items()}
    
    for pair_key, config in pair_config.items():
        # Parse pair key (e.g., "AB" -> type1="A", type2="B")
        if len(pair_key) == 2:
            type1_name, type2_name = pair_key[0], pair_key[1]
        else:
            continue
            
        if type1_name not in name_to_id or type2_name not in name_to_id:
            continue
            
        type1_id = name_to_id[type1_name]
        type2_id = name_to_id[type2_name]
        
        target_dist = config.get("target_dist", config.get("d0", 0.0))
        sigma = sigmas.get(pair_key, 1.0)
        
        # Extract positions for each type
        mask1 = particle_types == type1_id
        mask2 = particle_types == type2_id
        pos1 = positions[mask1]
        pos2 = positions[mask2]
        
        if pos1.shape[0] == 0 or pos2.shape[0] == 0:
            continue
        
        # Calculate score matrix
        score_matrix = calculate_pair_scores_matrix(pos1, pos2, target_dist, sigma)
        
        # Apply union-of-argmin strategy
        if type1_name == type2_name:
            pair_score = union_of_argmin_same_type(score_matrix)
        else:
            pair_score = union_of_argmin_different_types(score_matrix)
        
        total_score += pair_weight * pair_score
    
    return total_score


def harmonic_pair_score_simple(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    type_bounds: Dict[str, Tuple[int, int]],
    pair_config: Dict[str, Dict],
    sigmas: Dict[str, float],
    pair_weight: float = 1.0,
) -> float:
    """
    Simplified version using pre-computed type bounds for faster indexing.
    
    Args:
        positions: (N, 3) all particle positions
        particle_types: (N,) integer type labels (unused if type_bounds provided)
        type_bounds: Dict mapping type name to (start_idx, end_idx) tuple
        pair_config: Dict mapping pair keys to {"target_dist": float}
        sigmas: Dict mapping pair keys to sigma values
        pair_weight: weight for pairwise scores
    
    Returns:
        Total pairwise score (scalar)
    """
    total_score = 0.0
    
    for pair_key, config in pair_config.items():
        if len(pair_key) != 2:
            continue
            
        type1_name, type2_name = pair_key[0], pair_key[1]
        
        if type1_name not in type_bounds or type2_name not in type_bounds:
            continue
        
        start1, end1 = type_bounds[type1_name]
        start2, end2 = type_bounds[type2_name]
        
        target_dist = config.get("target_dist", config.get("d0", 0.0))
        sigma = sigmas.get(pair_key, 1.0)
        
        pos1 = positions[start1:end1]
        pos2 = positions[start2:end2]
        
        if pos1.shape[0] == 0 or pos2.shape[0] == 0:
            continue
        
        # Calculate score matrix
        score_matrix = calculate_pair_scores_matrix(pos1, pos2, target_dist, sigma)
        
        # Apply union-of-argmin strategy
        if type1_name == type2_name:
            pair_score = union_of_argmin_same_type(score_matrix)
        else:
            pair_score = union_of_argmin_different_types(score_matrix)
        
        total_score += pair_weight * pair_score
    
    return total_score


# JIT-compiled versions for maximum performance
# Note: These work best when pair_config and sigmas are static
# For dynamic sigmas (sampling), use functools.partial or pass as arrays

@partial(jax.jit, static_argnums=(2, 3))
def jit_pair_score_matrix(pos1, pos2, target_dist, sigma):
    """JIT-compiled score matrix calculation."""
    return calculate_pair_scores_matrix(pos1, pos2, target_dist, sigma)


# For use in sampling where sigma changes, create a closure:
def make_pair_scorer(
    particle_types: jnp.ndarray,
    type_bounds: Dict[str, Tuple[int, int]],
    pair_config: Dict[str, Dict],
    pair_weight: float = 1.0,
):
    """
    Factory function that returns a JIT-compilable scoring function.
    
    The returned function takes (positions, sigmas_array) and returns total score.
    This is useful for MCMC/SMC where positions and sigmas are sampled.
    
    Args:
        particle_types: (N,) integer type labels
        type_bounds: Dict mapping type name to (start_idx, end_idx)
        pair_config: Dict mapping pair keys to {"target_dist": float}
        pair_weight: weight for pairwise scores
    
    Returns:
        Callable (positions, sigmas_dict) -> score
    """
    # Pre-extract static config
    pair_keys = list(pair_config.keys())
    target_dists = {k: pair_config[k].get("target_dist", pair_config[k].get("d0", 0.0)) 
                    for k in pair_keys}
    bounds = {k: type_bounds.get(k[0], (0,0)) + type_bounds.get(k[1], (0,0)) 
              for k in pair_keys if len(k) == 2}
    
    def scorer(positions: jnp.ndarray, sigmas: Dict[str, float]) -> float:
        total = 0.0
        for pair_key in pair_keys:
            if len(pair_key) != 2:
                continue
            type1_name, type2_name = pair_key[0], pair_key[1]
            
            if type1_name not in type_bounds or type2_name not in type_bounds:
                continue
            
            start1, end1 = type_bounds[type1_name]
            start2, end2 = type_bounds[type2_name]
            
            pos1 = positions[start1:end1]
            pos2 = positions[start2:end2]
            
            if pos1.shape[0] == 0 or pos2.shape[0] == 0:
                continue
            
            target_dist = target_dists[pair_key]
            sigma = sigmas.get(pair_key, 1.0)
            
            score_matrix = calculate_pair_scores_matrix(pos1, pos2, target_dist, sigma)
            
            if type1_name == type2_name:
                pair_score = union_of_argmin_same_type(score_matrix)
            else:
                pair_score = union_of_argmin_different_types(score_matrix)
            
            total += pair_weight * pair_score
        
        return total
    
    return scorer


#------------------------------------------------------------------------------
# Convenience function matching the simple harmonic interface
#------------------------------------------------------------------------------

def jax_harmonic_pair_score(
    positions: jnp.ndarray,
    indices: jnp.ndarray,
    d0: float = 2.0,
    k: float = 0.1,
    sigma: float = 1.0,
) -> float:
    """
    Simple harmonic pair score for specified index pairs.
    
    This is a simpler interface that matches jax_harmonic_score but uses
    Gaussian NLL instead of pure harmonic energy.
    
    Energy = sum over pairs of: 0.5 * ((d - d0) / sigma)^2 + log(sigma)
    
    For backward compatibility with existing code that uses index-based restraints.
    
    Args:
        positions: (N, 3) particle coordinates
        indices: (M, 2) array of particle index pairs
        d0: equilibrium distance
        k: (unused, kept for API compatibility)
        sigma: standard deviation for Gaussian likelihood
    
    Returns:
        Total score (scalar)
    """
    pos_i = positions[indices[:, 0]]
    pos_j = positions[indices[:, 1]]
    
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    
    z = (dists - d0) / sigma
    normalization = jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi)
    scores = 0.5 * z**2 + normalization
    
    return jnp.sum(scores)


# JIT compiled version
jax_harmonic_pair_score_jit = jax.jit(jax_harmonic_pair_score, static_argnums=(2, 3, 4))
