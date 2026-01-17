"""
SMC Sampling Example
--------------------
Runs SMC to assemble a system of 8 A, 8 B, and 16 C particles using
matrix pair score (NLL) and excluded volume potentials.

Uses RMH (Random Walk Metropolis-Hastings) - gradient-free!
"""

import sys
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Ensure we can import from package root
current_file = Path(__file__).resolve()
package_root = current_file.parent.parent
sys.path.insert(0, str(package_root))

from representation.state import State
from sampling.smc import run_smc, SMCConfig
from scoring.basic import jax_excluded_volume, matrix_pair_score, matrix_pair_score_general
from io_utils.save_trajectory import save_trajectory

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
N_A, N_B, N_C = 8, 8, 16
N_TOTAL = N_A + N_B + N_C
DIM = N_TOTAL * 3

RADII = {'A': 24.0, 'B': 14.0, 'C': 16.0}
BOX_SIZE = 300.0

# Ideal distances between particle types (pair key = 'type_i' + 'type_j')
IDEAL_DISTANCES = {'00': 48.22, '01': 38.5, '12': 34.0}
SIGMAS = {'00': 5.0, '01': 5.0, '12': 5.0}  # Relaxed sigmas for sampling

TYPE_NAMES = ('A', 'B', 'C')
TYPE_RADII = jnp.array([RADII['A']]*N_A + [RADII['B']]*N_B + [RADII['C']]*N_C)
PARTICLE_TYPES = jnp.array([0]*N_A + [1]*N_B + [2]*N_C)

# IMPORTANT: Use Python tuple for copy_numbers, not jnp.array
COPY_NUMBERS = (N_A, N_B, N_C)  # Python tuple of ints


# -----------------------------------------------------------------------------
# Log Probability Functions (using scoring module)
# -----------------------------------------------------------------------------
def log_prior(flat_coords):
    """Uniform prior within box."""
    positions = flat_coords.reshape(-1, 3)
    in_box = jnp.all((positions >= 0.0) & (positions <= BOX_SIZE))
    return jnp.where(in_box, 0.0, -jnp.inf)


def log_likelihood(flat_coords):
    """
    Total log likelihood = - (Excluded Volume + NLL).
    
    Uses scoring functions from scoring.basic module.
    """
    positions = flat_coords.reshape(-1, 3)
    
    # 1. Excluded Volume (positive when overlapping)
    ev_score = jax_excluded_volume(positions, TYPE_RADII, k=10.0)
    
    # 2. Pairwise NLL (positive, lower is better fit)
    # Use the general version with Python tuple for copy_numbers
    nll_score = matrix_pair_score_general(
        positions,
        COPY_NUMBERS,  # Python tuple, not jnp.array
        IDEAL_DISTANCES,
        SIGMAS
    )
    
    # Return negative total (higher = better for MCMC)
    return -(ev_score + nll_score)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"SMC Sampling: {N_A} A, {N_B} B, {N_C} C particles")
    print("Using RMH kernel (gradient-free)")
    print("=" * 70)
    
    key = random.PRNGKey(42)
    
    # SMC Config
    smc_config = SMCConfig(
        n_particles=500,        # Population size
        n_mcmc_steps=50,        # MCMC steps per temperature
        rmh_step_size=5.0,      # Tune for ~20-40% acceptance
        target_ess=0.5,
        max_iterations=200
    )
    
    print(f"\nConfig:")
    print(f"  Population: {smc_config.n_particles}")
    print(f"  MCMC steps/iter: {smc_config.n_mcmc_steps}")
    print(f"  RMH step size: {smc_config.rmh_step_size}")
    print(f"  Target ESS: {smc_config.target_ess}")
    print(f"  Box size: {BOX_SIZE}")
    print(f"  Dimensions: {DIM} ({N_TOTAL} particles x 3)")
    
    # Generate initial particles uniformly in box
    init_key, smc_key = random.split(key)
    initial_particles = random.uniform(
        init_key,
        shape=(smc_config.n_particles, DIM),
        minval=0.0,
        maxval=BOX_SIZE
    )
    
    print("\nRunning SMC...")
    start_time = time.time()
    
    result = run_smc(
        smc_key,
        log_prior,
        log_likelihood,
        smc_config,
        dim=DIM,
        initial_particles=initial_particles,
        save_history=True
    )
    
    end_time = time.time()
    
    print(f"\n{'=' * 70}")
    print(f"SMC Complete!")
    print(f"{'=' * 70}")
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Iterations: {result.n_iterations}")
    print(f"Final lambda: {result.final_lambda:.4f}")
    print(f"Final log_prob (mean): {jnp.mean(result.log_probs):.2f}")
    print(f"Final log_prob (best): {jnp.max(result.log_probs):.2f}")
    
    # -------------------------------------------------------------------------
    # Save Trajectory (assembly movie)
    # -------------------------------------------------------------------------
    if result.trajectory:
        frames = []
        print(f"\nProcessing {len(result.trajectory)} frames for trajectory...")
        
        for flat_pop in result.trajectory:
            # Pick the first particle from the population for visualization
            flat_p = flat_pop[0]
            coords = np.array(flat_p).reshape(-1, 3)
            state = State(
                coordinates=coords,
                radius=np.array(TYPE_RADII),
                copy_number=np.array(COPY_NUMBERS),
                particle_types=np.array(PARTICLE_TYPES),
                type_names=TYPE_NAMES
            )
            frames.append(state)
        
        save_trajectory(frames, "smc_assembly_trajectory.h5")
        print(f"Saved assembly trajectory: smc_assembly_trajectory.h5")
    
    # -------------------------------------------------------------------------
    # Save Best Final Configurations
    # -------------------------------------------------------------------------
    indices = np.argsort(np.array(result.log_probs))[::-1]
    
    final_states = []
    for idx in indices[:10]:
        flat_p = result.particles[idx]
        coords = np.array(flat_p).reshape(-1, 3)
        state = State(
            coordinates=coords,
            radius=np.array(TYPE_RADII),
            copy_number=np.array(COPY_NUMBERS),
            particle_types=np.array(PARTICLE_TYPES),
            type_names=TYPE_NAMES
        )
        final_states.append(state)
    
    save_trajectory(final_states, "smc_final_best.h5")
    print(f"Saved top 10 configurations: smc_final_best.h5")
    
    # -------------------------------------------------------------------------
    # Print Best Configuration Stats
    # -------------------------------------------------------------------------
    best_idx = indices[0]
    best_coords = result.particles[best_idx].reshape(-1, 3)
    print(f"\nBest configuration (index {best_idx}):")
    print(f"  Log prob: {result.log_probs[best_idx]:.2f}")
    print(f"  Coord range: [{np.min(best_coords):.1f}, {np.max(best_coords):.1f}]")
    
    print("\n✅ Done! Convert to RMF3 with:")
    print("   python io_utils/h5py_to_rmf3.py smc_assembly_trajectory.h5")


if __name__ == "__main__":
    main()