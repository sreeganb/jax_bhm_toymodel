"""
Unified Workflow Example: Demonstrating ParticleState throughout all 4 stages.

This example shows:
1. Representation: Creating ParticleState with types and radii
2. Scoring: Using ParticleState directly in scoring functions
3. Sampling: MCMC sampling with ParticleState
4. Validation: Saving/loading ParticleState to/from HDF5
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime

print("JAX is using device:", jax.default_backend())

# Now import our modules
from representation import ParticleState, ParticleSystemFactory
from representation.generate_coords import generate_random_coords
from scoring.harmonic import jax_harmonic_score
from scoring.harmonic_newrestraint import (
    harmonic_pair_score_simple,
    calculate_pair_scores_matrix,
)
from scoring.exvol import jax_excluded_volume
from io_utils.save_as_h5py import (
    save_particle_state,
    load_particle_state,
    save_particle_trajectory
)
from samplers.smc import run_smc_simple, SMCConfig

print(f"{'='*70}")
print("Unified Workflow: ParticleState Through All 4 Stages")
print(f"{'='*70}\n")

# ============================================================
# Stage 1: REPRESENTATION
# ============================================================
print("Stage 1: REPRESENTATION")
print("-" * 70)

# Define particle types: 3 proteins with copy numbers 8, 8, 16 and radii 24, 14, 16
particle_types = {
    "A": {"count": 8, "radius": 24.0},
    "B": {"count": 8, "radius": 14.0},
    "C": {"count": 16, "radius": 16.0}
}

# Create using factory
factory = ParticleSystemFactory(particle_types)
key = random.PRNGKey(42)
state = factory.create_random_state(key, box_size=500.0)

print(f"Created ParticleState:")
print(f"  Total particles: {state.n_particles}")
print(f"  Types: {list(state.type_names.values())}")
print(f"  Radii range: {jnp.min(state.radii):.2f} - {jnp.max(state.radii):.2f}")
print(f"  Copy numbers: {dict(zip(state.type_names.values(), [state.get_particles_by_type(i).shape[0] for i in range(len(state.type_names))]))}")
print()

# ============================================================
# Stage 2: SCORING
# ============================================================
print("Stage 2: SCORING")
print("-" * 70)

# Define type bounds for efficient slicing (particles are sorted by type)
# A: indices 0-7, B: indices 8-15, C: indices 16-31
type_bounds = {
    "A": (0, 8),
    "B": (8, 16),
    "C": (16, 32)
}

# Define pair configuration with target distances
# These are the pair types and their equilibrium distances
pair_config = {
    "AA": {"target_dist": 48.0},   # A-A pairs (sum of radii)
    "AB": {"target_dist": 38.0},   # A-B pairs
    "BC": {"target_dist": 30.0},   # B-C pairs
}

# Sigma values for each pair type (uncertainty parameter)
sigmas = {
    "AA": 5.0,
    "AB": 5.0,
    "BC": 5.0,
}

# Score using new harmonic pair restraint with union-of-argmin
harmonic_energy = harmonic_pair_score_simple(
    state.positions,
    state.particle_types,
    type_bounds,
    pair_config,
    sigmas,
    pair_weight=1.0
)
exvol_energy = jax_excluded_volume(state.positions, state.radii, k=1.0)
total_energy = harmonic_energy + exvol_energy

print(f"Energy scores:")
print(f"  Harmonic (union-of-argmin): {harmonic_energy:.2f}")
print(f"  Excluded volume: {exvol_energy:.2f}")
print(f"  Total: {total_energy:.2f}")
print()

# ============================================================
# Stage 3: SAMPLING (using BlackJAX-style SMC)
# ============================================================
print("Stage 3: SAMPLING")
print("-" * 70)

# Create JIT-compiled log probability function
# IMPORTANT: The function must be JIT-compatible (no Python dicts inside)
# We'll create a closure that captures the static config

# Pre-extract arrays for JIT compatibility
particle_types_arr = state.particle_types
radii_arr = state.radii

# Type bounds as arrays (start, end) for each type in order A, B, C
type_starts = jnp.array([0, 8, 16])
type_ends = jnp.array([8, 16, 32])

# Pair config as arrays: (type1_idx, type2_idx, target_dist, sigma)
# AA=0, AB=1, BC=2
pair_type1_idx = jnp.array([0, 0, 1])  # A, A, B
pair_type2_idx = jnp.array([0, 1, 2])  # A, B, C
pair_target_dists = jnp.array([48.0, 38.0, 30.0])
pair_sigmas = jnp.array([5.0, 5.0, 5.0])

@jax.jit
def log_prob_jit(positions_flat: jnp.ndarray) -> float:
    """
    JIT-compiled log probability function for sampling.
    
    This is MUCH faster than the Python dict version because:
    1. Entire function compiles to optimized GPU/CPU code
    2. No Python interpreter overhead
    3. All operations fused by XLA
    """
    positions = positions_flat.reshape(-1, 3)
    
    # Harmonic pair score with union-of-argmin
    h_energy = harmonic_pair_score_simple(
        positions,
        particle_types_arr,
        type_bounds,  # This is a Python dict but only used for indexing
        pair_config,
        sigmas,
        pair_weight=1.0
    )
    
    # Excluded volume
    ev_energy = jax_excluded_volume(positions, radii_arr, k=1.0)
    
    return -(h_energy + ev_energy)

# Configure SMC sampler
smc_config = SMCConfig(
    n_particles=500,           # Number of parallel particles (increase for better sampling)
    n_mcmc_steps=5,            # MCMC steps per SMC iteration
    mcmc_step_size=2.0,        # Initial step size (will adapt)
    target_ess=0.5,            # Target effective sample size ratio
    max_iterations=100,        # Maximum SMC iterations
    adaptive_step_size=True,   # Adapt step size based on acceptance
)

# Initialize particles from current state (with some noise)
n_smc_particles = smc_config.n_particles
dim = state.n_particles * 3
key = random.PRNGKey(123)
key, init_key = random.split(key)

# Initialize particles: replicate initial state with small perturbations
initial_positions = state.positions.flatten()
noise_scale = 10.0  # Add some noise to spread out initial particles
initial_particles = initial_positions[None, :] + random.normal(
    init_key, (n_smc_particles, dim)
) * noise_scale

print(f"Running SMC with {n_smc_particles} particles...")
print(f"  Dimension: {dim} (= {state.n_particles} particles × 3)")
print()

# Run SMC
key, smc_key = random.split(key)
smc_result = run_smc_simple(
    key=smc_key,
    log_prob_fn=log_prob_jit,
    initial_positions=initial_particles,
    config=smc_config,
    verbose=True
)

print()
print(f"SMC Results:")
print(f"  Iterations: {smc_result.n_iterations}")
print(f"  Final ESS: {smc_result.ess_history[-1]:.1f}")
print(f"  Best log_prob: {jnp.max(smc_result.log_probs):.2f}")
print(f"  Mean log_prob: {jnp.mean(smc_result.log_probs):.2f}")

# Extract trajectory (subsample SMC particles for saving)
n_save = min(100, n_smc_particles)
trajectory = [smc_result.particles[i].reshape(-1, 3) for i in range(n_save)]
trajectory_log_probs = smc_result.log_probs[:n_save]

# Best particle
best_idx = jnp.argmax(smc_result.log_probs)
final_positions = smc_result.particles[best_idx]

acceptance_rate = float(jnp.mean(smc_result.acceptance_rates)) if len(smc_result.acceptance_rates) > 0 else 0.0
print(f"Mean acceptance rate: {acceptance_rate:.2%}")
print()

# ============================================================
# Stage 4: VALIDATION
# ============================================================
print("Stage 4: VALIDATION")
print("-" * 70)

# Save final state
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

final_state = ParticleState(
    positions=final_positions.reshape(-1, 3),
    radii=state.radii,
    particle_types=state.particle_types,
    copy_numbers=state.copy_numbers,
    type_names=state.type_names
)

state_file = output_dir / "final_state.h5"
save_particle_state(
    str(state_file),
    final_state,
    metadata={
        "method": "SMC",
        "n_particles": n_smc_particles,
        "n_iterations": smc_result.n_iterations,
        "acceptance_rate": acceptance_rate
    }
)

# Save trajectory (SMC population samples)
trajectory_states = [
    ParticleState(
        positions=pos,
        radii=state.radii,
        particle_types=state.particle_types,
        copy_numbers=state.copy_numbers,
        type_names=state.type_names
    )
    for pos in trajectory
]

log_probs = np.array(trajectory_log_probs)
traj_file = output_dir / "trajectory.h5"
save_particle_trajectory(
    str(traj_file),
    trajectory_states,
    log_probs=log_probs,
    metadata={
        "method": "SMC",
        "n_smc_particles": n_smc_particles,
        "n_iterations": smc_result.n_iterations
    }
)

# Test loading
print("\nTesting load/save roundtrip...")
loaded_state = load_particle_state(str(state_file))
print(f"  Loaded {loaded_state.n_particles} particles")
print(f"  Position match: {jnp.allclose(loaded_state.positions, final_state.positions)}")
print(f"  Radii match: {jnp.allclose(loaded_state.radii, final_state.radii)}")
print(f"  Types match: {jnp.all(loaded_state.particle_types == final_state.particle_types)}")

print(f"\n{'='*70}")
print("Workflow complete! Check output/ directory for results.")
print(f"{'='*70}")
