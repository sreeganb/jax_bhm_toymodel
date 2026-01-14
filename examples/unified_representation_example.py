#!/usr/bin/env python
"""
Example: Unified ParticleState representation for Bayesian inference.

Demonstrates the 4-stage pipeline:
1. REPRESENTATION - Create ParticleState with typed particles
2. SCORING - Compute energies using harmonic + excluded volume
3. SAMPLING - Run MCMC using blackjax
4. VALIDATION - Save trajectory to HDF5
"""

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime

print("JAX backend:", jax.default_backend())

# Local imports
from representation import ParticleState
from representation.generate_coords import (
    create_particle_state,
    create_random_particle_state,
    get_pairwise_indices
)
from scoring import jax_harmonic_score, jax_excluded_volume
from io_utils.save_as_h5py import save_particle_state, save_particle_trajectory

import blackjax


# ============================================================
# STAGE 1: REPRESENTATION
# ============================================================
print("\n" + "="*70)
print("STAGE 1: REPRESENTATION")
print("="*70)

# Define particle types with different radii
# Simulating a simple protein complex: 
#   - Type A: Large domain (radius 5.0)
#   - Type B: Medium subunit (radius 3.0)  
#   - Type C: Flexible linker beads (radius 1.5)

key = jax.random.PRNGKey(42)

# Create initial state with random positions
state = create_random_particle_state(
    key=key,
    n_dict={'A': 4, 'B': 6, 'C': 8},
    radii={'A': 5.0, 'B': 3.0, 'C': 1.5},
    copy_numbers={'A': 1, 'B': 2, 'C': 1},
    box_size=100.0,
    center=True
)

print(f"Created ParticleState:")
print(f"  Total particles: {state.n_particles}")
print(f"  Types: {state.type_names}")
print(f"  Radii range: {state.radii.min():.1f} - {state.radii.max():.1f}")
print(f"  Position bounds: [{state.positions.min():.1f}, {state.positions.max():.1f}]")


# ============================================================
# STAGE 2: SCORING  
# ============================================================
print("\n" + "="*70)
print("STAGE 2: SCORING")
print("="*70)

# Generate restraint indices (all pairs for simplicity)
indices = get_pairwise_indices(state)
print(f"Generated {len(indices)} pairwise restraints")

# Define log probability function
def log_prob_fn(positions_flat):
    """Combined log probability from harmonic and excluded volume."""
    positions = positions_flat.reshape(-1, 3)
    
    # Harmonic restraints (soft connectivity)
    E_harmonic = jax_harmonic_score(positions, indices, d0=10.0, k=0.1)
    
    # Excluded volume (using global radii from state)
    E_exvol = jax_excluded_volume(positions, state.radii, k=1.0)
    
    return -(E_harmonic + E_exvol)

# JIT compile
log_prob_jit = jax.jit(log_prob_fn)

# Test scoring
initial_log_prob = log_prob_jit(state.flat_positions)
print(f"Initial log probability: {initial_log_prob:.2f}")


# ============================================================
# STAGE 3: SAMPLING
# ============================================================
print("\n" + "="*70)
print("STAGE 3: SAMPLING (Random Walk Metropolis-Hastings)")
print("="*70)

# Setup MCMC kernel
step_size = jnp.ones(state.n_particles * 3) * 1.0  # Step size per DOF
proposal = blackjax.mcmc.random_walk.normal(step_size)
kernel = blackjax.rmh(log_prob_jit, proposal)

# Initialize sampler state
init_state = kernel.init(state.flat_positions)

# Run MCMC
n_steps = 5000
save_every = 50

@jax.jit
def step_fn(key, mcmc_state):
    return kernel.step(key, mcmc_state)

keys = jax.random.split(jax.random.PRNGKey(123), n_steps)
saved_positions = [np.array(state.positions)]
saved_log_probs = [float(init_state.logdensity)]
accepts = []

print(f"Running {n_steps} MCMC steps...")
mcmc_state = init_state

for i, k in enumerate(keys):
    mcmc_state, info = step_fn(k, mcmc_state)
    accepts.append(float(info.is_accepted))
    
    if (i + 1) % save_every == 0:
        saved_positions.append(np.array(mcmc_state.position.reshape(-1, 3)))
        saved_log_probs.append(float(mcmc_state.logdensity))
    
    if (i + 1) % 1000 == 0:
        acc_rate = np.mean(accepts[-1000:])
        print(f"  Step {i+1}: log_prob = {mcmc_state.logdensity:.2f}, accept = {acc_rate:.1%}")

print(f"\nSampling complete!")
print(f"  Final log_prob: {mcmc_state.logdensity:.2f}")
print(f"  Best log_prob: {max(saved_log_probs):.2f}")
print(f"  Acceptance rate: {np.mean(accepts):.1%}")


# ============================================================
# STAGE 4: VALIDATION (Save to HDF5)
# ============================================================
print("\n" + "="*70)
print("STAGE 4: VALIDATION (Save results)")
print("="*70)

# Create trajectory of ParticleStates
trajectory_states = [
    state.with_positions(jnp.array(pos)) 
    for pos in saved_positions
]

# Save trajectory with full metadata
output_file = "unified_mcmc_trajectory.h5"
save_particle_trajectory(
    filename=output_file,
    states=trajectory_states,
    log_probs=np.array(saved_log_probs),
    metadata={
        'method': 'RMH',
        'n_steps': n_steps,
        'acceptance_rate': float(np.mean(accepts)),
        'step_size': 1.0
    }
)

# Also save the best configuration separately
best_idx = np.argmax(saved_log_probs)
best_state = trajectory_states[best_idx]
save_particle_state(
    filename="unified_best_state.h5",
    state=best_state,
    metadata={
        'source_trajectory': output_file,
        'frame_index': int(best_idx),
        'log_probability': float(saved_log_probs[best_idx])
    }
)

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"Outputs:")
print(f"  - {output_file} (full trajectory)")
print(f"  - unified_best_state.h5 (best configuration)")
print(f"\nTo visualize, convert to RMF3:")
print(f"  python io_utils/h5py_to_rmf3.py {output_file}")
