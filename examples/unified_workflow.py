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
from scoring.exvol import jax_excluded_volume
from io_utils.save_as_h5py import (
    save_particle_state,
    load_particle_state,
    save_particle_trajectory
)

print(f"{'='*70}")
print("Unified Workflow: ParticleState Through All 4 Stages")
print(f"{'='*70}\n")

# ============================================================
# Stage 1: REPRESENTATION
# ============================================================
print("Stage 1: REPRESENTATION")
print("-" * 70)

# Define particle types
particle_types = {
    "ProteinA": {"count": 5, "radius": 2.0},
    "ProteinB": {"count": 3, "radius": 3.0},
    "DNA": {"count": 2, "radius": 1.5}
}

# Create using factory
factory = ParticleSystemFactory(particle_types)
key = random.PRNGKey(42)
state = factory.create_random_state(key, box_size=50.0)

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

# Define restraints (connect first particle of each type)
restraints = jnp.array([
    [0, 5],   # ProteinA[0] - ProteinB[0]
    [5, 8],   # ProteinB[0] - DNA[0]
])

# Score using ParticleState directly
harmonic_energy = jax_harmonic_score(state, restraints, d0=5.0, k=0.5)
exvol_energy = jax_excluded_volume(state, k=1.0)
total_energy = harmonic_energy + exvol_energy

print(f"Energy scores:")
print(f"  Harmonic: {harmonic_energy:.2f}")
print(f"  Excluded volume: {exvol_energy:.2f}")
print(f"  Total: {total_energy:.2f}")
print()

# ============================================================
# Stage 3: SAMPLING
# ============================================================
print("Stage 3: SAMPLING")
print("-" * 70)

def log_prob(positions_flat):
    """Log probability function for MCMC."""
    positions = positions_flat.reshape(-1, 3)
    # Create temporary state with updated positions
    temp_state = ParticleState(
        positions=positions,
        radii=state.radii,
        particle_types=state.particle_types,
        copy_numbers=state.copy_numbers,
        type_names=state.type_names
    )
    h_energy = jax_harmonic_score(temp_state, restraints, d0=5.0, k=0.5)
    ev_energy = jax_excluded_volume(temp_state, k=1.0)
    return -(h_energy + ev_energy)

# Simple random walk MCMC
def mcmc_step(key, positions, step_size=0.5):
    """Single MCMC step with Metropolis-Hastings."""
    key1, key2 = random.split(key)
    
    # Propose move
    proposal = positions + random.normal(key1, positions.shape) * step_size
    
    # Accept/reject
    log_prob_current = log_prob(positions)
    log_prob_proposal = log_prob(proposal)
    log_accept = log_prob_proposal - log_prob_current
    
    accept = jnp.log(random.uniform(key2)) < log_accept
    new_positions = jnp.where(accept, proposal, positions)
    
    return new_positions, accept

# Run MCMC
n_steps = 1000
positions = state.positions.flatten()
key = random.PRNGKey(123)
trajectory = []
accepts = []

print(f"Running {n_steps} MCMC steps...")
for i in range(n_steps):
    key, subkey = random.split(key)
    positions, accept = mcmc_step(subkey, positions)
    
    if i % 100 == 0:
        trajectory.append(positions.reshape(-1, 3))
        accepts.append(accept)
        if i % 200 == 0:
            print(f"  Step {i}: log_prob = {log_prob(positions):.2f}, accept = {accept}")

acceptance_rate = np.mean(accepts)
print(f"Acceptance rate: {acceptance_rate:.2%}")
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
    positions=positions.reshape(-1, 3),
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
        "method": "MCMC",
        "n_steps": n_steps,
        "acceptance_rate": float(acceptance_rate)
    }
)

# Save trajectory
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

log_probs = np.array([log_prob(pos.flatten()) for pos in trajectory])
traj_file = output_dir / "trajectory.h5"
save_particle_trajectory(
    str(traj_file),
    trajectory_states,
    log_probs=log_probs,
    metadata={"n_steps": n_steps}
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