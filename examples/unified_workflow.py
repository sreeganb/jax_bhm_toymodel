#------------------------------------------------------------------------------
# Example: Unified ParticleState Workflow
# Demonstrates all 4 stages: Representation → Scoring → Sampling → Validation
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from functools import partial

print("JAX is using device:", jax.default_backend())

# Import from our package
from representation import ParticleState
from representation.generate_coords import (
    create_particle_state,
    create_random_particle_state,
    create_from_ideal_coords,
    get_pairwise_indices
)
from scoring import (
    jax_harmonic_score,
    jax_excluded_volume,
    jax_excluded_volume_typed
)
from io_utils.save_as_h5py import (
    save_particle_state,
    save_particle_trajectory,
    load_particle_state
)

# ============================================================
# STAGE 1: REPRESENTATION
# Define particles with positions, radii, types
# ============================================================

print("\n" + "="*70)
print("STAGE 1: REPRESENTATION")
print("="*70)

# Method 1: Create from type dictionary (recommended)
coords = {
    'A': jnp.array([
        [0., 0., 0.],
        [5., 0., 0.],
        [10., 0., 0.],
        [15., 0., 0.]
    ]),
    'B': jnp.array([
        [2.5, 4., 0.],
        [7.5, 4., 0.],
        [12.5, 4., 0.]
    ])
}

# Different radii per type - this is the key feature!
radii = {'A': 2.0, 'B': 1.5}
copy_numbers = {'A': 1, 'B': 2}

state = create_particle_state(coords, radii=radii, copy_numbers=copy_numbers)

print(f"\nCreated ParticleState:")
print(f"  Total particles: {state.n_particles}")
print(f"  Particle types: {list(state.type_names.values())}")
print(f"  Radii: {np.unique(state.radii)}")
print(f"  Position shape: {state.positions.shape}")

# Method 2: Random initialization (for MCMC starting points)
key = jax.random.PRNGKey(42)
random_state = create_random_particle_state(
    key,
    n_dict={'A': 4, 'B': 3},
    radii=radii,
    box_size=50.0
)
print(f"\nRandom state: {random_state.n_particles} particles in [-25, 25]^3 box")

# Method 3: From ideal/reference coordinates
ideal_state = create_from_ideal_coords(radii={'A': 5.0, 'B': 5.0, 'C': 3.0})
print(f"Ideal state: {ideal_state.n_particles} particles (A:8, B:8, C:16)")


# ============================================================
# STAGE 2: SCORING
# Compute energies using positions + radii from ParticleState
# ============================================================

print("\n" + "="*70)
print("STAGE 2: SCORING")
print("="*70)

# Generate restraint indices (e.g., connect consecutive A particles)
indices_sequential = jnp.array([[0, 1], [1, 2], [2, 3]])  # A-A chain
indices_cross = get_pairwise_indices(state, type_pairs=[('A', 'B')])

print(f"\nRestraint indices:")
print(f"  Sequential (A chain): {indices_sequential.shape[0]} pairs")
print(f"  Cross-type (A-B): {indices_cross.shape[0]} pairs")

# Harmonic restraints - ParticleState passed directly!
E_harmonic = jax_harmonic_score(state, indices_sequential, d0=5.0, k=0.5)
print(f"\nHarmonic energy (A chain, d0=5.0): {E_harmonic:.4f}")

# Excluded volume - radii extracted automatically from state!
E_exvol = jax_excluded_volume(state, k=1.0)
print(f"Excluded volume energy: {E_exvol:.4f}")

# Type-aware excluded volume (different k for different type pairs)
k_matrix = jnp.array([
    [1.0, 0.5],  # A-A, A-B
    [0.5, 2.0]   # B-A, B-B
])
E_exvol_typed = jax_excluded_volume_typed(state, k_matrix=k_matrix)
print(f"Type-aware excluded volume: {E_exvol_typed:.4f}")

# Total energy as log probability
total_energy = E_harmonic + E_exvol
log_prob = -total_energy
print(f"\nTotal energy: {total_energy:.4f}")
print(f"Log probability: {log_prob:.4f}")


# ============================================================
# STAGE 3: SAMPLING
# Use BlackJAX with ParticleState for MCMC
# ============================================================

print("\n" + "="*70)
print("STAGE 3: SAMPLING (Mini MCMC Demo)")
print("="*70)

# Define log probability function using closure over state attributes
def make_log_prob(state_template, indices, d0=5.0, k_harm=0.5, k_exvol=1.0):
    """Create log prob function that preserves particle attributes."""
    radii = state_template.radii  # Captured in closure
    
    def log_prob_fn(positions_flat):
        positions = positions_flat.reshape(-1, 3)
        E_harm = jax_harmonic_score(positions, indices, d0=d0, k=k_harm)
        E_exvol = jax_excluded_volume(positions, radii, k=k_exvol)
        return -(E_harm + E_exvol)
    
    return log_prob_fn

log_prob = make_log_prob(state, indices_sequential, d0=5.0)

# Setup RMH sampler
step_size = jnp.ones(state.n_particles * 3) * 0.5
proposal = blackjax.mcmc.random_walk.normal(step_size)
kernel = blackjax.rmh(log_prob, proposal)

# Initialize sampler with flattened positions
init_position = state.flat_positions
rmh_state = kernel.init(init_position)

print(f"\nInitial log probability: {rmh_state.logdensity:.4f}")

# Run a few steps
key = jax.random.PRNGKey(123)
n_steps = 100
trajectory = []
log_probs = []

current_state = rmh_state
for i in range(n_steps):
    key, subkey = jax.random.split(key)
    current_state, info = kernel.step(subkey, current_state)
    
    if i % 10 == 0:
        # Store as ParticleState (preserving all attributes!)
        new_particle_state = state.with_positions(current_state.position)
        trajectory.append(new_particle_state)
        log_probs.append(float(current_state.logdensity))

print(f"Ran {n_steps} MCMC steps, saved {len(trajectory)} frames")
print(f"Log prob range: [{min(log_probs):.2f}, {max(log_probs):.2f}]")
print(f"Best log prob: {max(log_probs):.2f} at frame {np.argmax(log_probs)}")


# ============================================================
# STAGE 4: VALIDATION
# Save results with full attribute preservation
# ============================================================

print("\n" + "="*70)
print("STAGE 4: VALIDATION (Save/Load)")
print("="*70)

# Save trajectory with all particle attributes preserved
output_file = "unified_mcmc_trajectory.h5"
save_particle_trajectory(
    output_file,
    trajectory,
    log_probs=np.array(log_probs),
    metadata={
        'method': 'RMH-MCMC',
        'n_steps': n_steps,
        'd0': 5.0,
        'k_harmonic': 0.5,
        'k_exvol': 1.0
    }
)

# Demonstrate loading
print("\nLoading and verifying...")
loaded_trajectory, loaded_probs = __import__('io_utils.save_as_h5py', fromlist=['load_particle_trajectory']).load_particle_trajectory(output_file)

print(f"  Loaded {len(loaded_trajectory)} frames")
print(f"  Particles per frame: {loaded_trajectory[0].n_particles}")
print(f"  Preserved radii: {loaded_trajectory[0].radii[:4]}")  # First 4 = type A
print(f"  Preserved types: {loaded_trajectory[0].type_names}")

# Verify attributes are preserved
assert jnp.allclose(loaded_trajectory[0].radii, state.radii), "Radii should be preserved!"
assert loaded_trajectory[0].type_names == state.type_names, "Type names should be preserved!"
print("\n✓ All particle attributes successfully preserved through pipeline!")


# ============================================================
# BONUS: Gradient computation with ParticleState
# ============================================================

print("\n" + "="*70)
print("BONUS: Gradient Computation")
print("="*70)

# JAX gradients work seamlessly with our scoring functions
@jax.jit
def energy_fn(positions, radii, indices):
    E1 = jax_harmonic_score(positions, indices, d0=5.0, k=0.5)
    E2 = jax_excluded_volume(positions, radii, k=1.0)
    return E1 + E2

grad_fn = jax.grad(energy_fn)
gradients = grad_fn(state.positions, state.radii, indices_sequential)

print(f"Gradient shape: {gradients.shape}")
print(f"Gradient norm: {jnp.linalg.norm(gradients):.4f}")
print(f"Mean |gradient|: {jnp.mean(jnp.abs(gradients)):.4f}")

# This enables gradient-based sampling (HMC, NUTS) in the future!
print("\n✓ Gradients computed successfully - ready for HMC/NUTS!")

print("\n" + "="*70)
print("COMPLETE: All 4 Bayesian inference stages demonstrated")
print("="*70)
