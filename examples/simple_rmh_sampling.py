import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime

print("JAX is using device:", jax.default_backend())

import blackjax

from scoring.harmonic import jax_harmonic_score
from scoring.exvol import jax_excluded_volume

# ============================================================
# Initialize System
# ============================================================

def initialize_particle_positions(n_particles, box_size, seed=0):
    """Initialize random particle positions."""
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (n_particles, 3), minval=0.0, maxval=box_size)
    radii = jnp.ones((n_particles,)) * 2.0
    indices = jnp.array([[0, 1]])  # Single pair for 2 particles
    return positions, radii, indices

def log_prob_fn(positions_flat, radii, indices):
    """Combined log probability from harmonic and excluded volume scores."""
    # Reshape flat array to (N, 3)
    positions = positions_flat.reshape(-1, 3)
    
    # Compute energies
    harmonic_energy = jax_harmonic_score(positions, indices, d0=4.5, k=0.5)
    exvol_energy = jax_excluded_volume(positions, radii, k=1.0)
    total_energy = harmonic_energy + exvol_energy
    
    return -total_energy  # Negative energy as log prob

# ============================================================
# MCMC Functions
# ============================================================

def setup_coordinate_rmh(log_prob_fn, step_size):
    """Set up Random Walk Metropolis-Hastings."""
    proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
    rmh_kernel = blackjax.rmh(log_prob_fn, proposal_distribution)
    return rmh_kernel

def run_rmh_chain_with_logging(
    key,
    kernel,
    initial_position,
    n_steps,
    save_every=100,
    print_every=500,
    output_file="mcmc_coords.h5"
):
    """
    Run RMH chain with progress logging and HDF5 saving.
    
    Args:
        key: JAX random key
        kernel: BlackJAX RMH kernel
        initial_position: Initial flat coordinates (N*3,)
        n_steps: Total MCMC steps
        save_every: Save coordinates every N steps
        print_every: Print progress every N steps
        output_file: HDF5 output filename
    
    Returns:
        saved_positions, saved_log_probs, acceptance_rate
    """
    state = kernel.init(initial_position)
    
    @jax.jit
    def one_step(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, (new_state.position, new_state.logdensity, info.is_accepted)
    
    keys = random.split(key, n_steps)
    
    # Storage - SAVE INITIAL CONFIGURATION FIRST
    saved_positions = [np.array(initial_position)]  # <-- ADD INITIAL CONFIG
    saved_log_probs = [float(state.logdensity)]      # <-- ADD INITIAL LOG PROB
    accepts = []
    
    print(f"{'='*70}")
    print(f"Starting MCMC Sampling")
    print(f"{'='*70}")
    print(f"Total steps: {n_steps:,}")
    print(f"Save every: {save_every}")
    print(f"Initial log_prob: {state.logdensity:.4f}")
    print(f"{'='*70}\n")
    
    for i, k in enumerate(keys):
        state, (position, log_prob, accepted) = one_step(state, k)
        accepts.append(float(accepted))
        
        # Save at intervals (i+1 because we already saved initial at index 0)
        if (i + 1) % save_every == 0:
            saved_positions.append(np.array(position))
            saved_log_probs.append(float(log_prob))
        
        # Print progress
        if (i + 1) % print_every == 0:
            recent_accept = np.mean(accepts[-min(1000, len(accepts)):])
            print(
                f"Step {i+1:7,} / {n_steps:,} | "
                f"log_prob: {log_prob:10.2f} | "
                f"accept: {recent_accept:5.1%}"
            )
    
    # Final stats
    acceptance_rate = np.mean(accepts)
    saved_positions = np.array(saved_positions)
    saved_log_probs = np.array(saved_log_probs)
    
    print(f"\n{'='*70}")
    print(f"MCMC Complete!")
    print(f"{'='*70}")
    print(f"Final log_prob: {log_prob:.4f}")
    print(f"Best log_prob: {np.max(saved_log_probs):.4f}")
    print(f"Overall acceptance: {acceptance_rate:.1%}")
    print(f"Saved {len(saved_positions)} frames (including initial)")
    
    # Save to HDF5
    save_mcmc_to_hdf5(
        saved_positions,
        saved_log_probs,
        acceptance_rate,
        output_file,
        initial_position  # <-- PASS INITIAL CONFIG
    )
    
    return saved_positions, saved_log_probs, acceptance_rate

def save_mcmc_to_hdf5(
    positions: np.ndarray,
    log_probs: np.ndarray,
    acceptance_rate: float,
    filename: str,
    initial_position: np.ndarray = None
):
    """
    Save MCMC samples to HDF5.
    
    Structure:
        coordinates: (n_samples, n_particles, 3)
        log_probabilities: (n_samples,)
        initial_configuration: (n_particles, 3)  <-- NEW
        best_configuration: (n_particles, 3)
    """
    n_samples = positions.shape[0]
    n_particles = positions.shape[1] // 3
    
    # Reshape positions from (n_samples, n_particles*3) to (n_samples, n_particles, 3)
    coords_reshaped = positions.reshape(n_samples, n_particles, 3)
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['n_particles'] = n_particles
        f.attrs['acceptance_rate'] = acceptance_rate
        
        # Coordinates
        f.create_dataset('coordinates', data=coords_reshaped, compression='gzip')
        
        # Log probabilities
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Initial configuration (NEW)
        if initial_position is not None:
            init_grp = f.create_group('initial_configuration')
            initial_coords = initial_position.reshape(n_particles, 3)
            init_grp.create_dataset('coordinates', data=initial_coords)
            print(f"\nInitial configuration saved:")
            print(f"  Coordinates:\n{initial_coords}")
        
        # Best configuration
        best_idx = np.argmax(log_probs)
        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = int(best_idx)
        best_grp.attrs['log_probability'] = float(log_probs[best_idx])
        best_grp.create_dataset('coordinates', data=coords_reshaped[best_idx])
    
    print(f"\nSaved to: {filename}")
    print(f"  Frames: {n_samples}")
    print(f"  Best log_prob: {np.max(log_probs):.2f} (frame {best_idx})")

# ============================================================
# Run the MCMC
# ============================================================

if __name__ == "__main__":
    # Initialize system
    n_particles = 11
    box_size = 110.0
    positions, radii, indices = initialize_particle_positions(n_particles, box_size, seed=42)
    
    print("Initial Positions:")
    print(positions)
    print("\nInitial distances:")
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dist = float(jnp.linalg.norm(positions[j] - positions[i]))
            print(f"  Particle {i} <-> {j}: {dist:.2f}")
    
    # Flatten positions for MCMC (shape: (n_particles*3,))
    initial_position = positions.ravel()
    
    # Create log probability function with fixed radii and indices
    def log_prob(coords_flat):
        return log_prob_fn(coords_flat, radii, indices)
    
    # Setup MCMC
    step_size = 4.0  # Adjust for acceptance rate ~20-40%
    rmh_kernel = setup_coordinate_rmh(log_prob, step_size)
    
    # Run MCMC
    positions_out, log_probs_out, acc_rate = run_rmh_chain_with_logging(
        key=random.PRNGKey(19827),
        kernel=rmh_kernel,
        initial_position=initial_position,
        n_steps=100_000,
        save_every=75,
        print_every=1000,
        output_file="simple_mcmc_coords.h5"
    )
    
    # Analyze results
    print(f"\n{'='*70}")
    print("Analysis:")
    print(f"{'='*70}")
    
    # Check initial vs final
    initial_coords = positions_out[0].reshape(n_particles, 3)
    final_coords = positions_out[-1].reshape(n_particles, 3)
    best_idx = np.argmax(log_probs_out)
    best_coords = positions_out[best_idx].reshape(n_particles, 3)
    
    print(f"\nInitial configuration (frame 0):")
    print(initial_coords)
    print(f"\nFinal configuration (frame {len(positions_out)-1}):")
    print(final_coords)
    print(f"\nBest configuration (frame {best_idx}):")
    print(best_coords)
    
    # Check if initial config matches what we started with
    print(f"\nVerification:")
    print(f"Initial coords match input: {np.allclose(initial_coords, positions)}")
