#---------------------------------------------------------------
# testing the same code but using the SMC sampler in blackjax 
#---------------------------------------------------------------

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime
from functools import partial

print("JAX is using device:", jax.default_backend())

import blackjax
import blackjax.smc.resampling as resampling

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
    positions = positions_flat.reshape(-1, 3)
    harmonic_energy = jax_harmonic_score(positions, indices, d0=4.5, k=0.5)
    exvol_energy = jax_excluded_volume(positions, radii, k=1.0)
    total_energy = harmonic_energy + exvol_energy
    return -total_energy

# ============================================================
# SMC Functions
# ============================================================

def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the adaptive tempered SMC algorithm."""
    
    def cond(carry):
        i, state, _k = carry
        return state.tempering_param < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_kernel(subk, state)
        return i + 1, state, k

    print(f"{'='*70}")
    print(f"Starting Tempered SMC")
    print(f"{'='*70}")

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def save_smc_to_hdf5(particles, log_probs, filename):
    """Save SMC population to HDF5."""
    n_samples = particles.shape[0]
    n_particles = particles.shape[1] // 3
    coords_reshaped = particles.reshape(n_samples, n_particles, 3)
    
    with h5py.File(filename, 'w') as f:
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['method'] = "Tempered SMC"
        f.create_dataset('coordinates', data=coords_reshaped, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        best_idx = np.argmax(log_probs)
        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = int(best_idx)
        best_grp.attrs['log_probability'] = float(log_probs[best_idx])
        best_grp.create_dataset('coordinates', data=coords_reshaped[best_idx])
    
    print(f"\nSaved to: {filename}")
    print(f"  Population size: {n_samples}")
    print(f"  Best log_prob: {np.max(log_probs):.2f}")

# ============================================================
# Run the SMC
# ============================================================
if __name__ == "__main__":
    # 1. System Setup
    n_particles = 11
    box_size = 110.0
    _, radii, indices = initialize_particle_positions(n_particles, box_size, seed=42)
    
    # 2. Define Probabilities for SMC
    def prior_log_prob(x):
        return 0.0

    def log_likelihood(coords_flat):
        return log_prob_fn(coords_flat, radii, indices)

    # 3. Define RMH Kernel for SMC
    step_size = jnp.ones(n_particles * 3) * 2.5
    proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
    
    # Create a wrapped build_kernel that bakes in the proposal
    def rmh_build_kernel():
        base_kernel = blackjax.rmh.build_kernel()
        
        def wrapped_kernel(rng_key, state, logdensity_fn):
            return base_kernel(rng_key, state, logdensity_fn, proposal_distribution)
        
        return wrapped_kernel

    # 4. Initialize SMC Population
    num_smc_particles = 1000
    print(f"Initializing {num_smc_particles} replicas...")
    
    init_key = jax.random.PRNGKey(1234)
    initial_positions = jax.random.uniform(
        init_key, 
        (num_smc_particles, n_particles * 3), 
        minval=0.0, 
        maxval=box_size
    )

    # 5. Setup Tempered SMC Algorithm
    tempered = blackjax.adaptive_tempered_smc(
        prior_log_prob,
        log_likelihood,
        rmh_build_kernel(),         # Our wrapped kernel
        blackjax.rmh.init,          # RMH init
        {},                         # Empty params dict (proposal is baked in)
        resampling.systematic,
        target_ess=0.5,
        num_mcmc_steps=10
    )

    # Initialize the SMC state
    initial_smc_state = tempered.init(initial_positions)

    # 6. Run Inference
    rng_key = jax.random.PRNGKey(5678)
    n_iter, final_state = smc_inference_loop(rng_key, tempered.step, initial_smc_state)
    
    print(f"\nSMC Complete in {n_iter} adaptive steps.")
    print(f"Final tempering parameter lambda: {final_state.tempering_param:.4f}")

    # 7. Analyze and Save
    final_particles = final_state.particles
    final_log_probs = jax.vmap(log_likelihood)(final_particles)
    
    save_smc_to_hdf5(
        np.array(final_particles),
        np.array(final_log_probs),
        filename="smc_population.h5"
    )