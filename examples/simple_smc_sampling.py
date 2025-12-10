#---------------------------------------------------------------
# testing the same code but using the SMC sampler in blackjax 
#---------------------------------------------------------------

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime

print("JAX is using device:", jax.default_backend())

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

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
# SMC Functions
# ============================================================

def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the adaptive tempered SMC algorithm."""
    
    # Condition: Run until lambda (tempering parameter) reaches 1.0
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

    # Run the loop
    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def save_smc_to_hdf5(
    particles: np.ndarray,
    log_probs: np.ndarray,
    filename: str
):
    """
    Save SMC population to HDF5.
    """
    n_samples = particles.shape[0]
    n_particles = particles.shape[1] // 3
    
    coords_reshaped = particles.reshape(n_samples, n_particles, 3)
    
    with h5py.File(filename, 'w') as f:
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['method'] = "Tempered SMC"
        
        f.create_dataset('coordinates', data=coords_reshaped, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Best configuration
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
    # We use this just to get radii and indices, positions will be randomized for the population
    _, radii, indices = initialize_particle_positions(n_particles, box_size, seed=42)
    
    # 2. Define Probabilities for SMC
    
    # Prior: Flat prior (return 0). 
    # In SMC, lambda=0 samples from this. 
    # This implies we start with a random walk (ideal gas) behavior.
    def prior_log_prob(x):
        return 0.0

    # Likelihood: The actual physics (Harmonic + Excluded Volume)
    def log_likelihood(coords_flat):
        return log_prob_fn(coords_flat, radii, indices)

    # 3. Define RMH Kernel Factory for SMC
    # SMC needs a function that creates a kernel given a logdensity function
    def rmh_kernel_factory(logdensity_fn, step_size):
        proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
        return blackjax.rmh(logdensity_fn, proposal_distribution)

    # 4. Initialize SMC Population
    num_smc_particles = 1000  # Number of parallel chains/replicas
    print(f"Initializing {num_smc_particles} replicas...")
    
    # Initialize random positions for all replicas
    init_key = jax.random.PRNGKey(1234)
    initial_positions = jax.random.uniform(
        init_key, 
        (num_smc_particles, n_particles * 3), 
        minval=0.0, 
        maxval=box_size
    )

    # 5. Setup Tempered SMC Algorithm
    step_size = 2.5  # RMH step size
    rmh_parameters = dict(step_size=step_size)

    tempered = blackjax.adaptive_tempered_smc(
        prior_log_prob,
        log_likelihood,
        rmh_kernel_factory,     # Use RMH instead of HMC
        blackjax.rmh.init,      # Use RMH init
        extend_params(rmh_parameters),
        resampling.systematic,
        target_ess=0.5,         # Resample when effective sample size drops below 50%
        num_mcmc_steps=10       # Number of RMH steps to take between tempering steps
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
    # Recalculate log probs for the final population
    final_log_probs# filepath: /Users/sreeganeshbalasubramani/git/jax_bhm_toymodel/examples/simple_smc_sampling.py
#---------------------------------------------------------------
# testing the same code but using the SMC sampler in blackjax 
#---------------------------------------------------------------

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime

print("JAX is using device:", jax.default_backend())

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

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
# SMC Functions
# ============================================================

def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the adaptive tempered SMC algorithm."""
    
    # Condition: Run until lambda (tempering parameter) reaches 1.0
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

    # Run the loop
    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def save_smc_to_hdf5(
    particles: np.ndarray,
    log_probs: np.ndarray,
    filename: str
):
    """
    Save SMC population to HDF5.
    """
    n_samples = particles.shape[0]
    n_particles = particles.shape[1] // 3
    
    coords_reshaped = particles.reshape(n_samples, n_particles, 3)
    
    with h5py.File(filename, 'w') as f:
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['method'] = "Tempered SMC"
        
        f.create_dataset('coordinates', data=coords_reshaped, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Best configuration
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
    # We use this just to get radii and indices, positions will be randomized for the population
    _, radii, indices = initialize_particle_positions(n_particles, box_size, seed=42)
    
    # 2. Define Probabilities for SMC
    
    # Prior: Flat prior (return 0). 
    # In SMC, lambda=0 samples from this. 
    # This implies we start with a random walk (ideal gas) behavior.
    def prior_log_prob(x):
        return 0.0

    # Likelihood: The actual physics (Harmonic + Excluded Volume)
    def log_likelihood(coords_flat):
        return log_prob_fn(coords_flat, radii, indices)

    # 3. Define RMH Kernel Factory for SMC
    # SMC needs a function that creates a kernel given a logdensity function
    def rmh_kernel_factory(logdensity_fn, step_size):
        proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
        return blackjax.rmh(logdensity_fn, proposal_distribution)

    # 4. Initialize SMC Population
    num_smc_particles = 1000  # Number of parallel chains/replicas
    print(f"Initializing {num_smc_particles} replicas...")
    
    # Initialize random positions for all replicas
    init_key = jax.random.PRNGKey(1234)
    initial_positions = jax.random.uniform(
        init_key, 
        (num_smc_particles, n_particles * 3), 
        minval=0.0, 
        maxval=box_size
    )

    # 5. Setup Tempered SMC Algorithm
    step_size = 2.5  # RMH step size
    rmh_parameters = dict(step_size=step_size)

    tempered = blackjax.adaptive_tempered_smc(
        prior_log_prob,
        log_likelihood,
        rmh_kernel_factory,     # Use RMH instead of HMC
        blackjax.rmh.init,      # Use RMH init
        extend_params(rmh_parameters),
        resampling.systematic,
        target_ess=0.5,         # Resample when effective sample size drops below 50%
        num_mcmc_steps=10       # Number of RMH steps to take between tempering steps
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
    # Recalculate log probs for the final population
    final_log_probs = jax.vmap(log_likelihood)(final_particles)
    
    # save results
    save_smc_to_hdf5(
        np.array(final_particles),
        np.array(final_log_probs),
        filename="smc_population.h5"
    )