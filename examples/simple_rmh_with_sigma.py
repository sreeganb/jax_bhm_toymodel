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
# Modified Scoring Function with Sigma as Parameter
# ============================================================

def harmonic_score_with_sigma(positions, indices, d0=2.0, sigma=1.0):
    """
    Harmonic score with Gaussian likelihood.
    
    Likelihood: p(d | d0, sigma) = (1/sqrt(2*pi*sigma^2)) * exp(-0.5 * ((d - d0)/sigma)^2)
    Log-likelihood: log p(d | d0, sigma) = -0.5 * log(2*pi*sigma^2) - 0.5 * ((d - d0)/sigma)^2
    
    Returns the negative log-likelihood (to be minimized as energy).
    """
    pos_i, pos_j = positions[indices[:, 0]], positions[indices[:, 1]]
    dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
    
    # Negative log-likelihood
    n_pairs = len(indices)
    log_likelihood = -0.5 * n_pairs * jnp.log(2 * jnp.pi * sigma**2)
    log_likelihood -= jnp.sum(0.5 * ((dists - d0) / sigma)**2)
    
    return -log_likelihood  # Return as energy (to minimize)

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

def log_prob_fn_with_sigma(state_flat, radii, indices, d0=4.5):
    """
    Combined log probability with sigma as a parameter.
    
    State structure: [x1, y1, z1, ..., xN, yN, zN, sigma]
    
    Args:
        state_flat: Flattened state (n_particles*3 + 1,)
        radii: Particle radii
        indices: Bond indices
        d0: Equilibrium distance
    
    Returns:
        log_prob: Log probability (negative total energy)
    """
    # Extract coordinates and sigma
    n_coords = len(state_flat) - 1  # Last element is sigma
    positions_flat = state_flat[:n_coords]
    sigma = state_flat[-1]  # Last element
    
    # Reshape positions
    positions = positions_flat.reshape(-1, 3)
    
    # Prior on sigma: Log-normal prior (sigma > 0)
    # p(sigma) = LogNormal(mu=0, tau=1) 
    # This ensures sigma stays positive
    sigma_prior = -0.5 * (jnp.log(sigma))**2 - jnp.log(sigma)  # Log-normal prior
    
    # Alternatively, use a uniform prior with a soft boundary:
    # sigma_prior = jax.nn.relu(0.1 - sigma) * 1e6  # Penalty if sigma < 0.1
    
    # Compute energies
    harmonic_energy = harmonic_score_with_sigma(positions, indices, d0=d0, sigma=sigma)
    exvol_energy = jax_excluded_volume(positions, radii, k=1.0)
    total_energy = harmonic_energy + exvol_energy
    
    # Log probability = -energy + prior
    return -total_energy + sigma_prior

# ============================================================
# MCMC Functions
# ============================================================

def setup_coordinate_rmh_with_sigma(log_prob_fn, step_size_coords, step_size_sigma):
    """
    Set up Random Walk Metropolis-Hastings with different step sizes.
    
    Args:
        log_prob_fn: Log probability function
        step_size_coords: Step size for coordinate moves (3N values)
        step_size_sigma: Step size for sigma moves (1 value)
    """
    # Concatenate step sizes: [step_coords, step_coords, ..., step_sigma]
    step_sizes = jnp.concatenate([step_size_coords, jnp.array([step_size_sigma])])
    
    proposal_distribution = blackjax.mcmc.random_walk.normal(step_sizes)
    rmh_kernel = blackjax.rmh(log_prob_fn, proposal_distribution)
    return rmh_kernel

def run_rmh_chain_with_logging(
    key,
    kernel,
    initial_state,
    n_steps,
    save_every=100,
    print_every=500,
    output_file="mcmc_coords_sigma.h5"
):
    """Run RMH chain with sigma sampling."""
    state = kernel.init(initial_state)
    
    @jax.jit
    def one_step(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, (new_state.position, new_state.logdensity, info.is_accepted)
    
    keys = random.split(key, n_steps)
    
    # Storage
    saved_states = [np.array(initial_state)]
    saved_log_probs = [float(state.logdensity)]
    accepts = []
    
    print(f"{'='*70}")
    print(f"Starting MCMC Sampling (Coordinates + Sigma)")
    print(f"{'='*70}")
    print(f"Total steps: {n_steps:,}")
    print(f"Save every: {save_every}")
    print(f"Initial log_prob: {state.logdensity:.4f}")
    print(f"Initial sigma: {initial_state[-1]:.4f}")
    print(f"{'='*70}\n")
    
    for i, k in enumerate(keys):
        state, (position, log_prob, accepted) = one_step(state, k)
        accepts.append(float(accepted))
        
        if (i + 1) % save_every == 0:
            saved_states.append(np.array(position))
            saved_log_probs.append(float(log_prob))
        
        if (i + 1) % print_every == 0:
            recent_accept = np.mean(accepts[-min(1000, len(accepts)):])
            current_sigma = float(position[-1])
            print(
                f"Step {i+1:7,} / {n_steps:,} | "
                f"log_prob: {log_prob:10.2f} | "
                f"sigma: {current_sigma:6.3f} | "
                f"accept: {recent_accept:5.1%}"
            )
    
    acceptance_rate = np.mean(accepts)
    saved_states = np.array(saved_states)
    saved_log_probs = np.array(saved_log_probs)
    
    print(f"\n{'='*70}")
    print(f"MCMC Complete!")
    print(f"{'='*70}")
    print(f"Final log_prob: {log_prob:.4f}")
    print(f"Final sigma: {float(position[-1]):.4f}")
    print(f"Best log_prob: {np.max(saved_log_probs):.4f}")
    print(f"Overall acceptance: {acceptance_rate:.1%}")
    print(f"Saved {len(saved_states)} frames")
    
    # Analyze sigma
    sigma_trace = saved_states[:, -1]
    print(f"\nSigma Statistics:")
    print(f"  Mean: {np.mean(sigma_trace):.4f}")
    print(f"  Std:  {np.std(sigma_trace):.4f}")
    print(f"  Min:  {np.min(sigma_trace):.4f}")
    print(f"  Max:  {np.max(sigma_trace):.4f}")
    
    save_mcmc_to_hdf5(
        saved_states,
        saved_log_probs,
        acceptance_rate,
        output_file,
        initial_state
    )
    
    return saved_states, saved_log_probs, acceptance_rate

def save_mcmc_to_hdf5(
    states: np.ndarray,
    log_probs: np.ndarray,
    acceptance_rate: float,
    filename: str,
    initial_state: np.ndarray = None
):
    """Save MCMC samples including sigma to HDF5."""
    n_samples = states.shape[0]
    n_coords = states.shape[1] - 1  # Last column is sigma
    n_particles = n_coords // 3
    
    # Split coordinates and sigma
    coords = states[:, :-1].reshape(n_samples, n_particles, 3)
    sigmas = states[:, -1]
    
    with h5py.File(filename, 'w') as f:
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['n_particles'] = n_particles
        f.attrs['acceptance_rate'] = acceptance_rate
        
        f.create_dataset('coordinates', data=coords, compression='gzip')
        f.create_dataset('sigma', data=sigmas, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Initial configuration
        if initial_state is not None:
            init_grp = f.create_group('initial_configuration')
            init_grp.create_dataset('coordinates', data=initial_state[:-1].reshape(n_particles, 3))
            init_grp.attrs['sigma'] = float(initial_state[-1])
        
        # Best configuration
        best_idx = np.argmax(log_probs)
        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = int(best_idx)
        best_grp.attrs['log_probability'] = float(log_probs[best_idx])
        best_grp.create_dataset('coordinates', data=coords[best_idx])
        best_grp.attrs['sigma'] = float(sigmas[best_idx])
    
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
    
    # Initial sigma value
    initial_sigma = 1.0
    
    # Flatten positions and append sigma
    initial_state = jnp.concatenate([positions.ravel(), jnp.array([initial_sigma])])
    
    print("Initial State:")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Initial sigma: {initial_sigma}")
    print(f"  State shape: {initial_state.shape}")  # Should be (3*N + 1,)
    
    # Create log probability function
    def log_prob(state_flat):
        return log_prob_fn_with_sigma(state_flat, radii, indices, d0=4.5)
    
    # Setup MCMC with different step sizes
    step_size_coords = jnp.ones(n_particles * 3) * 4.0  # Coordinate step size
    step_size_sigma = 0.1  # Sigma step size (smaller because it's a single parameter)
    
    rmh_kernel = setup_coordinate_rmh_with_sigma(
        log_prob, 
        step_size_coords, 
        step_size_sigma
    )
    
    # Run MCMC
    states_out, log_probs_out, acc_rate = run_rmh_chain_with_logging(
        key=random.PRNGKey(19827),
        kernel=rmh_kernel,
        initial_state=initial_state,
        n_steps=100_000,
        save_every=75,
        print_every=1000,
        output_file="mcmc_coords_with_sigma.h5"
    )
    
    # Analyze sigma convergence
    import matplotlib.pyplot as plt
    
    sigma_trace = states_out[:, -1]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(sigma_trace)
    plt.xlabel('Sample')
    plt.ylabel('Sigma')
    plt.title('Sigma Trace')
    
    plt.subplot(132)
    plt.hist(sigma_trace, bins=50, density=True, alpha=0.7)
    plt.xlabel('Sigma')
    plt.ylabel('Density')
    plt.title('Sigma Posterior')
    
    plt.subplot(133)
    plt.plot(log_probs_out)
    plt.xlabel('Sample')
    plt.ylabel('Log Probability')
    plt.title('Log Probability Trace')
    
    plt.tight_layout()
    plt.savefig('sigma_analysis.png', dpi=150)
    print("\nSaved sigma analysis plot to: sigma_analysis.png")