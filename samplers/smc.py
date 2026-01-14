#------------------------------------------------------------------------------
# Sequential Monte Carlo (SMC) sampler using BlackJAX
# 
# SMC is a powerful sampling method that:
# 1. Gradually tempers from prior to posterior (avoids getting stuck)
# 2. Runs many particles in parallel (GPU-friendly)
# 3. Resamples to focus on high-probability regions
# 4. Works well for multi-modal distributions
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, NamedTuple, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial
import blackjax
import blackjax.smc.resampling as resampling


@dataclass
class SMCConfig:
    """Configuration for SMC sampler."""
    n_particles: int = 1000          # Number of parallel particles
    n_mcmc_steps: int = 5            # MCMC steps per SMC iteration
    mcmc_step_size: float = 1.0      # Initial MCMC step size
    target_ess: float = 0.5          # Target effective sample size ratio
    max_iterations: int = 100        # Maximum SMC iterations
    adaptive_step_size: bool = True  # Whether to adapt step size
    min_step_size: float = 0.01      # Minimum step size
    max_step_size: float = 10.0      # Maximum step size


class SMCState(NamedTuple):
    """State of SMC sampler."""
    particles: jnp.ndarray      # (n_particles, dim)
    log_weights: jnp.ndarray    # (n_particles,)
    log_probs: jnp.ndarray      # (n_particles,)
    temperature: float          # Current tempering parameter
    step_size: float            # Current MCMC step size
    iteration: int              # Current iteration


class SMCResult(NamedTuple):
    """Result from SMC sampling."""
    particles: jnp.ndarray      # Final particles (n_particles, dim)
    log_weights: jnp.ndarray    # Final log weights
    log_probs: jnp.ndarray      # Final log probabilities
    temperatures: jnp.ndarray   # Temperature schedule
    ess_history: jnp.ndarray    # ESS at each iteration
    acceptance_rates: jnp.ndarray  # Acceptance rates
    n_iterations: int           # Total iterations


def make_log_prob_fn(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
) -> Callable:
    """
    Create a log probability function from prior and likelihood.
    
    The tempering schedule interpolates:
        log_prob(x, temp) = log_prior(x) + temp * log_likelihood(x)
    
    At temp=0: samples from prior
    At temp=1: samples from full posterior
    """
    def log_prob(x: jnp.ndarray, temperature: float) -> float:
        return log_prior_fn(x) + temperature * log_likelihood_fn(x)
    return log_prob


def compute_ess(log_weights: jnp.ndarray) -> float:
    """
    Compute effective sample size from log weights.
    
    ESS = (sum(w))^2 / sum(w^2) = 1 / sum(normalized_w^2)
    """
    # Normalize in log space for numerical stability
    max_log_w = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log_w)
    normalized_weights = weights / jnp.sum(weights)
    ess = 1.0 / jnp.sum(normalized_weights**2)
    return ess


def find_next_temperature(
    log_likelihoods: jnp.ndarray,
    current_temp: float,
    target_ess_ratio: float,
    n_particles: int,
) -> float:
    """
    Find next temperature using bisection to achieve target ESS.
    
    Uses binary search to find the temperature increment that gives
    the desired effective sample size after reweighting.
    """
    target_ess = target_ess_ratio * n_particles
    
    def ess_at_temp(next_temp):
        delta_temp = next_temp - current_temp
        log_weights = delta_temp * log_likelihoods
        return compute_ess(log_weights)
    
    # Binary search for temperature
    low, high = current_temp, 1.0
    
    # Check if we can go directly to temp=1
    if ess_at_temp(1.0) >= target_ess:
        return 1.0
    
    # Binary search
    for _ in range(50):  # Max iterations for bisection
        mid = (low + high) / 2
        ess = ess_at_temp(mid)
        if ess > target_ess:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    
    return low


def rwm_kernel(
    key: jax.Array,
    position: jnp.ndarray,
    log_prob_fn: Callable,
    step_size: float,
) -> Tuple[jnp.ndarray, bool]:
    """
    Single Random Walk Metropolis step.
    
    This is a simple MCMC kernel that proposes Gaussian moves.
    """
    key1, key2 = random.split(key)
    
    # Propose
    proposal = position + random.normal(key1, position.shape) * step_size
    
    # Accept/reject
    log_prob_current = log_prob_fn(position)
    log_prob_proposal = log_prob_fn(proposal)
    log_accept_ratio = log_prob_proposal - log_prob_current
    
    accept = jnp.log(random.uniform(key2)) < log_accept_ratio
    new_position = jnp.where(accept, proposal, position)
    
    return new_position, accept


def run_mcmc_chain(
    key: jax.Array,
    initial_position: jnp.ndarray,
    log_prob_fn: Callable,
    n_steps: int,
    step_size: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Run multiple MCMC steps and return final position + acceptance rate.
    """
    def step(carry, key):
        position, n_accepts = carry
        new_position, accept = rwm_kernel(key, position, log_prob_fn, step_size)
        return (new_position, n_accepts + accept.astype(float)), None
    
    keys = random.split(key, n_steps)
    (final_position, n_accepts), _ = jax.lax.scan(step, (initial_position, 0.0), keys)
    acceptance_rate = n_accepts / n_steps
    
    return final_position, acceptance_rate


@partial(jax.jit, static_argnums=(2, 3, 4))
def smc_mutation_step(
    key: jax.Array,
    particles: jnp.ndarray,
    log_prob_fn: Callable,
    n_mcmc_steps: int,
    step_size: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Mutation step: run MCMC on each particle in parallel.
    
    JAX SPEEDUP: vmap parallelizes across all particles on GPU.
    """
    n_particles = particles.shape[0]
    keys = random.split(key, n_particles)
    
    # Vectorize over particles
    vmapped_chain = jax.vmap(
        lambda k, p: run_mcmc_chain(k, p, log_prob_fn, n_mcmc_steps, step_size)
    )
    
    new_particles, acceptance_rates = vmapped_chain(keys, particles)
    new_log_probs = jax.vmap(log_prob_fn)(new_particles)
    mean_acceptance = jnp.mean(acceptance_rates)
    
    return new_particles, new_log_probs, mean_acceptance


def systematic_resampling(
    key: jax.Array,
    particles: jnp.ndarray,
    log_weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Systematic resampling of particles based on weights.
    
    This is more efficient than multinomial resampling and
    reduces variance in the resampled population.
    """
    n_particles = particles.shape[0]
    
    # Normalize weights
    max_log_w = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log_w)
    weights = weights / jnp.sum(weights)
    
    # Cumulative sum
    cumsum = jnp.cumsum(weights)
    
    # Systematic resampling positions
    u = random.uniform(key)
    positions = (jnp.arange(n_particles) + u) / n_particles
    
    # Find indices
    indices = jnp.searchsorted(cumsum, positions)
    indices = jnp.clip(indices, 0, n_particles - 1)
    
    return particles[indices]


def run_smc(
    key: jax.Array,
    log_prior_fn: Callable[[jnp.ndarray], float],
    log_likelihood_fn: Callable[[jnp.ndarray], float],
    initial_positions: jnp.ndarray,
    config: Optional[SMCConfig] = None,
    verbose: bool = True,
) -> SMCResult:
    """
    Run Sequential Monte Carlo sampling.
    
    SMC gradually tempers from prior to posterior:
    - Start at temperature=0 (prior)
    - Gradually increase to temperature=1 (full posterior)
    - At each step: reweight → resample → mutate (MCMC)
    
    This avoids getting stuck in local modes and is highly parallel.
    
    Args:
        key: JAX random key
        log_prior_fn: Function computing log prior (must be JIT-compatible)
        log_likelihood_fn: Function computing log likelihood (must be JIT-compatible)
        initial_positions: (n_particles, dim) initial particle positions
        config: SMC configuration options
        verbose: Whether to print progress
    
    Returns:
        SMCResult with final particles, weights, and diagnostics
    """
    if config is None:
        config = SMCConfig()
    
    n_particles, dim = initial_positions.shape
    
    # Create tempered log probability function
    def log_prob_tempered(x, temp):
        return log_prior_fn(x) + temp * log_likelihood_fn(x)
    
    # Initialize
    particles = initial_positions
    temperature = 0.0
    step_size = config.mcmc_step_size
    
    # Compute initial log likelihoods
    log_likelihoods = jax.vmap(log_likelihood_fn)(particles)
    log_probs = jax.vmap(log_prior_fn)(particles)  # At temp=0, just prior
    log_weights = jnp.zeros(n_particles)
    
    # Storage for diagnostics
    temperatures = [0.0]
    ess_history = [float(n_particles)]
    acceptance_rates = []
    
    iteration = 0
    
    if verbose:
        print(f"Starting SMC with {n_particles} particles, dim={dim}")
        print("-" * 60)
    
    while temperature < 1.0 and iteration < config.max_iterations:
        key, key_resample, key_mutate = random.split(key, 3)
        
        # 1) Find next temperature (adaptive)
        next_temp = find_next_temperature(
            log_likelihoods,
            temperature,
            config.target_ess,
            n_particles
        )
        delta_temp = next_temp - temperature
        
        # 2) Reweight particles
        log_weights = delta_temp * log_likelihoods
        ess = compute_ess(log_weights)
        
        # 3) Resample if ESS too low
        if ess < config.target_ess * n_particles:
            particles = systematic_resampling(key_resample, particles, log_weights)
            log_weights = jnp.zeros(n_particles)
            # Recompute log likelihoods after resampling
            log_likelihoods = jax.vmap(log_likelihood_fn)(particles)
        
        # 4) Mutate with MCMC at new temperature
        def log_prob_current(x):
            return log_prob_tempered(x, next_temp)
        
        particles, log_probs, mean_accept = smc_mutation_step(
            key_mutate,
            particles,
            log_prob_current,
            config.n_mcmc_steps,
            step_size
        )
        
        # Update log likelihoods
        log_likelihoods = jax.vmap(log_likelihood_fn)(particles)
        
        # 5) Adapt step size based on acceptance rate
        if config.adaptive_step_size:
            if mean_accept < 0.2:
                step_size = max(config.min_step_size, step_size * 0.8)
            elif mean_accept > 0.4:
                step_size = min(config.max_step_size, step_size * 1.2)
        
        # Update temperature
        temperature = next_temp
        iteration += 1
        
        # Store diagnostics
        temperatures.append(temperature)
        ess_history.append(float(ess))
        acceptance_rates.append(float(mean_accept))
        
        if verbose and iteration % 1 == 0:
            print(f"  Iter {iteration:3d}: temp={temperature:.4f}, "
                  f"ESS={ess:.1f}, accept={mean_accept:.2%}, step={step_size:.3f}")
    
    if verbose:
        print("-" * 60)
        print(f"SMC complete: {iteration} iterations, final temp={temperature:.4f}")
    
    # Final log probabilities at temp=1
    final_log_probs = jax.vmap(lambda x: log_prob_tempered(x, 1.0))(particles)
    
    return SMCResult(
        particles=particles,
        log_weights=log_weights,
        log_probs=final_log_probs,
        temperatures=jnp.array(temperatures),
        ess_history=jnp.array(ess_history),
        acceptance_rates=jnp.array(acceptance_rates),
        n_iterations=iteration
    )


def run_smc_simple(
    key: jax.Array,
    log_prob_fn: Callable[[jnp.ndarray], float],
    initial_positions: jnp.ndarray,
    config: Optional[SMCConfig] = None,
    verbose: bool = True,
) -> SMCResult:
    """
    Simplified SMC interface when you have a single log_prob function.
    
    Uses uniform prior and treats log_prob_fn as the likelihood.
    """
    def log_prior(x):
        # Flat/uniform prior (returns 0)
        return 0.0
    
    return run_smc(
        key=key,
        log_prior_fn=log_prior,
        log_likelihood_fn=log_prob_fn,
        initial_positions=initial_positions,
        config=config,
        verbose=verbose
    )
