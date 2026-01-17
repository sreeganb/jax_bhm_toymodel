#------------------------------------------------------------------------------
# Sequential Monte Carlo (SMC) sampler using BlackJAX
# Uses Random Walk Metropolis-Hastings (RMH) - no gradients required
#------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from jax import random
import blackjax
import blackjax.smc.resampling as resampling
from dataclasses import dataclass
from typing import Callable, Optional, NamedTuple, List
import numpy as np


@dataclass
class SMCConfig:
    """Configuration for SMC sampler."""
    n_particles: int = 200          # Number of SMC particles (population size)
    n_mcmc_steps: int = 50          # MCMC steps per SMC iteration
    rmh_step_size: float = 5.0      # Step size for random walk proposal
    target_ess: float = 0.5         # Target effective sample size ratio
    max_iterations: int = 200       # Max SMC iterations


class SMCResult(NamedTuple):
    """Result from SMC sampling."""
    particles: jnp.ndarray          # Final particles (n_particles, dim)
    log_probs: jnp.ndarray          # Log probabilities of final particles
    n_iterations: int               # Number of SMC iterations
    final_lambda: float             # Final tempering parameter
    trajectory: Optional[List]      # History of particles at each step


def get_tempering_param(state):
    """Get tempering parameter from SMC state (handles different BlackJAX versions)."""
    # Try different attribute names used in various BlackJAX versions
    if hasattr(state, 'lmbda'):
        return float(state.lmbda)
    elif hasattr(state, 'tempering_param'):
        return float(state.tempering_param)
    elif hasattr(state, 'lambda_'):
        return float(state.lambda_)
    else:
        # Print available attributes for debugging
        print(f"Warning: Unknown state type. Attributes: {dir(state)}")
        return 0.0


def run_smc(
    key: jax.Array,
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    config: SMCConfig,
    dim: int,
    initial_particles: Optional[jnp.ndarray] = None,
    save_history: bool = False,
) -> SMCResult:
    """
    Run adaptive tempered SMC with BlackJAX RMH kernel (gradient-free).
    
    Args:
        key: JAX random key
        log_prior_fn: Function computing log prior (position -> scalar)
        log_likelihood_fn: Function computing log likelihood (position -> scalar)
        config: SMC configuration
        dim: Dimension of the state space
        initial_particles: Optional initial particles (n_particles, dim)
        save_history: Whether to save particle history
    
    Returns:
        SMCResult with final particles and diagnostics
    """
    init_key, run_key = random.split(key)
    
    # Generate initial particles if not provided
    if initial_particles is None:
        initial_particles = random.uniform(
            init_key,
            shape=(config.n_particles, dim),
            minval=-100.0,
            maxval=100.0
        )
    
    n_pop, _ = initial_particles.shape
    
    # Create RMH proposal distribution
    step_size = jnp.ones(dim) * config.rmh_step_size
    proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
    
    # Create wrapped RMH kernel builder (bakes in the proposal)
    def rmh_build_kernel():
        base_kernel = blackjax.rmh.build_kernel()
        
        def wrapped_kernel(rng_key, state, logdensity_fn):
            return base_kernel(rng_key, state, logdensity_fn, proposal_distribution)
        
        return wrapped_kernel
    
    # Build adaptive tempered SMC
    tempered = blackjax.adaptive_tempered_smc(
        log_prior_fn,
        log_likelihood_fn,
        rmh_build_kernel(),
        blackjax.rmh.init,
        {},  # Empty params (proposal baked into kernel)
        resampling.systematic,
        target_ess=config.target_ess,
        num_mcmc_steps=config.n_mcmc_steps,
    )
    
    # Initialize SMC state
    state = tempered.init(initial_particles)
    
    print(f"Starting SMC with {n_pop} particles, dim={dim}")
    print(f"RMH step size: {config.rmh_step_size}")
    print(f"Target ESS: {config.target_ess}")
    print(f"MCMC steps/iter: {config.n_mcmc_steps}")
    
    # Run inference loop
    history = [] if save_history else None
    n_iter = 0
    current_lambda = get_tempering_param(state)
    
    while current_lambda < 1.0 and n_iter < config.max_iterations:
        step_key, run_key = random.split(run_key)
        
        # Save current state
        if save_history:
            history.append(np.array(state.particles))
        
        # Perform one SMC step
        state, info = tempered.step(step_key, state)
        n_iter += 1
        current_lambda = get_tempering_param(state)
        
        if n_iter % 5 == 0 or current_lambda >= 1.0:
            print(f"  Iter {n_iter:3d}: lambda={current_lambda:.4f}")
    
    # Save final state
    if save_history:
        history.append(np.array(state.particles))
    
    # Compute final log probs
    final_log_probs = jax.vmap(
        lambda x: log_prior_fn(x) + log_likelihood_fn(x)
    )(state.particles)
    
    print(f"\nSMC complete: {n_iter} iterations, final lambda={current_lambda:.4f}")
    
    return SMCResult(
        particles=state.particles,
        log_probs=final_log_probs,
        n_iterations=n_iter,
        final_lambda=current_lambda,
        trajectory=history,
    )