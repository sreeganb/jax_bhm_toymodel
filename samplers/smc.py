#------------------------------------------------------------------------------
# Sequential Monte Carlo (SMC) sampler using BlackJAX
# 
# This module wraps BlackJAX's adaptive_tempered_smc for easy use.
# BlackJAX provides:
# 1. Adaptive tempering from prior to posterior
# 2. Parallel particle evolution (GPU-friendly, also works on CPU)
# 3. Systematic resampling
# 4. HMC (Hamiltonian Monte Carlo) for efficient mutation steps
#
# Platform compatibility:
# - Linux/Windows with NVIDIA GPU: Full speed with CUDA
# - Linux/Windows CPU: Works, uses all cores via XLA
# - macOS Intel: Works on CPU
# - macOS Apple Silicon: Works on CPU (Metal support is experimental)
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, NamedTuple, Optional
from dataclasses import dataclass

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params


@dataclass
class SMCConfig:
    """Configuration for SMC sampler."""
    n_particles: int = 1000          # Number of parallel particles
    n_mcmc_steps: int = 5            # MCMC steps per SMC iteration
    hmc_step_size: float = 0.1       # HMC leapfrog step size
    hmc_n_leapfrog: int = 10         # Number of leapfrog steps per HMC iteration
    target_ess: float = 0.5          # Target effective sample size ratio
    max_iterations: int = 100        # Maximum SMC iterations


class SMCResult(NamedTuple):
    """Result from SMC sampling."""
    particles: jnp.ndarray      # Final particles (n_particles, dim)
    log_weights: jnp.ndarray    # Final log weights
    log_probs: jnp.ndarray      # Final log probabilities
    temperatures: jnp.ndarray   # Temperature schedule used
    n_iterations: int           # Total iterations


def _inference_loop(kernel_step, rng_key, initial_state, max_iterations, verbose=True):
    """
    Run adaptive SMC iterations until temperature reaches 1.0.
    
    This is a while loop that steps the SMC kernel until the tempering
    parameter reaches 1.0 (full posterior).
    """
    temperatures = [0.0]
    state = initial_state
    iteration = 0
    
    while state.tempering_param < 1.0 and iteration < max_iterations:
        rng_key, step_key = random.split(rng_key)
        state, info = kernel_step(step_key, state)
        iteration += 1
        temperatures.append(float(state.tempering_param))
        
        if verbose:
            print(f"  Iter {iteration:3d}: temp={state.tempering_param:.4f}")
    
    return state, jnp.array(temperatures), iteration


def run_smc(
    key: jax.Array,
    log_prior_fn: Callable[[jnp.ndarray], float],
    log_likelihood_fn: Callable[[jnp.ndarray], float],
    initial_positions: jnp.ndarray,
    config: Optional[SMCConfig] = None,
    verbose: bool = True,
) -> SMCResult:
    """
    Run BlackJAX's adaptive tempered SMC.
    
    This uses blackjax.adaptive_tempered_smc which automatically finds
    the temperature schedule to maintain a target ESS.
    
    SMC gradually tempers from prior to posterior:
    - Start at temperature=0 (prior)
    - Gradually increase to temperature=1 (full posterior)
    - At each step: reweight → resample → mutate (HMC)
    
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
    
    if verbose:
        print(f"Starting BlackJAX SMC with {n_particles} particles, dim={dim}")
        print("-" * 60)
    
    # Build MCMC kernel: HMC (Hamiltonian Monte Carlo) for efficient mutation
    hmc_kernel = blackjax.hmc.build_kernel()
    hmc_init = blackjax.hmc.init
    
    # HMC parameters
    hmc_parameters = extend_params({
        'step_size': config.hmc_step_size,
        'inverse_mass_matrix': jnp.eye(dim),
        'num_integration_steps': config.hmc_n_leapfrog,
    })
    
    # Build the adaptive tempered SMC kernel
    smc = blackjax.adaptive_tempered_smc(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=hmc_init,
        mcmc_parameters=hmc_parameters,
        resampling_fn=resampling.systematic,
        target_ess=config.target_ess,
        num_mcmc_steps=config.n_mcmc_steps,
    )
    
    # Initialize SMC state with particles from prior
    state = smc.init(initial_positions)
    
    # Run SMC iterations
    state, temperatures, n_iterations = _inference_loop(
        smc.step, key, state, config.max_iterations, verbose
    )
    
    if verbose:
        print("-" * 60)
        print(f"SMC complete: {n_iterations} iterations, final temp={state.tempering_param:.4f}")
    
    # Compute final log probabilities at temperature=1
    final_log_probs = jax.vmap(
        lambda x: log_prior_fn(x) + log_likelihood_fn(x)
    )(state.particles)
    
    return SMCResult(
        particles=state.particles,
        log_weights=state.weights,
        log_probs=final_log_probs,
        temperatures=temperatures,
        n_iterations=n_iterations
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
    
    Uses a flat (uniform) prior and treats log_prob_fn as the likelihood.
    This is the recommended interface for most use cases.
    
    Args:
        key: JAX random key
        log_prob_fn: Log probability function (treated as log-likelihood)
        initial_positions: (n_particles, dim) initial particle positions
        config: SMC configuration options
        verbose: Whether to print progress
    
    Returns:
        SMCResult with final particles and diagnostics
    """
    # Flat/uniform prior that returns 0 everywhere
    def log_prior(x):
        return 0.0
    
    return run_smc(
        key=key,
        log_prior_fn=log_prior,
        log_likelihood_fn=log_prob_fn,
        initial_positions=initial_positions,
        config=config,
        verbose=verbose
    )