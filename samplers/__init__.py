# Samplers module for Bayesian inference
# Provides SMC and MCMC samplers using pure JAX (CPU/GPU compatible)

from .smc import run_smc, run_smc_simple, SMCConfig, SMCResult
