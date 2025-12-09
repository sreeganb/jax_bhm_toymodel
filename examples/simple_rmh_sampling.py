#-------------------------------------------------------------------------
# Just a simple example, 2 particles, initiated at random positions
# inside a box, and then there is a spring potential between them 
# and then there is excluded volume potential between these particles, 
# so implement a SMC sampling scheme with these restraints and sample the 
# configurations
#-------------------------------------------------------------------------

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from scoring.harmonic import jax_harmonic_score
from scoring.exvol import jax_excluded_volume

# Initialize particle coordinates as jax numpy arrays
def initialize_particle_positions(n_particles, box_size, seed=0):
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (n_particles, 3), minval=0.0, maxval=box_size)
    radii = jnp.ones((n_particles,)) * 1.0  # uniform radii for simplicity
    indices = jnp.arange(n_particles)
    return positions, radii, indices

def setup_coordinate_rmh(log_prob_fn, step_size):
    """
    Set up Random Walk Metropolis-Hastings for coordinate sampling.
    
    step_size: scalar or array matching coordinate dimensions
    """
    proposal_distribution = blackjax.mcmc.random_walk.normal(step_size)
    rmh_kernel = blackjax.rmh(log_prob_fn, proposal_distribution)
    return rmh_kernel

# Use the code
pos, radii, indices = initialize_particle_positions(n_particles=2, box_size=100.0)
print("Initial Positions:\n", pos)
