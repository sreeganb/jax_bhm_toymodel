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
from representation.generate_coords import generate_random_coords

#---------------------------------------------------------------
# Create a system of particles, 3 types, different radii and 
# make the system
#---------------------------------------------------------------
parts = {"A" : 8, "B" : 8, "C" : 16}
key = jax.random.PRNGKey(123)
coords = generate_random_coords(key, parts, minval=-200.0, maxval=200.0)

print("Generated coordinates shape:", {k: v.shape for k, v in coords.items()})
print("coordinates: ", coords)