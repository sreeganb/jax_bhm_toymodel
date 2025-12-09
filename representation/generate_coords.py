# Your ideal ground truth (fixed N_A=8, N_B=8, N_C=16)
import jax
import jax.numpy as jnp
import numpy as np

def get_ideal_coords() -> dict[str, jnp.ndarray]:
    array_A = np.array([
        [63., 0., 0.],
        [44.55, 44.55, 0.],
        [0., 63., 0.],
        [-44.55, 44.55, 0.],
        [-63., 0., 0.],
        [-44.55, -44.55, 0.],
        [-0., -63., 0.],
        [44.55, -44.55, 0.]
    ])
    array_B = np.array([
        [63., 0., -38.5],
        [44.55, 44.55, -38.5],
        [0., 63., -38.5],
        [-44.55, 44.55, -38.5],
        [-63., 0., -38.5],
        [-44.55, -44.55, -38.5],
        [-0., -63., -38.5],
        [44.55, -44.55, -38.5]
    ])
    array_C = np.array([
        [47.00, 0.00, -68.50],
        [79.00, 0.00, -68.50],
        [55.86, 55.86, -68.50],
        [33.23, 33.23, -68.50],
        [0.00, 47.00, -68.50],
        [0.00, 79.00, -68.50],
        [-55.86, 55.86, -68.50],
        [-33.23, 33.23, -68.50],
        [-47.00, 0.00, -68.50],
        [-79.00, 0.00, -68.50],
        [-55.86, -55.86, -68.50],
        [-33.23, -33.23, -68.50],
        [0.00, -47.00, -68.50],
        [0.00, -79.00, -68.50],
        [55.86, -55.86, -68.50],
        [33.23, -33.23, -68.50],
    ])
    return {
        'A': jnp.array(array_A),  # slight offset to avoid zero distances
        'B': jnp.array(array_B),
        'C': jnp.array(array_C)
    }

# Generalized function to generate random coords for arbitrary N_A, N_B, N_C
def generate_random_coords(
    key: jax.Array,
    n_dict: dict[str, int],
    minval: float = -200.0,
    maxval: float = 200.0
) -> dict[str, jnp.ndarray]:
    coords = {}
    subkeys = jax.random.split(key, len(n_dict))
    for i, (label, n) in enumerate(n_dict.items()):
        coords[label] = jax.random.uniform(subkeys[i], shape=(n, 3), minval=minval, maxval=maxval)
    return coords
