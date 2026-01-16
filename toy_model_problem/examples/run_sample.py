"""
Example: Run simple MCMC and save trajectory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from representation.state import State
from io_utils import save_trajectory, load_trajectory, load_best_frame, get_trajectory_info


def create_initial_state():
    """Create initial state."""
    key = jax.random.PRNGKey(42)
    n_A, n_B, n_C = 8, 8, 16
    n_total = n_A + n_B + n_C
    
    coords = jax.random.uniform(key, (n_total, 3), minval=-100.0, maxval=100.0)
    radii = jnp.concatenate([
        jnp.full(n_A, 24.0),
        jnp.full(n_B, 14.0),
        jnp.full(n_C, 16.0),
    ])
    copy_number = jnp.array([n_A, n_B, n_C])
    
    return State(coords, radii, copy_number, type_names=('ProteinA', 'ProteinB', 'ProteinC'))


def simple_energy(state):
    """Simple harmonic potential from origin."""
    return jnp.sum(state.coordinates ** 2) / 1000.0


@jax.jit
def mcmc_step(state, key, step_size=2.0):
    """One MCMC step with Metropolis-Hastings."""
    # Propose new coordinates
    noise = jax.random.normal(key, state.coordinates.shape) * step_size
    new_coords = state.coordinates + noise
    new_state = state.with_coordinates(new_coords)
    
    # Metropolis acceptance
    old_energy = simple_energy(state)
    new_energy = simple_energy(new_state)
    delta_e = new_energy - old_energy
    
    key, subkey = jax.random.split(key)
    accept = jax.random.uniform(subkey) < jnp.exp(-delta_e)
    
    final_state = jax.lax.cond(accept, lambda: new_state, lambda: state)
    log_prob = -simple_energy(final_state)
    
    return final_state, log_prob, accept


def run_mcmc(n_steps=1000, save_every=10):
    """Run MCMC and collect trajectory."""
    print("="*70)
    print("Running MCMC")
    print("="*70)
    
    state = create_initial_state()
    key = jax.random.PRNGKey(123)
    
    # Storage
    trajectory = []
    log_probs = []
    n_accepted = 0
    
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        state, log_prob, accepted = mcmc_step(state, subkey)
        
        n_accepted += int(accepted)
        
        # Save every N steps
        if step % save_every == 0:
            trajectory.append(state)
            log_probs.append(float(log_prob))
            
            if step % 100 == 0:
                print(f"Step {step}: log_prob={log_prob:.3f}, "
                      f"accept_rate={n_accepted/(step+1):.3f}")
    
    print(f"\nFinal acceptance rate: {n_accepted/n_steps:.3f}")
    
    return trajectory, jnp.array(log_probs)


def main():
    # Run MCMC
    trajectory, log_probs = run_mcmc(n_steps=1000, save_every=10)
    
    # Save trajectory
    print("\n" + "="*70)
    print("Saving Trajectory")
    print("="*70)
    
    save_trajectory(
        trajectory,
        "mcmc_trajectory.h5",
        log_probs=log_probs,
        metadata={"method": "simple_mcmc", "step_size": 2.0}
    )
    
    # Get info
    print("\n" + "="*70)
    print("Trajectory Info")
    print("="*70)
    info = get_trajectory_info("mcmc_trajectory.h5")
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    # Load best frame only
    print("\n" + "="*70)
    print("Loading Best Frame")
    print("="*70)
    best_state, best_log_prob = load_best_frame("mcmc_trajectory.h5")
    print(f"Best state: {best_state}")
    
    # Load specific frames
    print("\n" + "="*70)
    print("Loading Specific Frames")
    print("="*70)
    frames_to_load = [0, 10, 20, 30, 40, 50]  # Load every 10th frame
    states, probs, meta = load_trajectory("mcmc_trajectory.h5", frame_indices=frames_to_load)
    print(f"Loaded {len(states)} frames")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()