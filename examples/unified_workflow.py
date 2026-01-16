"""
Unified Workflow Example: Demonstrating ParticleState throughout all 4 stages.

This example shows:
1. Representation: Creating ParticleState with types and radii
2. Scoring: Using ParticleState directly in scoring functions
3. Sampling: MCMC sampling with ParticleState using BlackJAX (RMH or SMC)
4. Validation: Saving/loading ParticleState to/from HDF5

Usage:
    python unified_workflow.py --sampler rmh   # Use Random Walk Metropolis-Hastings
    python unified_workflow.py --sampler smc   # Use Sequential Monte Carlo
"""
import sys
from pathlib import Path
import os
import argparse

# Force JAX to use CPU to avoid Metal/MPS issues on Mac
# For Mac users with M1/M2 chips/ otherwise for the linux machines use GPU
os.environ['JAX_PLATFORMS'] = 'cpu'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime

print("JAX is using device:", jax.default_backend())

# Now import our modules
from representation import ParticleState, ParticleSystemFactory
from representation.generate_coords import generate_random_coords
from scoring.harmonic import jax_harmonic_score
from scoring.exvol import jax_excluded_volume
from io_utils.save_as_h5py import (
    save_particle_state,
    load_particle_state,
    save_particle_trajectory
)

# Import BlackJAX-based samplers
from samplers.rmh import run_rmh, run_rmh_trajectory, RMHConfig
from samplers.smc import run_smc_simple, SMCConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified workflow with BlackJAX samplers"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["rmh", "smc"],
        default="rmh",
        help="Sampler to use: 'rmh' (Random Walk Metropolis-Hastings) or 'smc' (Sequential Monte Carlo)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to collect (for RMH) or particles (for SMC)"
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help="Number of warmup/burn-in steps (RMH only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    return parser.parse_args()


def create_log_prob_fn(state_template, restraints, harmonic_d0=5.0, harmonic_k=0.5, exvol_k=1.0):
    """
    Create a log probability function for sampling.
    
    This returns a function that takes a flat position array and returns
    the log probability (negative energy).
    
    Args:
        state_template: ParticleState template (provides radii, types, etc.)
        restraints: Restraint indices array
        harmonic_d0: Equilibrium distance for harmonic restraints
        harmonic_k: Harmonic force constant
        exvol_k: Excluded volume force constant
    
    Returns:
        log_prob_fn: Function mapping flat positions to log probability
    """
    def log_prob(positions_flat):
        """Log probability function for MCMC."""
        positions = positions_flat.reshape(-1, 3)
        # Create temporary state with updated positions
        temp_state = ParticleState(
            positions=positions,
            radii=state_template.radii,
            particle_types=state_template.particle_types,
            copy_numbers=state_template.copy_numbers,
            type_names=state_template.type_names
        )
        h_energy = jax_harmonic_score(temp_state, restraints, d0=harmonic_d0, k=harmonic_k)
        ev_energy = jax_excluded_volume(temp_state, k=exvol_k)
        return -(h_energy + ev_energy)
    
    return log_prob


def run_rmh_sampling(key, log_prob_fn, initial_positions, args):
    """
    Run Random Walk Metropolis-Hastings sampling using BlackJAX.
    
    Returns:
        trajectory: List of position arrays
        log_probs: Array of log probabilities
        acceptance_rate: Float
    """
    print("\nUsing BlackJAX Random Walk Metropolis-Hastings (RMH)")
    print("-" * 70)
    
    config = RMHConfig(
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        sigma=0.5,
        thin=10,  # Keep every 10th sample to reduce autocorrelation
    )
    
    # Run RMH with trajectory saving
    result, trajectory = run_rmh_trajectory(
        key=key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_positions,
        config=config,
        save_every=100,
        verbose=args.verbose
    )
    
    return result.samples, result.log_probs, result.acceptance_rate, trajectory


def run_smc_sampling(key, log_prob_fn, initial_positions, args, n_particles_system):
    """
    Run Sequential Monte Carlo sampling using BlackJAX.
    
    Returns:
        samples: Final particle positions
        log_probs: Array of log probabilities
        temperatures: Temperature schedule used
    """
    print("\nUsing BlackJAX Sequential Monte Carlo (SMC)")
    print("-" * 70)
    
    # For SMC, we need multiple initial particles
    # Generate them by perturbing the initial position
    n_smc_particles = min(args.n_samples, 500)  # SMC particles (not system particles)
    
    key1, key2 = random.split(key)
    perturbations = random.normal(key1, (n_smc_particles, len(initial_positions))) * 1.0
    initial_smc_particles = initial_positions + perturbations
    
    config = SMCConfig(
        n_particles=n_smc_particles,
        n_mcmc_steps=5,
        hmc_step_size=0.1,
        hmc_n_leapfrog=10,
        target_ess=0.5,
        max_iterations=100
    )
    
    result = run_smc_simple(
        key=key2,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_smc_particles,
        config=config,
        verbose=args.verbose
    )
    
    return result.particles, result.log_probs, result.temperatures


def main():
    args = parse_args()
    
    print(f"{'='*70}")
    print("Unified Workflow: ParticleState Through All 4 Stages")
    print(f"Sampler: {args.sampler.upper()}")
    print(f"{'='*70}\n")

    # ============================================================
    # Stage 1: REPRESENTATION
    # ============================================================
    print("Stage 1: REPRESENTATION")
    print("-" * 70)

    # Define particle types
    particle_types = {
        "ProteinA": {"count": 8, "radius": 24.0},
        "ProteinB": {"count": 8, "radius": 14.0},
        "ProteinC": {"count": 16, "radius": 16.0}
    }

    # Create using factory
    factory = ParticleSystemFactory(particle_types)
    key = random.PRNGKey(42)
    state = factory.create_random_state(key, box_size=500.0)

    print(f"Created ParticleState:")
    print(f"  Total particles: {state.n_particles}")
    print(f"  Types: {list(state.type_names.values())}")
    print(f"  Radii range: {jnp.min(state.radii):.2f} - {jnp.max(state.radii):.2f}")
    print(f"  Copy numbers: {dict(zip(state.type_names.values(), [state.get_particles_by_type(i).shape[0] for i in range(len(state.type_names))]))}")
    print()

    # ============================================================
    # Stage 2: SCORING
    # ============================================================
    print("Stage 2: SCORING")
    print("-" * 70)

    # Define restraints (connect first particle of each type)
    restraints = jnp.array([
        [0, 5],   # ProteinA[0] - ProteinB[0]
        [0, 16],  # ProteinA[0] - ProteinC[0]
        [5, 16],  # ProteinB[0] - ProteinC[0]
        [16, 20],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])

    # Score using ParticleState directly
    harmonic_energy = jax_harmonic_score(state, restraints, d0=30.0, k=0.5)
    exvol_energy = jax_excluded_volume(state, k=1.0)
    total_energy = harmonic_energy + exvol_energy

    print(f"Initial energy scores:")
    print(f"  Harmonic: {harmonic_energy:.2f}")
    print(f"  Excluded volume: {exvol_energy:.2f}")
    print(f"  Total: {total_energy:.2f}")
    print()

    # ============================================================
    # Stage 3: SAMPLING (using BlackJAX)
    # ============================================================
    print("Stage 3: SAMPLING")
    print("-" * 70)

    # Create log probability function
    log_prob_fn = create_log_prob_fn(
        state_template=state,
        restraints=restraints,
        harmonic_d0=5.0,
        harmonic_k=0.5,
        exvol_k=1.0
    )

    # Initial positions (flattened)
    initial_positions = state.positions.flatten()
    
    # Run sampling based on selected method
    key = random.PRNGKey(123)
    
    if args.sampler == "rmh":
        samples, log_probs, acceptance_rate, trajectory = run_rmh_sampling(
            key, log_prob_fn, initial_positions, args
        )
        print(f"\nRMH Results:")
        print(f"  Samples collected: {len(samples)}")
        print(f"  Acceptance rate: {acceptance_rate:.2%}")
        print(f"  Trajectory frames: {len(trajectory)}")
        
        # Use trajectory for saving
        trajectory_positions = [pos.reshape(-1, 3) for pos in trajectory]
        final_positions = samples[-1].reshape(-1, 3)
        
    else:  # smc
        samples, log_probs, temperatures = run_smc_sampling(
            key, log_prob_fn, initial_positions, args, state.n_particles
        )
        print(f"\nSMC Results:")
        print(f"  Final particles: {len(samples)}")
        print(f"  Temperature steps: {len(temperatures)}")
        print(f"  Final temperature: {temperatures[-1]:.4f}")
        
        # For SMC, use the best particle (highest log prob)
        best_idx = jnp.argmax(log_probs)
        final_positions = samples[best_idx].reshape(-1, 3)
        
        # Create trajectory from all SMC particles (for visualization)
        trajectory_positions = [p.reshape(-1, 3) for p in samples[:100]]  # First 100 particles
        acceptance_rate = 1.0  # SMC doesn't have traditional acceptance rate

    print()

    # ============================================================
    # Stage 4: VALIDATION
    # ============================================================
    print("Stage 4: VALIDATION")
    print("-" * 70)

    # Save final state
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    final_state = ParticleState(
        positions=final_positions,
        radii=state.radii,
        particle_types=state.particle_types,
        copy_numbers=state.copy_numbers,
        type_names=state.type_names
    )

    # Compute final energy
    final_harmonic = jax_harmonic_score(final_state, restraints, d0=5.0, k=0.5)
    final_exvol = jax_excluded_volume(final_state, k=1.0)
    print(f"\nFinal energy scores:")
    print(f"  Harmonic: {final_harmonic:.2f}")
    print(f"  Excluded volume: {final_exvol:.2f}")
    print(f"  Total: {final_harmonic + final_exvol:.2f}")

    state_file = output_dir / f"final_state_{args.sampler}.h5"
    save_particle_state(
        str(state_file),
        final_state,
        metadata={
            "method": args.sampler.upper(),
            "n_samples": args.n_samples,
            "acceptance_rate": float(acceptance_rate) if args.sampler == "rmh" else None,
            "final_log_prob": float(log_probs[-1]) if args.sampler == "rmh" else float(jnp.max(log_probs))
        }
    )

    # Save trajectory
    trajectory_states = [
        ParticleState(
            positions=pos,
            radii=state.radii,
            particle_types=state.particle_types,
            copy_numbers=state.copy_numbers,
            type_names=state.type_names
        )
        for pos in trajectory_positions
    ]

    # Compute log probs for trajectory
    traj_log_probs = np.array([log_prob_fn(pos.flatten()) for pos in trajectory_positions])
    traj_file = output_dir / f"trajectory_{args.sampler}.h5"
    save_particle_trajectory(
        str(traj_file),
        trajectory_states,
        log_probs=traj_log_probs,
        metadata={
            "method": args.sampler.upper(),
            "n_samples": args.n_samples
        }
    )

    # Test loading
    print("\nTesting load/save roundtrip...")
    loaded_state = load_particle_state(str(state_file))
    print(f"  Loaded {loaded_state.n_particles} particles")
    print(f"  Position match: {jnp.allclose(loaded_state.positions, final_state.positions)}")
    print(f"  Radii match: {jnp.allclose(loaded_state.radii, final_state.radii)}")
    print(f"  Types match: {jnp.all(loaded_state.particle_types == final_state.particle_types)}")

    print(f"\n{'='*70}")
    print(f"Workflow complete using {args.sampler.upper()}!")
    print(f"Check output/ directory for results:")
    print(f"  - {state_file.name}")
    print(f"  - {traj_file.name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
