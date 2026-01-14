#------------------------------------------------------------------------------
# Tests for ParticleState and unified representation system
#------------------------------------------------------------------------------

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import h5py

from representation import ParticleState
from representation.generate_coords import (
    create_particle_state,
    create_random_particle_state,
    get_pairwise_indices
)
from scoring import (
    jax_harmonic_score,
    jax_excluded_volume,
    jax_excluded_volume_typed
)
from io_utils.save_as_h5py import (
    save_particle_state,
    save_particle_trajectory,
    load_particle_state
)


def test_particle_state_creation():
    """Test basic ParticleState creation."""
    print("\n" + "="*70)
    print("TEST 1: ParticleState Creation")
    print("="*70)
    
    positions = jnp.array([[0., 0., 0.], [1., 1., 1.]])
    radii = jnp.array([1.0, 1.5])
    particle_types = jnp.array([0, 1])
    
    state = ParticleState(positions, radii, particle_types)
    
    assert state.positions.shape == (2, 3)
    assert state.radii.shape == (2,)
    assert state.particle_types.shape == (2,)
    assert state.n_particles == 2
    
    print("✓ Basic creation works")
    print(f"  n_particles: {state.n_particles}")
    print(f"  positions shape: {state.positions.shape}")


def test_particle_state_with_names():
    """Test ParticleState with optional fields."""
    print("\n" + "="*70)
    print("TEST 2: ParticleState with Optional Fields")
    print("="*70)
    
    positions = jnp.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    radii = jnp.array([1.0, 1.5, 2.0])
    particle_types = jnp.array([0, 1, 0])
    copy_numbers = jnp.array([1, 1, 2])
    type_names = {0: "A", 1: "B"}
    
    state = ParticleState(
        positions, radii, particle_types,
        copy_numbers=copy_numbers,
        type_names=type_names
    )
    
    assert state.copy_numbers.shape == (3,)
    assert state.type_names == {0: "A", 1: "B"}
    
    print("✓ Optional fields work")
    print(f"  copy_numbers: {state.copy_numbers}")
    print(f"  type_names: {state.type_names}")


def test_factory_functions():
    """Test particle state factory functions."""
    print("\n" + "="*70)
    print("TEST 3: Factory Functions")
    print("="*70)
    
    # Test create_particle_state
    positions = jnp.array([[0., 0., 0.], [1., 1., 1.]])
    radii = jnp.array([1.0, 1.5])
    
    state = create_particle_state(positions, radii)
    assert state.n_particles == 2
    print("✓ create_particle_state works")
    
    # Test create_random_particle_state
    key = jax.random.PRNGKey(42)
    n_particles = {"A": 3, "B": 2}
    radii_dict = {"A": 1.0, "B": 1.5}
    
    state = create_random_particle_state(
        key, n_particles, radii_dict, 
        box_min=-10.0, box_max=10.0
    )
    
    assert state.n_particles == 5
    assert len(jnp.unique(state.particle_types)) == 2
    print(f"✓ create_random_particle_state works")
    print(f"  Created {state.n_particles} particles of {len(n_particles)} types")


def test_scoring_functions():
    """Test that scoring functions work with ParticleState."""
    print("\n" + "="*70)
    print("TEST 4: Scoring Functions")
    print("="*70)
    
    # Create simple 2-particle system
    positions = jnp.array([[0., 0., 0.], [5., 0., 0.]])
    radii = jnp.array([1.0, 1.0])
    particle_types = jnp.array([0, 0])
    
    state = ParticleState(positions, radii, particle_types)
    
    # Test harmonic score
    indices = jnp.array([[0, 1]])
    harmonic_energy = jax_harmonic_score(state, indices, d0=4.5, k=0.5)
    print(f"✓ Harmonic energy: {harmonic_energy:.4f}")
    
    # Test excluded volume
    exvol_energy = jax_excluded_volume(state, k=1.0)
    print(f"✓ Excluded volume energy: {exvol_energy:.4f}")
    
    # Test typed excluded volume
    type_radii = {0: 1.0, 1: 1.5}
    type_k = {(0, 0): 1.0, (0, 1): 1.5, (1, 1): 2.0}
    
    state_typed = ParticleState(
        jnp.array([[0., 0., 0.], [3., 0., 0.]]),
        jnp.array([1.0, 1.5]),
        jnp.array([0, 1]),
        type_names={0: "A", 1: "B"}
    )
    
    typed_energy = jax_excluded_volume_typed(state_typed, type_radii, type_k)
    print(f"✓ Typed excluded volume energy: {typed_energy:.4f}")


def test_hdf5_io():
    """Test HDF5 save/load functionality."""
    print("\n" + "="*70)
    print("TEST 5: HDF5 I/O")
    print("="*70)
    
    # Create particle state
    positions = jnp.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    radii = jnp.array([1.0, 1.5, 2.0])
    particle_types = jnp.array([0, 1, 0])
    copy_numbers = jnp.array([1, 1, 2])
    type_names = {0: "A", 1: "B"}
    
    state = ParticleState(
        positions, radii, particle_types,
        copy_numbers=copy_numbers,
        type_names=type_names
    )
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = f.name
    
    save_particle_state(state, temp_file)
    print(f"✓ Saved to {temp_file}")
    
    # Load and verify
    loaded_state = load_particle_state(temp_file)
    
    assert loaded_state.n_particles == state.n_particles
    assert jnp.allclose(loaded_state.positions, state.positions)
    assert jnp.allclose(loaded_state.radii, state.radii)
    assert jnp.array_equal(loaded_state.particle_types, state.particle_types)
    assert jnp.array_equal(loaded_state.copy_numbers, state.copy_numbers)
    assert loaded_state.type_names == state.type_names
    
    print("✓ Loaded and verified state matches original")
    
    # Clean up
    import os
    os.remove(temp_file)


def test_trajectory_io():
    """Test trajectory save/load."""
    print("\n" + "="*70)
    print("TEST 6: Trajectory I/O")
    print("="*70)
    
    # Create a trajectory (list of states)
    n_frames = 10
    n_particles = 3
    
    trajectory = []
    for i in range(n_frames):
        positions = jnp.array([[i*0.1, 0., 0.], 
                               [i*0.1, 1., 0.], 
                               [i*0.1, 2., 0.]])
        radii = jnp.array([1.0, 1.0, 1.0])
        particle_types = jnp.array([0, 0, 1])
        trajectory.append(ParticleState(positions, radii, particle_types))
    
    log_probs = np.linspace(-100, -10, n_frames)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = f.name
    
    save_particle_trajectory(trajectory, log_probs, temp_file)
    print(f"✓ Saved trajectory with {n_frames} frames")
    
    # Verify file structure
    with h5py.File(temp_file, 'r') as f:
        assert 'coordinates' in f
        assert f['coordinates'].shape == (n_frames, n_particles, 3)
        assert 'log_probabilities' in f
        assert f['log_probabilities'].shape == (n_frames,)
        assert 'radii' in f
        assert 'particle_types' in f
        print("✓ Trajectory structure verified")
    
    # Clean up
    import os
    os.remove(temp_file)


def test_get_pairwise_indices():
    """Test pairwise indices generation."""
    print("\n" + "="*70)
    print("TEST 7: Pairwise Indices")
    print("="*70)
    
    # Test all pairs
    indices = get_pairwise_indices(4)
    assert indices.shape == (6, 2)  # C(4,2) = 6
    print(f"✓ All pairs: {indices.shape[0]} pairs for 4 particles")
    
    # Test specific pairs
    specific = jnp.array([[0, 1], [1, 2], [2, 3]])
    indices = get_pairwise_indices(4, specific_pairs=specific)
    assert indices.shape == (3, 2)
    assert jnp.array_equal(indices, specific)
    print(f"✓ Specific pairs work")
    
    # Test type-filtered pairs
    particle_types = jnp.array([0, 0, 1, 1])
    indices = get_pairwise_indices(4, particle_types=particle_types, type_pairs=[(0, 1)])
    # Should only get pairs between type 0 and type 1
    print(f"✓ Type-filtered pairs: {indices.shape[0]} cross-type pairs")
    print(f"  Indices: {indices}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING PARTICLE STATE TESTS")
    print("="*70)
    
    test_particle_state_creation()
    test_particle_state_with_names()
    test_factory_functions()
    test_scoring_functions()
    test_hdf5_io()
    test_trajectory_io()
    test_get_pairwise_indices()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
