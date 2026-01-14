#------------------------------------------------------------------------------
# Tests for the ParticleState representation system
# Tests all 4 stages: Representation → Scoring → Sampling → Validation
#------------------------------------------------------------------------------

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from representation import ParticleState, RigidBodyState
from representation.generate_coords import (
    create_particle_state,
    create_random_particle_state,
    create_from_ideal_coords,
    get_pairwise_indices,
    get_ideal_coords,
    generate_random_coords
)
from scoring import (
    jax_harmonic_score,
    harmonic_upperbound_restraint,
    harmonic_lowerbound_restraint,
    jax_excluded_volume,
    jax_excluded_volume_typed
)
from io_utils.save_as_h5py import (
    save_particle_state,
    load_particle_state,
    save_particle_trajectory,
    load_particle_trajectory
)


class TestParticleState:
    """Tests for ParticleState dataclass."""
    
    def test_from_type_dict_basic(self):
        """Test basic creation from type dictionary."""
        coords = {
            'A': jnp.array([[0., 0., 0.], [1., 0., 0.]]),
            'B': jnp.array([[5., 0., 0.]])
        }
        state = ParticleState.from_type_dict(coords, radii={'A': 1.0, 'B': 2.0})
        
        assert state.n_particles == 3
        assert state.positions.shape == (3, 3)
        assert state.radii.shape == (3,)
        assert len(state.type_names) == 2
        
    def test_from_type_dict_uniform_radius(self):
        """Test creation with uniform radius for all types."""
        coords = {
            'A': jnp.array([[0., 0., 0.]]),
            'B': jnp.array([[5., 0., 0.]])
        }
        state = ParticleState.from_type_dict(coords, radii=1.5)
        
        assert jnp.allclose(state.radii, jnp.array([1.5, 1.5]))
        
    def test_from_arrays(self):
        """Test creation from raw arrays."""
        positions = jnp.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        radii = jnp.array([1.0, 1.5, 2.0])
        
        state = ParticleState.from_arrays(positions, radii)
        
        assert state.n_particles == 3
        assert jnp.allclose(state.radii, radii)
        
    def test_from_arrays_flat_positions(self):
        """Test creation from flattened position array."""
        positions_flat = jnp.array([0., 0., 0., 1., 0., 0., 2., 0., 0.])
        
        state = ParticleState.from_arrays(positions_flat, radii=1.0)
        
        assert state.n_particles == 3
        assert state.positions.shape == (3, 3)
        
    def test_with_positions(self):
        """Test immutable position update."""
        coords = {'A': jnp.array([[0., 0., 0.], [1., 0., 0.]])}
        state = ParticleState.from_type_dict(coords, radii=1.0)
        
        new_positions = jnp.array([[10., 0., 0.], [11., 0., 0.]])
        new_state = state.with_positions(new_positions)
        
        # Original unchanged
        assert jnp.allclose(state.positions[0], jnp.array([0., 0., 0.]))
        # New state updated
        assert jnp.allclose(new_state.positions[0], jnp.array([10., 0., 0.]))
        # Radii preserved
        assert jnp.allclose(new_state.radii, state.radii)
        
    def test_flat_positions(self):
        """Test flattened position property."""
        coords = {'A': jnp.array([[0., 0., 0.], [1., 2., 3.]])}
        state = ParticleState.from_type_dict(coords)
        
        flat = state.flat_positions
        assert flat.shape == (6,)
        assert jnp.allclose(flat, jnp.array([0., 0., 0., 1., 2., 3.]))
        
    def test_get_particles_by_type(self):
        """Test filtering particles by type."""
        coords = {
            'A': jnp.array([[0., 0., 0.], [1., 0., 0.]]),
            'B': jnp.array([[5., 0., 0.], [6., 0., 0.], [7., 0., 0.]])
        }
        state = ParticleState.from_type_dict(coords)
        
        # Type A has id 0 (alphabetically first)
        type_a_coords = state.get_particles_by_type(0)
        assert type_a_coords.shape == (2, 3)
        
    def test_to_dict_from_dict_roundtrip(self):
        """Test dictionary serialization roundtrip."""
        coords = {'A': jnp.array([[0., 0., 0.]]), 'B': jnp.array([[5., 0., 0.]])}
        state = ParticleState.from_type_dict(coords, radii={'A': 1.0, 'B': 2.0})
        
        data = state.to_dict()
        restored = ParticleState.from_dict(data)
        
        assert restored.n_particles == state.n_particles
        assert jnp.allclose(restored.positions, state.positions)
        assert jnp.allclose(restored.radii, state.radii)


class TestGenerateCoords:
    """Tests for coordinate generation utilities."""
    
    def test_create_particle_state(self):
        """Test factory function with validation."""
        coords = {'A': jnp.array([[0., 0., 0.]])}
        state = create_particle_state(coords, radii=2.0)
        
        assert state.n_particles == 1
        assert state.radii[0] == 2.0
        
    def test_create_particle_state_invalid_radius(self):
        """Test that negative radius raises error."""
        coords = {'A': jnp.array([[0., 0., 0.]])}
        
        with pytest.raises(ValueError, match="positive"):
            create_particle_state(coords, radii=-1.0)
            
    def test_create_particle_state_empty_coords(self):
        """Test that empty coords raises error."""
        with pytest.raises(ValueError, match="empty"):
            create_particle_state({})
            
    def test_create_random_particle_state(self):
        """Test random initialization."""
        key = jax.random.PRNGKey(42)
        state = create_random_particle_state(
            key, 
            n_dict={'A': 5, 'B': 3},
            radii={'A': 1.0, 'B': 2.0},
            box_size=100.0
        )
        
        assert state.n_particles == 8
        assert jnp.all(state.positions >= -50.0)
        assert jnp.all(state.positions <= 50.0)
        
    def test_create_from_ideal_coords(self):
        """Test ideal coordinate initialization."""
        state = create_from_ideal_coords(radii={'A': 5.0, 'B': 5.0, 'C': 3.0})
        
        # A: 8, B: 8, C: 16 = 32 particles
        assert state.n_particles == 32
        
    def test_get_pairwise_indices_all_pairs(self):
        """Test all-pairs index generation."""
        coords = {'A': jnp.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])}
        state = ParticleState.from_type_dict(coords)
        
        indices = get_pairwise_indices(state)
        
        # 3 choose 2 = 3 pairs
        assert indices.shape == (3, 2)
        
    def test_get_pairwise_indices_filtered(self):
        """Test type-filtered index generation."""
        coords = {
            'A': jnp.array([[0., 0., 0.], [1., 0., 0.]]),
            'B': jnp.array([[5., 0., 0.], [6., 0., 0.]])
        }
        state = ParticleState.from_type_dict(coords)
        
        # Only A-B pairs
        indices = get_pairwise_indices(state, type_pairs=[('A', 'B')])
        
        # 2 A particles × 2 B particles = 4 pairs
        assert indices.shape[0] == 4


class TestScoringWithParticleState:
    """Tests for scoring functions with ParticleState input."""
    
    def test_harmonic_score_with_state(self):
        """Test harmonic score accepts ParticleState."""
        coords = {'A': jnp.array([[0., 0., 0.], [5., 0., 0.]])}
        state = ParticleState.from_type_dict(coords)
        indices = jnp.array([[0, 1]])
        
        # Using ParticleState
        E1 = jax_harmonic_score(state, indices, d0=4.0, k=1.0)
        
        # Using raw arrays (backward compatible)
        E2 = jax_harmonic_score(state.positions, indices, d0=4.0, k=1.0)
        
        assert jnp.allclose(E1, E2)
        # Distance is 5, d0 is 4, so E = 0.5 * 1.0 * (4-5)^2 = 0.5
        assert jnp.isclose(E1, 0.5)
        
    def test_harmonic_upperbound(self):
        """Test upperbound restraint (only penalizes d > d0)."""
        coords = {'A': jnp.array([[0., 0., 0.], [5., 0., 0.]])}
        state = ParticleState.from_type_dict(coords)
        indices = jnp.array([[0, 1]])
        
        # d=5 > d0=4, should have penalty
        E_over = harmonic_upperbound_restraint(state, indices, d0=4.0, k=1.0)
        assert E_over > 0
        
        # d=5 < d0=10, should have no penalty
        E_under = harmonic_upperbound_restraint(state, indices, d0=10.0, k=1.0)
        assert jnp.isclose(E_under, 0.0)
        
    def test_harmonic_lowerbound(self):
        """Test lowerbound restraint (only penalizes d < d0)."""
        coords = {'A': jnp.array([[0., 0., 0.], [3., 0., 0.]])}
        state = ParticleState.from_type_dict(coords)
        indices = jnp.array([[0, 1]])
        
        # d=3 < d0=5, should have penalty
        E_under = harmonic_lowerbound_restraint(state, indices, d0=5.0, k=1.0)
        assert E_under > 0
        
        # d=3 > d0=2, should have no penalty
        E_over = harmonic_lowerbound_restraint(state, indices, d0=2.0, k=1.0)
        assert jnp.isclose(E_over, 0.0)
        
    def test_excluded_volume_with_state(self):
        """Test excluded volume accepts ParticleState."""
        coords = {'A': jnp.array([[0., 0., 0.], [3., 0., 0.]])}
        state = ParticleState.from_type_dict(coords, radii=2.0)
        
        # Using ParticleState (radii extracted automatically)
        E1 = jax_excluded_volume(state, k=1.0)
        
        # Using raw arrays
        E2 = jax_excluded_volume(state.positions, state.radii, k=1.0)
        
        assert jnp.allclose(E1, E2)
        # overlap = (2+2) - 3 = 1, E = 0.5 * 1.0 * 1^2 = 0.5
        assert jnp.isclose(E1, 0.5)
        
    def test_excluded_volume_no_overlap(self):
        """Test no energy when particles don't overlap."""
        coords = {'A': jnp.array([[0., 0., 0.], [10., 0., 0.]])}
        state = ParticleState.from_type_dict(coords, radii=1.0)
        
        E = jax_excluded_volume(state, k=1.0)
        assert jnp.isclose(E, 0.0)
        
    def test_excluded_volume_typed(self):
        """Test type-aware excluded volume."""
        coords = {
            'A': jnp.array([[0., 0., 0.]]),
            'B': jnp.array([[3., 0., 0.]])
        }
        state = ParticleState.from_type_dict(coords, radii=2.0)
        
        # Different k for A-A, A-B, B-B
        k_matrix = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        
        E = jax_excluded_volume_typed(state, k_matrix=k_matrix)
        
        # A-B pair: overlap=1, k=0.5, E = 0.5 * 0.5 * 1^2 = 0.25
        assert jnp.isclose(E, 0.25)


class TestHDF5IO:
    """Tests for HDF5 save/load functionality."""
    
    def test_save_load_particle_state(self):
        """Test single state save/load roundtrip."""
        coords = {
            'A': jnp.array([[0., 0., 0.], [1., 0., 0.]]),
            'B': jnp.array([[5., 0., 0.]])
        }
        state = ParticleState.from_type_dict(
            coords, 
            radii={'A': 1.0, 'B': 2.0},
            copy_numbers={'A': 1, 'B': 2}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filename = f.name
            
        try:
            save_particle_state(filename, state, metadata={'test': 'value'})
            loaded = load_particle_state(filename)
            
            assert loaded.n_particles == state.n_particles
            assert jnp.allclose(loaded.positions, state.positions)
            assert jnp.allclose(loaded.radii, state.radii)
            assert jnp.allclose(loaded.particle_types, state.particle_types)
            assert jnp.allclose(loaded.copy_numbers, state.copy_numbers)
            assert loaded.type_names == state.type_names
        finally:
            os.unlink(filename)
            
    def test_save_load_trajectory(self):
        """Test trajectory save/load roundtrip."""
        coords = {'A': jnp.array([[0., 0., 0.], [1., 0., 0.]])}
        base_state = ParticleState.from_type_dict(coords, radii=1.0)
        
        # Create trajectory with 5 frames
        states = []
        for i in range(5):
            new_pos = base_state.positions + i * 0.5
            states.append(base_state.with_positions(new_pos))
            
        log_probs = np.array([-10.0, -8.0, -5.0, -6.0, -7.0])
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filename = f.name
            
        try:
            save_particle_trajectory(filename, states, log_probs)
            loaded_states, loaded_probs = load_particle_trajectory(filename)
            
            assert len(loaded_states) == 5
            assert loaded_probs is not None
            assert np.argmax(loaded_probs) == 2  # Best frame
            
            # Check position evolution
            assert jnp.allclose(loaded_states[0].positions, states[0].positions)
            assert jnp.allclose(loaded_states[4].positions, states[4].positions)
        finally:
            os.unlink(filename)


class TestIntegration:
    """Integration tests covering full workflow."""
    
    def test_full_pipeline(self):
        """Test representation → scoring → sampling-like update → validation."""
        # Stage 1: Representation
        key = jax.random.PRNGKey(42)
        state = create_random_particle_state(
            key,
            n_dict={'A': 4, 'B': 4},
            radii={'A': 2.0, 'B': 3.0},
            box_size=50.0
        )
        
        assert state.n_particles == 8
        
        # Stage 2: Scoring
        indices = get_pairwise_indices(state, type_pairs=[('A', 'B')])
        
        E_harmonic = jax_harmonic_score(state, indices, d0=10.0, k=0.5)
        E_exvol = jax_excluded_volume(state, k=1.0)
        total_energy = E_harmonic + E_exvol
        log_prob = -total_energy
        
        assert jnp.isfinite(log_prob)
        
        # Stage 3: Sampling (simulate one step)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, state.positions.shape) * 0.1
        new_positions = state.positions + noise
        new_state = state.with_positions(new_positions)
        
        # Compute new energy
        new_E = jax_harmonic_score(new_state, indices, d0=10.0, k=0.5) + \
                jax_excluded_volume(new_state, k=1.0)
        new_log_prob = -new_E
        
        # Stage 4: Validation (save)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filename = f.name
            
        try:
            save_particle_state(filename, new_state, metadata={
                'log_probability': float(new_log_prob),
                'n_types': 2
            })
            
            # Verify we can reload
            loaded = load_particle_state(filename)
            assert loaded.n_particles == 8
            assert jnp.allclose(loaded.radii[:4], 2.0)  # A particles
            assert jnp.allclose(loaded.radii[4:], 3.0)  # B particles
        finally:
            os.unlink(filename)
            
    def test_jit_compatibility(self):
        """Test that scoring functions work with JIT compilation."""
        coords = {'A': jnp.array([[0., 0., 0.], [5., 0., 0.]])}
        state = ParticleState.from_type_dict(coords, radii=1.0)
        indices = jnp.array([[0, 1]])
        
        # Define a function that uses ParticleState fields
        @jax.jit
        def compute_energy(positions, radii, indices):
            E1 = jax_harmonic_score(positions, indices, d0=4.0, k=1.0)
            E2 = jax_excluded_volume(positions, radii, k=1.0)
            return E1 + E2
        
        E = compute_energy(state.positions, state.radii, indices)
        assert jnp.isfinite(E)
        
        # Test gradient computation
        grad_fn = jax.grad(lambda pos: compute_energy(pos, state.radii, indices))
        grads = grad_fn(state.positions)
        assert grads.shape == state.positions.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
