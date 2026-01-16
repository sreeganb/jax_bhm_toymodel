"""
Basic tests for State class - checking dimensions and JAX compatibility.
"""
import pytest
import jax
import jax.numpy as jnp
from representation.state import State


def test_state_creation():
    """Test that State can be created with correct dimensions."""
    coords = jnp.array([[0., 1., 2.], [3., 4., 5.]])
    radius = jnp.array([1.0, 2.0])
    copy_number = jnp.array([2])
    
    state = State(coords, radius, copy_number)
    state.validate()
    
    assert state.n_particles == 2
    assert state.coordinates.shape == (2, 3)
    assert state.radius.shape == (2,)


def test_multi_type_state():
    """Test State with multiple particle types."""
    # 2 particles of type A, 3 of type B
    coords = jnp.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]])
    radius = jnp.array([1.0, 1.0, 2.0, 2.0, 2.0])
    copy_number = jnp.array([2, 3])
    
    state = State(coords, radius, copy_number, type_names=('A', 'B'))
    state.validate()
    
    assert state.n_particles == 5
    assert state.n_types == 2


def test_jit_works():
    """Test that State works with JAX JIT compilation."""
    coords = jnp.array([[0., 1., 2.], [3., 4., 5.]])
    radius = jnp.array([1.0, 2.0])
    copy_number = jnp.array([2])
    
    state = State(coords, radius, copy_number)
    
    @jax.jit
    def sum_coords(s):
        return jnp.sum(s.coordinates)
    
    result = sum_coords(state)
    assert jnp.isclose(result, 15.0)  # 0+1+2+3+4+5 = 15

def test_immutable_update():
    """Test that State updates are immutable."""
    coords = jnp.array([[0., 1., 2.], [3., 4., 5.]])
    radius = jnp.array([1.0, 2.0])
    copy_number = jnp.array([2])
    
    state = State(coords, radius, copy_number)
    new_coords = jnp.array([[10., 11., 12.], [13., 14., 15.]])
    new_state = state.with_coordinates(new_coords)
    
    # Original unchanged
    assert jnp.array_equal(state.coordinates, coords)
    # New state updated
    assert jnp.array_equal(new_state.coordinates, new_coords)


def test_get_coords_by_type():
    """Test getting coordinates for specific particle type."""
    coords = jnp.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    radius = jnp.array([1.0, 1.0, 2.0, 2.0])
    copy_number = jnp.array([2, 2])  # 2 of type 0, 2 of type 1
    
    state = State(coords, radius, copy_number)
    
    type0_coords = state.get_coords_by_type(0)
    assert type0_coords.shape == (2, 3)
    assert jnp.array_equal(type0_coords, coords[:2])
    
    type1_coords = state.get_coords_by_type(1)
    assert jnp.array_equal(type1_coords, coords[2:])