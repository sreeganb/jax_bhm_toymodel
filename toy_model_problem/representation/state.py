"""
State class for representing particle system configurations in MCMC simulations.

This class is a JAX pytree, making it compatible with:
- JIT compilation
- Automatic differentiation (grad, value_and_grad)
- Vectorization (vmap)
- Other JAX transformations
"""
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
from dataclasses import dataclass

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    """
    Immutable state representing a particle system configuration.
    
    Attributes:
        coordinates: (n_particles, 3) array of particle positions
        radius: (n_particles,) array of particle radii
        copy_number: (n_types,) array indicating number of particles per type
        particle_types: Optional (n_particles,) array of type indices (0, 1, 2, ...)
        rigid_body_ids: Optional (n_particles,) array for rigid body constraints
        type_names: Optional tuple of type names (e.g., ('A', 'B', 'C'))
    
    Example:
        >>> coords = jnp.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
        >>> radii = jnp.array([1.0, 1.5, 2.0])
        >>> copy_nums = jnp.array([1, 1, 1])  # 3 types, 1 particle each
        >>> state = State(coords, radii, copy_nums)
    """
    coordinates: jnp.ndarray  # (n_particles, 3)
    radius: jnp.ndarray       # (n_particles,)
    copy_number: jnp.ndarray  # (n_types,)
    particle_types: Optional[jnp.ndarray] = None  # (n_particles,)
    rigid_body_ids: Optional[jnp.ndarray] = None  # (n_particles,)
    type_names: Optional[tuple] = None  # ('A', 'B', 'C', ...)
    
    def __post_init__(self):
        """Convert inputs to JAX arrays (no validation - must be JAX-traceable)."""
        # Convert to JAX arrays (frozen dataclass requires object.__setattr__)
        object.__setattr__(self, 'coordinates', jnp.asarray(self.coordinates))
        object.__setattr__(self, 'radius', jnp.asarray(self.radius))
        object.__setattr__(self, 'copy_number', jnp.asarray(self.copy_number))
        
        if self.particle_types is not None:
            object.__setattr__(self, 'particle_types', jnp.asarray(self.particle_types))
        
        if self.rigid_body_ids is not None:
            object.__setattr__(self, 'rigid_body_ids', jnp.asarray(self.rigid_body_ids))
    
    def validate(self):
        """
        Validate state consistency (call this OUTSIDE of JIT/vmap/grad).
        
        Raises:
            ValueError: If state is invalid
        """
        n_particles = self.coordinates.shape[0]
        
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(f"coordinates must be (n_particles, 3), got {self.coordinates.shape}")
        
        if self.radius.shape != (n_particles,):
            raise ValueError(f"radius must be (n_particles,), got {self.radius.shape}")
        
        if int(jnp.sum(self.copy_number)) != n_particles:
            raise ValueError(
                f"sum(copy_number)={int(jnp.sum(self.copy_number))} must equal "
                f"n_particles={n_particles}"
            )
        
        if self.particle_types is not None and self.particle_types.shape != (n_particles,):
            raise ValueError(f"particle_types must be (n_particles,), got {self.particle_types.shape}")
        
        if self.rigid_body_ids is not None and self.rigid_body_ids.shape != (n_particles,):
            raise ValueError(f"rigid_body_ids must be (n_particles,), got {self.rigid_body_ids.shape}")
    
    @property
    def n_particles(self) -> int:
        """Total number of particles."""
        return self.coordinates.shape[0]
    
    @property
    def n_types(self) -> int:
        """Number of particle types."""
        return self.copy_number.shape[0]
    
    def tree_flatten(self):
        """
        Flatten for JAX pytree.
        
        Returns:
            children: Tuple of arrays (JAX will process these)
            aux_data: Static data (not processed by JAX)
        """
        children = (
            self.coordinates,
            self.radius,
            self.copy_number,
            self.particle_types,
            self.rigid_body_ids,
        )
        aux_data = {"type_names": self.type_names}
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct State from flattened representation."""
        coords, radius, copy_num, ptypes, rigid_ids = children
        return cls(
            coordinates=coords,
            radius=radius,
            copy_number=copy_num,
            particle_types=ptypes,
            rigid_body_ids=rigid_ids,
            type_names=aux_data["type_names"],
        )
    
    def with_coordinates(self, new_coords: jnp.ndarray) -> "State":
        """Create new State with updated coordinates (immutable update)."""
        return State(
            coordinates=new_coords,
            radius=self.radius,
            copy_number=self.copy_number,
            particle_types=self.particle_types,
            rigid_body_ids=self.rigid_body_ids,
            type_names=self.type_names,
        )
    
    def get_type_slice(self, type_idx: int) -> tuple:
        """
        Get slice indices for a specific particle type.
        
        Args:
            type_idx: Type index (0, 1, 2, ...)
        
        Returns:
            (start, end) tuple for slicing
        """
        start = jnp.sum(self.copy_number[:type_idx])
        end = start + self.copy_number[type_idx]
        return (int(start), int(end))
    
    def get_coords_by_type(self, type_idx: int) -> jnp.ndarray:
        """Get coordinates for a specific particle type."""
        start, end = self.get_type_slice(type_idx)
        return self.coordinates[start:end]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (useful for saving/loading)."""
        return {
            "coordinates": self.coordinates,
            "radius": self.radius,
            "copy_number": self.copy_number,
            "particle_types": self.particle_types,
            "rigid_body_ids": self.rigid_body_ids,
            "type_names": self.type_names,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create State from dictionary."""
        return cls(**data)
    
    def __repr__(self):
        type_info = f", types={self.type_names}" if self.type_names else ""
        return (
            f"State(n_particles={self.n_particles}, "
            f"n_types={self.n_types}{type_info})"
        )