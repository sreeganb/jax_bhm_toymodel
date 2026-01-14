#------------------------------------------------------------------------------
# Core particle state representation for Bayesian inference
# Following Sali's 4 stages: Representation → Scoring → Sampling → Validation
#------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Tuple
from functools import partial


@dataclass
class ParticleState:
    """
    Unified particle state representation for coarse-grained systems.
    
    This is the central data structure that flows through all 4 stages:
    - Representation: Initialize particles with attributes
    - Scoring: Compute energies from positions + radii
    - Sampling: Update positions while preserving attributes
    - Validation: Save/load complete state to HDF5
    
    Attributes:
        positions: Particle coordinates, shape (N, 3)
        radii: Particle radii, shape (N,)
        particle_types: Integer type labels, shape (N,) - e.g., 0=A, 1=B, 2=C
        copy_numbers: Copy number per particle, shape (N,) - for stoichiometry
        names: Optional string names per particle
        type_names: Mapping from integer types to string names
        
    Example:
        >>> state = ParticleState.from_type_dict(
        ...     coords={'A': jnp.array([[0,0,0], [1,0,0]]), 'B': jnp.array([[2,0,0]])},
        ...     radii={'A': 1.0, 'B': 2.0}
        ... )
        >>> state.n_particles
        3
        >>> state.positions.shape
        (3, 3)
    """
    positions: jnp.ndarray  # (N, 3)
    radii: jnp.ndarray      # (N,)
    particle_types: jnp.ndarray  # (N,) integer labels
    copy_numbers: jnp.ndarray    # (N,) integer copy numbers
    names: Optional[List[str]] = None  # Optional per-particle names
    type_names: Dict[int, str] = field(default_factory=dict)  # type_id -> name
    
    def __post_init__(self):
        """Validate shapes and types."""
        n = self.positions.shape[0]
        assert self.positions.shape == (n, 3), f"positions must be (N, 3), got {self.positions.shape}"
        assert self.radii.shape == (n,), f"radii must be (N,), got {self.radii.shape}"
        assert self.particle_types.shape == (n,), f"particle_types must be (N,), got {self.particle_types.shape}"
        assert self.copy_numbers.shape == (n,), f"copy_numbers must be (N,), got {self.copy_numbers.shape}"
        if self.names is not None:
            assert len(self.names) == n, f"names must have length N={n}, got {len(self.names)}"
    
    @property
    def n_particles(self) -> int:
        """Total number of particles."""
        return self.positions.shape[0]
    
    @property
    def flat_positions(self) -> jnp.ndarray:
        """Flattened position array for samplers, shape (N*3,)."""
        return self.positions.flatten()
    
    def with_positions(self, new_positions: jnp.ndarray) -> 'ParticleState':
        """Return new state with updated positions (immutable update)."""
        if new_positions.ndim == 1:
            new_positions = new_positions.reshape(-1, 3)
        return ParticleState(
            positions=new_positions,
            radii=self.radii,
            particle_types=self.particle_types,
            copy_numbers=self.copy_numbers,
            names=self.names,
            type_names=self.type_names
        )
    
    def get_particles_by_type(self, type_id: int) -> jnp.ndarray:
        """Get positions of particles with given type."""
        mask = self.particle_types == type_id
        return self.positions[mask]
    
    def get_type_name(self, type_id: int) -> str:
        """Get string name for type ID."""
        return self.type_names.get(type_id, f"type_{type_id}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for HDF5 storage."""
        return {
            'positions': np.array(self.positions),
            'radii': np.array(self.radii),
            'particle_types': np.array(self.particle_types),
            'copy_numbers': np.array(self.copy_numbers),
            'type_names': self.type_names,
            'names': self.names
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ParticleState':
        """Reconstruct from dictionary (e.g., loaded from HDF5)."""
        return cls(
            positions=jnp.array(data['positions']),
            radii=jnp.array(data['radii']),
            particle_types=jnp.array(data['particle_types']),
            copy_numbers=jnp.array(data['copy_numbers']),
            type_names=data.get('type_names', {}),
            names=data.get('names', None)
        )
    
    @classmethod
    def from_type_dict(
        cls,
        coords: Dict[str, jnp.ndarray],
        radii: Union[Dict[str, float], float] = 1.0,
        copy_numbers: Union[Dict[str, int], int] = 1
    ) -> 'ParticleState':
        """
        Create ParticleState from dictionary of coordinates per type.
        
        This is the primary factory method for building systems from
        type-labeled coordinate dictionaries (e.g., {'A': coords_A, 'B': coords_B}).
        
        Args:
            coords: Dict mapping type names to coordinate arrays, each (N_type, 3)
            radii: Either a dict mapping type names to radii, or a single float
            copy_numbers: Either a dict mapping type names to copy numbers, or single int
            
        Returns:
            ParticleState with concatenated coordinates and per-particle attributes
        """
        type_names_list = sorted(coords.keys())
        type_name_to_id = {name: i for i, name in enumerate(type_names_list)}
        type_id_to_name = {i: name for name, i in type_name_to_id.items()}
        
        all_positions = []
        all_radii = []
        all_types = []
        all_copy_numbers = []
        all_names = []
        
        for type_name in type_names_list:
            type_coords = coords[type_name]
            n_particles = type_coords.shape[0]
            type_id = type_name_to_id[type_name]
            
            # Get radius for this type
            if isinstance(radii, dict):
                r = radii.get(type_name, 1.0)
            else:
                r = radii
            
            # Get copy number for this type
            if isinstance(copy_numbers, dict):
                cn = copy_numbers.get(type_name, 1)
            else:
                cn = copy_numbers
            
            all_positions.append(type_coords)
            all_radii.extend([r] * n_particles)
            all_types.extend([type_id] * n_particles)
            all_copy_numbers.extend([cn] * n_particles)
            all_names.extend([f"{type_name}_{i}" for i in range(n_particles)])
        
        return cls(
            positions=jnp.concatenate(all_positions, axis=0),
            radii=jnp.array(all_radii),
            particle_types=jnp.array(all_types),
            copy_numbers=jnp.array(all_copy_numbers),
            names=all_names,
            type_names=type_id_to_name
        )
    
    @classmethod
    def from_arrays(
        cls,
        positions: jnp.ndarray,
        radii: Union[jnp.ndarray, float] = 1.0,
        particle_types: Optional[jnp.ndarray] = None,
        copy_numbers: Optional[jnp.ndarray] = None,
        names: Optional[List[str]] = None
    ) -> 'ParticleState':
        """
        Create ParticleState from raw arrays (backward compatibility).
        
        Args:
            positions: Coordinates, shape (N, 3) or (N*3,)
            radii: Per-particle radii (N,) or single float
            particle_types: Per-particle type IDs (N,), defaults to all 0
            copy_numbers: Per-particle copy numbers (N,), defaults to all 1
            names: Optional list of particle names
            
        Returns:
            ParticleState instance
        """
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        n = positions.shape[0]
        
        if isinstance(radii, (int, float)):
            radii = jnp.ones(n) * radii
        if particle_types is None:
            particle_types = jnp.zeros(n, dtype=jnp.int32)
        if copy_numbers is None:
            copy_numbers = jnp.ones(n, dtype=jnp.int32)
            
        return cls(
            positions=positions,
            radii=radii,
            particle_types=particle_types,
            copy_numbers=copy_numbers,
            names=names
        )


@dataclass
class RigidBodyState:
    """
    State representation for rigid body systems.
    
    Extends ParticleState with rigid body structure information,
    storing the internal DOF representation (translation + quaternion)
    alongside reference coordinates.
    
    Attributes:
        particle_state: Underlying particle positions and attributes
        translations: Per-rigid-body translations, shape (n_rb, 3)
        quaternions: Per-rigid-body quaternions [w,x,y,z], shape (n_rb, 4)
        ref_coords_list: List of reference coordinate arrays per rigid body
        rb_assignments: Which rigid body each particle belongs to, shape (N,)
                       Value -1 indicates flexible particle
    """
    particle_state: ParticleState
    translations: jnp.ndarray  # (n_rb, 3)
    quaternions: jnp.ndarray   # (n_rb, 4)
    ref_coords_list: List[jnp.ndarray]  # List of (n_i, 3) per RB
    rb_assignments: jnp.ndarray  # (N,) which RB each particle belongs to, -1 = flexible
    
    @property
    def n_rigid_bodies(self) -> int:
        return self.translations.shape[0]
    
    @property
    def n_flexible(self) -> int:
        return int(jnp.sum(self.rb_assignments == -1))
    
    @property
    def state_vector(self) -> jnp.ndarray:
        """
        Get compact state vector for sampling: [trans + quat per RB, flex coords].
        Shape: (n_rb * 7 + n_flex * 3,)
        """
        rb_dof = jnp.concatenate([
            jnp.concatenate([self.translations[i], self.quaternions[i]])
            for i in range(self.n_rigid_bodies)
        ]) if self.n_rigid_bodies > 0 else jnp.array([])
        
        flex_mask = self.rb_assignments == -1
        flex_coords = self.particle_state.positions[flex_mask].flatten()
        
        return jnp.concatenate([rb_dof, flex_coords])
    
    @property
    def state_size(self) -> int:
        """Total DOF count."""
        return self.n_rigid_bodies * 7 + self.n_flexible * 3


# Utility functions for working with ParticleState in JAX
def positions_from_state(state: ParticleState) -> jnp.ndarray:
    """Extract positions array (for use in jit-compiled functions)."""
    return state.positions


def radii_from_state(state: ParticleState) -> jnp.ndarray:
    """Extract radii array (for use in jit-compiled functions)."""
    return state.radii


class ParticleSystemFactory:
    """Small helper to build ParticleState objects from a type spec.

    Expected spec format (matches unified_workflow.py):
        {
            "ProteinA": {"count": 5, "radius": 2.0},
            "ProteinB": {"count": 3, "radius": 3.0},
            ...
        }
    Optional key: "copy_number"; defaults to 1 if omitted.
    """

    def __init__(self, type_spec: Dict[str, Dict[str, float | int]]):
        self.type_spec = type_spec

    def create_random_state(self, key: jax.Array, box_size: float = 100.0, center: float = 0.0) -> ParticleState:
        """Generate uniformly random coordinates in a cube and return ParticleState."""
        # Build deterministic type ordering for reproducibility
        type_names_list = sorted(self.type_spec.keys())
        type_name_to_id = {name: i for i, name in enumerate(type_names_list)}

        positions_list = []
        radii_list: list[float] = []
        types_list: list[int] = []
        copy_numbers_list: list[int] = []
        names_list: list[str] = []

        half_box = box_size / 2.0
        # Split key for each type block
        subkeys = jax.random.split(key, len(type_names_list)) if len(type_names_list) > 0 else []

        for idx, type_name in enumerate(type_names_list):
            spec = self.type_spec[type_name]
            count = int(spec.get("count", 0))
            radius = float(spec.get("radius", 1.0))
            copy_number = int(spec.get("copy_number", 1))

            if count <= 0:
                continue

            # Uniform in cube [center-half_box, center+half_box]
            coords = center + jax.random.uniform(
                subkeys[idx],
                shape=(count, 3),
                minval=-half_box,
                maxval=half_box,
                dtype=jnp.float32,
            )

            positions_list.append(coords)
            radii_list.extend([radius] * count)
            types_list.extend([type_name_to_id[type_name]] * count)
            copy_numbers_list.extend([copy_number] * count)
            names_list.extend([f"{type_name}_{i}" for i in range(count)])

        if not positions_list:
            raise ValueError("No particles generated; check type_spec counts.")

        positions = jnp.concatenate(positions_list, axis=0)
        radii = jnp.array(radii_list, dtype=jnp.float32)
        particle_types = jnp.array(types_list, dtype=jnp.int32)
        copy_numbers = jnp.array(copy_numbers_list, dtype=jnp.int32)
        type_id_to_name = {v: k for k, v in type_name_to_id.items()}

        return ParticleState(
            positions=positions,
            radii=radii,
            particle_types=particle_types,
            copy_numbers=copy_numbers,
            names=names_list,
            type_names=type_id_to_name,
        )