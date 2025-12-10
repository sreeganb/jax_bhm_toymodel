"""Modular Rigid Body MCMC with Quaternion Rotations."""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import blackjax

# ============================================================
# Quaternion Operations
# ============================================================

def normalize_quaternion(q):
    """Normalize quaternion to unit length: q = [w, x, y, z]."""
    return q / jnp.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def apply_rigid_transform(ref_coords, translation, quaternion):
    """Apply rigid body transform: rotate then translate."""
    q = normalize_quaternion(quaternion)
    R = quaternion_to_rotation_matrix(q)
    return (R @ ref_coords.T).T + translation

# ============================================================
# Scoring Functions
# ============================================================

def harmonic_score(positions, restraints, k=0.5):
    """Harmonic restraint energy: E = k * sum((d - d0)^2)."""
    if len(restraints) == 0:
        return 0.0
    i_idx = restraints[:, 0].astype(int)
    j_idx = restraints[:, 1].astype(int)
    d0 = restraints[:, 2]
    distances = jnp.linalg.norm(positions[i_idx] - positions[j_idx], axis=1)
    return k * jnp.sum((distances - d0) ** 2)

def excluded_volume(positions, radii, k=1.0):
    """Soft excluded volume: E = k * sum(max(0, overlap)^2)."""
    n = positions.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = jnp.linalg.norm(positions[i] - positions[j])
            overlap = (radii[i] + radii[j]) - d
            energy += k * jnp.maximum(0.0, overlap) ** 2
    return energy

# ============================================================
# Particle and RigidBody Definitions
# ============================================================

@dataclass
class Particle:
    """Single particle definition."""
    coords: jnp.ndarray  # (3,)
    radius: float = 1.0
    copy_number: int = 1
    name: str = ""

@dataclass 
class RigidBody:
    """Rigid body with reference coordinates."""
    ref_coords: jnp.ndarray  # (N, 3)
    radii: jnp.ndarray       # (N,)
    copy_number: int = 1
    name: str = ""

# ============================================================
# System Definition
# ============================================================

class ModularSystem:
    """Modular system with rigid bodies and flexible particles."""
    
    def __init__(self, rigid_bodies: List[RigidBody], 
                 flexible_particles: Optional[List[Particle]] = None,
                 restraints: Optional[jnp.ndarray] = None):
        """
        Args:
            rigid_bodies: List of RigidBody objects
            flexible_particles: List of Particle objects  
            restraints: (M, 3) array of [idx_i, idx_j, d0] restraints
        """
        self.rigid_bodies = rigid_bodies
        self.flexible_particles = flexible_particles or []
        self.restraints = restraints if restraints is not None else jnp.array([]).reshape(0, 3)
        
        # Count particles
        self.n_rb = len(rigid_bodies)
        self.n_rb_particles = sum(rb.ref_coords.shape[0] for rb in rigid_bodies)
        self.n_flex = len(self.flexible_particles)
        self.n_total = self.n_rb_particles + self.n_flex
        
        # State: 7 per RB (3 trans + 4 quat) + 3 per flex
        self.state_size = 7 * self.n_rb + 3 * self.n_flex
        
        # Build global radii array
        radii_list = [rb.radii for rb in rigid_bodies]
        radii_list += [jnp.array([p.radius]) for p in self.flexible_particles]
        self.radii = jnp.concatenate(radii_list) if radii_list else jnp.array([])
        
        print(f"System: {self.n_rb} RBs ({self.n_rb_particles} particles), "
              f"{self.n_flex} flex, {self.state_size} DOF")
    
    def state_to_coords(self, state):
        """Convert state [trans(3), quat(4)]×n_rb + flex(3×n_flex) → coords."""
        coords_list = []
        offset = 0
        
        for rb in self.rigid_bodies:
            trans = state[offset:offset+3]
            quat = state[offset+3:offset+7]
            coords_list.append(apply_rigid_transform(rb.ref_coords, trans, quat))
            offset += 7
        
        if self.n_flex > 0:
            flex_coords = state[offset:].reshape(self.n_flex, 3)
            coords_list.append(flex_coords)
        
        return jnp.vstack(coords_list)
    
    def normalize_state(self, state):
        """Normalize all quaternions in state vector."""
        state = jnp.array(state)
        for i in range(self.n_rb):
            offset = i * 7 + 3
            q = state[offset:offset+4]
            state = state.at[offset:offset+4].set(normalize_quaternion(q))
        return state
    
    def init_state(self, rb_positions: List[jnp.ndarray], 
                   flex_positions: Optional[jnp.ndarray] = None):
        """Initialize state from RB centers and flex positions."""
        parts = []
        for pos in rb_positions:
            trans = jnp.array(pos)
            quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
            parts.append(jnp.concatenate([trans, quat]))
        
        if flex_positions is not None:
            parts.append(flex_positions.ravel())
        
        return jnp.concatenate(parts)

# ============================================================
# Log Probability
# ============================================================

def make_log_prob(system: ModularSystem, k_harmonic=0.5, k_exvol=1.0):
    """Create log probability function for system."""
    
    def log_prob(state):
        state = system.normalize_state(state)
        coords = system.state_to_coords(state)
        
        E_harm = harmonic_score(coords, system.restraints, k=k_harmonic)
        E_exvol = excluded_volume(coords, system.radii, k=k_exvol)
        
        # Quaternion prior: prefer unit quaternions (regularization)
        quat_penalty = 0.0
        for i in range(system.n_rb):
            q = state[i*7+3 : i*7+7]
            quat_penalty += 0.1 * (jnp.linalg.norm(q) - 1.0)**2
        
        return -(E_harm + E_exvol + quat_penalty)
    
    return log_prob

# ============================================================
# MCMC
# ============================================================

def setup_mcmc(log_prob, n_rb, n_flex, step_trans=0.5, step_quat=0.02, step_flex=1.0):
    """Setup RMH kernel with per-DOF step sizes."""
    steps = []
    for _ in range(n_rb):
        steps.extend([step_trans]*3 + [step_quat]*4)
    steps.extend([step_flex] * n_flex * 3)
    
    proposal = blackjax.mcmc.random_walk.normal(jnp.array(steps))
    return blackjax.rmh(log_prob, proposal)

def run_mcmc(key, kernel, system, init_state, n_steps, save_every=100, print_every=1000):
    """Run MCMC sampling."""
    state = kernel.init(init_state)
    
    @jax.jit
    def step(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, (new_state.position, new_state.logdensity, info.is_accepted)
    
    keys = random.split(key, n_steps)
    saved = {'states': [np.array(init_state)], 
             'coords': [np.array(system.state_to_coords(init_state))],
             'log_probs': [float(state.logdensity)]}
    accepts = []
    
    print(f"Running {n_steps:,} steps, init log_prob: {state.logdensity:.2f}")
    
    for i, k in enumerate(keys):
        state, (pos, lp, acc) = step(state, k)
        accepts.append(float(acc))
        
        if (i+1) % save_every == 0:
            saved['states'].append(np.array(pos))
            saved['coords'].append(np.array(system.state_to_coords(pos)))
            saved['log_probs'].append(float(lp))
        
        if (i+1) % print_every == 0:
            acc_rate = np.mean(accepts[-1000:])
            print(f"Step {i+1:,}: log_prob={lp:.2f}, accept={acc_rate:.1%}")
    
    saved['states'] = np.array(saved['states'])
    saved['coords'] = np.array(saved['coords'])
    saved['log_probs'] = np.array(saved['log_probs'])
    saved['acceptance'] = np.mean(accepts)
    
    print(f"Done! Final: {lp:.2f}, Best: {np.max(saved['log_probs']):.2f}, "
          f"Accept: {saved['acceptance']:.1%}")
    return saved

def save_mcmc_to_hdf5(
    coords: np.ndarray,
    states: np.ndarray,
    log_probs: np.ndarray,
    acceptance_rate: float,
    system: ModularSystem,
    filename: str,
    initial_state: Optional[np.ndarray] = None
):
    """
    Save rigid body MCMC results to HDF5 in standard format.
    
    Args:
        coords: (n_samples, n_particles, 3) Cartesian coordinates
        states: (n_samples, state_size) rigid body DOFs
        log_probs: (n_samples,) log probabilities
        acceptance_rate: Overall acceptance rate
        system: ModularSystem instance
        filename: Output HDF5 filename
        initial_state: Optional initial state vector
    
    The saved HDF5 file is compatible with existing h5_to_rmf3 converters.
    """
    n_samples = coords.shape[0]
    n_particles = coords.shape[1]
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['n_particles'] = n_particles
        f.attrs['acceptance_rate'] = acceptance_rate
        
        # System information
        f.attrs['n_rigid_bodies'] = system.n_rb
        f.attrs['n_rigid_particles'] = system.n_rb_particles
        f.attrs['n_flexible_particles'] = system.n_flex
        f.attrs['state_dof'] = system.state_size
        
        # Main datasets - CARTESIAN COORDINATES (for visualization)
        f.create_dataset('coordinates', data=coords, compression='gzip')
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Store radii for visualization
        f.create_dataset('radii', data=system.radii)
        
        # Rigid body states (for analysis)
        rb_grp = f.create_group('rigid_body_states')
        rb_grp.create_dataset('states', data=states, compression='gzip')
        
        # Store reference configurations
        rb_ref_grp = f.create_group('rigid_body_references')
        for i, rb in enumerate(system.rigid_bodies):
            rb_ref_grp.create_dataset(f'body_{i}_coords', data=rb.ref_coords)
            rb_ref_grp.create_dataset(f'body_{i}_radii', data=rb.radii)
            rb_ref_grp.attrs[f'body_{i}_name'] = rb.name
            rb_ref_grp.attrs[f'body_{i}_copy_number'] = rb.copy_number
            rb_ref_grp.attrs[f'body_{i}_n_particles'] = rb.ref_coords.shape[0]
        
        # Initial configuration
        if initial_state is not None:
            init_grp = f.create_group('initial_configuration')
            init_coords = system.state_to_coords(initial_state)
            init_grp.create_dataset('coordinates', data=init_coords)
            init_grp.create_dataset('state', data=initial_state)
            
            # Store initial rigid body DOFs
            offset = 0
            for i in range(system.n_rb):
                rb_state = initial_state[offset:offset+7]
                init_grp.attrs[f'rb{i}_translation'] = rb_state[:3]
                init_grp.attrs[f'rb{i}_quaternion'] = rb_state[3:7]
                offset += 7
        
        # Best configuration
        best_idx = np.argmax(log_probs)
        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = int(best_idx)
        best_grp.attrs['log_probability'] = float(log_probs[best_idx])
        best_grp.create_dataset('coordinates', data=coords[best_idx])
        best_grp.create_dataset('state', data=states[best_idx])
        
        # Store best rigid body DOFs
        offset = 0
        for i in range(system.n_rb):
            rb_state = states[best_idx, offset:offset+7]
            best_grp.attrs[f'rb{i}_translation'] = rb_state[:3]
            best_grp.attrs[f'rb{i}_quaternion'] = rb_state[3:7]
            offset += 7
        
        # Restraint information (for reproducibility)
        if system.restraints.shape[0] > 0:
            f.create_dataset('restraints', data=system.restraints)
    
    print(f"\n{'='*70}")
    print(f"Saved to: {filename}")
    print(f"{'='*70}")
    print(f"  Frames: {n_samples}")
    print(f"  Particles: {n_particles} ({system.n_rb_particles} rigid + {system.n_flex} flex)")
    print(f"  Rigid bodies: {system.n_rb}")
    print(f"  Acceptance rate: {acceptance_rate:.1%}")
    print(f"  Best log_prob: {np.max(log_probs):.2f} (frame {best_idx})")
    print(f"{'='*70}")
    print("\nDatasets saved:")
    print(f"  - coordinates (n_samples={n_samples}, n_particles={n_particles}, 3)")
    print(f"  - log_probabilities (n_samples={n_samples})")
    print(f"  - radii (n_particles={n_particles})")
    print(f"  - rigid_body_states/states (internal DOFs)")
    print(f"  - rigid_body_references/* (reference structures)")
    print(f"\n✓ Compatible with existing h5_to_rmf3 converter!")
    print(f"{'='*70}\n")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Define two rigid bodies with different copy numbers
    rb1 = RigidBody(
        ref_coords=jnp.array([[0., 0., 0.], [2., 0., 0.], [1., 1.73, 0.]]),
        radii=jnp.array([1.0, 1.0, 1.0]),
        copy_number=1, name="triangle"
    )
    rb2 = RigidBody(
        ref_coords=jnp.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]]),
        radii=jnp.array([1.2, 1.2, 1.2, 1.2]),  # Different radii
        copy_number=2, name="square"
    )
    
    # Flexible linker particles
    flex = [Particle(jnp.array([5., 0., 0.]), radius=0.8, name=f"flex_{i}") for i in range(3)]
    
    # Restraints: [particle_i, particle_j, ideal_distance]
    restraints = jnp.array([
        [2, 7, 3.5],   # RB1 to flex
        [7, 8, 3.0],   # flex internal
        [8, 9, 3.0],   # flex internal  
        [9, 3, 4.0],   # flex to RB2 (different d0)
    ])
    
    # Create system
    system = ModularSystem([rb1, rb2], flex, restraints)
    
    # Initialize: RB1 at (2,2,0), RB2 at (12,2,0), flex spread between
    init_state = system.init_state(
        rb_positions=[jnp.array([2., 2., 0.]), jnp.array([12., 2., 0.])],
        flex_positions=jnp.array([[5., 1., 0.], [7., 1., 0.], [9., 1., 0.]])
    )
    
    # Setup and run
    log_prob = make_log_prob(system, k_harmonic=0.5, k_exvol=1.0)
    kernel = setup_mcmc(log_prob, system.n_rb, system.n_flex)
    
    results = run_mcmc(
        random.PRNGKey(42), kernel, system, init_state,
        n_steps=20_000, save_every=50, print_every=2000
    )
    
    # Save using standard format (compatible with h5_to_rmf3)
    save_mcmc_to_hdf5(
        coords=results['coords'],
        states=results['states'],
        log_probs=results['log_probs'],
        acceptance_rate=results['acceptance'],
        system=system,
        filename="rigid_body_mcmc.h5",
        initial_state=init_state
    )
    
    # Verify HDF5 structure
    print("\nVerifying HDF5 structure:")
    with h5py.File("rigid_body_mcmc.h5", 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        print(f"  coordinates shape: {f['coordinates'].shape}")
        print(f"  radii shape: {f['radii'].shape}")
        print(f"  Has initial_configuration: {'initial_configuration' in f}")
        print(f"  Has best_configuration: {'best_configuration' in f}")
        print(f"  Rigid body references: {list(f['rigid_body_references'].keys())}")