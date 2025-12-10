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

def save_results(saved, filename):
    """Save MCMC results to HDF5."""
    with h5py.File(filename, 'w') as f:
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['acceptance'] = saved['acceptance']
        f.create_dataset('coordinates', data=saved['coords'], compression='gzip')
        f.create_dataset('states', data=saved['states'], compression='gzip')
        f.create_dataset('log_probs', data=saved['log_probs'], compression='gzip')
        
        best_idx = np.argmax(saved['log_probs'])
        f.attrs['best_idx'] = best_idx
        f.attrs['best_log_prob'] = saved['log_probs'][best_idx]
    print(f"Saved {len(saved['coords'])} frames to {filename}")

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
    save_results(results, "rigid_body_quat.h5")