import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Dict, Optional, List, Tuple


def detect_hdf5_structure(hdf5_file: str) -> Dict:
    """
    Detect the structure of the HDF5 file.
    
    Returns a dictionary with:
        - has_coordinates: bool
        - has_sigma: bool
        - has_other_params: list of other parameter names
        - n_samples: int
        - n_particles: int (if coordinates exist)
    """
    structure = {
        'has_coordinates': False,
        'has_sigma': False,
        'has_other_params': [],
        'n_samples': 0,
        'n_particles': 0,
        'datasets': []
    }
    
    with h5py.File(hdf5_file, 'r') as f:
        # Check for standard datasets
        if 'coordinates' in f:
            structure['has_coordinates'] = True
            coords = f['coordinates'][:]
            structure['n_samples'] = coords.shape[0]
            structure['n_particles'] = coords.shape[1]
        
        if 'sigma' in f:
            structure['has_sigma'] = True
            if structure['n_samples'] == 0:
                structure['n_samples'] = len(f['sigma'][:])
        
        if 'log_probabilities' in f:
            if structure['n_samples'] == 0:
                structure['n_samples'] = len(f['log_probabilities'][:])
        
        # Detect other numeric datasets (potential parameters)
        for key in f.keys():
            if key not in ['coordinates', 'sigma', 'log_probabilities', 
                          'initial_configuration', 'best_configuration']:
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset) and len(dataset.shape) == 1:
                    structure['has_other_params'].append(key)
        
        structure['datasets'] = list(f.keys())
    
    return structure


def compute_pairwise_distances(coords: np.ndarray, 
                               pair_indices: Optional[List[Tuple[int, int]]] = None) -> Dict[str, np.ndarray]:
    """
    Compute pairwise distances between particles.
    
    Args:
        coords: Shape (n_samples, n_particles, 3)
        pair_indices: List of (i, j) tuples. If None, compute all pairs.
    
    Returns:
        Dictionary with keys like "d_0_1", "d_1_2", etc.
    """
    n_samples, n_particles, _ = coords.shape
    distances = {}
    
    if pair_indices is None:
        # Default: compute first pair only (backward compatibility)
        if n_particles >= 2:
            pair_indices = [(0, 1)]
        else:
            return {}
    
    for i, j in pair_indices:
        if i < n_particles and j < n_particles:
            dists = np.linalg.norm(coords[:, j] - coords[:, i], axis=1)
            distances[f'd_{i}_{j}'] = dists
    
    return distances


def plot_trace(ax, data: np.ndarray, ylabel: str, title: str, 
               target_value: Optional[float] = None):
    """Generic trace plot."""
    ax.plot(data, linewidth=0.5, alpha=0.8)
    if target_value is not None:
        ax.axhline(target_value, color='red', linestyle='--', 
                   linewidth=2, label=f'Target ({target_value:.2f})')
        ax.legend()
    ax.set_xlabel('Sample', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_histogram(ax, data: np.ndarray, xlabel: str, title: str,
                   target_value: Optional[float] = None, bins: int = 50):
    """Generic histogram plot."""
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', density=True)
    if target_value is not None:
        ax.axvline(target_value, color='red', linestyle='--', 
                   linewidth=2, label=f'Target ({target_value:.2f})')
    ax.axvline(np.mean(data), color='blue', linestyle='--', 
               linewidth=2, label=f'Mean ({np.mean(data):.3f})')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')


def plot_autocorrelation(ax, data: np.ndarray, max_lag: int = 500):
    """Plot autocorrelation function."""
    n_samples = len(data)
    max_lag = min(max_lag, n_samples // 2)
    
    # Compute autocorrelation
    autocorr = np.correlate(data - np.mean(data), 
                           data - np.mean(data), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]
    
    ax.plot(autocorr[:max_lag], linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=14)
    ax.set_ylabel('Autocorrelation', fontsize=14)
    ax.set_title('Autocorrelation Function', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_mcmc_diagnostics(hdf5_file: str, 
                         output_dir: str = "output_figures",
                         target_distances: Optional[Dict[str, float]] = None,
                         target_sigma: Optional[float] = None,
                         pair_indices: Optional[List[Tuple[int, int]]] = None):
    """
    Generate MCMC diagnostic plots from HDF5 file (flexible version).
    
    Args:
        hdf5_file: Path to MCMC HDF5 file
        output_dir: Directory to save plots
        target_distances: Dict like {"d_0_1": 4.5} for target distances
        target_sigma: Target value for sigma (if applicable)
        pair_indices: List of (i, j) pairs to compute distances for
    
    Returns:
        Dictionary with statistics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"{'='*70}")
    print(f"Generating MCMC Diagnostics")
    print(f"{'='*70}")
    print(f"Input: {hdf5_file}")
    print(f"Output directory: {output_dir}")
    
    # Detect file structure
    structure = detect_hdf5_structure(hdf5_file)
    print(f"\nDetected structure:")
    print(f"  Samples: {structure['n_samples']}")
    print(f"  Particles: {structure['n_particles']}")
    print(f"  Has coordinates: {structure['has_coordinates']}")
    print(f"  Has sigma: {structure['has_sigma']}")
    print(f"  Other parameters: {structure['has_other_params']}")
    print(f"{'='*70}\n")
    
    # Load data
    data = {}
    with h5py.File(hdf5_file, 'r') as f:
        if 'log_probabilities' in f:
            data['log_probs'] = f['log_probabilities'][:]
        
        if structure['has_coordinates']:
            data['coords'] = f['coordinates'][:]
        
        if structure['has_sigma']:
            data['sigma'] = f['sigma'][:]
        
        # Load other parameters
        for param_name in structure['has_other_params']:
            data[param_name] = f[param_name][:]
        
        data['acceptance_rate'] = f.attrs.get('acceptance_rate', None)
        
        # Check initial configuration
        if 'initial_configuration' in f:
            print("Initial configuration found:")
            if 'coordinates' in f['initial_configuration']:
                initial_coords = f['initial_configuration/coordinates'][:]
                print(f"  Coordinates shape: {initial_coords.shape}")
                if structure['has_coordinates'] and np.allclose(data['coords'][0], initial_coords):
                    print("  ✓ Frame 0 matches initial coordinates")
            if 'sigma' in f['initial_configuration'].attrs:
                initial_sigma = f['initial_configuration'].attrs['sigma']
                print(f"  Initial sigma: {initial_sigma:.4f}")
                if structure['has_sigma'] and np.allclose(data['sigma'][0], initial_sigma):
                    print("  ✓ Frame 0 matches initial sigma")
    
    n_samples = structure['n_samples']
    stats = {'n_samples': n_samples}
    
    # ============================================================
    # 1. Log Probability Diagnostics
    # ============================================================
    if 'log_probs' in data:
        print("Generating log probability diagnostics...")
        log_probs = data['log_probs']
        
        # Trace
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_trace(ax, log_probs, 'Log Probability', 'MCMC Trace: Log Probability')
        plt.tight_layout()
        plt.savefig(output_path / "trace_log_prob.png", dpi=250)
        plt.close()
        
        # Running mean
        window = min(500, n_samples // 10)
        running_mean = np.convolve(log_probs, np.ones(window)/window, mode='valid')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(running_mean, linewidth=2, color='C2')
        ax.set_xlabel('Sample', fontsize=14)
        ax.set_ylabel(f'Running Mean (window={window})', fontsize=14)
        ax.set_title('Log Probability Running Mean', fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "running_mean_log_prob.png", dpi=250)
        plt.close()
        
        # Autocorrelation
        if n_samples > 100:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_autocorrelation(ax, log_probs)
            ax.set_title('Log Probability Autocorrelation', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / "autocorr_log_prob.png", dpi=250)
            plt.close()
        
        stats['log_prob_mean'] = float(np.mean(log_probs))
        stats['log_prob_std'] = float(np.std(log_probs))
        stats['log_prob_min'] = float(np.min(log_probs))
        stats['log_prob_max'] = float(np.max(log_probs))
    
    # ============================================================
    # 2. Distance Diagnostics (if coordinates exist)
    # ============================================================
    if structure['has_coordinates']:
        print("Generating distance diagnostics...")
        coords = data['coords']
        
        # Compute distances
        distances = compute_pairwise_distances(coords, pair_indices)
        
        for dist_name, dist_values in distances.items():
            target_val = target_distances.get(dist_name) if target_distances else None
            
            # Trace
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_trace(ax, dist_values, 'Distance (Å)', 
                      f'Particle Distance: {dist_name}', target_val)
            plt.tight_layout()
            plt.savefig(output_path / f"trace_{dist_name}.png", dpi=250)
            plt.close()
            
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_histogram(ax, dist_values, 'Distance (Å)', 
                          f'Distance Distribution: {dist_name}', target_val)
            plt.tight_layout()
            plt.savefig(output_path / f"hist_{dist_name}.png", dpi=250)
            plt.close()
            
            # Autocorrelation
            if n_samples > 100:
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_autocorrelation(ax, dist_values)
                ax.set_title(f'Autocorrelation: {dist_name}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path / f"autocorr_{dist_name}.png", dpi=250)
                plt.close()
            
            stats[f'{dist_name}_mean'] = float(np.mean(dist_values))
            stats[f'{dist_name}_std'] = float(np.std(dist_values))
    
    # ============================================================
    # 3. Sigma Diagnostics (if sigma exists)
    # ============================================================
    if structure['has_sigma']:
        print("Generating sigma diagnostics...")
        sigma = data['sigma']
        
        # Trace
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_trace(ax, sigma, 'Sigma', 'MCMC Trace: Sigma Parameter', target_sigma)
        plt.tight_layout()
        plt.savefig(output_path / "trace_sigma.png", dpi=250)
        plt.close()
        
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_histogram(ax, sigma, 'Sigma', 'Sigma Posterior Distribution', target_sigma)
        plt.tight_layout()
        plt.savefig(output_path / "hist_sigma.png", dpi=250)
        plt.close()
        
        # Running mean
        window = min(500, n_samples // 10)
        running_mean = np.convolve(sigma, np.ones(window)/window, mode='valid')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(running_mean, linewidth=2, color='C3')
        if target_sigma is not None:
            ax.axhline(target_sigma, color='red', linestyle='--', 
                      linewidth=2, label=f'Target ({target_sigma:.2f})')
            ax.legend()
        ax.set_xlabel('Sample', fontsize=14)
        ax.set_ylabel(f'Running Mean (window={window})', fontsize=14)
        ax.set_title('Sigma Running Mean', fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "running_mean_sigma.png", dpi=250)
        plt.close()
        
        # Autocorrelation
        if n_samples > 100:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_autocorrelation(ax, sigma)
            ax.set_title('Sigma Autocorrelation', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / "autocorr_sigma.png", dpi=250)
            plt.close()
        
        stats['sigma_mean'] = float(np.mean(sigma))
        stats['sigma_std'] = float(np.std(sigma))
        stats['sigma_min'] = float(np.min(sigma))
        stats['sigma_max'] = float(np.max(sigma))
    
    # ============================================================
    # 4. Other Parameters (if any)
    # ============================================================
    for param_name in structure['has_other_params']:
        print(f"Generating diagnostics for {param_name}...")
        param_data = data[param_name]
        
        # Trace
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_trace(ax, param_data, param_name, f'MCMC Trace: {param_name}')
        plt.tight_layout()
        plt.savefig(output_path / f"trace_{param_name}.png", dpi=250)
        plt.close()
        
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_histogram(ax, param_data, param_name, f'{param_name} Distribution')
        plt.tight_layout()
        plt.savefig(output_path / f"hist_{param_name}.png", dpi=250)
        plt.close()
        
        stats[f'{param_name}_mean'] = float(np.mean(param_data))
        stats[f'{param_name}_std'] = float(np.std(param_data))
    
    # ============================================================
    # 5. Summary Panel
    # ============================================================
    print("Generating summary plot...")
    
    # Determine layout based on what's available
    n_plots = 1  # Always have log_prob
    if structure['has_coordinates']:
        n_plots += 1
    if structure['has_sigma']:
        n_plots += 1
    
    # Create figure with dynamic layout
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        axes = [axes]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes = np.atleast_1d(axes).flatten()
    
    plot_idx = 0
    
    # Panel: Log prob
    if 'log_probs' in data:
        axes[plot_idx].plot(data['log_probs'], linewidth=0.5, alpha=0.8)
        axes[plot_idx].set_xlabel('Sample')
        axes[plot_idx].set_ylabel('Log Probability')
        axes[plot_idx].set_title('Log Probability Trace')
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Panel: Distance (first pair)
    if structure['has_coordinates'] and distances:
        first_dist_name = list(distances.keys())[0]
        first_dist = distances[first_dist_name]
        target_val = target_distances.get(first_dist_name) if target_distances else None
        
        axes[plot_idx].plot(first_dist, linewidth=0.5, alpha=0.8, color='C1')
        if target_val is not None:
            axes[plot_idx].axhline(target_val, color='red', linestyle='--', label='Target')
            axes[plot_idx].legend()
        axes[plot_idx].set_xlabel('Sample')
        axes[plot_idx].set_ylabel('Distance (Å)')
        axes[plot_idx].set_title(f'Distance: {first_dist_name}')
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Panel: Sigma
    if structure['has_sigma']:
        axes[plot_idx].plot(data['sigma'], linewidth=0.5, alpha=0.8, color='C2')
        if target_sigma is not None:
            axes[plot_idx].axhline(target_sigma, color='red', linestyle='--', label='Target')
            axes[plot_idx].legend()
        axes[plot_idx].set_xlabel('Sample')
        axes[plot_idx].set_ylabel('Sigma')
        axes[plot_idx].set_title('Sigma Parameter')
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Panel: Statistics text
    if len(axes) > plot_idx:
        axes[plot_idx].axis('off')
        
        stats_lines = [
            "MCMC Statistics",
            "=" * 35,
            f"Samples: {n_samples:,}",
            ""
        ]
        
        if 'log_probs' in data:
            stats_lines.extend([
                "Log Probability:",
                f"  Mean: {stats['log_prob_mean']:.2f}",
                f"  Std:  {stats['log_prob_std']:.2f}",
                ""
            ])
        
        if structure['has_coordinates'] and distances:
            first_dist_name = list(distances.keys())[0]
            stats_lines.extend([
                f"Distance ({first_dist_name}):",
                f"  Mean: {stats[f'{first_dist_name}_mean']:.3f} Å",
                f"  Std:  {stats[f'{first_dist_name}_std']:.3f} Å",
                ""
            ])
        
        if structure['has_sigma']:
            stats_lines.extend([
                "Sigma:",
                f"  Mean: {stats['sigma_mean']:.4f}",
                f"  Std:  {stats['sigma_std']:.4f}",
                ""
            ])
        
        if data['acceptance_rate'] is not None:
            stats_lines.append(f"Acceptance: {data['acceptance_rate']:.1%}")
        
        stats_text = "\n".join(stats_lines)
        axes[plot_idx].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path / "summary.png", dpi=250)
    plt.close()
    
    # ============================================================
    # Print Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("Diagnostics Complete!")
    print(f"{'='*70}")
    print("Generated plots:")
    for file in sorted(output_path.glob("*.png")):
        print(f"  - {file.name}")
    print(f"\nAll saved to: {output_path}/")
    print(f"{'='*70}")
    
    stats['acceptance_rate'] = data['acceptance_rate']
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostics_mcmc.py <input.h5> [output_dir]")
        print("\nExample:")
        print("  python diagnostics_mcmc.py mcmc_coords.h5")
        print("  python diagnostics_mcmc.py mcmc_coords_with_sigma.h5 my_output/")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_figures"
    
    # Optional: specify target values
    target_distances = {"d_0_1": 4.5}  # Target distance for pair (0,1)
    target_sigma = None  # Or set to expected value
    
    stats = plot_mcmc_diagnostics(
        hdf5_file, 
        output_dir, 
        target_distances=target_distances,
        target_sigma=target_sigma,
        pair_indices=[(0, 1)]  # Compute distance for first pair
    )
    
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")