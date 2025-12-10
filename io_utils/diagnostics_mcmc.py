import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path


def plot_mcmc_diagnostics(hdf5_file, output_dir="output_figures"):
    """
    Generate MCMC diagnostic plots from HDF5 file.
    
    Creates:
        - trace_plot.png: Log probability trace
        - distance_trace.png: Particle distance over time
        - distance_histogram.png: Distance distribution
        - acceptance_rate.png: Running acceptance rate
        - convergence_summary.png: Multi-panel summary
    
    Args:
        hdf5_file: Path to MCMC HDF5 file
        output_dir: Directory to save plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"{'='*70}")
    print(f"Generating MCMC Diagnostics")
    print(f"{'='*70}")
    print(f"Input: {hdf5_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load data
    with h5py.File(hdf5_file, 'r') as f:
        coords = f['coordinates'][:]  # (n_samples, n_particles, 3)
        log_probs = f['log_probabilities'][:]
        acceptance_rate = f.attrs.get('acceptance_rate', None)
        n_samples = coords.shape[0]
    
    # Calculate distances between particles
    distances = np.linalg.norm(coords[:, 1] - coords[:, 0], axis=1)
    
    print(f"Loaded {n_samples} samples")
    print(f"Overall acceptance rate: {acceptance_rate:.1%}" if acceptance_rate else "")
    
    # ============================================================
    # 1. Log Probability Trace
    # ============================================================
    print("Generating trace plot...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(log_probs, linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Log Probability', fontsize=12)
    ax.set_title('MCMC Trace: Log Probability', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "trace_plot.png", dpi=150)
    plt.close()
    
    # ============================================================
    # 2. Distance Trace
    # ============================================================
    print("Generating distance trace...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(distances, linewidth=0.5, alpha=0.8, color='C1')
    ax.axhline(10.0, color='red', linestyle='--', linewidth=2, label='Target (d₀=10)')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Distance (Å)', fontsize=12)
    ax.set_title('Particle Distance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "distance_trace.png", dpi=150)
    plt.close()
    
    # ============================================================
    # 3. Distance Histogram
    # ============================================================
    print("Generating distance histogram...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(distances, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(10.0, color='red', linestyle='--', linewidth=2, label='Target (d₀=10)')
    ax.axvline(np.mean(distances), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean ({np.mean(distances):.2f})')
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "distance_histogram.png", dpi=150)
    plt.close()
    
    # ============================================================
    # 4. Running Mean (Convergence Check)
    # ============================================================
    print("Generating convergence check...")
    window = min(500, n_samples // 10)
    running_mean = np.convolve(log_probs, np.ones(window)/window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(running_mean, linewidth=2, color='C2')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel(f'Running Mean (window={window})', fontsize=12)
    ax.set_title('Log Probability Running Mean', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "running_mean.png", dpi=150)
    plt.close()
    
    # ============================================================
    # 5. Autocorrelation (if enough samples)
    # ============================================================
    if n_samples > 100:
        print("Generating autocorrelation plot...")
        max_lag = min(500, n_samples // 2)
        autocorr = np.correlate(log_probs - np.mean(log_probs), 
                                log_probs - np.mean(log_probs), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(autocorr[:max_lag], linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title('Log Probability Autocorrelation', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "autocorrelation.png", dpi=150)
        plt.close()
    
    # ============================================================
    # 6. Summary Panel (4 subplots)
    # ============================================================
    print("Generating summary plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Log prob trace
    axes[0, 0].plot(log_probs, linewidth=0.5, alpha=0.8)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Log Probability')
    axes[0, 0].set_title('Log Probability Trace')
    axes[0, 0].grid(alpha=0.3)
    
    # Panel 2: Distance trace
    axes[0, 1].plot(distances, linewidth=0.5, alpha=0.8, color='C1')
    axes[0, 1].axhline(10.0, color='red', linestyle='--', label='Target')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Distance (Å)')
    axes[0, 1].set_title('Particle Distance')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Panel 3: Distance histogram
    axes[1, 0].hist(distances, bins=40, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(10.0, color='red', linestyle='--', linewidth=2, label='Target')
    axes[1, 0].set_xlabel('Distance (Å)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distance Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Panel 4: Statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
    MCMC Statistics
    {'='*30}
    
    Samples: {n_samples:,}
    
    Log Probability:
      Min: {np.min(log_probs):.2f}
      Max: {np.max(log_probs):.2f}
      Mean: {np.mean(log_probs):.2f}
      Std: {np.std(log_probs):.2f}
    
    Distance:
      Min: {np.min(distances):.2f} Å
      Max: {np.max(distances):.2f} Å
      Mean: {np.mean(distances):.2f} Å
      Std: {np.std(distances):.2f} Å
      Target: 10.00 Å
    
    Acceptance Rate: {acceptance_rate:.1%}
    """ if acceptance_rate else f"""
    MCMC Statistics
    {'='*30}
    
    Samples: {n_samples:,}
    
    Log Probability:
      Min: {np.min(log_probs):.2f}
      Max: {np.max(log_probs):.2f}
      Mean: {np.mean(log_probs):.2f}
      Std: {np.std(log_probs):.2f}
    
    Distance:
      Min: {np.min(distances):.2f} Å
      Max: {np.max(distances):.2f} Å
      Mean: {np.mean(distances):.2f} Å
      Std: {np.std(distances):.2f} Å
      Target: 10.00 Å
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path / "convergence_summary.png", dpi=150)
    plt.close()
    
    # ============================================================
    # Print Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("Diagnostics Complete!")
    print(f"{'='*70}")
    print(f"Generated plots:")
    print(f"  - trace_plot.png")
    print(f"  - distance_trace.png")
    print(f"  - distance_histogram.png")
    print(f"  - running_mean.png")
    if n_samples > 100:
        print(f"  - autocorrelation.png")
    print(f"  - convergence_summary.png")
    print(f"\nAll saved to: {output_path}/")
    print(f"{'='*70}")
    
    return {
        'n_samples': n_samples,
        'log_prob_mean': float(np.mean(log_probs)),
        'log_prob_std': float(np.std(log_probs)),
        'distance_mean': float(np.mean(distances)),
        'distance_std': float(np.std(distances)),
        'acceptance_rate': acceptance_rate
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_mcmc_diagnostics.py <input.h5> [output_dir]")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_figures"
    
    stats = plot_mcmc_diagnostics(hdf5_file, output_dir)