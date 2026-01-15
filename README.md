# jax_bhm_toymodel

JAX implementation of Bayesian Hierarchical Modeling for protein structure sampling.

## Features
- GPU-accelerated scoring functions (harmonic restraints, excluded volume)
- Union-of-argmin pairing strategy for ambiguous data
- Sequential Monte Carlo (SMC) sampler
- HDF5 trajectory storage with RMF3 export for visualization

## Installation

### All platforms (CPU)
```bash
pip install -e .
```

### Linux/Windows with NVIDIA GPU
```bash
pip install -e ".[cuda]"
# or manually:
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### macOS (Intel or Apple Silicon)
```bash
# JAX runs on CPU by default on Mac
pip install -e .

# Note: Metal (Apple GPU) support is experimental and not recommended yet
```

## Quick Start

```python
python examples/unified_workflow.py
```

## Platform Compatibility

| Platform | Backend | Status |
|----------|---------|--------|
| Linux + NVIDIA GPU | CUDA | ✅ Full speed |
| Linux CPU | XLA | ✅ Works |
| Windows + NVIDIA GPU | CUDA | ✅ Full speed |
| Windows CPU | XLA | ✅ Works |
| macOS Intel | CPU | ✅ Works |
| macOS Apple Silicon | CPU | ✅ Works |
| macOS Apple Silicon | Metal | ⚠️ Experimental |

## Dependencies
- JAX >= 0.4.0
- NumPy >= 1.20
- h5py >= 3.0
- IMP (optional, for RMF3 export)
