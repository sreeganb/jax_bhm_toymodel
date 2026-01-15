from setuptools import setup, find_packages

setup(
    name="jax_bhm_toymodel",
    version="0.1.0",
    description="JAX-based Bayesian Hierarchical Modeling toy model for protein sampling",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20",
        "h5py>=3.0",
    ],
    extras_require={
        # For NVIDIA GPU support (Linux/Windows)
        "cuda": ["jax[cuda12]"],
        # For development/testing
        "dev": ["pytest", "matplotlib"],
        # For RMF3 conversion (requires IMP separately)
        "rmf": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
