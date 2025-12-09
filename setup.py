from setuptools import setup, find_packages

setup(
    name="jax_bhm_toymodel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "blackjax",
    ],
)
