#!/usr/bin/env python3
"""
Setup script for Markov Chain of ER Graphs package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="markov-chain-er-graphs",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Greedy reconstruction algorithms for sequences of Erdős-Rényi graphs with noisy permutations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/markov-chain-of-er-graphs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "greedy-reorder-mc=src.greedy_reorder_mc:main",
            "compare-modes=src.compare_modes:main", 
            "debug-trial=src.debug_trial:main",
            "analyze-results=src.analyze_debug:main",
        ],
    },
    keywords="graph-theory, markov-chains, greedy-algorithms, pytorch, monte-carlo",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/markov-chain-of-er-graphs/issues",
        "Source": "https://github.com/yourusername/markov-chain-of-er-graphs",
        "Documentation": "https://github.com/yourusername/markov-chain-of-er-graphs#readme",
    },
)
