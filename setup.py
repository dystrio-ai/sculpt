"""Compatibility shim for older setuptools that can't read pyproject.toml [project]."""
from setuptools import setup, find_packages

setup(
    name="dystrio-sculpt",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "dystrio=dystrio_sculpt.cli:app",
        ],
    },
)
