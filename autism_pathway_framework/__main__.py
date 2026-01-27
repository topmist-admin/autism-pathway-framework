"""
Entry point for running the package as a module.

Usage:
    python -m autism_pathway_framework --config configs/demo.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
