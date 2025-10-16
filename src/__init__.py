"""
Augmented Lagrangian method for constrained optimization.

This package provides a Python implementation of the Augmented Lagrangian method
for solving equality-constrained optimization problems.
"""

from .aug_lag import AugmentedLagrangian

__version__ = "0.1.1"
__author__ = "Hongwei Jin"
__email__ = "jinh@anl.gov"

__all__ = ["AugmentedLagrangian"]
