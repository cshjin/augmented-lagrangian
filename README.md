# Augmented Lagrangian

[![PyPI](https://img.shields.io/pypi/v/augmented_lagrangian.svg)](https://pypi.org/project/augmented-lagrangian/)
[![Python Version](https://img.shields.io/pypi/pyversions/augmented_lagrangian.svg)](https://pypi.org/project/augmented-lagrangian/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


A Python implementation of the Augmented Lagrangian method for constrained optimization problems.

## Overview

The Augmented Lagrangian method is a powerful technique for solving constrained optimization problems of the form:

```
minimize   f(x)
subject to c_i(x) = 0  for i = 1, ..., m
```

This package provides a flexible and easy-to-use implementation that can handle both single and multiple equality constraints.

## Features

- **Multiple backends**: Support for SciPy (BFGS) and PyTorch (SGD) optimization backends
- **Flexible constraint handling**: Support for single or multiple equality constraints
- **Customizable parameters**: Control penalty parameters, tolerances, and iteration limits
- **Convergence monitoring**: Track optimization progress with detailed history
- **Easy-to-use API**: Simple interface for defining objective and constraint functions
- **GPU acceleration**: PyTorch backend supports GPU acceleration when available

## Installation

```bash
# Basic installation (SciPy backend only)
pip install augmented-lagrangian

# With PyTorch backend support
pip install augmented-lagrangian[pytorch]
```

## Quick Start

Here's a simple example of using the Augmented Lagrangian solver:

```python
import numpy as np
from aug_lag import AugmentedLagrangian

# Define objective function: minimize (x1 - 1)^2 + (x2 - 2)^2
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

# Define constraint: x1 + x2 - 3 = 0
def constraint(x):
    return x[0] + x[1] - 3

# Create solver instance
solver = AugmentedLagrangian(
    objective_func=objective,
    constraint_funcs=constraint,
    tolerance=1e-6,
    verbose=True
)

# Solve the problem
x0 = np.array([0.0, 0.0])  # Initial guess
result = solver.solve(x0)

print(f"Solution: x = {result['x']}")
print(f"Objective value: {result['fun']}")
print(f"Constraint violation: {result['constraint_violation']}")
```

## API Reference

### AugmentedLagrangian Class

#### Constructor Parameters

- `objective_func`: Function to minimize f(x)
- `constraint_funcs`: Single constraint function or list of constraint functions
- `backend`: Optimization backend - "scipy" (default) or "pytorch"
- `mu_0`: Initial penalty parameter (default: 1.0)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `rho`: Factor to increase penalty parameter (default: 1.5)
- `max_mu`: Maximum penalty parameter value (default: 1000.0)
- `constraint_tolerance`: Tolerance for constraint satisfaction (default: 1e-4)
- `max_outer_iterations`: Maximum outer iterations (default: 20)
- `max_inner_iterations`: Maximum inner iterations per subproblem (default: 50)
- `verbose`: Whether to print optimization progress (default: True)

#### Methods

- `solve(x0, max_outer_iterations=100, tolerance=1e-6)`: Solve the optimization problem
- `set_functions(objective_func, constraint_funcs)`: Set objective and constraint functions

## PyTorch Backend Example

The PyTorch backend uses SGD optimization and supports GPU acceleration:

```python
import numpy as np
from aug_lag import AugmentedLagrangian

# Define objective and constraint functions (same as before)
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def constraint(x):
    return x[0] + x[1] - 3

# Create solver with PyTorch backend
solver = AugmentedLagrangian(
    objective_func=objective,
    constraint_funcs=constraint,
    backend="pytorch",           # Use PyTorch backend
    max_inner_iterations=200,    # More epochs for SGD
    tolerance=1e-6,
    verbose=True
)

# Solve the problem
x0 = np.array([0.0, 0.0])
result = solver.solve(x0)

print(f"Solution: x = {result['x']}")
print(f"Backend used: {solver.backend}")
```

**Backend Comparison:**
- **SciPy backend**: Uses BFGS algorithm, typically faster convergence, CPU-only
- **PyTorch backend**: Uses SGD algorithm, supports GPU acceleration, good for large-scale problems

## Multiple Constraints Example

```python
import numpy as np
from augmented_lagrangian import AugmentedLagrangian

# Objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Multiple constraints
def constraint1(x):
    return x[0] + x[1] - 1

def constraint2(x):
    return x[0] - x[1]

# Create solver with multiple constraints
solver = AugmentedLagrangian(
    objective_func=objective,
    constraint_funcs=[constraint1, constraint2]
)

# Solve
x0 = np.array([0.0, 0.0])
result = solver.solve(x0)
```

## Algorithm Details

The Augmented Lagrangian method combines the objective function with penalty terms for constraint violations:

```
L_A(x, λ, μ) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ c_i(x)²
```

Where:
- `f(x)` is the objective function
- `c_i(x)` are the constraint functions
- `λ_i` are the Lagrange multipliers
- `μ` is the penalty parameter

The algorithm iteratively:
1. Minimizes the augmented Lagrangian with respect to x
2. Updates the Lagrange multipliers: λ := λ - μ * c(x)
3. Increases the penalty parameter if needed
4. Repeats until convergence

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please consider citing:

```bibtex
@software{augmented_lagrangian,
  title={Augmented Lagrangian: A Python Implementation},
  author={Hongwei Jin},
  year={2025},
  url={https://github.com/cshjin/augmented-lagrangian}
}
```
