"""
Example 4: Pytorch Neural Network Parameter Optimization using augmented Lagrangian

This example demonstrates how to use the AugmentedLagrangian optimizer
with PyTorch backend for constrained optimization problems.

Problem:
    minimize   sum(w_i^2)     (L2 regularization)
    subject to sum(w_i) = 1   (weight normalization)

This is the same problem as Example 1, but solved using PyTorch backend
with SGD optimizer instead of SciPy's BFGS.

Analytical solution: all weights equal to 1/n
"""

import numpy as np
import sys
import os


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not available. Please install it with: pip install torch")
    sys.exit(1)

from aug_lag import AugmentedLagrangian


def main():
    """
    Example using PyTorch for a simple neural network parameter optimization.

    This demonstrates how the PyTorch backend can be useful for optimizing
    neural network parameters subject to constraints.
    """
    print("\n" + "=" * 60)
    print("Neural Network Parameter Example (PyTorch Backend)")
    print("=" * 60)

    def nn_objective(x):
        """Objective: minimize weights squared (regularization)"""
        return np.sum(x**2)

    def nn_constraint(x):
        """Constraint: sum of weights should equal 1 (normalization)"""
        return np.sum(x) - 1.0

    print("\nProblem:")
    print("  minimize   sum(w_i^2)     (L2 regularization)")
    print("  subject to sum(w_i) = 1   (weight normalization)")
    print("\nThis simulates constraining neural network weights.")

    # Create solver with PyTorch backend
    solver = AugmentedLagrangian(
        objective_func=nn_objective,
        constraint_funcs=nn_constraint,
        backend="pytorch",
        mu_0=0.1,                    # Start with smaller penalty parameter
        tolerance=1e-5,
        rho=1.2,                     # Conservative penalty increase
        max_outer_iterations=50,
        max_inner_iterations=100,     # Fewer epochs to avoid instability
        verbose=True
    )

    # Initial weights (could represent neural network layer weights)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 weights
    print(f"\nInitial weights: {x0}")

    # Solve
    print("\n" + "-" * 40)
    print("Starting PyTorch optimization...")
    print("-" * 40)

    result = solver.solve(x0, tolerance=1e-6)

    # Display results
    print("\n" + "=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)

    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final weights: {result['x']}")
    print(f"Objective value: {result['objective']:.8f}")
    print(f"Constraint violation: {result['constraint_violation']:.2e}")

    # Verify constraint
    weight_sum = np.sum(result['x'])
    print(f"\nConstraint check:")
    print(f"  Sum of weights = {weight_sum:.8f} (should be 1.0)")
    print(f"  Constraint violation = {abs(weight_sum - 1.0):.2e}")

    # Analytical solution: all weights equal to 1/n
    n_weights = len(x0)
    analytical_solution = np.ones(n_weights) / n_weights
    print(f"\nAnalytical solution: {analytical_solution}")
    print(f"Error from analytical: {np.linalg.norm(result['x'] - analytical_solution):.2e}")


if __name__ == "__main__":
    main()
