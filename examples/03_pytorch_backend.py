"""
Example 3: PyTorch backend optimization.

This example demonstrates how to use the AugmentedLagrangian optimizer
with PyTorch backend for constrained optimization problems.

Problem:
    minimize   (x1 - 1)^2 + (x2 - 2)^2
    subject to x1 + x2 - 3 = 0

This is the same problem as Example 1, but solved using PyTorch backend
with SGD optimizer instead of SciPy's BFGS.

Analytical solution: x1 = 1.5, x2 = 1.5
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


def objective_function(x):
    """
    Objective function: minimize (x1 - 1)^2 + (x2 - 2)^2

    This is a quadratic function with minimum at (1, 2) if unconstrained.
    """
    return (x[0] - 1)**2 + (x[1] - 2)**2


def constraint_function(x):
    """
    Constraint function: x1 + x2 - 3 = 0

    This constraint forces the sum of variables to equal 3.
    """
    return x[0] + x[1] - 3


def neural_network_example():
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


def main():
    if not TORCH_AVAILABLE:
        print("This example requires PyTorch. Please install it with:")
        print("pip install torch")
        return

    print("=" * 60)
    print("Augmented Lagrangian Example 3: PyTorch Backend")
    print("=" * 60)

    print("\nProblem:")
    print("  minimize   (x1 - 1)² + (x2 - 2)²")
    print("  subject to x1 + x2 - 3 = 0")

    print("\nAnalytical solution: x1 = 1.5, x2 = 1.5")
    print("Expected objective value: 0.5")

    # Create the solver with PyTorch backend
    solver = AugmentedLagrangian(
        objective_func=objective_function,
        constraint_funcs=constraint_function,
        backend="pytorch",           # Use PyTorch backend
        mu_0=0.1,                   # Start with smaller penalty parameter
        tolerance=1e-5,
        rho=1.2,                    # Conservative penalty increase
        max_outer_iterations=50,
        max_inner_iterations=100,   # Moderate epochs for PyTorch SGD
        verbose=True
    )

    # Initial guess
    x0 = np.array([0.0, 0.0])
    print(f"\nInitial guess: x0 = {x0}")
    print(f"Backend: {solver.backend}")

    # Solve the problem
    print("\n" + "-" * 40)
    print("Starting PyTorch optimization with SGD...")
    print("-" * 40)

    result = solver.solve(x0, tolerance=1e-6)

    # Display results
    print("\n" + "=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)

    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Solution: x = [{result['x'][0]:.6f}, {result['x'][1]:.6f}]")
    print(f"Objective value: {result['objective']:.8f}")
    print(f"Constraint violation: {result['constraint_violation']:.2e}")
    print(f"Final Lagrange multiplier: λ = {result['lambda']}")
    print(f"Final penalty parameter: μ = {result['mu']:.2f}")

    # Verify constraint satisfaction
    constraint_value = constraint_function(result['x'])
    print(f"\nConstraint check:")
    print(f"  x1 + x2 - 3 = {result['x'][0]:.6f} + {result['x'][1]:.6f} - 3 = {constraint_value:.8f}")

    # Compare with analytical solution
    analytical_solution = np.array([1.5, 1.5])
    analytical_objective = objective_function(analytical_solution)

    print(f"\nComparison with analytical solution:")
    print(f"  Analytical: x = [{analytical_solution[0]}, {analytical_solution[1]}]")
    print(f"  PyTorch:    x = [{result['x'][0]:.6f}, {result['x'][1]:.6f}]")
    print(f"  Error: ||x_pytorch - x_analytical|| = {np.linalg.norm(result['x'] - analytical_solution):.2e}")
    print(f"  Analytical objective: {analytical_objective}")
    print(f"  PyTorch objective:    {result['objective']:.8f}")

    # Compare with SciPy backend
    print("\n" + "=" * 40)
    print("COMPARISON WITH SCIPY BACKEND")
    print("=" * 40)

    solver_scipy = AugmentedLagrangian(
        objective_func=objective_function,
        constraint_funcs=constraint_function,
        backend="scipy",
        mu_0=1.0,
        tolerance=1e-6,
        rho=2.0,
        max_outer_iterations=50,
        verbose=False
    )

    result_scipy = solver_scipy.solve(x0, tolerance=1e-6)

    print(f"SciPy result:   x = [{result_scipy['x'][0]:.6f}, {result_scipy['x'][1]:.6f}]")
    print(f"PyTorch result: x = [{result['x'][0]:.6f}, {result['x'][1]:.6f}]")
    print(f"Difference: ||x_pytorch - x_scipy|| = {np.linalg.norm(result['x'] - result_scipy['x']):.2e}")
    print(f"SciPy iterations: {result_scipy['iterations']}")
    print(f"PyTorch iterations: {result['iterations']}")

    # Run neural network example
    neural_network_example()


if __name__ == "__main__":
    main()
