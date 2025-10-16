"""
Example 1: Simple optimization with a single constraint.

This example demonstrates how to use the AugmentedLagrangian optimizer
to solve a constrained optimization problem with a single equality constraint.

Problem:
    minimize   (x1 - 1)^2 + (x2 - 2)^2
    subject to x1 + x2 - 3 = 0

Analytical solution: x1 = 1, x2 = 2, with optimal objective value f(x)=0.
"""

import numpy as np

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


def main():
    print("=" * 60)
    print("Augmented Lagrangian Example 1: Single Constraint")
    print("=" * 60)

    print("\nProblem:")
    print("  minimize   (x1 - 1)² + (x2 - 2)²")
    print("  subject to x1 + x2 - 3 = 0")

    print("\nAnalytical solution: x1 = 1, x2 = 2")
    print("Expected objective value: 0")

    # Create the solver
    solver = AugmentedLagrangian(
        objective_func=objective_function,
        constraint_funcs=constraint_function,
        mu_0=1.0,                      # Initial penalty parameter
        tolerance=1e-8,                # Convergence tolerance
        rho=2.0,                       # Penalty parameter increase factor
        max_outer_iterations=50,       # Maximum outer iterations
        max_inner_iterations=2,        # Maximum inner iterations
        backend="scipy",               # Optimization backend
        verbose=True                   # Print progress
    )

    # Initial guess
    x0 = np.array([0.0, 0.0])
    x0 = np.random.rand(2)
    print(f"\nInitial guess: x0 = {x0}")

    # Solve the problem
    print("\n" + "-" * 40)
    print("Starting optimization...")
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
    analytical_solution = np.array([1, 2])
    analytical_objective = objective_function(analytical_solution)

    print(f"\nComparison with analytical solution:")
    print(f"  Analytical: x = [{analytical_solution[0]}, {analytical_solution[1]}]")
    print(f"  Numerical:  x = [{result['x'][0]:.6f}, {result['x'][1]:.6f}]")
    print(f"  Error: ||x_num - x_analytical|| = {np.linalg.norm(result['x'] - analytical_solution):.2e}")
    print(f"  Analytical objective: {analytical_objective}")
    print(f"  Numerical objective:  {result['objective']:.8f}")


if __name__ == "__main__":
    main()
