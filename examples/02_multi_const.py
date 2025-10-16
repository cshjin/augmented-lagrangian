"""
Example 2: Multi-constraint optimization problem.

This example demonstrates how to solve an optimization problem with
multiple equality constraints using the Augmented Lagrangian method.

Problem:
    minimize   x1^2 + x2^2
    subject to x1 + x2 - 1 = 0
               x1 - x2 = 0

Analytical solution: x1 = x2 = 0.5
"""

import numpy as np

from aug_lag import AugmentedLagrangian

np.set_printoptions(precision=3, suppress=True)


def objective_function(x):
    """
    Objective function: minimize x1^2 + x2^2

    This is a quadratic function with minimum at (0, 0) if unconstrained.
    """
    return x[0]**2 + x[1]**2


def constraint1(x):
    """First constraint: x1 + x2 - 1 = 0"""
    return x[0] + x[1] - 1


def constraint2(x):
    """Second constraint: x1 - x2 = 0"""
    return x[0] - x[1]


def main():
    print("=" * 60)
    print("Augmented Lagrangian Example 2: Multiple Constraints")
    print("=" * 60)

    print("\nProblem:")
    print("  minimize   x1² + x2²")
    print("  subject to x1 + x2 - 1 = 0")
    print("             x1 - x2 = 0")

    print("\nAnalytical solution: x1 = x2 = 0.5")
    print("Expected objective value: 0.5")

    # Create the solver with multiple constraints
    solver = AugmentedLagrangian(
        objective_func=objective_function,
        constraint_funcs=[constraint1, constraint2],  # List of constraint functions
        mu_0=1.0,
        tolerance=1e-6,
        rho=1.5,
        max_outer_iterations=50,
        verbose=True
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
    print(f"Final Lagrange multipliers: λ = {result['lambda']}")
    print(f"Final penalty parameter: μ = {result['mu']:.2f}")

    # Verify constraint satisfaction
    c1_value = constraint1(result['x'])
    c2_value = constraint2(result['x'])

    print("\nConstraint verification:")
    print(f"  Constraint 1: x1 + x2 - 1 = {result['x'][0]:.6f} + {result['x'][1]:.6f} - 1 = {c1_value:.8f}")
    print(f"  Constraint 2: x1 - x2 = {result['x'][0]:.6f} - {result['x'][1]:.6f} = {c2_value:.8f}")

    # Compare with analytical solution
    analytical_solution = np.array([0.5, 0.5])
    analytical_objective = objective_function(analytical_solution)

    print("\nComparison with analytical solution:")
    print(f"  Analytical: x = [{analytical_solution[0]}, {analytical_solution[1]}]")
    print(f"  Numerical:  x = [{result['x'][0]:.6f}, {result['x'][1]:.6f}]")
    print(f"  Error: ||x_num - x_analytical|| = {np.linalg.norm(result['x'] - analytical_solution):.2e}")
    print(f"  Analytical objective: {analytical_objective}")
    print(f"  Numerical objective:  {result['objective']:.8f}")

    # Demonstrate optimization history
    if hasattr(solver, 'constraint_history') and len(solver.constraint_history) > 1:
        print("\nOptimization progress:")
        print("  Iter  |  Objective  |  Constraint Violation  |  Penalty μ")
        print("  ------|-------------|------------------------|------------")
        for i, (obj, constr, mu) in enumerate(zip(
            solver.objective_history[:5],
            solver.constraint_history[:5],
            solver.mu_history[:5]
        )):
            violation = np.linalg.norm(constr)
            print(f"  {i:4d}  |  {obj:9.6f}  |  {violation:18.2e}  |  {mu:8.2f}")
        if len(solver.objective_history) > 5:
            print("  ...   |      ...    |         ...            |     ...")


if __name__ == "__main__":
    main()
