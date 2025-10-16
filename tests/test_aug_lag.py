"""
Tests for the AugmentedLagrangian class.
"""

import numpy as np
import pytest
from aug_lag import AugmentedLagrangian


class TestAugmentedLagrangian:
    """Test cases for the AugmentedLagrangian optimizer."""

    def test_simple_quadratic_with_linear_constraint(self):
        """Test a simple quadratic objective with linear constraint."""
        # Minimize (x1 - 1)^2 + (x2 - 2)^2
        #       subject to x1 + x2 - 3 = 0
        #
        # Analytical solution: x1 = 1, x2 = 2

        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        def constraint(x):
            return x[0] + x[1] - 3

        solver = AugmentedLagrangian(
            objective_func=objective,
            constraint_funcs=constraint,
            tolerance=1e-6,
            verbose=False
        )

        x0 = np.array([0.0, 0.0])
        result = solver.solve(x0, tolerance=1e-6)

        # Check convergence
        assert result['success'], "Optimization should converge"
        assert result['constraint_violation'] < 1e-5, "Constraint should be satisfied"

        # Check solution accuracy (approximate analytical solution)
        expected_x = np.array([1, 2])
        np.testing.assert_allclose(result['x'], expected_x, atol=1e-3)

    def test_multiple_constraints(self):
        """Test with multiple constraints."""
        # Minimize x1^2 + x2^2
        #       subject to x1 + x2 - 1 = 0 and x1 - x2 = 0
        #
        # Solution should be x1 = x2 = 0.5

        def objective(x):
            return x[0]**2 + x[1]**2

        def constraint1(x):
            return x[0] + x[1] - 1

        def constraint2(x):
            return x[0] - x[1]

        solver = AugmentedLagrangian(
            objective_func=objective,
            constraint_funcs=[constraint1, constraint2],
            verbose=False
        )

        x0 = np.array([0.0, 0.0])
        result = solver.solve(x0, tolerance=1e-6)

        # Check convergence
        assert result['success'], "Optimization should converge"
        assert result['constraint_violation'] < 1e-5, "Constraints should be satisfied"

        # Check solution
        expected_x = np.array([0.5, 0.5])
        np.testing.assert_allclose(result['x'], expected_x, atol=1e-3)

    def test_set_functions(self):
        """Test setting functions after initialization."""
        solver = AugmentedLagrangian()

        def objective(x):
            return x[0]**2

        def constraint(x):
            return x[0] - 1

        solver.set_functions(objective, constraint)

        x0 = np.array([0.0])
        result = solver.solve(x0, tolerance=1e-6)

        assert result['success'], "Optimization should converge"
        np.testing.assert_allclose(result['x'], [1.0], atol=1e-3)

    def test_single_variable_problem(self):
        """Test with a single variable."""
        # Minimize (x - 2)^2
        #       subject to x - 1 = 0
        #
        # Analytical solution: x = 1

        def objective(x):
            return (x[0] - 2)**2

        def constraint(x):
            return x[0] - 1

        solver = AugmentedLagrangian(
            objective_func=objective,
            constraint_funcs=constraint,
            verbose=False
        )

        x0 = np.array([0.0])
        result = solver.solve(x0, tolerance=1e-6)

        assert result['success'], "Optimization should converge"
        np.testing.assert_allclose(result['x'], [1.0], atol=1e-3)

    def test_parameter_validation(self):
        """Test that parameters are properly validated."""
        solver = AugmentedLagrangian(
            mu_0=2.0,
            tolerance=1e-8,
            rho=2.0,
            max_mu=500.0
        )

        assert solver.mu_0 == 2.0
        assert solver.tolerance == 1e-8
        assert solver.rho == 2.0
        assert solver.max_mu == 500.0

    def test_constraint_history_tracking(self):
        """Test that optimization history is properly tracked."""
        def objective(x):
            return x[0]**2 + x[1]**2

        def constraint(x):
            return x[0] + x[1] - 1

        solver = AugmentedLagrangian(
            objective_func=objective,
            constraint_funcs=constraint,
            verbose=False
        )

        x0 = np.array([0.0, 0.0])
        solver.solve(x0, max_outer_iterations=5)

        # Check that histories are populated
        assert len(solver.constraint_history) > 0
        assert len(solver.lambda_history) > 0
        assert len(solver.mu_history) > 0
        assert len(solver.objective_history) > 0

        # Check that histories have consistent lengths
        n_iterations = len(solver.constraint_history)
        assert len(solver.lambda_history) == n_iterations
        assert len(solver.mu_history) == n_iterations
        assert len(solver.objective_history) == n_iterations


if __name__ == "__main__":
    pytest.main([__file__])
