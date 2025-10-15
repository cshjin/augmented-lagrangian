import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class AugmentedLagrangian(object):
    """
    General Augmented Lagrangian class for constrained optimization.

    Solves problems of the form:
        min f(x)
        s.t. c_i(x) = 0  for i = 1, ..., m (equality constraints)

    The augmented Lagrangian function is:
        L_A(x, λ, μ) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ c_i(x)²
    """

    def __init__(self,
                 objective_func=None,
                 constraint_funcs=None,
                 mu_0: float = 1.0,
                 tolerance: float = 1e-6,
                 mu_increase_factor: float = 1.5,
                 max_mu: float = 1000.0,
                 constraint_tolerance: float = 1e-4,
                 max_outer_iterations: int = 20,
                 max_inner_iterations: int = 50,
                 verbose: bool = True):
        """
        Initialize Augmented Lagrangian solver.

        Args:
            objective_func: Function f(x) to minimize
            constraint_funcs: Single constraint function or list of constraint functions.
                             Each function should return constraint values that should equal 0
            mu_0: Initial penalty parameter
            tolerance: Convergence tolerance
            mu_increase_factor: Factor to increase penalty parameter
            max_mu: Maximum penalty parameter value
            constraint_tolerance: Tolerance for constraint satisfaction
            max_outer_iterations: Maximum outer iterations
            max_inner_iterations: Maximum inner iterations per subproblem
            verbose: Whether to print optimization progress
        """
        super().__init__()

        # User-provided functions
        self.objective_func = objective_func

        # Handle constraint functions - support both single function and list
        if callable(constraint_funcs):
            self.constraint_func = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_func = list(constraint_funcs)
        else:
            self.constraint_func = constraint_funcs

        # Algorithm parameters
        self.mu_0 = mu_0
        self.tolerance = tolerance
        self.mu_increase_factor = mu_increase_factor
        self.max_mu = max_mu
        self.constraint_tolerance = constraint_tolerance
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.verbose = verbose

        # Current algorithm state
        self.mu_k = mu_0
        self.lambda_k = None  # Lagrange multipliers
        self.outer_iteration = 0

        # Constraint tracking
        self.constraint_history = []
        self.lambda_history = []
        self.mu_history = []
        self.objective_history = []

    def set_functions(self, objective_func,
                      constraint_funcs):
        """Set the objective and constraint functions.

        Args:
            objective_func: Single objective function f(x)
            constraint_funcs: Single constraint function or list of constraint functions
        """
        self.objective_func = objective_func

        # Handle both single function and list of functions
        if callable(constraint_funcs):
            self.constraint_func = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_func = list(constraint_funcs)
        else:
            self.constraint_func = constraint_funcs

    def f(self, x: np.ndarray) -> float:
        """Objective function f(x).

        Args:
            x: Current variable values

        Returns:
            (float): Objective function value at x
        """
        if self.objective_func is None:
            raise NotImplementedError("Objective function not set. Use set_functions() or override this method.")
        return self.objective_func(x)

    def c(self, x: np.ndarray) -> np.ndarray:
        """
        Constraint function c(x). Evaluates all constraint functions and returns combined array.

        Args:
            x: Current variable values

        Returns:
            Combined constraint values from all constraint functions
        """
        if self.constraint_func is None:
            raise NotImplementedError("Constraint function not set. Use set_functions() or override this method.")

        # Handle list of constraint functions (always the case after initialization)
        if isinstance(self.constraint_func, (list, tuple)):
            all_constraints = []

            for i, constraint_fn in enumerate(self.constraint_func):
                constraint_val = constraint_fn(x)

                # # Convert to array and flatten if needed
                # if np.isscalar(constraint_val):
                #     constraint_array = np.array([constraint_val])
                # else:
                #     constraint_array = np.array(constraint_val).flatten()

                all_constraints.append(constraint_val)

            return np.array(all_constraints)

        # Fallback for single function stored as non-callable (should not occur)
        return np.array(self.constraint_func(x))

    def augmented_lagrangian(self, x: np.ndarray) -> float:
        """Compute the augmented Lagrangian function.

        Args:
            x: Current variable values

        Returns:
            Augmented Lagrangian value at x
        """
        obj = self.f(x)
        constraints = self.c(x)

        # Handle both scalar and vector constraints
        if np.isscalar(constraints):
            constraints = np.array([constraints])
        else:
            constraints = np.array(constraints)

        # Augmented Lagrangian: L(x,λ,μ) = f(x) - λᵀc(x) + (μ/2)||c(x)||²
        lagrangian_term = -np.dot(self.lambda_k, constraints)
        penalty_term = 0.5 * self.mu_k * np.sum(constraints**2)

        return obj + lagrangian_term + penalty_term

    def update_multipliers(self, x: np.ndarray):
        """Update Lagrange multipliers: λ = λ - μ * c(x).

        Args:
            x: Current variable values
        """
        constraints = self.c(x)
        if hasattr(constraints, 'shape'):
            constraint_array = constraints
        else:
            constraint_array = np.array([constraints]) if np.isscalar(constraints) else np.array(constraints)

        self.lambda_k = self.lambda_k - self.mu_k * constraint_array

    def update_penalty_parameter(self):
        """Increase penalty parameter μ.

        .. math::
            μ_k = min(μ * ρ, μ_{max}),
            where ρ > 1 is the increase factor, and μ_{max} is the maximum allowed value.

        """
        self.mu_k = min(self.mu_k * self.mu_increase_factor, self.max_mu)

    def solve(self, x0: np.ndarray, max_outer_iterations: int = 100, tolerance: float = 1e-6) -> dict:
        """
        Solve the constrained optimization problem using augmented Lagrangian method.

        Args:
            x0: Initial guess for variables
            max_outer_iterations: Maximum number of outer iterations
            tolerance: Convergence tolerance for constraint violation

        Returns:
            Dictionary with solution results
        """
        x = np.array(x0, dtype=float)

        # Initialize constraints to determine dimensions
        constraints_init = self.c(x)
        obj_init = self.f(x)

        self.constraint_history.append(constraints_init)
        self.lambda_history.append(self.lambda_k)
        self.mu_history.append(self.mu_k)
        self.objective_history.append(obj_init)

        # Handle both scalar and vector constraints
        if np.isscalar(constraints_init):
            n_constraints = 1
            constraints_init = np.array([constraints_init])
        else:
            constraints_init = np.array(constraints_init)
            n_constraints = len(constraints_init)

        # Initialize Lagrange multipliers
        if self.lambda_k is None:
            self.lambda_k = np.zeros(n_constraints)

        converged = False

        for iteration in range(max_outer_iterations):
            # Minimize augmented Lagrangian using scipy
            result = minimize(
                self.augmented_lagrangian,
                x,
                method='L-BFGS-B',
                options={'maxiter': self.max_inner_iterations,
                         'disp': False}
            )

            x = result.x

            # Check convergence
            constraints = self.c(x)
            # Handle both scalar and vector constraints
            if np.isscalar(constraints):
                constraints = np.array([constraints])
            else:
                constraints = np.array(constraints)

            constraint_violation = np.linalg.norm(constraints)

            if self.verbose:
                obj_val = self.f(x)
                print(f"Iteration {iteration}: obj={obj_val:.6f}, constraint_violation={constraint_violation:.6f}")

            if constraint_violation < tolerance:
                converged = True
                break

            # # Update Lagrange multipliers: λ := λ - μ * c(x)
            self.lambda_k -= self.mu_k * constraints
            # self.update_multipliers(x)

            # Update penalty parameter
            self.update_penalty_parameter()

            self.constraint_history.append(constraints)
            self.lambda_history.append(self.lambda_k)
            self.mu_history.append(self.mu_k)
            self.objective_history.append(self.f(x))

        return {
            'x': x,
            'fun': self.f(x),
            'success': converged,
            'objective': self.f(x),
            'constraint_violation': constraint_violation,
            'converged': converged,
            'nit': iteration + 1,
            'iterations': iteration + 1,
            'lambda': self.lambda_k,
            'mu': self.mu_k
        }
