import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=3)


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
                 with_noise: bool = False,
                 tolerance: float = 1e-6,
                 rho: float = 1.5,
                 max_mu: float = 1000.0,
                 constraint_tolerance: float = 1e-4,
                 max_outer_iterations: int = 20,
                 max_inner_iterations: int = 50,
                 backend: str = "scipy",
                 verbose: bool = True):
        """
        Initialize Augmented Lagrangian solver.

        Args:
            objective_func: Function f(x) to minimize
            constraint_funcs: Single constraint function or list of constraint functions.
                             Each function should return constraint values that should equal 0
            mu_0: Initial penalty parameter
            with_noise: Whether to add noise to escape stationary points
            tolerance: Convergence tolerance
            rho: Factor to increase penalty parameter
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
            self.constraint_funcs = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_funcs = list(constraint_funcs)
        else:
            self.constraint_funcs = constraint_funcs

        # Algorithm parameters
        self.mu_0 = np.random.rand() if mu_0 is None else mu_0
        self.with_noise = with_noise
        self.tolerance = tolerance
        self.rho = rho
        self.max_mu = max_mu
        self.constraint_tolerance = constraint_tolerance
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.backend = backend
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
            self.constraint_funcs = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_funcs = list(constraint_funcs)
        else:
            self.constraint_funcs = constraint_funcs

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
        if self.constraint_funcs is None:
            raise NotImplementedError("Constraint function not set. Use set_functions() or override this method.")

        # Handle list of constraint functions (always the case after initialization)
        if isinstance(self.constraint_funcs, (list, tuple)):
            all_constraints = []

            for i, constraint_fn in enumerate(self.constraint_funcs):
                if callable(constraint_fn):
                    constraint_val = constraint_fn(x)
                    # Handle case where constraint function returns multiple constraints
                    if isinstance(constraint_val, (list, tuple, np.ndarray)):
                        all_constraints.extend(constraint_val)
                    else:
                        all_constraints.append(constraint_val)

            return np.array(all_constraints)

        # Fallback for single function stored as non-callable (should not occur)
        return np.array(self.constraint_funcs(x))

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
        self.mu_k = min(self.mu_k * self.rho, self.max_mu)

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
        const_k = self.c(x)
        obj_k = self.f(x)

        # Handle both scalar and vector constraints
        if np.isscalar(const_k):
            n_constraints = 1
            const_k = np.array([const_k])
        else:
            const_k = np.array(const_k)
            n_constraints = len(const_k)

        # Store initial values in history
        self.constraint_history = [const_k]
        self.lambda_history = [self.lambda_k]
        self.mu_history = [self.mu_k]
        self.objective_history = [obj_k]

        # Initialize Lagrange multipliers
        if self.lambda_k is None:
            self.lambda_k = np.zeros(n_constraints)
        elif len(self.lambda_k) != n_constraints:
            warnings.warn("Dimension of initial lambda does not match number of constraints. Reinitializing to zeros.")
            self.lambda_k = np.zeros(n_constraints)
        self.lambda_history = [self.lambda_k]

        converged = False

        for iter in range(max_outer_iterations):
            # Minimize augmented Lagrangian using scipy
            result = minimize(
                self.augmented_lagrangian,
                x,
                method='BFGS',
                options={'maxiter': self.max_inner_iterations,
                         'disp': False}
            )

            x = result.x

            if self.with_noise:
                # Add small random noise to escape stationary points
                noise = np.random.normal(scale=self.tolerance, size=x.shape)
                x += noise

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
                x_str = np.array2string(x, precision=3, separator=', ', suppress_small=True)
                print(f"Iter {iter:<4} |"
                      f"x_k: {x_str:<15} |"
                      f"x_norm: {np.linalg.norm(x):>10.3e} |"
                      f"obj: {obj_val:>10.3e} |"
                      f"const_vio_norm: {constraint_violation:>10.3e} |")

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

        return {'x': x,
                'fun': self.f(x),
                'success': converged,
                'objective': self.f(x),
                'constraint_violation': constraint_violation,
                'converged': converged,
                'nit': iter + 1,
                'iterations': iter + 1,
                'lambda': self.lambda_k,
                'mu': self.mu_k
                }
