""" Augmented Lagrangian method using scipy.optimize and numpy

Problem:
min   f(x) = x1 + x2
s.t.  e(x) = x1^2 + x2^2 - 2 = 0

- init x = [1, 1] will converge to local stationary point [1, 1]
  - either pick a better initial point, or
  - add small noise to x after each iteration to escape local stationary points
- lambda: Lagrange multiplier, update using dual ascent
- mu (> 0): penalty parameter, typically increased over iterations
"""
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=3)
np.random.seed(42)


def f(x):
    return x[0] + x[1]


def e(x):
    return x[0]**2 + x[1]**2 - 2


def augmented_lagrangian(x, mu, lam):
    return f(x) - lam * e(x) + 0.5 * mu * e(x)**2


def quadratic_penalty(x, mu):
    return f(x) + 0.5 * mu * e(x)**2


def update_lambda(lam, mu, x):
    return lam - mu * e(x)


def main(x0, mu, lam,
         steps=20,
         with_lam_update=True,
         with_noise=False,
         with_penalty_update=False,
         rho=1.2,
         verbose=False):
    """ Augmented Lagrangian Optimization

    Args:
        x0 (np.ndarray): Initial guess for the variables.
        mu (float): Penalty parameter.
        lam (float): Lagrange multiplier.
        steps (int, optional): Number of iterations. Defaults to 20.
        with_lam_update (bool, optional): Whether to update lambda. Defaults to True.
        with_noise (bool, optional): Whether to add noise to escape local minima. Defaults to False.
        with_penalty_update (bool, optional): Whether to update the penalty parameter. Defaults to False.
        rho (float, optional): Penalty parameter update multiplier. Defaults to 1.2.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """

    x = x0
    for k in range(steps):
        obj = f(x)
        con = e(x)
        x_str = f"[{x[0]:.3f} {x[1]:.3f}]"
        alm = augmented_lagrangian(x, mu, lam)
        if verbose:
            print(f"iter={k:<2} "
                  f"| x={x_str:<15} "
                  f"| x_norm={np.linalg.norm(x):>6.3f} "
                  f"| f(x)={obj:>6.3f} "
                  f"| c(x)={con:>6.3f} "
                  f"| ALM={alm:>6.3f} "
                  f"| λ={lam:>6.3f} "
                  f"| μ={mu:>6.3f}")

        # Minimize the augmented Lagrangian with respect to x
        # NOTE: with solution not close to the optimum, the optimization may get stuck in local minima or even diverge
        result = minimize(lambda x_var: augmented_lagrangian(x_var, mu, lam), x, method='L-BFGS-B')
        x = result.x

        # Update lambda using augmented Lagrangian method
        if with_lam_update:
            lam = update_lambda(lam, mu, x)

        # NOTE: add a small noise to avoid getting stuck in local saddle points
        if with_noise:
            x += np.random.randn(*x.shape) * 1e-5

        # Update mu (penalty parameter) - typically increased
        if with_penalty_update:
            mu = mu * rho

    print("=" * 30)
    print("Final solution:", x)
    print(f"f(x) = {f(x):.6f}")
    print(f"c(x) = {e(x):.6f}")
    print(f"lambda = {lam:.6f}")
    print(f"mu = {mu:.6f}")


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])
    initial_points = [
        [1.0, 1.0],  # with_noise=False, mu~>=0.84 will get stuck at [1, 1]
        [-1.0, -1.0],
        [0.5, -0.5],
        [2.0, 0.0],
        np.random.randn(2)
    ]
    # NOTE: mu ranged from (0, 1], and with approximate > 0.84, it will not converge
    mu = np.random.rand()
    # NOTE: different initial mu may lead to different local solutions
    lam = 0.0
    main(x0,
         mu,
         lam,
         steps=10,
         with_lam_update=True,
         with_noise=False,
         with_penalty_update=True,
         rho=1.02,
         verbose=True)
