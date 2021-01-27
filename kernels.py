import itertools

import numpy as np
from scipy.optimize import minimize

import utils

def rbf(X, Z, theta=None):
    """Calculates the RBF kernel matrix for sets of row vectors.

    Parameters
    ----------
    X : {(m,), (m, d)} array_like
        First set of row vectors.
    Z : {(n,), (n, d)} array like
        Second set of row vectors.
    theta : (d + 1,) array like, optional
        Kernel hyperparameters. theta[0] is the square root of the prefactor,
        while theta[1:] are the length scales for each coordinate -- that is,
        the argument of the exponential is
            -0.5 * (X[i] - Z[j]).T @ 1 / np.diag(theta[1:] ** 2) @ (X[i] - Z[j])
        If unspecified, all hyperparameters are set to one.

    Returns
    -------
    K : (m, n) array_like
        Kernel covariance matrix such that K[i, j] = k(X[i], Z[j]), where k
        is the RBF kernel function.

    """

    X, Z = utils.as_row_vectors(X), utils.as_row_vectors(Z)

    _, d = X.shape
    if theta is None:
        theta = np.ones(d + 1)

    W = 1 / (theta[1:] ** 2)

    return (theta[0] ** 2) * np.exp(-0.5 * utils.squared_distances(X, Z, W))

def select_rbf_hyperparameters(X_train, y_train):
    """Selects hyperparameters for the RBF kernel to maximize the log likelihood
    of the training data.

    Parameters
    ----------
    X_train : {(m,), (m, d)} array_like
        Array of training inputs stored as row vectors.
    y_train : (m,) array_like
        Array of training labels.

    Returns
    -------
    theta : {(2,), (d + 2,)} array_like
        Array such that theta[:-1] are the prefactor and length scale
        hyperparemeters to be bassed to rbf and theta[-1] is the
        standard deviation of the training label noise.

    Raises
    ------
    RuntimeError
        If convergence of hyperparameters is not achieved.

    """

    X = utils.as_row_vectors(X_train)
    y = y_train.reshape(-1, 1)
    m, d = X.shape

    D_sqs = [utils.squared_distances(X, X, np.eye(d)[i]) for i in range(d)]

    def nll_and_grad(theta):
        """Returns the negative log likelihood and its gradient at theta.

        Parameters
        ----------
        theta : {(2,), (d + 2,)} array_like
            Array as described in the docstring of the parent function.

        Returns
        -------
        nll : float
            Negative log likelihood of the training data at theta.
        grad : {(2,), (d + 2,)} array_like
            Gradient of the negative log likelihood at theta.

        """

        theta = theta.flatten()

        K_a = rbf(X, X, theta[:-1])
        K = K_a + (theta[-1] ** 2) * np.eye(m)
        _, logdet = np.linalg.slogdet(K)
        K_i = np.linalg.inv(K)

        nll = 0.5 * (y.T @ K_i @ y + logdet + m * np.log(2 * np.pi)).item()

        dKs = np.zeros((d + 2, m, m))
        dKs[0] = (2 * K_a) / theta[0]
        dKs[-1] = 2 * theta[-1] * np.eye(m)

        for i in range(d):
            dKs[i + 1] = (K_a * D_sqs[i]) / (theta[i + 1] ** 3)

        grad = np.zeros(theta.shape)
        for i, dK in enumerate(dKs):
            grad[i] = -0.5 * np.trace(((K_i @ y @ y.T @ K_i.T) - K_i) @ dK)

        return nll, grad

    grid = [10 ** k for k in range(-2, 3)]
    best_score, best_theta, success = np.inf, None, False

    for guess in itertools.product(*(grid for _ in range(d + 2))):
        res = minimize(nll_and_grad, x0=np.array(guess), jac=True)
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_theta = abs(res.x)
            success = True

    if not success:
        raise RuntimeError('Hyperparameters did not converge.')

    return best_theta
