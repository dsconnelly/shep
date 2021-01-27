import warnings

import numpy as np

import kernels
import utils

def predict(X_train, y_train, X_test, kernel, theta=None):
    """Computes a Gaussian process prediction of function values at X_test.

    Parameters
    ----------
    X_train : {(m,), (m, d)}
        Array of training inputs stored as row vectors.
    y_train : (m,)
        Training labels.
    X_test : {(n,), (n, d)}
        Array of test inputs stored as row vectors.
    kernel : callable
        Kernel function. Should take two arrays of row vectors and an optional
        array of hyperparameters as arguments.
    theta : (p,) array_like, optional
        Array of hyperparameters. theta[:-1] will be passed to kernel and should
        be appropriate for use therein. theta[-1] will be interpreted as the
        standard deviation of normally distributed noise on the training labels,
        and should be set to zero if the training data is noise-free.

        If unspecified, the kernel hyperparameters are the kernel defaults and
        the training label noise is zero.

    Returns
    -------
    y_test : (n,) array_like
        Means of the posterior over function values at X_test.
    S_test : (n, n) array like
        Covariance matrix of the posterior over function values at X_test.

    Warns
    -----
    RuntimeWarning
        If there are negative values along the diagonal of the posterior
        covariance matrix S_test. This is usually numerical. The warning message
        shows the minimum negative diagonal value, and the warning can be safely
        ignored if the absolute value is small.

    Notes
    -----
    If negative values are found along the diagonal of S_test, they are set to
    zero and the warning message described above is displayed.

    """

    X_train = utils.as_row_vectors(X_train)
    X_test = utils.as_row_vectors(X_test)

    if theta is None:
        hpar, training_noise = None, 0
    else:
        hpar = theta[:-1]
        training_noise = (theta[-1] ** 2) * np.eye(y_train.shape[0])

    K = kernel(X_train, X_train, hpar) + training_noise
    K_s = kernel(X_train, X_test, hpar)
    K_ss = kernel(X_test, X_test, hpar)
    K_i = np.linalg.inv(K)

    y_test = K_s.T @ K_i @ y_train
    S_test = K_ss - K_s.T @ K_i @ K_s

    if np.diag(S_test).min() < 0:
        message = ' '.join((
            f'Minimum posterior variance of {np.diag(S_test).min()}.',
            'Setting negative diagonal entries of S_test to zero.'
        ))
        warnings.warn(message, RuntimeWarning)

        np.fill_diagonal(S_test, np.maximum(np.diag(S_test), 0))

    return y_test, S_test
