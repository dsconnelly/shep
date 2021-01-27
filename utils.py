import numpy as np

def as_row_vectors(X):
    """Reshapes a potentially one-dimensional array into two dimensions.

    Parameters
    ----------
    X : {(m,), (m, d)} array_like
        Array of row vectors or scalars.

    Returns
    -------
    X : {(m, d)} array_like
        (Potentially reshaped) array of row vectors.

    """

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X

def squared_distances(X, Z, W=None):
    """Calculates pairwise squared distances between sets of row vectors, with
    optional individual weighting of coordinates.

    Parameters
    ----------
    X : {(m,), (m, d)} array_like
        First set of row vectors.
    Z : {(n,), (n, d)} array_like
        Second set of row vectors.
    W : (d,) array_like, optional
        Weights for each coordinate. If unspecified, all weights are unity, and
        the distance calculated is Euclidean.

    Returns
    -------
    D_sq : (m, n) array_like
        Matrix where D[i, j] = (X[i] - Z[j]).T @ W @ (X[i] - Z[j]) is the square
        of the distance between vectors X[i] and Z[j], with coordinates weighted
        by the entries of W.

    Raises
    ------
    ValueError
        If the row vectors in X and Z do not have the same dimension.

    Notes
    -----
    If X and Z are one-dimensional, they are treated as sets of one-dimensional
    vectors and recast as such.

    """

    X, Z = as_row_vectors(X), as_row_vectors(Z)

    m, d1 = X.shape
    n, d2 = Z.shape

    if d1 != d2:
        raise ValueError("Incompatible dimensions")

    if W is None:
        W = np.eye(d1)
    else:
        W = np.diag(W)

    A = np.broadcast_to(np.diag(X @ W @ X.T)[:, np.newaxis], (m, n))
    B = X @ W @ Z.T
    C = np.broadcast_to(np.diag(Z @ W @ Z.T)[np.newaxis, :], (m, n))

    return A - (2 * B) + C
