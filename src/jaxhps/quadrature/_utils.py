"""Utility functions. I would like to remove these utilities."""

import jax
import jax.numpy as jnp


@jax.jit
def meshgrid_to_lst_of_pts(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """
    :meta private:
    Given X and Y, which each have shape (n, n) and are
    the output of a Numpy meshgrid-like function, stacks these points into a tall 2D array of shape (n**2, 2)

    Args:
        X (jnp.ndarray): Has shape (n,n)
        Y (jnp.ndarray): Has shape (n,n)

    Returns:
        jnp.ndarray: Has shape (n**2, 2)
    """
    n = X.shape[0]
    X_exp = jnp.expand_dims(X, -1)
    Y_exp = jnp.expand_dims(Y, -1)
    together = jnp.concatenate((X_exp, Y_exp), axis=-1)
    return jnp.reshape(together, (n**2, 2))
