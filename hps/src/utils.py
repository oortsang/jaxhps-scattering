"""
This file contains general utility functions for the HPS package.
"""

from typing import List, Tuple
import jax
import jax.numpy as jnp


def points_to_2d_lst_of_points(x: jnp.ndarray) -> jnp.ndarray:
    """Given a set of n points <x> which discretizes a 1-D interval, this function
    returns an array with shape (n**2, 2), which discretizes a 2-D area by
    taking the Cartesian product of <x> with itself, and then flattening the
    resulting 2-D grid in a column-rasterized way.

    Args:
        x (jnp.ndarray): Has shape (n,)

    Returns:
        jnp.ndarray: Has shape (n**2, 2)
    """
    xx, yy = jnp.meshgrid(x, jnp.flipud(x), indexing="ij")
    xx = jnp.expand_dims(xx, -1)
    yy = jnp.expand_dims(yy, -1)
    pts = jnp.concatenate((xx, yy), axis=-1)
    pts = pts.reshape(-1, 2)
    return pts


def lst_of_points_to_meshgrid(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Given a list of n points which create a regular grid over a 2D plane,
    this function returns a meshgrid of the points.

    Args:
        x (jnp.ndarray): Has shape (n,2)
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Each tensor has shape (n,n).
    """
    s = int(jnp.sqrt(x.shape[0]))
    xx = x[:, 0].reshape(s, s)
    yy = x[:, 1].reshape(s, s)
    return xx, yy


@jax.jit
def meshgrid_to_lst_of_pts(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Given X and Y, which each have shape (n, n) and are
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


def get_opposite_boundary_str(bdry_str: str) -> str:
    """Returns the string for the boundary opposite to bdry_str"""
    bdry_dict = {"N": "S", "E": "W", "S": "N", "W": "E"}
    return bdry_dict[bdry_str]
