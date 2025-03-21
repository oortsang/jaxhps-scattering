"""Defines different scattering potentials for use in forward wave scattering experiments."""

import jax.numpy as jnp
import jax


key = jax.random.key(0)
N_ATOMS = 10
ATOM_CENTERS = jax.random.uniform(key, minval=-0.5, maxval=0.5, shape=(N_ATOMS, 2))
DELTA = 50


@jax.jit
def q_luneburg(x: jnp.array) -> jnp.array:
    """
    q(x) = max(0, 1 - ||x||^2 / 0.5)

    Args:
        x (jnp.array): Has shape [..., 2]

    Returns:
        jnp.array: Has shape [...]
    """
    return jnp.maximum(0, 1 - jnp.linalg.norm(x, axis=-1) ** 2 / 0.5)


@jax.jit
def q_gaussian_bumps(x: jnp.array) -> jnp.array:
    """
    Args:
        x: Array of points with shape (..., 2) where ... can be any number of batch dimensions

    Returns:
        Array of charge densities with shape (...)

    Note:
        ATOM_CENTERS should have shape (n_atoms, 2)
        DELTA should be a scalar
    """
    # Add two new axes for broadcasting:
    # 1. One before the final dimension for the atoms dimension
    # 2. One at the end to match the coordinate dimension
    # This makes x have shape (..., 1, 2)
    x_expanded = jnp.expand_dims(x, axis=-2)

    # Reshape ATOM_CENTERS to (1, ..., 1, n_atoms, 2) where the number of 1s
    # matches the number of batch dimensions in x
    # This ensures proper broadcasting with x_expanded
    n_batch_dims = x.ndim - 1  # number of dimensions before the coordinate dimension
    atom_centers_shape = (1,) * n_batch_dims + (ATOM_CENTERS.shape[0], 2)
    atoms_expanded = jnp.reshape(ATOM_CENTERS, atom_centers_shape)

    # Compute differences - will broadcast to shape (..., n_atoms, 2)
    diffs = x_expanded - atoms_expanded

    # Square and sum over coordinates (axis=-1)
    squared_diffs = diffs**2
    summed_squared_diffs = jnp.sum(squared_diffs, axis=-1)

    # Compute exponential decay
    e = jnp.exp(-DELTA * summed_squared_diffs)

    # Sum over atoms dimension (now axis=-1 after previous sum reduced last dim)
    return jnp.sum(e, axis=-1)


@jax.jit
def q_horizontally_graded(x: jnp.array) -> jnp.array:
    """
    q(x,y) = 4(x - 0.2)[erf(25(|x| - 0.3)) - 1]
    Args:
        x: Array of points with shape (..., 2)

    Returns:
        Array of charge densities with shape (...)
    """
    x_nrm = jnp.linalg.norm(x, axis=-1)
    x = x[..., 0]

    return 4 * (x - 0.2) * (jax.scipy.special.erf(25 * (jnp.abs(x_nrm) - 0.3)) - 1)


@jax.jit
def q_vertically_graded(x: jnp.array) -> jnp.array:
    """
    q(x,y) = 4(y - 0.2)[erf(25(|x| - 0.3)) - 1]
    Args:
        x: Array of points with shape (..., 2)

    Returns:
        Array of charge densities with shape (...)
    """
    x_nrm = jnp.linalg.norm(x, axis=-1)
    y = x[..., 1]
    x = x[..., 0]

    return 4 * (y - 0.2) * (jax.scipy.special.erf(25 * (jnp.abs(x_nrm) - 0.3)) - 1)


def q_GBM_1(x: jnp.ndarray) -> jnp.ndarray:
    """
    Bumps 1 example from Barnett, Gillman, Martinsson
    x has shape (..., 2)
    q(x) = 1.5e^{-160 ||x||^2}
    """
    # return jnp.zeros_like(x[..., 0])
    return 1.5 * jnp.exp(-160 * jnp.linalg.norm(x, axis=-1) ** 2)
