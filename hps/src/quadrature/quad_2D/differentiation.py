from functools import partial
from typing import Tuple
import jax.numpy as jnp
import jax

from hps.src.quadrature.quad_2D.indexing import _rearrange_indices
from hps.src.quadrature.quadrature_utils import (
    chebyshev_points,
    differentiation_matrix_1d,
    barycentric_lagrange_interpolation_matrix,
)


def precompute_diff_operators(p: int, half_side_len: float) -> Tuple[jnp.ndarray]:
    """
    Returns D_x, D_y, D_xx, D_yy, D_xy
    """
    rearrange_indices = _rearrange_indices(p)

    n_cheby_bdry_pts = 4 * (p - 1)

    pts = chebyshev_points(p)[0]
    cheby_diff_matrix = differentiation_matrix_1d(pts) / half_side_len

    # Precompute du/dx and du/dy
    du_dx = jnp.kron(cheby_diff_matrix, jnp.eye(p))
    du_dy = -1 * jnp.kron(jnp.eye(p), cheby_diff_matrix)

    # Permute the rows and cols of du_dx and du_dy to match the ordering of the Chebyshev points.
    du_dx = du_dx[rearrange_indices, :]
    du_dx = du_dx[:, rearrange_indices]

    du_dy = du_dy[rearrange_indices, :]
    du_dy = du_dy[:, rearrange_indices]

    return (
        du_dx,
        du_dy,
        du_dx @ du_dx,
        du_dy @ du_dy,
        du_dx @ du_dy,
    )


def precompute_N_matrix(du_dx: jnp.array, du_dy: jnp.array, p: int) -> jnp.array:
    """
    The N matrix is a 4p x p^2 matrix that maps a solution on the
    Cheby points to the outward normal derivatives on the Cheby boundaries.
    This matrix double-counts the corners, i.e. the derivative
    at the corner is evaluated twice, once for each side
    it is on.

    Args:
        du_dx (jnp.array): Has shape (p**2, p**2)
        du_dy (jnp.array): Has shape (p**2, p**2)

    Returns:
        jnp.array: Has shape (4p, p**2)
    """

    N_dbl = jnp.full((4 * p, p**2), 0.0, dtype=jnp.float64)

    # S boundary
    N_dbl = N_dbl.at[:p].set(-1 * du_dy[:p])
    # E boundary
    N_dbl = N_dbl.at[p : 2 * p].set(du_dx[p - 1 : 2 * p - 1])
    # N boundary
    N_dbl = N_dbl.at[2 * p : 3 * p].set(du_dy[2 * p - 2 : 3 * p - 2])
    # W boundary
    N_dbl = N_dbl.at[3 * p :].set(-1 * du_dx[3 * p - 3 : 4 * p - 3])
    N_dbl = N_dbl.at[-1].set(-1 * du_dx[0])

    return N_dbl


def precompute_N_tilde_matrix(du_dx: jnp.array, du_dy: jnp.array, p: int) -> jnp.array:
    """
    Implements an operator mapping from samples on the Chebyshev grid points to normal derivatives at the 4*(p-1) boundary points.

    Args:
        du_dx (jnp.array): Precomputed differential operator for x-direction. Has shape (p**2, p**2)
        du_dy (jnp.array): Precomputed differential operator for y-direction. Has shape (p**2, p**2)
        p (int): Shape parameter.

    Returns:
        jnp.array: Has shape (4*(p-1), p**2)
    """

    N_tilde = jnp.full((4 * (p - 1), p**2), 0.0, dtype=jnp.float64)

    N_tilde = N_tilde.at[: p - 1].set(-1 * du_dy[: p - 1])
    N_tilde = N_tilde.at[p - 1 : 2 * (p - 1)].set(du_dx[p - 1 : 2 * (p - 1)])
    N_tilde = N_tilde.at[2 * (p - 1) : 3 * (p - 1)].set(
        du_dy[2 * (p - 1) : 3 * (p - 1)]
    )
    N_tilde = N_tilde.at[3 * (p - 1) :].set(-1 * du_dx[3 * (p - 1) : 4 * (p - 1)])
    return N_tilde


def precompute_G_matrix(N: jnp.array, p: int, eta: float) -> jnp.array:
    """
    G is the matrix which maps functions on a
    2D Chebyshev grid to outgoing impedance data on the 4p
    Chebyshev boundary points, which include each corner twice.

    Args:
        N (jnp.array): Has shape (4p, p**2). Is the result of precompute_N_matrix().
        p (int): Shape parameter.
        eta (float): Real number

    Returns:
        jnp.array: Has shape (4p, p**2)
    """
    G = N.astype(jnp.complex128)
    # S side rows 0:p and cols 0:p
    G = G.at[:p, :p].set(G[:p, :p] - 1j * eta * jnp.eye(p))
    # E side rows p:2p and cols p-1:2p-1
    G = G.at[p : 2 * p, p - 1 : 2 * p - 1].set(
        G[p : 2 * p, p - 1 : 2 * p - 1] - 1j * eta * jnp.eye(p)
    )
    # N side rows 2p:3p and cols 2p-2:3p-2
    G = G.at[2 * p : 3 * p, 2 * p - 2 : 3 * p - 2].set(
        G[2 * p : 3 * p, 2 * p - 2 : 3 * p - 2] - 1j * eta * jnp.eye(p)
    )
    # W side rows 3p:4p, cols 3p-3:4p-4, 0
    G = G.at[3 * p : 4 * p - 1, 3 * p - 3 : 4 * p - 4].set(
        G[3 * p : 4 * p - 1, 3 * p - 3 : 4 * p - 4] - 1j * eta * jnp.eye(p - 1)
    )
    G = G.at[4 * p - 1, 0].set(G[4 * p - 1, 0] - 1j * eta)
    return G


def precompute_F_matrix(N_tilde: jnp.array, p: int, eta: float) -> jnp.array:
    """
    F = N_tilde + i eta I[:4(p-1)] is the matrix which maps functions on a
    2D Chebyshev grid to incoming impedance data on the 4(p - 1)
    Chebyshev boundary points.

    Args:
        N_tilde (jnp.array): Has shape (4(p - 1), p**2). Is the result of precompute_N_tilde_matrix().
        p (int): Shape parameter
        eta (float): Real number

    Returns:
        jnp.array: Has shape (4(p - 1), p**2)
    """
    shape_0 = N_tilde.shape[0]
    F = N_tilde.astype(jnp.complex128)
    # Add i eta I to the top block of F
    F = F.at[:shape_0, :shape_0].set(
        F[:shape_0, :shape_0] + 1j * eta * jnp.eye(shape_0)
    )

    return F
