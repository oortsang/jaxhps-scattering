from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax

from ._grid_creation_3D import rearrange_indices_ext_int
from .quadrature import (
    chebyshev_points,
    gauss_points,
    differentiation_matrix_1D,
    barycentric_lagrange_interpolation_matrix_2D,
)


@partial(jax.jit, static_argnums=(0,))
def precompute_diff_operators_3D(
    p: int, half_side_len: float
) -> Tuple[jnp.ndarray]:
    """
    Returns D_x, D_y, D_z, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    """

    r = rearrange_indices_ext_int(p)

    pts = chebyshev_points(p)
    cheby_diff_matrix = differentiation_matrix_1D(pts) / half_side_len

    # Precompute du/dx, du/dy, du/dz
    du_dx = jnp.kron(
        cheby_diff_matrix,
        jnp.kron(jnp.eye(p), jnp.eye(p)),
    )
    du_dy = jnp.kron(
        jnp.eye(p),
        jnp.kron(cheby_diff_matrix, jnp.eye(p)),
    )
    du_dz = jnp.kron(
        jnp.eye(p),
        jnp.kron(jnp.eye(p), cheby_diff_matrix),
    )

    du_dx = du_dx[r, :]
    du_dx = du_dx[:, r]

    du_dy = du_dy[r, :]
    du_dy = du_dy[:, r]

    du_dz = du_dz[r, :]
    du_dz = du_dz[:, r]

    # Precompute D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    d_xx = du_dx @ du_dx
    d_yy = du_dy @ du_dy
    d_zz = du_dz @ du_dz
    d_xy = du_dx @ du_dy
    d_xz = du_dx @ du_dz
    d_yz = du_dy @ du_dz

    return (du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz)


@partial(jax.jit, static_argnums=(0, 1))
def precompute_P_3D_DtN(p: int, q: int) -> jnp.ndarray:
    """Puts together an operator interpolating from the Gauss grid on the boundary to the Cheby points on the boundary of the 3D cube.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.

    Returns:
        jnp.ndarray: Has shape (p**3 - (p-2)**3, 6 * q**2)
    """
    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)

    n_cheby_bdry_pts = p**3 - (p - 2) ** 3

    P = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, cheby_pts, cheby_pts
    )

    I_P = jnp.zeros((n_cheby_bdry_pts, 6 * q**2), dtype=jnp.float64)

    # First face
    idxes = get_face_1_idxes(p)
    I_P = I_P.at[idxes, 0 : q**2].set(P)

    # Second face
    idxes = get_face_2_idxes(p)
    I_P = I_P.at[idxes, q**2 : 2 * q**2].set(P)

    # Third face
    idxes = get_face_3_idxes(p)
    I_P = I_P.at[idxes, 2 * q**2 : 3 * q**2].set(P)

    # Fourth face
    idxes = get_face_4_idxes(p)
    I_P = I_P.at[idxes, 3 * q**2 : 4 * q**2].set(P)

    # Fifth face
    idxes = get_face_5_idxes(p)
    I_P = I_P.at[idxes, 4 * q**2 : 5 * q**2].set(P)

    # Sixth face
    idxes = get_face_6_idxes(p)
    I_P = I_P.at[idxes, 5 * q**2 :].set(P)

    # Take averages of the corners of the cube. They occur at
    # indices 0, p-1, p**2 - p, p**2 - 1, p**2, p**2 + p - 1, 2 * p**2 - p, 2 * p**2 - 1.
    corner_idxes = jnp.array(
        [
            0,
            p - 1,
            p**2 - p,
            p**2 - 1,
            p**2,
            p**2 + p - 1,
            2 * p**2 - p,
            2 * p**2 - 1,
        ]
    )
    I_P = I_P.at[corner_idxes, :].set(I_P[corner_idxes, :] / 3)

    # Take averages of the edges of the cube. There are 12 edges. They are all of length p-2 because we have
    # already averaged the corners.
    edge_idxes = jnp.concatenate(
        [
            jnp.arange(1, p - 1),
            jnp.arange(p**2 + 1, p**2 + p - 1),
            jnp.arange(p**2 - p + 1, p**2 - 1),
            jnp.arange(2 * p**2 - p + 1, 2 * p**2 - 1),
            jnp.arange(1, p - 1) * p,
            jnp.arange(p, p**2 - p, p) + p**2,
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p),
            jnp.arange(2 * p - 1, p**2 - p, p),
            jnp.arange(p**2 + 2 * p - 1, 2 * p**2 - p, p),
            jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p),
        ]
    )

    I_P = I_P.at[edge_idxes, :].set(I_P[edge_idxes, :] / 2)

    return I_P


@partial(jax.jit, static_argnums=(0, 1))
def precompute_Q_3D_DtN(
    p: int, q: int, du_dx: jnp.ndarray, du_dy: jnp.ndarray, du_dz: jnp.ndarray
) -> jnp.ndarray:
    """Precomputes an operator which maps a solution on the
    3D Chebyshev points to the outward normal derivative of the solution on the boundary Gauss grid.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.
        du_dx (jnp.ndarray): Precomputed derivative operator in the x-direction. Has shape (p**3, p**3).
        du_dy (jnp.ndarray): Precomputed derivative operator in the y-direction. Has shape (p**3, p**3).
        du_dz (jnp.ndarray): Precomputed derivative operator in the z-direction. Has shape (p**3, p**3).

    Returns:
        jnp.ndarray: Has shape (6 * q**2, p**3)
    """

    # First assemble N, a matrix of shape (6 p**2, p**3) which maps from a solution on all of the
    # Chebyshev points to the outward normal derivative of the solution on the boundary Chebyshev points.
    # Note this operator is double-counting
    # grid points on the edges.
    N = jnp.full((6 * p**2, p**3), jnp.nan, dtype=jnp.float64)
    # Grab the rows of the appropriate derivative operator and put them into N.
    face_1_idxes = get_face_1_idxes(p)
    N = N.at[: p**2].set(-1 * du_dx[face_1_idxes])
    face_2_idxes = get_face_2_idxes(p)
    N = N.at[p**2 : 2 * p**2].set(du_dx[face_2_idxes])
    face_3_idxes = get_face_3_idxes(p)
    N = N.at[2 * p**2 : 3 * p**2].set(-1 * du_dy[face_3_idxes])
    face_4_idxes = get_face_4_idxes(p)
    N = N.at[3 * p**2 : 4 * p**2].set(du_dy[face_4_idxes])
    face_5_idxes = get_face_5_idxes(p)
    N = N.at[4 * p**2 : 5 * p**2].set(-1 * du_dz[face_5_idxes])
    face_6_idxes = get_face_6_idxes(p)
    N = N.at[5 * p**2 :].set(du_dz[face_6_idxes])

    # Now assemble Q_D, a matrix of shape (6 * q**2, p**3) which maps from the Chebyshev points
    # on the boundary to the Gauss points on the boundary.

    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)
    Q = barycentric_lagrange_interpolation_matrix_2D(
        cheby_pts, cheby_pts, gauss_pts, gauss_pts
    )
    Q_I = jnp.kron(jnp.eye(6), Q)
    # Q_I = jnp.zeros((6 * q**2, 6 * p**2), dtype=jnp.float64)
    # # Face 1
    # Q_I = Q_I.at[0 : q**2, 0 : p**2].set(jnp.flip(Q, axis=(0, 1)))
    # # Face 2
    # # Q_I = Q_I.at[q**2 : 2 * q**2].set(Q)
    # # # Face 3
    # # Q_I = Q_I.at[2 * q**2 : 3 * q**2].set(Q)
    # # # Face 4
    # # Q_I = Q_I.at[3 * q**2 : 4 * q**2].set(Q)
    # # # Face 5
    # # Q_I = Q_I.at[4 * q**2 : 5 * q**2].set(Q)
    # # # Face 6
    # # Q_I = Q_I.at[5 * q**2 :].set(Q)
    Q_D = Q_I @ N
    return Q_D


############################################
# These next functions are designed to get the
# indices of certain faces of the p^3 Chebyshev grid.
# I am using numbers [1,...,6] for the faces; look at my notes for
# the definition of these numbers.


@partial(jax.jit, static_argnums=(0,))
def get_face_1_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (y,z) plane farthest in the -x direction."""
    return jnp.arange(p**2)


@partial(jax.jit, static_argnums=(0,))
def get_face_2_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (y,z) plane farthest in the +x direction."""
    return jnp.arange(p**2) + p**2


@partial(jax.jit, static_argnums=(0,))
def get_face_3_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,z) plane farthest in the -y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p),
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p),
            jnp.arange(p**2, p**2 + p),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_4_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,z) plane farthest in the +y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p * (p - 1), p**2),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p),
            jnp.arange(2 * p**2 - p, 2 * p**2),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_5_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,y) plane farthest in the -z direction."""
    first_col = jnp.arange(0, p**2, p)
    last_col = jnp.arange(p**2, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p)

    int_idxes = jnp.arange(4 * p**2 - 4 * p, 4 * p**2 - 4 - p + (p - 2) ** 2)

    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out


@partial(jax.jit, static_argnums=(0,))
def get_face_6_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,y) plane farthest in the +z direction."""
    first_col = jnp.arange(p - 1, p**2, p)
    last_col = jnp.arange(p**2 + p - 1, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p)

    u = p**3 - (p - 2) ** 3
    l = u - (p - 2) ** 2
    int_idxes = jnp.arange(l, u)
    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out
