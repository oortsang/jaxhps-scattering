from typing import Tuple, List
from functools import partial
import jax.numpy as jnp
import jax


from hps.src.quadrature.quad_3D.indexing import rearrange_indices_ext_int
from hps.src.quadrature.quadrature_utils import (
    chebyshev_points,
    affine_transform,
    differentiation_matrix_1d,
)


@partial(jax.jit, static_argnums=(0,))
def precompute_diff_operators(p: int, half_side_len: float) -> Tuple[jnp.ndarray]:
    """
    Returns D_x, D_y, D_z, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    """

    r = rearrange_indices_ext_int(p)

    pts = chebyshev_points(p)[0]
    cheby_diff_matrix = differentiation_matrix_1d(pts) / half_side_len

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
