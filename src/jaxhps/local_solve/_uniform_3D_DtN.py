import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from typing import Tuple

from ._uniform_2D_DtN import (
    vmapped_get_DtN_uniform,
    vmapped_assemble_diff_operator,
)


def local_solve_stage_uniform_3D_DtN(
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    This function performs the local solve stage for 3D problems with a uniform quadtree, creating DtN matrices.

    The 3D adaptive version of the local solve stage is similar to the 3D uniform version. The major difference
    appears when dealing with the pre-computed operator Q, which must be scaled by the side length of each leaf.
    This means Q must be re-computed for each leaf solve.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.
    device : jax.Device
        Where to perform the computation. Defaults to ``jax.devices()[0]``.
    host_device : jax.Device
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    Y : jax.Array
        Solution operators mapping from Dirichlet boundary data to homogeneous solutions on the leaf interiors. Has shape (n_leaves, p^3, 6q^2)
    T : jax.Array
        Dirichlet-to-Neumann matrices for each leaf. Has shape (n_leaves, 6q^2, 6q^2)
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^3) or (n_leaves, p^3, n_src) if multi-source.
    h : jax.Array
        Outgoing boundary data. This is the outward-pointing normal derivative of the particular solution. Has shape (n_leaves, 6q^2) or (n_leaves, 6q^2, n_src) if multi-source.
    """
    coeffs_gathered, which_coeffs = _gather_coeffs_3D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_xz_coeffs=pde_problem.D_xz_coefficients,
        D_yz_coeffs=pde_problem.D_yz_coefficients,
        D_zz_coeffs=pde_problem.D_zz_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        D_z_coeffs=pde_problem.D_z_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    source_term = pde_problem.source
    bool_multi_source = source_term.ndim == 3
    source_term = jax.device_put(source_term, device)

    # stack the precomputed differential operators into a single array
    diff_ops = jnp.stack(
        [
            pde_problem.D_xx,
            pde_problem.D_xy,
            pde_problem.D_yy,
            pde_problem.D_xz,
            pde_problem.D_yz,
            pde_problem.D_zz,
            pde_problem.D_x,
            pde_problem.D_y,
            pde_problem.D_z,
            jnp.eye(pde_problem.domain.p**3),
        ]
    )
    # Now that arrays are in contiguous blocks of memory, we can move them to the GPU.
    diff_ops = jax.device_put(diff_ops, device)
    coeffs_gathered = jax.device_put(coeffs_gathered, device)
    diff_operators = vmapped_assemble_diff_operator(
        coeffs_gathered, which_coeffs, diff_ops
    )
    if not bool_multi_source:
        source_term = jnp.expand_dims(source_term, axis=-1)
    Y_arr, T_arr, v, h = vmapped_get_DtN_uniform(
        source_term, diff_operators, pde_problem.Q, pde_problem.P
    )

    if not bool_multi_source:
        # Remove the last dimension if it was added
        h = h[..., 0]
        v = v[..., 0]

    # Return data to the requested device
    T_arr_host = jax.device_put(T_arr, host_device)
    del T_arr
    v_host = jax.device_put(v, host_device)
    del v
    h_host = jax.device_put(h, host_device)
    del h
    Y_arr_host = jax.device_put(Y_arr, host_device)
    del Y_arr

    # Return the DtN arrays, particular solutions, particular
    # solution fluxes, and the solution operators. The solution
    # operators are not moved to the host.
    return Y_arr_host, T_arr_host, v_host, h_host


@jax.jit
def _gather_coeffs_3D(
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_xz_coeffs: jnp.ndarray | None = None,
    D_yz_coeffs: jnp.ndarray | None = None,
    D_zz_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    D_z_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """If not None, expects each input to have shape (n_leaf_nodes, p**2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: coeffs_gathered and which_coeffs
            coeffs_gathered is an array of shape (?, n_leaf_nodes, p**3) containing the non-None coefficients.
            which_coeffs is an array of shape (10) containing boolean values specifying which coefficients were not None.
    """
    coeffs_lst = [
        D_xx_coeffs,
        D_xy_coeffs,
        D_yy_coeffs,
        D_xz_coeffs,
        D_yz_coeffs,
        D_zz_coeffs,
        D_x_coeffs,
        D_y_coeffs,
        D_z_coeffs,
        I_coeffs,
    ]
    which_coeffs = jnp.array([coeff is not None for coeff in coeffs_lst])
    coeffs_gathered = jnp.array(
        [coeff for coeff in coeffs_lst if coeff is not None]
    )
    return coeffs_gathered, which_coeffs
