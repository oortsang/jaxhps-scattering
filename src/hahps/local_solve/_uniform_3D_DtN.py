import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from .._device_config import DEVICE_ARR, HOST_DEVICE
from typing import Tuple

from ._uniform_2D_DtN import (
    vmapped_get_DtN_uniform,
    vmapped_assemble_diff_operator,
)


def local_solve_stage_uniform_3D_DtN(
    pde_problem: PDEProblem,
    host_device: jax.Device = HOST_DEVICE,
    device: jax.Device = DEVICE_ARR[0],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    Y_arr, T_arr, v, h = vmapped_get_DtN_uniform(
        source_term, diff_operators, pde_problem.Q, pde_problem.P
    )

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
