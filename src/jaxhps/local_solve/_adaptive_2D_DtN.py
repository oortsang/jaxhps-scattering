import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from .._domain import get_all_leaves
from ._uniform_2D_DtN import _gather_coeffs_2D, get_DtN, assemble_diff_operator
from .._precompute_operators_2D import precompute_Q_2D_DtN
from typing import Tuple
import logging
from functools import partial


def local_solve_stage_adaptive_2D_DtN(
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    This function performs the local solve stage for 2D adaptive discretization problems, creating DtN matrices.

    The 2D adaptive version of the local solve stage is similar to the 2D uniform version. The major difference
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
        Solution operators mapping from Dirichlet boundary data to homogeneous solutions on the leaf interiors. Has shape (n_leaves, p^2, 4q)
    T : jax.Array
        Dirichlet-to-Neumann matrices for each leaf. Has shape (n_leaves, 4q, 4q)
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^2)
    h : jax.Array
        Outgoing boundary data. This is the outward-pointing normal derivative of the particular solution. Has shape (n_leaves, 4q)
    """

    logging.debug("local_solve_stage_adaptive_2D_DtN: started")

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs_2D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    source_term = pde_problem.source
    logging.debug(
        "local_solve_stage_adaptive_2D_DtN: input source_term devices = %s",
        source_term.devices(),
    )
    source_term = jax.device_put(
        source_term,
        device,
    )
    # stack the precomputed differential operators into a single array
    diff_ops = jnp.stack(
        [
            pde_problem.D_xx,
            pde_problem.D_xy,
            pde_problem.D_yy,
            pde_problem.D_x,
            pde_problem.D_y,
            jnp.eye(pde_problem.D_xx.shape[0], dtype=jnp.float64),
        ]
    )

    # Put the input data on the device
    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )
    # Have to generate Q matrices for each leaf by re-scaling the differential
    # operators
    sidelens = jnp.array(
        [l.xmax - l.xmin for l in get_all_leaves(pde_problem.domain.root)]
    )
    sidelens = jax.device_put(sidelens, device)

    all_diff_operators, Q_Ds = (
        vmapped_prep_nonuniform_refinement_diff_operators_2D(
            sidelens,
            coeffs_gathered,
            which_coeffs,
            diff_ops,
            pde_problem.domain.p,
            pde_problem.domain.q,
        )
    )

    # Add a dummy source dimension
    source_term = jnp.expand_dims(source_term, axis=-1)
    Y_arr, T_arr, v, h = vmapped_get_DtN_adaptive(
        source_term, all_diff_operators, Q_Ds, pde_problem.P
    )

    # Remove the dummy source dimension from v and h
    v = v[..., 0]
    h = h[..., 0]

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


vmapped_get_DtN_adaptive = jax.vmap(
    get_DtN,
    in_axes=(0, 0, 0, None),
    out_axes=(0, 0, 0, 0),
)


@partial(jax.jit, static_argnums=(4, 5))
def _prep_nonuniform_refinement_diff_operators_2D(
    sidelen: jax.Array,
    coeffs_arr: jax.Array,
    which_coeffs: jax.Array,
    diff_ops_2D: jax.Array,
    p: int,
    q: int,
) -> Tuple[jax.Array, jax.Array]:
    """Prepares the differential operators for nonuniform refinement.

    Args:
        sidelen (jax.Array): Array of shape (n_leaves,) containing the sidelengths of each leaf.
        coeffs_arr (jax.Array): Array of shape (?, n_leaves, p**2) containing the PDE coefficients.
        which_coeffs (jax.Array): Array of shape (6,) containing boolean values specifying which coefficients are not None.
        diff_ops_2D (jax.Array): Array of shape (6, p**2, p**2) containing the precomputed differential operators.
        p (int): The number of Chebyshev nodes in each dimension.
        q (int): The number of Gauss nodes in each dimension.

    Returns:
        Tuple[jax.Array, jax.Array]: The precomputed differential operators for nonuniform refinement.
    """
    half_side_len = sidelen / 2
    # Second-order differential operators
    diff_ops_2D = diff_ops_2D.at[:3].set(diff_ops_2D[:3] / (half_side_len**2))
    # First-order differential operators
    diff_ops_2D = diff_ops_2D.at[3:5].set(diff_ops_2D[3:5] / half_side_len)
    diff_operator = assemble_diff_operator(
        coeffs_arr, which_coeffs, diff_ops_2D
    )
    Q_D = precompute_Q_2D_DtN(p, q, diff_ops_2D[3], diff_ops_2D[4])
    return diff_operator, Q_D


vmapped_prep_nonuniform_refinement_diff_operators_2D = jax.vmap(
    _prep_nonuniform_refinement_diff_operators_2D,
    in_axes=(0, 1, None, None, None, None),
    out_axes=(0, 0),
)
