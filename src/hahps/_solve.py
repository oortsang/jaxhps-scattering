import jax
import jax.numpy as jnp
from typing import List
from ._pdeproblem import PDEProblem

from ._device_config import (
    HOST_DEVICE,
    DEVICE_ARR,
)

from .down_pass._adaptive_2D_DtN import down_pass_adaptive_2D_DtN
from .down_pass._adaptive_3D_DtN import down_pass_adaptive_3D_DtN
from .down_pass._uniform_2D_DtN import down_pass_uniform_2D_DtN
from .down_pass._uniform_2D_ItI import down_pass_uniform_2D_ItI
from .down_pass._uniform_3D_DtN import down_pass_uniform_3D_DtN


def solve(
    pde_problem: PDEProblem,
    boundary_data: jax.Array | List[jax.Array],
    compute_device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> jax.Array:
    """
    This function performs the downward pass of the HPS algorithm, after the solution operators have
    been formed by a call to :func:`hahps.build_solver`.


    Args:
        pde_problem (PDEProblem): Specifies the differential operator, source, domain, and precomputed interpolation and differentiation matrices. Also contains all of the solution operators computed by hahps.build_solver().

        boundary_data (jax.Array | List[jax.Array]): This specifies the data on the boundary of the domain that will be propagated down to the interior of the leaves. If using an adaptive discretization, this must be specified as a list of arrays, one for each side or face of the root boundary. This list can be specified using the :func:`hahps.Domain.get_adaptive_boundary_data_lst` utility. For uniform discretizations, this argument can be a jax.Array of shape (n_bdry,) or a list.

        compute_device (jax.Device, optional): Where the computation should happen. Defaults to jax.devices()[0].

        host_device (jax.Device, optional): Where the solution operators should be stored. Defaults to jax.devices("cpu")[0].

    Returns:
        jax.Array: The solution on the HPS grid. This has shape (n_leaves, p^d).
    """
    if not pde_problem.domain.bool_uniform:
        # interface is different for the adaptive down pass so we'll
        # pass to a different function.
        return _adaptive_solve(
            pde_problem=pde_problem,
            boundary_data_lst=boundary_data,
        )

    if isinstance(boundary_data, list):
        # If the boundary data is a list, we need to concatenate it
        # into a single array.
        boundary_data = jnp.concatenate(boundary_data)

    if pde_problem.use_ItI:
        down_pass_fn = down_pass_uniform_2D_ItI

    elif pde_problem.domain.bool_2D:
        down_pass_fn = down_pass_uniform_2D_DtN
    else:
        down_pass_fn = down_pass_uniform_3D_DtN

    return down_pass_fn(
        boundary_data,
        pde_problem.S_lst,
        pde_problem.g_tilde_lst,
        pde_problem.Y,
        pde_problem.v,
        device=compute_device,
        host_device=host_device,
    )


def _adaptive_solve(
    pde_problem: PDEProblem,
    boundary_data_lst: List[jax.Array],
) -> jax.Array:
    if not isinstance(boundary_data_lst, list):
        raise ValueError(
            "For adaptive solves, boundary data needs to be a list. Try using the Domain.get_adaptive_boundary_data_lst() utility."
        )
    # Figure out which function to use
    if pde_problem.domain.bool_2D:
        down_pass_fn = down_pass_adaptive_2D_DtN
    else:
        down_pass_fn = down_pass_adaptive_3D_DtN

    return down_pass_fn(
        pde_problem=pde_problem, boundary_data=boundary_data_lst
    )
