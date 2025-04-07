from typing import List, Tuple

import jax

from .._device_config import HOST_DEVICE, DEVICE_ARR
from .._pdeproblem import PDEProblem
from ..local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI


def up_pass_uniform_2D_ItI(
    source: jax.Array,
    pde_problem: PDEProblem,
    D_inv_lst: List[jax.Array],
    BD_inv_lst: List[jax.Array],
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> Tuple[jax.Array, List[jax.Array]]:
    """
    This function performs the upward pass for 2D ItI problems. It recomputes the local solve stage to get
    outgoing impedance data from the particular solution, which is now known because the source is specified.



    Parameters
    ----------
    source : jax.Array
        _description_

    pde_probem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    D_inv_lst : List[jax.Array]
        List of pre-computed D^{-1} matrices for each level of the quadtree.

    BD_inv_lst : List[jax.Array]
        List of pre-computed BD^{-1} matrices for each level of the quadtree.

    device : jax.Device, optional
        Where to perform the computation. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^2)

    g_tilde_lst : List[jax.Array]
        List of pre-computed g_tilde matrices for each level of the quadtree.
    """

    # Re-do a full local solve.
    pde_problem.source = source

    Y, T, v, h = local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem,
        device=device,
        host_device=host_device,
    )


@jax.jit
def assemble_boundary_data(
    h_in: jax.Array,
    D_inv: jax.Array,
    BD_inv: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """


    Args:
        h_in (jax.Array): Has shape (4, 4 * nside) where nside is the number of discretization points along each side of the nodes being merged.
        D_inv (jax.Array): Has shape (8 * nside, 8 * nside)
        BD_inv (jax.Array): Has shape (8 * nside, 8 * nside)

    Returns:
        h : jax.Array
            Has shape (8 * nside) and is the outgoing impedance data due to the particular solution on the merged node.
        g_tilde : jax.Array
            Has shape (8 * nside) and is the incoming impedance data due to the particular solution on the merged node, evaluated along the merge interfaces.
    """
    pass
