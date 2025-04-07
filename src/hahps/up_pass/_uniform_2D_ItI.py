from typing import List, Tuple
import jax
import jax.numpy as jnp

from .._device_config import HOST_DEVICE, DEVICE_ARR
from .._pdeproblem import PDEProblem
from ..local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI


def up_pass_uniform_2D_ItI(
    source: jax.Array,
    pde_problem: PDEProblem,
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

    # Get the saved D_inv_lst and BD_inv_lst
    D_inv_lst = pde_problem.D_inv_lst
    BD_inv_lst = pde_problem.BD_inv_lst

    Y, T, v, h_in = local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem,
        device=device,
        host_device=host_device,
    )

    g_tilde_lst = []

    for i in range(len(D_inv_lst)):
        # Get h and g_tilde for this level
        nbdry = h_in.shape[-1]
        h_in = h_in.reshape(4, -1, nbdry)
        D_inv = D_inv_lst[i]
        BD_inv = BD_inv_lst[i]
        h_in, g_tilde = vmapped_assemble_boundary_data(h_in, D_inv, BD_inv)
        g_tilde_lst.append(g_tilde)

    # Return v and g_tilde_lst
    return v, g_tilde_lst


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

    nside = h_in.shape[1] // 4

    # Remember, the slices along the merge interface go from OUTSIDE to INSIDE
    h_a_1 = jnp.concatenate([h_in[0, -nside:], h_in[0, :nside]])
    h_a_5 = h_in[0, nside : 2 * nside]
    h_a_8 = jnp.flipud(h_in[0, 2 * nside : 3 * nside])

    h_b_2 = h_in[1, : 2 * nside]
    h_b_6 = h_in[1, 2 * nside : 3 * nside]
    h_b_5 = jnp.flipud(h_in[1, 3 * nside : 4 * nside])

    h_c_6 = jnp.flipud(h_in[2, :nside])
    h_c_3 = h_in[2, nside : 3 * nside]
    h_c_7 = h_in[2, 3 * nside : 4 * nside]

    h_d_8 = h_in[3, :nside]
    h_d_7 = jnp.flipud(h_in[3, nside : 2 * nside])
    h_d_4 = h_in[3, 2 * nside : 4 * nside]

    h_int_child = jnp.concatenate(
        [h_b_5, h_d_8, h_b_6, h_d_7, h_a_5, h_c_6, h_c_7, h_a_8]
    )

    h_ext_child = jnp.concatenate([h_a_1, h_b_2, h_c_3, h_d_4])

    g_tilde = -1 * D_inv @ h_int_child

    h = h_ext_child - BD_inv @ h_int_child

    return h, g_tilde


vmapped_assemble_boundary_data = jax.vmap(
    assemble_boundary_data, in_axes=(1, 0, 0), out_axes=(0, 0)
)
