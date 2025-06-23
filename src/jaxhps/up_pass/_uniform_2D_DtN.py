from typing import List, Tuple
import jax
import jax.numpy as jnp
from .._pdeproblem import PDEProblem
import logging


def up_pass_uniform_2D_DtN(
    source: jax.Array,
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_h_last: bool = False,
) -> Tuple[jax.Array, List[jax.Array], jax.Array]:
    """
    This function performs the upward pass for 2D DtN problems. It recomputes the local solve stage to get
    outgoing impedance data from the particular solution, which is now known because the source is specified.



    Parameters
    ----------
    source : jax.Array
        Specifies the source term(s). Can have shape (nleaves, p^2) or (nleaves, p^2, nsrc).

    pde_probem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    device : jax.Device, optional
        Where to perform the computation. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    return_h_last : bool, optional
        If True, return the last h vector, which gives the outward pointing normal derivative data for the particular solution evaluated at the domain boundary.

    Returns
    -------
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^2)

    g_tilde_lst : List[jax.Array]
        List of pre-computed g_tilde matrices for each level of the quadtree.

    h_last : jax.Array
        Outward pointing normal derivative data for the particular solution evaluated at the domain boundary. Has shape (nbdry, nsrc). Is only returned if ``return_h_last=True``.
    """

    bool_multi_source = source.ndim == 3
    if not bool_multi_source:
        # Add a source dimension to the source term
        source = jnp.expand_dims(source, axis=-1)
        logging.debug(
            "up_pass_uniform_2D_DtN: new source shape = %s", source.shape
        )

    # Re-do a full local solve.
    pde_problem.source = source

    # Get the saved D_inv_lst and BD_inv_lst
    D_inv_lst = pde_problem.D_inv_lst
    BD_inv_lst = pde_problem.BD_inv_lst

    # Compute v via an einsum between the source term and the Phi matrices
    # which are stored in pde_problem.Phi
    # first init v to zeros then fill in the interior values.
    v = jnp.zeros_like(source)
    n_cheby_bdry = 4 * (pde_problem.domain.p - 1)
    v = v.at[:, n_cheby_bdry:].set(
        jnp.einsum(
            "ijk,ikl->ijl",
            pde_problem.Phi,
            source[:, n_cheby_bdry:],
        )
    )
    h_in = jnp.einsum("ij,kjl->kil", pde_problem.Q, v)

    # Y, T, v, h_in = local_solve_stage_uniform_2D_DtN(
    #     pde_problem=pde_problem,
    #     device=device,
    #     host_device=host_device,
    # )
    logging.debug(
        "up_pass_uniform_2D_DtN: after local solve, h_in shape = %s",
        h_in.shape,
    )

    g_tilde_lst = []

    for i in range(len(D_inv_lst)):
        # Get h and g_tilde for this level

        nnodes, nbdry, nsrc = h_in.shape
        h_in = h_in.reshape(nnodes // 4, 4, nbdry, nsrc)
        D_inv = D_inv_lst[i]
        BD_inv = BD_inv_lst[i]

        h_in, g_tilde = vmapped_assemble_boundary_data_DtN(h_in, D_inv, BD_inv)
        g_tilde_lst.append(g_tilde)

    out = (v, g_tilde_lst)

    if return_h_last:
        out = (v, g_tilde_lst, jnp.squeeze(h_in, axis=0))

    return out


@jax.jit
def assemble_boundary_data_DtN(
    h_in: jax.Array,
    D_inv: jax.Array,
    BD_inv: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """


    Args:
        h_in (jax.Array): Has shape (4, 4 * nside, n_src) where nside is the number of discretization points along each side of the nodes being merged.
        D_inv (jax.Array): Has shape (8 * nside, 8 * nside)
        BD_inv (jax.Array): Has shape (8 * nside, 8 * nside)

    Returns:
        h : jax.Array
            Has shape (8 * nside, n_src) and is the outgoing impedance data due to the particular solution on the merged node.
        g_tilde : jax.Array
            Has shape (8 * nside, n_src) and is the incoming impedance data due to the particular solution on the merged node, evaluated along the merge interfaces.
    """

    nside = h_in.shape[1] // 4

    # Remember, the slices along the merge interface go from OUTSIDE to INSIDE
    h_a = h_in[0]
    h_a_1 = jnp.concatenate([h_a[-nside:], h_a[:nside]])
    h_a_5 = h_a[nside : 2 * nside]
    h_a_8 = jnp.flipud(h_a[2 * nside : 3 * nside])

    h_b = h_in[1]
    h_b_2 = h_b[: 2 * nside]
    h_b_6 = h_b[2 * nside : 3 * nside]
    h_b_5 = jnp.flipud(h_b[3 * nside : 4 * nside])

    h_c = h_in[2]
    h_c_6 = jnp.flipud(h_c[:nside])
    h_c_3 = h_c[nside : 3 * nside]
    h_c_7 = h_c[3 * nside : 4 * nside]

    h_d = h_in[3]
    h_d_8 = h_d[:nside]
    h_d_7 = jnp.flipud(h_d[nside : 2 * nside])
    h_d_4 = h_d[2 * nside : 4 * nside]

    h_int_child = jnp.concatenate(
        [h_a_5 + h_b_5, h_b_6 + h_c_6, h_c_7 + h_d_7, h_d_8 + h_a_8]
    )

    h_ext_child = jnp.concatenate([h_a_1, h_b_2, h_c_3, h_d_4])

    g_tilde = -1 * D_inv @ h_int_child

    h = h_ext_child - BD_inv @ h_int_child

    # h is ordered like h_a_1, h_b_2, h_c_3, h_d_4. Need to
    # roll it so that it's ordered like [bottom, left, right, top].
    h = jnp.roll(h, -nside, axis=0)

    return h, g_tilde


vmapped_assemble_boundary_data_DtN = jax.vmap(
    assemble_boundary_data_DtN, in_axes=(0, 0, 0), out_axes=(0, 0)
)
