from typing import List

import jax
import jax.numpy as jnp
import logging


def down_pass_uniform_2D_ItI(
    boundary_data: jax.Array,
    S_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    Y_arr: jax.Array,
    v_arr: jax.Array,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> None:
    """
    Computes the downward pass of the HPS algorithm. This function takes the incoming impedance data
    at the boundary of the domain:

    .. math::

       u_n(x) + i \\eta u(x) = g(x)

    and propagates it down to the leaves.

    If Y_arr is None, the function will exit early after doing all of the downward propagation operations.

    Parameters
    ----------

    boundary_data : jax.Array
        An array specifying Dirichlet data on the boundary of the domain.  Has shape (n_bdry,) or (n_bdry, nsrc).

    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst : List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    Y_arr : jax.Array
        Matrices mapping the solution to the interior of the leaf nodes. Has shape (n_leaf, p^2, 4q).

    v_arr : jax.Array
        Particular solution data at the interior of the leaves. Has shape (n_leaf, p^2) or (n_leaf, p^2, nsrc).

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].


    Returns
    -------

    solns : jax.Array
        Has shape (n_leaves, p^2). Interior solution on all of the leaf nodes.

    """
    logging.debug(
        "down_pass_uniform_2D_ItI: started. boundary_imp_data shape: %s, len(g_tilde_lst): %s, len(S_lst): %s",
        boundary_data.shape,
        len(g_tilde_lst),
        len(S_lst),
    )

    bdry_data = jax.device_put(boundary_data, device)
    Y_arr = jax.device_put(Y_arr, device)
    v_arr = jax.device_put(v_arr, device)
    S_lst = [jax.device_put(S_arr, device) for S_arr in S_lst]
    g_tilde_lst = [jax.device_put(g_tilde, device) for g_tilde in g_tilde_lst]

    n_levels = len(S_lst)

    bool_multi_source = len(g_tilde_lst) and g_tilde_lst[0].ndim == 3
    if bool_multi_source and bdry_data.ndim == 1:
        raise ValueError(
            "For multi-source downward pass, need to specify boundary data for each source."
        )

    # Reshape to (1, n_bdry) or (1, n_bdry, nsrc)
    if (bool_multi_source and bdry_data.ndim == 2) or (
        not bool_multi_source and bdry_data.ndim == 1
    ):
        bdry_data = jnp.expand_dims(bdry_data, axis=0)
    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_propogate_down_2D_ItI(S_arr, bdry_data, g_tilde)

        n_bdry = bdry_data.shape[2]
        if bool_multi_source:
            nsrc = bdry_data.shape[-1]
            bdry_data = bdry_data.reshape((-1, n_bdry, nsrc))
        else:
            bdry_data = bdry_data.reshape((-1, n_bdry))

    # Once we have the leaf node incoming impedance data, compute solution on the interior
    # of each leaf node using the Y maps.
    root_incoming_imp_data = bdry_data

    if Y_arr is None:
        return root_incoming_imp_data

    if bool_multi_source:
        leaf_homog_solns = jnp.einsum(
            "ijk,ikl->ijl", Y_arr, root_incoming_imp_data
        )
    else:
        leaf_homog_solns = jnp.einsum(
            "ijk,ik->ij", Y_arr, root_incoming_imp_data
        )
    leaf_solns = leaf_homog_solns + v_arr
    leaf_solns = jax.device_put(leaf_solns, host_device)
    return leaf_solns


@jax.jit
def _propogate_down_2D_ItI(
    S_arr: jax.Array,
    bdry_data: jax.Array,
    f_data: jax.Array,
) -> jax.Array:
    """
    Given homogeneous data on the boundary, interface homogeneous solution operator S, and
    interface particular solution data, this function returns the solution on the boundaries
    of the four children.

    suppose n_child is the number of quadrature points on EACH SIDE of a child node.

    Args:
        S_arr (jax.Array): Has shape (8 * n_child, 8 * n_child)
        bdry_data (jax.Array): 8 * n_child
        f_data (jax.Array): 8 * n_child

    Returns:
        jax.Array: Has shape (4, 4 * n_child)
    """

    n_child = bdry_data.shape[0] // 8

    t_int_homog = S_arr @ bdry_data
    # print("_propogate_down_quad_ItI: t_int_homog.shape ", t_int_homog.shape)
    # print("_propogate_down_quad_ItI: f_data.shape ", f_data.shape)
    t_int = t_int_homog + f_data

    # All of these slices of g_int are propogating from OUTSIDE to INSIDE
    t_a_5 = t_int[:n_child]
    t_b_5 = jnp.flipud(t_int[n_child : 2 * n_child])
    t_b_6 = t_int[2 * n_child : 3 * n_child]
    t_c_6 = jnp.flipud(t_int[3 * n_child : 4 * n_child])
    t_c_7 = t_int[4 * n_child : 5 * n_child]
    t_d_7 = jnp.flipud(t_int[5 * n_child : 6 * n_child])
    t_d_8 = t_int[6 * n_child : 7 * n_child]
    t_a_8 = jnp.flipud(t_int[7 * n_child :])

    g_a = jnp.concatenate(
        [
            bdry_data[:n_child],  # S edge
            t_a_5,  # E edge
            t_a_8,  # N edge
            bdry_data[7 * n_child :],  # W edge
        ]
    )

    g_b = jnp.concatenate(
        [
            bdry_data[n_child : 3 * n_child],  # S edge, E edge
            t_b_6,  # N edge
            t_b_5,  # W edge
        ]
    )

    g_c = jnp.concatenate(
        [
            t_c_6,  # S edge
            bdry_data[3 * n_child : 5 * n_child],  # E edge, N edge
            t_c_7,  # W edge
        ]
    )

    g_d = jnp.concatenate(
        [
            t_d_8,  # S edge
            t_d_7,  # E edge
            bdry_data[5 * n_child : 7 * n_child],  # N edge, W edge
        ]
    )
    return jnp.stack([g_a, g_b, g_c, g_d])


vmapped_propogate_down_2D_ItI = jax.vmap(
    _propogate_down_2D_ItI, in_axes=(0, 0, 0), out_axes=0
)
