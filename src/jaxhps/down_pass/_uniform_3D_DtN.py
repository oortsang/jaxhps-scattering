import logging
from typing import List

import jax
import jax.numpy as jnp


def down_pass_uniform_3D_DtN(
    boundary_data: jax.Array,
    S_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    Y_arr: jax.Array,
    v_arr: jax.Array,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> None:
    """
    Computes the downward pass of the HPS algorithm. This function takes the Dirichlet data
    at the boundary of the domain and propagates it down the tree to the leaf nodes.


    If Y_arr is None, the function will exit early after doing all of the downward propagation operations.

    Parameters
    ----------

    boundary_data : jax.Array
        An array specifying Dirichlet data on the boundary of the domain.  Has shape (n_bdry,) or (n_bdry, nsrc) for multi-source problems.

    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst : List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    Y_arr : jax.Array
        Matrices mapping the solution to the interior of the leaf nodes. Has shape (n_leaf, p^3, 6q^2).

    v_arr : jax.Array
        Particular solution data at the interior of the leaves. Has shape (n_leaf, p^3).

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].


    Returns
    -------

    solns : jax.Array
        Has shape (n_leaves, p^3). Interior solution on all of the leaf nodes.


    """
    logging.debug("_down_pass_3D: started")

    # leaf_Y_maps = jax.device_put(leaf_Y_maps, DEVICE)
    # v_array = jax.device_put(v_array, DEVICE)

    boundary_data = jax.device_put(boundary_data, device)
    Y_arr = jax.device_put(Y_arr, device)
    v_arr = jax.device_put(v_arr, device)
    S_lst = [jax.device_put(S_arr, device) for S_arr in S_lst]
    g_tilde_lst = [jax.device_put(g_tilde, device) for g_tilde in g_tilde_lst]

    n_levels = len(S_lst)

    bool_multi_source = len(g_tilde_lst) > 1 and g_tilde_lst[0].ndim == 3
    if bool_multi_source and boundary_data.ndim == 1:
        raise ValueError(
            "For multi-source downward pass, need to specify boundary data for each source."
        )
    # Reshape to (1, n_bdry, nsrc)
    if (bool_multi_source and boundary_data.ndim == 2) or (
        not bool_multi_source and boundary_data.ndim == 1
    ):
        bdry_data = jnp.expand_dims(boundary_data, axis=0)

    # if not bool_multi_source:
    #     # Reshape to (1, n_bdry)
    #     bdry_data = jnp.expand_dims(bdry_data, axis=-1)

    # Change the last entry of the S_lst and v_int_lst to have batch dimension 1
    S_lst[-1] = jnp.expand_dims(S_lst[-1], axis=0)
    g_tilde_lst[-1] = jnp.expand_dims(g_tilde_lst[-1], axis=0)

    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_propogate_down_oct_DtN(S_arr, bdry_data, g_tilde)
        # Reshape from (-1, 4, n_bdry) to (-1, n_bdry)
        n_bdry = bdry_data.shape[2]
        if bool_multi_source:
            nsrc = bdry_data.shape[-1]
            bdry_data = bdry_data.reshape((-1, n_bdry, nsrc))
        else:
            bdry_data = bdry_data.reshape((-1, n_bdry))

    root_dirichlet_data = bdry_data
    # Batched matrix multiplication to compute homog solution on all leaves
    if bool_multi_source:
        leaf_homog_solns = jnp.einsum(
            "ijk,ikl->ijl", Y_arr, root_dirichlet_data
        )
    else:
        leaf_homog_solns = jnp.einsum("ijk,ik->ij", Y_arr, root_dirichlet_data)
    leaf_solns = leaf_homog_solns + v_arr
    leaf_solns = jax.device_put(leaf_solns, host_device)
    return leaf_solns


@jax.jit
def _propogate_down_oct_DtN(
    S_arr: jax.Array,
    bdry_data: jax.Array,
    v_int_data: jax.Array,
) -> jax.Array:
    """_summary_

    Args:
        S_arr (jax.Array): Has shape (12 * n_per_face, 24 * n_per_face)
        bdry_data (jax.Array): Has shape (24 * n_per_face,)
        v_int_data (jax.Array): Has shape (12 * n_per_face,)

    Returns:
        jax.Array: Has shape (8, 6 * n_per_face)
    """
    n_per_face = bdry_data.shape[0] // 24

    n = 4 * n_per_face

    g_int = S_arr @ bdry_data + v_int_data

    g_int_9 = g_int[:n_per_face]
    g_int_10 = g_int[n_per_face : 2 * n_per_face]
    g_int_11 = g_int[2 * n_per_face : 3 * n_per_face]
    g_int_12 = g_int[3 * n_per_face : 4 * n_per_face]
    g_int_13 = g_int[4 * n_per_face : 5 * n_per_face]
    g_int_14 = g_int[5 * n_per_face : 6 * n_per_face]
    g_int_15 = g_int[6 * n_per_face : 7 * n_per_face]
    g_int_16 = g_int[7 * n_per_face : 8 * n_per_face]
    g_int_17 = g_int[8 * n_per_face : 9 * n_per_face]
    g_int_18 = g_int[9 * n_per_face : 10 * n_per_face]
    g_int_19 = g_int[10 * n_per_face : 11 * n_per_face]
    g_int_20 = g_int[11 * n_per_face :]

    bdry_data_0 = bdry_data[:n]
    bdry_data_1 = bdry_data[n : 2 * n]
    bdry_data_2 = bdry_data[2 * n : 3 * n]
    bdry_data_3 = bdry_data[3 * n : 4 * n]
    bdry_data_4 = bdry_data[4 * n : 5 * n]
    bdry_data_5 = bdry_data[5 * n : 6 * n]

    g_a = jnp.concatenate(
        [
            bdry_data_0[-n_per_face:],
            g_int_9,
            bdry_data_2[-n_per_face:],
            g_int_12,
            g_int_17,
            bdry_data_5[:n_per_face],
        ]
    )

    g_b = jnp.concatenate(
        [
            g_int_9,
            bdry_data_1[-n_per_face:],
            bdry_data_2[2 * n_per_face : 3 * n_per_face],
            g_int_10,
            g_int_18,
            bdry_data_5[n_per_face : 2 * n_per_face],
        ]
    )

    g_c = jnp.concatenate(
        [
            g_int_11,
            bdry_data_1[2 * n_per_face : 3 * n_per_face],
            g_int_10,
            bdry_data_3[2 * n_per_face : 3 * n_per_face],
            g_int_19,
            bdry_data_5[2 * n_per_face : 3 * n_per_face],
        ]
    )

    g_d = jnp.concatenate(
        [
            bdry_data_0[2 * n_per_face : 3 * n_per_face],
            g_int_11,
            g_int_12,
            bdry_data_3[-n_per_face:],
            g_int_20,
            bdry_data_5[3 * n_per_face : 4 * n_per_face],
        ]
    )

    g_e = jnp.concatenate(
        [
            bdry_data_0[:n_per_face],
            g_int_13,
            bdry_data_2[:n_per_face],
            g_int_16,
            bdry_data_4[:n_per_face],
            g_int_17,
        ]
    )

    g_f = jnp.concatenate(
        [
            g_int_13,
            bdry_data_1[:n_per_face],
            bdry_data_2[n_per_face : 2 * n_per_face],
            g_int_14,
            bdry_data_4[n_per_face : 2 * n_per_face],
            g_int_18,
        ]
    )

    g_g = jnp.concatenate(
        [
            g_int_15,
            bdry_data_1[n_per_face : 2 * n_per_face],
            g_int_14,
            bdry_data_3[n_per_face : 2 * n_per_face],
            bdry_data_4[2 * n_per_face : 3 * n_per_face],
            g_int_19,
        ]
    )

    g_h = jnp.concatenate(
        [
            bdry_data_0[n_per_face : 2 * n_per_face],
            g_int_15,
            g_int_16,
            bdry_data_3[:n_per_face],
            bdry_data_4[3 * n_per_face : 4 * n_per_face],
            g_int_20,
        ]
    )

    return jnp.stack([g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h])


vmapped_propogate_down_oct_DtN = jax.vmap(
    _propogate_down_oct_DtN, in_axes=(0, 0, 0), out_axes=0
)
