import logging
from typing import List

import jax
import jax.numpy as jnp


from .._device_config import DEVICE_ARR, HOST_DEVICE


def down_pass_uniform_3D_DtN(
    boundary_data: jax.Array,
    S_maps_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    Y_arr: jax.Array,
    v_arr: jax.Array,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:
    logging.debug("_down_pass_3D: started")

    # leaf_Y_maps = jax.device_put(leaf_Y_maps, DEVICE)
    # v_array = jax.device_put(v_array, DEVICE)

    boundary_data = jax.device_put(boundary_data, device)
    Y_arr = jax.device_put(Y_arr, device)
    v_arr = jax.device_put(v_arr, device)
    S_maps_lst = [jax.device_put(S_arr, device) for S_arr in S_maps_lst]
    g_tilde_lst = [jax.device_put(g_tilde, device) for g_tilde in g_tilde_lst]

    n_levels = len(S_maps_lst)
    bdry_data = jnp.expand_dims(boundary_data, axis=0)

    # Change the last entry of the S_maps_lst and v_int_lst to have batch dimension 1
    S_maps_lst[-1] = jnp.expand_dims(S_maps_lst[-1], axis=0)
    g_tilde_lst[-1] = jnp.expand_dims(g_tilde_lst[-1], axis=0)

    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_maps_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_propogate_down_oct_DtN(S_arr, bdry_data, g_tilde)
        # Reshape from (-1, 4, n_bdry) to (-1, n_bdry)
        n_bdry = bdry_data.shape[-1]
        bdry_data = bdry_data.reshape((-1, n_bdry))

    root_dirichlet_data = bdry_data
    # Batched matrix multiplication to compute homog solution on all leaves
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
