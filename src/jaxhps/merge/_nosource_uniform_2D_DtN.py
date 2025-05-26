from typing import Tuple, List

import jax
import jax.numpy as jnp


from ._schur_complement import (
    nosource_assemble_merge_outputs_DtN,
)
from ._uniform_2D_DtN import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)
import logging


def nosource_merge_stage_uniform_2D_DtN(
    T_arr: jax.Array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_T: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Implements uniform 2D merges of DtN matrices. Merges the nodes in the quadtree four at a time.
    This function uses a Schur complement strategy to reduce the size of matrix inverted in each merge operation.
    This function returns lists containing :math:`S` and :math:`\\tilde{g}`, giving enough information
    to propagate boundary data back down the tree in a later part of the algorithm.

    If this function is called with the argument ``return_T=True``, the top-level DtN matrix is also returned.

    Parameters
    ----------

    T_arr : jax.Array
        Array of DtN matrices from the local solve stage. Has shape (n_leaves, 4q, 4q)

    l : int
        Number of levels to merge

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    return_T : bool
        A flag to return the top-level T matrix. Defaults to False.

    Returns
    -------
    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    D_inv_lst : List[jax.Array]
        List of pre-computed D^{-1} matrices for each level of the quadtree.

    BD_inverse_lst : List[jax.Array]
        List of pre-computed BD^{-1} matrices for each level of the quadtree.

    T_last : jax.Array
        The top-level DtN matrix, which is only returned if ``return_T=True``. Has shape (4q, 4q).

    """

    # Start lists to output data
    S_lst = []
    D_inv_lst = []
    BD_inverse_lst = []

    T_arr = jax.device_put(T_arr, device)

    if len(T_arr.shape) < 4:
        logging.debug(
            "merge_stage_uniform_2D_DtN: T_arr.shape = %s", T_arr.shape
        )
        n_leaves, n_ext, _ = T_arr.shape
        T_arr = T_arr.reshape(n_leaves // 4, 4, n_ext, n_ext)

    for i in range(l - 1):
        S_arr, T_arr_new, D_inv_arr, BD_inv_arr = (
            vmapped_nosource_uniform_quad_merge_DtN(T_arr)
        )

        # TODO: Figure out how to safely delete these arrays
        # when using autodiff.

        # T_arr.delete()
        # h_arr.delete()

        if host_device != device:
            S_host = jax.device_put(S_arr, host_device)
            S_lst.append(S_host)
            S_arr.delete()

            D_inv_host = jax.device_put(D_inv_arr, host_device)
            D_inv_lst.append(D_inv_host)
            D_inv_arr.delete()

            BD_inv_host = jax.device_put(BD_inv_arr, host_device)
            BD_inverse_lst.append(BD_inv_host)
            BD_inv_arr.delete()
        else:
            S_lst.append(S_arr)
            D_inv_lst.append(D_inv_arr)
            BD_inverse_lst.append(BD_inv_arr)

        T_arr = T_arr_new

    S_last, T_last, D_inv_last, BD_inv_last = _nosource_uniform_quad_merge_DtN(
        T_arr[0, 0],
        T_arr[0, 1],
        T_arr[0, 2],
        T_arr[0, 3],
    )

    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    D_inv_lst.append(
        jax.device_put(jnp.expand_dims(D_inv_last, axis=0), host_device)
    )
    BD_inverse_lst.append(
        jax.device_put(jnp.expand_dims(BD_inv_last, axis=0), host_device)
    )

    if return_T:
        T_last_out = jax.device_put(T_last, host_device)
        return (S_lst, D_inv_lst, BD_inverse_lst, T_last_out)
    else:
        return (S_lst, D_inv_lst, BD_inverse_lst)


@jax.jit
def _nosource_uniform_quad_merge_DtN(
    T_a: jax.Array,
    T_b: jax.Array,
    T_c: jax.Array,
    T_d: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_a = T_a.shape[0]
    dummy_h = jnp.zeros((n_a,), dtype=T_a.dtype)
    # First, find all of the necessary submatrices and sub-vectors
    (
        _,
        _,
        _,
        T_a_11,
        T_a_15,
        T_a_18,
        T_a_51,
        T_a_55,
        T_a_58,
        T_a_81,
        T_a_85,
        T_a_88,
    ) = get_quadmerge_blocks_a(T_a, dummy_h)

    (
        _,
        _,
        _,
        T_b_22,
        T_b_26,
        T_b_25,
        T_b_62,
        T_b_66,
        T_b_65,
        T_b_52,
        T_b_56,
        T_b_55,
    ) = get_quadmerge_blocks_b(T_b, dummy_h)

    (
        _,
        _,
        _,
        T_c_66,
        T_c_63,
        T_c_67,
        T_c_36,
        T_c_33,
        T_c_37,
        T_c_76,
        T_c_73,
        T_c_77,
    ) = get_quadmerge_blocks_c(T_c, dummy_h)

    (
        _,
        _,
        _,
        T_d_88,
        T_d_87,
        T_d_84,
        T_d_78,
        T_d_77,
        T_d_74,
        T_d_48,
        T_d_47,
        T_d_44,
    ) = get_quadmerge_blocks_d(T_d, dummy_h)

    n_int, n_ext = T_a_51.shape

    zero_block_ei = jnp.zeros((n_ext, n_int))
    zero_block_ie = jnp.zeros((n_int, n_ext))
    zero_block_ii = jnp.zeros((n_int, n_int))

    # A t_ext + B t_int = g_ext - h_ext
    B = jnp.block(
        [
            [T_a_15, zero_block_ei, zero_block_ei, T_a_18],
            [T_b_25, T_b_26, zero_block_ei, zero_block_ei],
            [zero_block_ei, T_c_36, T_c_37, zero_block_ei],
            [zero_block_ei, zero_block_ei, T_d_47, T_d_48],
        ]
    )
    C = jnp.block(
        [
            [T_a_51, T_b_52, zero_block_ie, zero_block_ie],
            [zero_block_ie, T_b_62, T_c_63, zero_block_ie],
            [zero_block_ie, zero_block_ie, T_c_73, T_d_74],
            [T_a_81, zero_block_ie, zero_block_ie, T_d_84],
        ]
    )

    D = jnp.block(
        [
            [T_a_55 + T_b_55, T_b_56, zero_block_ii, T_a_58],
            [T_b_65, T_b_66 + T_c_66, T_c_67, zero_block_ii],
            [zero_block_ii, T_c_76, T_c_77 + T_d_77, T_d_78],
            [T_a_85, zero_block_ii, T_d_87, T_d_88 + T_a_88],
        ]
    )
    A_lst = [T_a_11, T_b_22, T_c_33, T_d_44]

    T, S, D_inv, BD_inv = nosource_assemble_merge_outputs_DtN(A_lst, B, C, D)

    # Roll the exterior by n_int to get the correct ordering
    # of the exterior discretization points. Right now, the exterior points are ordered like [a_1, b_2, c_3, d_4]
    # but we want [bottom, left, top, right]. This requires
    # rolling the exterior points by n_int.
    T = jnp.roll(T, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=1)
    S = jnp.roll(S, -n_int, axis=1)

    return S, T, D_inv, BD_inv


_vmapped_nosource_uniform_quad_merge_DtN = jax.vmap(
    _nosource_uniform_quad_merge_DtN,
    in_axes=(0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_nosource_uniform_quad_merge_DtN(
    R_in: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    S, R, D_inv, BD_inv = _vmapped_nosource_uniform_quad_merge_DtN(
        R_in[:, 0],
        R_in[:, 1],
        R_in[:, 2],
        R_in[:, 3],
    )

    n_merges, n_int, n_ext = S.shape
    R = R.reshape((n_merges // 4, 4, n_ext, n_ext))
    return S, R, D_inv, BD_inv
