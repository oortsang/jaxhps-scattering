from typing import Tuple, List

import jax
import jax.numpy as jnp


from ._schur_complement import (
    assemble_merge_outputs_ItI,
)
from ._uniform_2D_DtN import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)
import logging


def merge_stage_uniform_2D_ItI(
    T_arr: jnp.array,
    h_arr: jnp.array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    subtree_recomp: bool = False,
    return_T: bool = False,
    return_h: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Implements uniform 2D merges of ItI matrices. Merges the nodes in the quadtree four at a time.
    This function uses a Schur complement strategy to reduce the size of matrix inverted in each merge operation.
    This function returns lists containing :math:`S` and :math:`\\tilde{g}`, giving enough information
    to propagate boundary data back down the tree in a later part of the algorithm.

    If this function is called with the argument ``return_T=True``, the top-level DtN matrix is also returned.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    T_arr : jax.Array
        Array of ItI matrices from the local solve stage. Has shape (n_leaves, 4q, 4q)

    h_arr : jax.Array
        Array of outgoing boundary data from the local solve stage. Has shape (n_leaves, 4q) or (n_leaves, 4q, nsrc)

    l : int
        Number of levels to merge

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    subtree_recomp : bool
        A flag for used by the subtree recomputation methods, which triggers an early return with just the top-level T and h.

    return_T : bool
        A flag to return the top-level T matrix. Defaults to False.

    Returns
    -------
    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst: List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    T_last : jax.Array
        The top-level DtN matrix, which is only returned if ``return_T=True``. Has shape (4q, 4q).

    """
    logging.debug("merge_stage_uniform_2D_ItI: started. device=%s", device)

    if not subtree_recomp:
        # Start lists to output data
        S_lst = []
        g_tilde_lst = []

    T_arr = jax.device_put(T_arr, device)
    h_arr = jax.device_put(h_arr, device)

    bool_multi_source = h_arr.ndim == 3

    if len(T_arr.shape) < 4:
        logging.debug(
            "merge_stage_uniform_2D_ItI: T_arr.shape = %s", T_arr.shape
        )
        n_leaves, n_ext, _ = T_arr.shape
        T_arr = T_arr.reshape(n_leaves // 4, 4, n_ext, n_ext)
        if bool_multi_source:
            n_src = h_arr.shape[-1]
            h_arr = h_arr.reshape(n_leaves // 4, 4, n_ext, n_src)
        else:
            h_arr = h_arr.reshape(n_leaves // 4, 4, n_ext, 1)

    for i in range(l - 1):
        S_arr, T_arr_new, h_arr_new, g_tilde_arr = (
            vmapped_uniform_quad_merge_ItI(T_arr, h_arr)
        )

        # TODO: Figure out how to safely delete these arrays
        # when using autodiff.

        # T_arr.delete()
        # h_arr.delete()

        if not bool_multi_source:
            # Remove source dimension from g_tilde_arr
            g_tilde_arr = jnp.squeeze(g_tilde_arr, axis=-1)

        if host_device != device:
            if not subtree_recomp:
                S_host = jax.device_put(S_arr, host_device)
                S_lst.append(S_host)
                g_tilde_host = jax.device_put(g_tilde_arr, host_device)
                g_tilde_lst.append(g_tilde_host)

            S_arr.delete()
            g_tilde_arr.delete()
        elif not subtree_recomp:
            S_lst.append(S_arr)
            g_tilde_lst.append(g_tilde_arr)

        T_arr = T_arr_new
        h_arr = h_arr_new

    S_last, T_last, h_last, g_tilde_last = _uniform_quad_merge_ItI(
        T_arr[0, 0],
        T_arr[0, 1],
        T_arr[0, 2],
        T_arr[0, 3],
        h_arr[0, 0],
        h_arr[0, 1],
        h_arr[0, 2],
        h_arr[0, 3],
    )
    if not bool_multi_source:
        # Remove source dimension from g_tilde_last and h_last
        g_tilde_last = jnp.squeeze(g_tilde_last, axis=-1)
        h_last = jnp.squeeze(h_last, axis=-1)

    if subtree_recomp:
        # In this branch, we only return T_last and h_last
        S_last.delete()
        g_tilde_last.delete()

        # Expand the dimensions of T_last and h_last so they stack nicely
        T_last = jnp.expand_dims(T_last, 0)
        h_last = jnp.expand_dims(h_last, 0)

        # Move the data to the requested device
        T_last_out = jax.device_put(T_last, host_device)
        h_last_out = jax.device_put(h_last, host_device)

        return (T_last_out, h_last_out)

    else:
        # In this branch, we are returning S_lst, g_tilde_lst, and optionally
        # T_last
        S_lst.append(
            jax.device_put(jnp.expand_dims(S_last, axis=0), host_device)
        )
        g_tilde_lst.append(
            jax.device_put(jnp.expand_dims(g_tilde_last, axis=0), host_device)
        )
        out = (S_lst, g_tilde_lst)

        if return_T:
            T_last_out = jax.device_put(T_last, host_device)
            out = out + (T_last_out,)

        if return_h:
            h_last_out = jax.device_put(h_last, host_device)
            out = out + (h_last_out,)

        return out


@jax.jit
def _uniform_quad_merge_ItI(
    R_a: jnp.array,
    R_b: jnp.array,
    R_c: jnp.array,
    R_d: jnp.array,
    h_a: jnp.array,
    h_b: jnp.array,
    h_c: jnp.array,
    h_d: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    # print("_quad_merge_ItI: h_a", h_a.shape)
    # First, find all of the necessary submatrices and sub-vectors
    (
        h_a_1,
        h_a_5,
        h_a_8,
        R_a_11,
        R_a_15,
        R_a_18,
        R_a_51,
        R_a_55,
        R_a_58,
        R_a_81,
        R_a_85,
        R_a_88,
    ) = get_quadmerge_blocks_a(R_a, h_a)

    (
        h_b_2,
        h_b_6,
        h_b_5,
        R_b_22,
        R_b_26,
        R_b_25,
        R_b_62,
        R_b_66,
        R_b_65,
        R_b_52,
        R_b_56,
        R_b_55,
    ) = get_quadmerge_blocks_b(R_b, h_b)

    (
        h_c_6,
        h_c_3,
        h_c_7,
        R_c_66,
        R_c_63,
        R_c_67,
        R_c_36,
        R_c_33,
        R_c_37,
        R_c_76,
        R_c_73,
        R_c_77,
    ) = get_quadmerge_blocks_c(R_c, h_c)

    (
        h_d_8,
        h_d_7,
        h_d_4,
        R_d_88,
        R_d_87,
        R_d_84,
        R_d_78,
        R_d_77,
        R_d_74,
        R_d_48,
        R_d_47,
        R_d_44,
    ) = get_quadmerge_blocks_d(R_d, h_d)

    n_int, n_ext = R_a_51.shape

    zero_block_ei = jnp.zeros((n_ext, n_int))
    zero_block_ie = jnp.zeros((n_int, n_ext))
    zero_block_ii = jnp.zeros((n_int, n_int))

    # print("_quad_merge_ItI: h_a_1", h_a_1.shape)

    # A t_ext + B t_int = g_ext - h_ext
    B = jnp.block(
        [
            [
                R_a_15,
                R_a_18,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                R_b_25,
                R_b_26,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                R_c_36,
                R_c_37,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                R_d_47,
                R_d_48,
            ],
        ]
    )

    # C t_ext + D t_int + h_int = 0
    C = jnp.block(
        [
            [zero_block_ie, R_b_52, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, zero_block_ie, R_d_84],
            [zero_block_ie, R_b_62, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, zero_block_ie, R_d_74],
            [R_a_51, zero_block_ie, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, R_c_63, zero_block_ie],
            [zero_block_ie, zero_block_ie, R_c_73, zero_block_ie],
            [R_a_81, zero_block_ie, zero_block_ie, zero_block_ie],
        ]
    )

    D_12 = jnp.block(
        [
            [R_b_55, R_b_56, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_d_87, R_d_88],
            [R_b_65, R_b_66, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_d_77, R_d_78],
        ]
    )

    D_21 = jnp.block(
        [
            [R_a_55, R_a_58, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_c_66, R_c_67],
            [zero_block_ii, zero_block_ii, R_c_76, R_c_77],
            [R_a_85, R_a_88, zero_block_ii, zero_block_ii],
        ]
    )

    h_int = jnp.concatenate(
        [h_b_5, h_d_8, h_b_6, h_d_7, h_a_5, h_c_6, h_c_7, h_a_8]
    )
    h_ext = jnp.concatenate([h_a_1, h_b_2, h_c_3, h_d_4])
    A_lst = [R_a_11, R_b_22, R_c_33, R_d_44]

    T, S, h_ext_out, g_tilde_int = assemble_merge_outputs_ItI(
        A_lst, B, C, D_12, D_21, h_ext, h_int
    )

    # Roll the exterior by n_int to get the correct ordering
    h_ext_out = jnp.roll(h_ext_out, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=1)
    S = jnp.roll(S, -n_int, axis=1)

    # rows of S are ordered like a_5, a_8, c_6, c_7, b_5, b_6, d_7, d_8.
    # Want to rearrange them so they are ordered like
    # a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8
    r = jnp.concatenate(
        [
            jnp.arange(n_int),  # a5
            jnp.arange(4 * n_int, 5 * n_int),  # b5
            jnp.arange(5 * n_int, 6 * n_int),  # b6
            jnp.arange(2 * n_int, 3 * n_int),  # c6
            jnp.arange(3 * n_int, 4 * n_int),  # c7
            jnp.arange(6 * n_int, 7 * n_int),  # d7
            jnp.arange(7 * n_int, 8 * n_int),  # d8
            jnp.arange(n_int, 2 * n_int),  # a8
        ]
    )
    S = S[r]
    g_tilde_int = g_tilde_int[r]

    return S, T, h_ext_out, g_tilde_int


_vmapped_uniform_quad_merge_ItI = jax.vmap(
    _uniform_quad_merge_ItI,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_quad_merge_ItI(
    R_in: jnp.array,
    h_in: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    S, R, h, f = _vmapped_uniform_quad_merge_ItI(
        R_in[:, 0],
        R_in[:, 1],
        R_in[:, 2],
        R_in[:, 3],
        h_in[:, 0],
        h_in[:, 1],
        h_in[:, 2],
        h_in[:, 3],
    )

    n_merges, n_int, n_ext = S.shape
    n_src = h.shape[-1]
    R = R.reshape((n_merges // 4, 4, n_ext, n_ext))
    h = h.reshape((n_merges // 4, 4, n_ext, n_src))
    return S, R, h, f
