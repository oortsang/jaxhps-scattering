from functools import partial
from typing import Tuple, List
import numpy as np
import jax
import jax.numpy as jnp


# @partial(jax.jit, static_argnums=(4, 5, 6))
def get_a_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_9: jnp.array,
    need_interp_12: jnp.array,
    need_interp_17: jnp.array,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:
    idxes_1 = jnp.concatenate(
        [
            jnp.arange(n_0),
            jnp.arange(n_0 + n_1, n_0 + n_1 + n_2),
            jnp.arange(T.shape[0] - n_5, T.shape[0]),
        ]
    )
    idxes_9 = jnp.arange(n_0, n_0 + n_1)
    idxes_12 = jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3)
    idxes_17 = jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4)

    # n_pts_per_panel = L_2f1.shape[1]
    # n_panels_1 = idxes_1.shape[0] // n_pts_per_panel
    # need_interp_1 = jnp.full((n_panels_1,), False)

    # print("get_a_submatrices: idxes_1.shape = ", idxes_1.shape)
    # print("get_a_submatrices: idxes_9.shape = ", idxes_9.shape)
    # print("get_a_submatrices: idxes_12.shape = ", idxes_12.shape)
    # print("get_a_submatrices: idxes_17.shape = ", idxes_17.shape)
    # print("get_a_submatrices: L_2f1.shape = ", L_2f1.shape)
    # print("get_a_submatrices: L_1f2.shape = ", L_1f2.shape)
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_1,
        idxes_9,
        idxes_12,
        idxes_17,
        L_2f1,
        L_1f2,
        need_interp_9,
        need_interp_12,
        need_interp_17,
    )


def get_b_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_9: bool,
    need_interp_10: bool,
    need_interp_18: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:
    idxes_2 = jnp.concatenate(
        [jnp.arange(n_0, n_0 + n_1 + n_2), jnp.arange(T.shape[0] - n_5, T.shape[0])]
    )
    idxes_9 = jnp.arange(n_0)
    idxes_10 = jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3)
    idxes_18 = jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4)
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_2,
        idxes_9,
        idxes_10,
        idxes_18,
        L_2f1,
        L_1f2,
        need_interp_9,
        need_interp_10,
        need_interp_18,
    )


def get_c_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_10: bool,
    need_interp_11: bool,
    need_interp_19: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:
    idxes_3 = jnp.concatenate(
        [
            jnp.arange(n_0, n_0 + n_1),
            jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3),
            jnp.arange(T.shape[0] - n_5, T.shape[0]),
        ]
    )
    idxes_10 = jnp.arange(n_0 + n_1, n_0 + n_1 + n_2)
    idxes_11 = jnp.arange(n_0)
    idxes_19 = jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4)
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_3,
        idxes_10,
        idxes_11,
        idxes_19,
        L_2f1,
        L_1f2,
        need_interp_10,
        need_interp_11,
        need_interp_19,
    )


def get_d_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_11: bool,
    need_interp_12: bool,
    need_interp_20: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:

    idxes_4 = jnp.concatenate(
        [
            jnp.arange(n_0),
            jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3),
            jnp.arange(T.shape[0] - n_5, T.shape[0]),
        ]
    )

    idxes_11 = jnp.arange(n_0, n_0 + n_1)
    idxes_12 = jnp.arange(n_0 + n_1, n_0 + n_1 + n_2)
    idxes_20 = jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4)
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_4,
        idxes_11,
        idxes_12,
        idxes_20,
        L_2f1,
        L_1f2,
        need_interp_11,
        need_interp_12,
        need_interp_20,
    )


def get_e_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_13: bool,
    need_interp_16: bool,
    need_interp_17: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:

    idxes_5 = jnp.concatenate(
        [
            jnp.arange(n_0),
            jnp.arange(n_0 + n_1, n_0 + n_1 + n_2),
            jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4),
        ]
    )
    idxes_13 = jnp.arange(n_0, n_0 + n_1)
    idxes_16 = jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3)
    idxes_17 = jnp.arange(T.shape[0] - n_5, T.shape[0])

    return _return_submatrices_subvecs(
        T,
        v,
        idxes_5,
        idxes_13,
        idxes_16,
        idxes_17,
        L_2f1,
        L_1f2,
        need_interp_13,
        need_interp_16,
        need_interp_17,
    )


def get_f_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_13: bool,
    need_interp_14: bool,
    need_interp_18: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:
    idxes_6 = jnp.concatenate(
        [
            jnp.arange(n_0, n_0 + n_1 + n_2),
            jnp.arange(n_0 + n_1 + n_2 + n_3, n_0 + n_1 + n_2 + n_3 + n_4),
        ]
    )
    idxes_13 = jnp.arange(n_0)
    idxes_14 = jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3)
    idxes_18 = jnp.arange(T.shape[0] - n_5, T.shape[0])
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_6,
        idxes_13,
        idxes_14,
        idxes_18,
        L_2f1,
        L_1f2,
        need_interp_13,
        need_interp_14,
        need_interp_18,
    )


def get_g_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_14: bool,
    need_interp_15: bool,
    need_interp_19: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:

    idxes_7 = jnp.concatenate(
        [
            jnp.arange(n_0, n_0 + n_1),
            jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3 + n_4),
        ]
    )
    idxes_14 = jnp.arange(n_0 + n_1, n_0 + n_1 + n_2)
    idxes_15 = jnp.arange(n_0)
    idxes_19 = jnp.arange(T.shape[0] - n_5, T.shape[0])
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_7,
        idxes_14,
        idxes_15,
        idxes_19,
        L_2f1,
        L_1f2,
        need_interp_14,
        need_interp_15,
        need_interp_19,
    )


def get_h_submatrices(
    T: jnp.array,
    v: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_15: bool,
    need_interp_16: bool,
    need_interp_20: bool,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
    n_4: int,
    n_5: int,
) -> Tuple[jnp.array]:

    idxes_8 = jnp.concatenate(
        [
            jnp.arange(n_0),
            jnp.arange(n_0 + n_1 + n_2, n_0 + n_1 + n_2 + n_3 + n_4),
        ]
    )
    idxes_15 = jnp.arange(n_0, n_0 + n_1)
    idxes_16 = jnp.arange(n_0 + n_1, n_0 + n_1 + n_2)
    idxes_20 = jnp.arange(T.shape[0] - n_5, T.shape[0])
    return _return_submatrices_subvecs(
        T,
        v,
        idxes_8,
        idxes_15,
        idxes_16,
        idxes_20,
        L_2f1,
        L_1f2,
        need_interp_15,
        need_interp_16,
        need_interp_20,
    )


@jax.jit
def _get_from_idxes(
    T: jnp.array,
    v: jnp.array,
    idxes_0: jnp.array,
    idxes_1: jnp.array,
    idxes_2: jnp.array,
    idxes_3: jnp.array,
) -> Tuple[jnp.array]:
    T_0_0 = T[idxes_0][:, idxes_0]
    T_0_1 = T[idxes_0][:, idxes_1]
    T_0_2 = T[idxes_0][:, idxes_2]
    T_0_3 = T[idxes_0][:, idxes_3]
    T_1_0 = T[idxes_1][:, idxes_0]
    T_1_1 = T[idxes_1][:, idxes_1]
    T_1_2 = T[idxes_1][:, idxes_2]
    T_1_3 = T[idxes_1][:, idxes_3]
    T_2_0 = T[idxes_2][:, idxes_0]
    T_2_1 = T[idxes_2][:, idxes_1]
    T_2_2 = T[idxes_2][:, idxes_2]
    T_2_3 = T[idxes_2][:, idxes_3]
    T_3_0 = T[idxes_3][:, idxes_0]
    T_3_1 = T[idxes_3][:, idxes_1]
    T_3_2 = T[idxes_3][:, idxes_2]
    T_3_3 = T[idxes_3][:, idxes_3]
    v_0 = v[idxes_0]
    v_1 = v[idxes_1]
    v_2 = v[idxes_2]
    v_3 = v[idxes_3]
    return (
        T_0_0,
        T_0_1,
        T_0_2,
        T_0_3,
        T_1_0,
        T_1_1,
        T_1_2,
        T_1_3,
        T_2_0,
        T_2_1,
        T_2_2,
        T_2_3,
        T_3_0,
        T_3_1,
        T_3_2,
        T_3_3,
        v_0,
        v_1,
        v_2,
        v_3,
    )


def _return_submatrices_subvecs(
    T: jnp.array,
    v: jnp.array,
    idxes_0: jnp.array,
    idxes_1: jnp.array,
    idxes_2: jnp.array,
    idxes_3: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_1: jnp.array,
    need_interp_2: jnp.array,
    need_interp_3: jnp.array,
) -> Tuple[jnp.array]:

    (
        T_0_0,
        T_0_1,
        T_0_2,
        T_0_3,
        T_1_0,
        T_1_1,
        T_1_2,
        T_1_3,
        T_2_0,
        T_2_1,
        T_2_2,
        T_2_3,
        T_3_0,
        T_3_1,
        T_3_2,
        T_3_3,
        v_0,
        v_1,
        v_2,
        v_3,
    ) = _get_from_idxes(T, v, idxes_0, idxes_1, idxes_2, idxes_3)

    if jnp.any(need_interp_1):
        # Do all of the compression ops specified by need_interp_1:
        T_0_1 = _compress_cols_from_lst(T_0_1, L_2f1, need_interp_1)
        T_1_0 = _compress_rows_from_lst(T_1_0, L_1f2, need_interp_1)
        T_1_1 = _compress_rows_from_lst(T_1_1, L_1f2, need_interp_1)
        T_1_1 = _compress_cols_from_lst(T_1_1, L_2f1, need_interp_1)
        T_1_2 = _compress_rows_from_lst(T_1_2, L_1f2, need_interp_1)
        T_1_3 = _compress_rows_from_lst(T_1_3, L_1f2, need_interp_1)
        T_2_1 = _compress_cols_from_lst(T_2_1, L_2f1, need_interp_1)
        T_3_1 = _compress_cols_from_lst(T_3_1, L_2f1, need_interp_1)
        v_1 = _compress_rows_from_lst(v_1, L_1f2, need_interp_1)

    if jnp.any(need_interp_2):
        # Do all of the compression ops specified by need_interp_2:
        T_0_2 = _compress_cols_from_lst(T_0_2, L_2f1, need_interp_2)
        T_1_2 = _compress_cols_from_lst(T_1_2, L_2f1, need_interp_2)
        T_2_0 = _compress_rows_from_lst(T_2_0, L_1f2, need_interp_2)
        T_2_1 = _compress_rows_from_lst(T_2_1, L_1f2, need_interp_2)
        T_2_2 = _compress_rows_from_lst(T_2_2, L_1f2, need_interp_2)
        T_2_2 = _compress_cols_from_lst(T_2_2, L_2f1, need_interp_2)
        T_2_3 = _compress_rows_from_lst(T_2_3, L_1f2, need_interp_2)
        T_3_2 = _compress_cols_from_lst(T_3_2, L_2f1, need_interp_2)
        v_2 = _compress_rows_from_lst(v_2, L_1f2, need_interp_2)

    if jnp.any(need_interp_3):
        # Do all of the compression ops specified by need_interp_3:
        T_0_3 = _compress_cols_from_lst(T_0_3, L_2f1, need_interp_3)
        T_1_3 = _compress_cols_from_lst(T_1_3, L_2f1, need_interp_3)
        T_2_3 = _compress_cols_from_lst(T_2_3, L_2f1, need_interp_3)
        T_3_0 = _compress_rows_from_lst(T_3_0, L_1f2, need_interp_3)
        T_3_1 = _compress_rows_from_lst(T_3_1, L_1f2, need_interp_3)
        T_3_2 = _compress_rows_from_lst(T_3_2, L_1f2, need_interp_3)
        T_3_3 = _compress_rows_from_lst(T_3_3, L_1f2, need_interp_3)
        T_3_3 = _compress_cols_from_lst(T_3_3, L_2f1, need_interp_3)
        v_3 = _compress_rows_from_lst(v_3, L_1f2, need_interp_3)

    return (
        T_0_0,
        T_0_1,
        T_0_2,
        T_0_3,
        T_1_0,
        T_1_1,
        T_1_2,
        T_1_3,
        T_2_0,
        T_2_1,
        T_2_2,
        T_2_3,
        T_3_0,
        T_3_1,
        T_3_2,
        T_3_3,
        v_0,
        v_1,
        v_2,
        v_3,
    )


def _compress_cols_from_lst(
    T: jnp.array,
    L: jnp.array,
    compression_bools: jnp.array,
) -> jnp.array:
    """
    Suppose compression_bools has length (n,). Then, we expect T to be a block matrix
    T = [T_1 T_2, ..., T_n].

    We iterate through compression_bools, and if compression_bools[i] :
    We replace [T_i ...  T_{i+3}] with [T_i ... T_{i+3} ] @ L
    """

    n_per_panel = L.shape[1]
    n = compression_bools.shape[0]
    out_lst = []
    i = 0
    while i < n:
        if compression_bools[i]:
            X = _compress_col(L, T, i, n_per_panel)
            i += 4
        else:
            X = T[:, i * n_per_panel : (i + 1) * n_per_panel]
            i += 1
        out_lst.append(X)
    out_T = jnp.concatenate(out_lst, axis=1)
    return out_T


def _compress_rows_from_lst(
    T: jnp.array,
    L: jnp.array,
    compression_bools: jnp.array,
) -> jnp.array:
    """
    Suppose compression_bools has length (n,). Then we expect T to be a block matrix
    T = [T_1
         T_2
         ...
         T_n]

    We iterate through compression_bools, and if compression_bools[i] :
    We replace [T_i
                ...
                 T_{i+3}]
            with L @ [T_i
                        ...
                        T_{i+3}]
    """

    n_per_panel = L.shape[0]
    n = compression_bools.shape[0]
    out_lst = []
    i = 0
    while i < n:
        if compression_bools[i]:
            X = _compress_row(L, T, i, n_per_panel)
            i += 4
        else:
            X = T[i * n_per_panel : (i + 1) * n_per_panel]
            i += 1
        out_lst.append(X)
    out_T = jnp.concatenate(out_lst, axis=0)
    return out_T


@partial(jax.jit, static_argnums=(3,))
def _compress_col(L: jnp.array, T: jnp.array, i: int, n_per_panel: int) -> jnp.array:
    a, b = T.shape
    X = jax.lax.dynamic_slice(T, (0, i * n_per_panel), (a, 4 * n_per_panel))

    return X @ L


@partial(jax.jit, static_argnums=(3,))
def _compress_row(L: jnp.array, T: jnp.array, i: int, n_per_panel: int) -> jnp.array:
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    b = T.shape[1]
    X = jax.lax.dynamic_slice(T, (i * n_per_panel, 0), (4 * n_per_panel, b))

    if b == 1:
        out = L @ X
        out = out.reshape(-1)
    else:
        out = L @ X

    return out


##############################################
# These functions are the last part of the _oct_merge
# function. It is designed to give the indices for re-arranging
# the outputs to comply with the expected ordering.
def get_rearrange_indices(
    idxes: jnp.array,
    n_a_0: int,
    n_a_2: int,
    n_a_5: int,
    n_b_1: int,
    n_b_2: int,
    n_b_5: int,
    n_c_1: int,
    n_c_3: int,
    n_c_5: int,
    n_d_0: int,
    n_d_3: int,
    n_d_5: int,
    n_e_0: int,
    n_e_2: int,
    n_e_4: int,
    n_f_1: int,
    n_f_2: int,
    n_f_4: int,
    n_g_1: int,
    n_g_3: int,
    n_g_4: int,
    n_h_0: int,
    n_h_3: int,
    n_h_4: int,
) -> jnp.array:
    """
    After the _oct_merge computation, the outputs are ordered
    [region_1, ..., region_8]. This function returns the indices
    which will re-arrange the outputs to be in the order
    [face 1, ..., face 6], where the quad points in each face
    are ordered in column-first order


    Args:
        idxes (jnp.array): Has shape (24 * q**2,)

    Returns:
        jnp.array: Has shape (24 * q**2,)
    """
    # n_per_region = idxes.shape[0] // 8
    # n_per_region_face = n_per_region // 3

    n_a = n_a_0 + n_a_2 + n_a_5
    region_a_idxes = idxes[:n_a]
    region_a_yz = region_a_idxes[:n_a_0]
    region_a_xz = region_a_idxes[n_a_0 : n_a_0 + n_a_2]
    region_a_xy = region_a_idxes[n_a_0 + n_a_2 :]

    n_b = n_b_1 + n_b_2 + n_b_5
    region_b_idxes = idxes[n_a : n_a + n_b]
    region_b_yz = region_b_idxes[:n_b_1]
    region_b_xz = region_b_idxes[n_b_1 : n_b_1 + n_b_2]
    region_b_xy = region_b_idxes[n_b_1 + n_b_2 :]

    n_c = n_c_1 + n_c_3 + n_c_5
    region_c_idxes = idxes[n_a + n_b : n_a + n_b + n_c]
    region_c_yz = region_c_idxes[:n_c_1]
    region_c_xz = region_c_idxes[n_c_1 : n_c_1 + n_c_3]
    region_c_xy = region_c_idxes[n_c_1 + n_c_3 :]

    n_d = n_d_0 + n_d_3 + n_d_5
    region_d_idxes = idxes[n_a + n_b + n_c : n_a + n_b + n_c + n_d]
    region_d_yz = region_d_idxes[:n_d_0]
    region_d_xz = region_d_idxes[n_d_0 : n_d_0 + n_d_3]
    region_d_xy = region_d_idxes[n_d_0 + n_d_3 :]

    n_e = n_e_0 + n_e_2 + n_e_4
    region_e_idxes = idxes[n_a + n_b + n_c + n_d : n_a + n_b + n_c + n_d + n_e]
    region_e_yz = region_e_idxes[:n_e_0]
    region_e_xz = region_e_idxes[n_e_0 : n_e_0 + n_e_2]
    region_e_xy = region_e_idxes[n_e_0 + n_e_2 :]

    n_f = n_f_1 + n_f_2 + n_f_4
    region_f_idxes = idxes[
        n_a + n_b + n_c + n_d + n_e : n_a + n_b + n_c + n_d + n_e + n_f
    ]
    region_f_yz = region_f_idxes[:n_f_1]
    region_f_xz = region_f_idxes[n_f_1 : n_f_1 + n_f_2]
    region_f_xy = region_f_idxes[n_f_1 + n_f_2 :]

    n_g = n_g_1 + n_g_3 + n_g_4
    region_g_idxes = idxes[
        n_a + n_b + n_c + n_d + n_e + n_f : n_a + n_b + n_c + n_d + n_e + n_f + n_g
    ]
    region_g_yz = region_g_idxes[:n_g_1]
    region_g_xz = region_g_idxes[n_g_1 : n_g_1 + n_g_3]
    region_g_xy = region_g_idxes[n_g_1 + n_g_3 :]

    n_h = n_h_0 + n_h_3 + n_h_4
    region_h_idxes = idxes[n_a + n_b + n_c + n_d + n_e + n_f + n_g :]
    region_h_yz = region_h_idxes[:n_h_0]
    region_h_xz = region_h_idxes[n_h_0 : n_h_0 + n_h_3]
    region_h_xy = region_h_idxes[n_h_0 + n_h_3 :]

    out = jnp.concatenate(
        [
            # Face 0 [e, h, d, a]
            region_e_yz,
            region_h_yz,
            region_d_yz,
            region_a_yz,
            # Face 1 [f, g, c, b]
            region_f_yz,
            region_g_yz,
            region_c_yz,
            region_b_yz,
            # Face 2 [e, f, b, a]
            region_e_xz,
            region_f_xz,
            region_b_xz,
            region_a_xz,
            # Face 3 [h, g, c, d]
            region_h_xz,
            region_g_xz,
            region_c_xz,
            region_d_xz,
            # Face 4
            region_e_xy,
            region_f_xy,
            region_g_xy,
            region_h_xy,
            # Face 5
            region_a_xy,
            region_b_xy,
            region_c_xy,
            region_d_xy,
        ]
    )
    return out
