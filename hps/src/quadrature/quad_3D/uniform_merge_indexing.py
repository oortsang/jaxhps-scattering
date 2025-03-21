from typing import Tuple
import jax
import jax.numpy as jnp


##############################################
# These functions are for getting submatrices and subvectors
# of the DtN maps and v_prime arrays. These are used in the
# _oct_merge function.


@jax.jit
def get_a_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_1 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[2 * n_per_face : 3 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_9 = idxes[n_per_face : 2 * n_per_face]
    idxes_12 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_17 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(T, v, idxes_1, idxes_9, idxes_12, idxes_17)


@jax.jit
def get_b_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_2 = jnp.concatenate(
        [
            idxes[n_per_face : 3 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_9 = idxes[:n_per_face]
    idxes_10 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_18 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(T, v, idxes_2, idxes_9, idxes_10, idxes_18)


@jax.jit
def get_c_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_3 = jnp.concatenate(
        [
            idxes[n_per_face : 2 * n_per_face],
            idxes[3 * n_per_face : 4 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_10 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_11 = idxes[:n_per_face]
    idxes_19 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(T, v, idxes_3, idxes_10, idxes_11, idxes_19)


@jax.jit
def get_d_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_4 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[3 * n_per_face : 4 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_11 = idxes[n_per_face : 2 * n_per_face]
    idxes_12 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_20 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(T, v, idxes_4, idxes_11, idxes_12, idxes_20)


@jax.jit
def get_e_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_5 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[2 * n_per_face : 3 * n_per_face],
            idxes[4 * n_per_face : 5 * n_per_face],
        ]
    )
    idxes_13 = idxes[n_per_face : 2 * n_per_face]
    idxes_16 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_17 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(T, v, idxes_5, idxes_13, idxes_16, idxes_17)


@jax.jit
def get_f_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_6 = jnp.concatenate(
        [idxes[n_per_face : 3 * n_per_face], idxes[4 * n_per_face : 5 * n_per_face]]
    )
    idxes_13 = idxes[:n_per_face]
    idxes_14 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_18 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(T, v, idxes_6, idxes_13, idxes_14, idxes_18)


@jax.jit
def get_g_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_7 = jnp.concatenate(
        [idxes[n_per_face : 2 * n_per_face], idxes[3 * n_per_face : 5 * n_per_face]]
    )
    idxes_14 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_15 = idxes[:n_per_face]
    idxes_19 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(T, v, idxes_7, idxes_14, idxes_15, idxes_19)


@jax.jit
def get_h_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_8 = jnp.concatenate(
        [idxes[:n_per_face], idxes[3 * n_per_face : 5 * n_per_face]]
    )
    idxes_15 = idxes[n_per_face : 2 * n_per_face]
    idxes_16 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_20 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(T, v, idxes_8, idxes_15, idxes_16, idxes_20)


@jax.jit
def _return_submatrices_subvecs(
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


##############################################
# These functions are the last part of the _oct_merge
# function. It is designed to give the indices for re-arranging
# the outputs to comply with the expected ordering.


@jax.jit
def get_rearrange_indices(idxes: jnp.array, q_idxes: jnp.array) -> jnp.array:
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
    n_per_region = idxes.shape[0] // 8
    n_per_face = n_per_region // 3

    n_a = n_per_region
    region_a_idxes = idxes[:n_a]
    region_a_yz = region_a_idxes[:n_per_face]
    region_a_xz = region_a_idxes[n_per_face : 2 * n_per_face]
    region_a_xy = region_a_idxes[2 * n_per_face :]

    n_b = 2 * n_per_region
    region_b_idxes = idxes[n_a:n_b]
    region_b_yz = region_b_idxes[:n_per_face]
    region_b_xz = region_b_idxes[n_per_face : 2 * n_per_face]
    region_b_xy = region_b_idxes[2 * n_per_face :]

    n_c = 3 * n_per_region
    region_c_idxes = idxes[n_b:n_c]
    region_c_yz = region_c_idxes[:n_per_face]
    region_c_xz = region_c_idxes[n_per_face : 2 * n_per_face]
    region_c_xy = region_c_idxes[2 * n_per_face :]

    n_d = 4 * n_per_region
    region_d_idxes = idxes[n_c:n_d]
    region_d_yz = region_d_idxes[:n_per_face]
    region_d_xz = region_d_idxes[n_per_face : 2 * n_per_face]
    region_d_xy = region_d_idxes[2 * n_per_face :]

    n_e = 5 * n_per_region
    region_e_idxes = idxes[n_d:n_e]
    region_e_yz = region_e_idxes[:n_per_face]
    region_e_xz = region_e_idxes[n_per_face : 2 * n_per_face]
    region_e_xy = region_e_idxes[2 * n_per_face :]

    n_f = 6 * n_per_region
    region_f_idxes = idxes[n_e:n_f]
    region_f_yz = region_f_idxes[:n_per_face]
    region_f_xz = region_f_idxes[n_per_face : 2 * n_per_face]
    region_f_xy = region_f_idxes[2 * n_per_face :]

    n_g = 7 * n_per_region
    region_g_idxes = idxes[n_f:n_g]
    region_g_yz = region_g_idxes[:n_per_face]
    region_g_xz = region_g_idxes[n_per_face : 2 * n_per_face]
    region_g_xy = region_g_idxes[2 * n_per_face :]

    region_h_idxes = idxes[n_g:]
    region_h_yz = region_h_idxes[:n_per_face]
    region_h_xz = region_h_idxes[n_per_face : 2 * n_per_face]
    region_h_xy = region_h_idxes[2 * n_per_face :]

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
