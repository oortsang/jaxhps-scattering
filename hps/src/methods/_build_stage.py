import logging
from typing import Tuple, List

import jax
import jax.numpy as jnp

from hps.src.quadrature.quad_3D.indexing import (
    get_a_submatrices,
    get_b_submatrices,
    get_c_submatrices,
    get_d_submatrices,
    get_e_submatrices,
    get_f_submatrices,
    get_g_submatrices,
    get_h_submatrices,
    get_rearrange_indices,
)

from hps.src.config import HOST_DEVICE, DEVICE_ARR


def _build_stage_3D(
    DtN_maps: jnp.ndarray,
    v_prime_arr: jnp.ndarray,
    l: int,
    device: jax.Device = DEVICE_ARR[0],
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    logging.debug("_build_stage_3D: started")

    DtN_arr = DtN_maps
    S_lst = []
    DtN_lst = []
    v_int_lst = []

    logging.debug("_build_stage_3D: DtN_maps.device %s", DtN_maps.devices())

    DtN_arr = jax.device_put(DtN_arr, device)
    v_prime_arr = jax.device_put(v_prime_arr, device)
    logging.debug("_build_stage_3D: DtN_arr.device %s", DtN_arr.devices())

    q = int(jnp.sqrt(DtN_arr.shape[-1] // 6))

    for i in range(l - 1):
        logging.debug("_build_stage_3D: starting with level %i", i)
        logging.debug("_build_stage_3D: DtN_arr.shape = %s", DtN_arr.shape)

        S_arr, DtN_arr, v_prime_arr, v_int_arr = vmapped_oct_merge(
            jnp.arange(q), DtN_arr, v_prime_arr
        )
        S_lst.append(jax.device_put(S_arr, HOST_DEVICE))
        v_int_lst.append(jax.device_put(v_int_arr, HOST_DEVICE))
        q = int(jnp.sqrt(DtN_arr.shape[-1] // 6))
        del S_arr, v_int_arr
        DtN_lst.append(jax.device_put(DtN_arr, HOST_DEVICE))

    logging.debug("_build_stage_3D: done with the loop. Doing final merge.")
    logging.debug("_build_stage_3D: DtN_arr.shape = %s", DtN_arr.shape)
    # Do the last oct-merge without the reshaping operation.
    S_last, T_last, v_prime_last, v_int_last = _oct_merge(
        jnp.arange(q),
        DtN_arr[0, 0],
        DtN_arr[0, 1],
        DtN_arr[0, 2],
        DtN_arr[0, 3],
        DtN_arr[0, 4],
        DtN_arr[0, 5],
        DtN_arr[0, 6],
        DtN_arr[0, 7],
        v_prime_arr[0, 0],
        v_prime_arr[0, 1],
        v_prime_arr[0, 2],
        v_prime_arr[0, 3],
        v_prime_arr[0, 4],
        v_prime_arr[0, 5],
        v_prime_arr[0, 6],
        v_prime_arr[0, 7],
    )
    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), HOST_DEVICE))
    v_int_lst.append(jax.device_put(jnp.expand_dims(v_int_last, axis=0), HOST_DEVICE))
    DtN_lst.append(jax.device_put(T_last, HOST_DEVICE))

    # logging.debug("_build_stage: done with merging.")
    return S_lst, DtN_lst, v_int_lst


def _build_stage_2D(
    DtN_maps: jnp.array,
    v_prime_arr: jnp.array,
    l: int,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    return_fused_info: bool = False,
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    Implements the build stage of the HPS algorithm for 2D problems. Given a list of
    leaf-level DtN matrices and a list of leaf-level particular solution boundary fluxes,
    this function recursively merges the DtN matrices and the particular solution boundary
    fluxes to produce the DtN matrices and the particular solution boundary fluxes at each
    level of the quadtree.

    Expects input data living on the CPU; will move data to the specified device and return
    data to the CPU.

    Args:
        DtN_maps (jnp.array): Has shape (n_quad_merges, 4, n_ext, n_ext)
        v_prime_arr (jnp.array): Has shape (n_quad_merges, 4, n_ext)
        l (int): Number of levels in the quadtree. 4**(l-1) = n_quad_merges.
        device (jax.Device): Where to perform computation.

    Returns:
        Tuple[List[jnp.array], List[jnp.array], List[jnp.array]]:
            S_lst: List of S matrices at each level of the quadtree. Each S matrix has shape (n_quad_merges, n_int, n_ext).
            DtN_lst: List of DtN matrices at each level of the quadtree. Each DtN matrix has shape (n_quad_merges, 4, n_ext, n_ext).
            v_int_lst: List of particular solution boundary fluxes at each level of the quadtree. Each v_int matrix has shape (n_quad_merges, 4, n_int).
    """
    logging.debug("_build_stage_2D: started")

    DtN_arr = DtN_maps
    # logging.debug("_build_stage_2D: input DtN_maps.device %s", DtN_arr.devices())
    logging.debug("_build_stage_2D: host_device: %s", host_device)
    # print("_build_stage_2D: device: ", device)
    # print("_build_stage_2D: host_device: ", host_device)

    # Move the data to the compute device if necessary
    DtN_arr = jax.device_put(DtN_arr, device)
    v_prime_arr = jax.device_put(v_prime_arr, device)

    # Start lists to store S, DtN, and v_int arrays
    S_lst = []
    DtN_lst = []
    v_int_lst = []

    # Working on merging the merge pairs at level i
    for i in range(l - 1):
        logging.debug("_build_stage_2D: i: %i", i)

        (
            S_arr,
            DtN_arr_new,
            v_prime_arr_new,
            v_int_arr,
        ) = vmapped_quad_merge(DtN_arr, v_prime_arr)
        DtN_arr.delete()
        v_prime_arr.delete()
        # Only do these copies and GPU -> CPU moves if necessary
        if host_device != device:
            logging.debug("_build_stage_2D: Moving data to CPU")
            # print("_build_stage_2D: Moving data to CPU")
            # print("_build_stage_2D: S_arr.device: ", S_arr.devices())
            S_host = jax.device_put(S_arr, host_device)
            # print("_build_stage_2D: S_host.device: ", S_host.devices())
            S_lst.append(S_host)
            v_int_host = jax.device_put(v_int_arr, host_device)
            v_int_lst.append(v_int_host)
            # DtN_host = jax.device_put(DtN_arr_new, host_device)
            # DtN_lst.append(DtN_host)

            S_arr.delete()
            v_int_arr.delete()
            DtN_arr.delete()
        else:
            S_lst.append(S_arr)
            v_int_lst.append(v_int_arr)
            # DtN_lst.append(DtN_arr_new)

        DtN_arr = DtN_arr_new
        v_prime_arr = v_prime_arr_new

    if return_fused_info:
        T_last = DtN_arr
        v_prime_last = v_prime_arr
        return (
            jax.device_put(T_last, host_device),
            jax.device_put(v_prime_last, host_device),
            S_lst,
            v_int_lst,
        )

    S_last, T_last, v_prime_last, v_int_last = _quad_merge(
        DtN_arr[0, 0],
        DtN_arr[0, 1],
        DtN_arr[0, 2],
        DtN_arr[0, 3],
        v_prime_arr[0, 0],
        v_prime_arr[0, 1],
        v_prime_arr[0, 2],
        v_prime_arr[0, 3],
    )
    # print("_build_stage: T_last shape: ", T_last.shape)
    # print("_build_stage: v_int_last shape: ", v_int_last.shape)

    # DtN_lst.append(jax.device_put(T_last, HOST_DEVICE))
    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    v_int_lst.append(jax.device_put(jnp.expand_dims(v_int_last, axis=0), host_device))
    # DtN_lst.append(jax.device_put(jnp.expand_dims(T_last, axis=0), host_device))

    S_last.delete()
    T_last.delete()
    v_int_last.delete()
    v_prime_last.delete()

    # logging.debug("_build_stage: done with merging.")
    return S_lst, DtN_lst, v_int_lst


def _build_stage_2D_ItI(
    R_maps: jnp.ndarray,
    h_arr: jnp.ndarray,
    l: int,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    return_fused_info: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """Implements the upward pass for merging ItI maps

    Args:
        R_maps (jnp.ndarray): Has shape (n_merge_pairs, 2, p**2, p**2)
        f_arr (jnp.ndarray): Has shape (n_merge_pairs, 2, 4*q)

    Returns:
        Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]: S_arr, DtN_arr, f_arr.
            S_arr is a list of arrays containing the S maps for each level.
            R_arr is a list of arrays containing the ItI maps for each level.
            f_arr is a list of arrays containing the interface particular soln incoming impedance data for each level.
    """
    logging.debug("_build_stage_2D_ItI: started")
    # logging.debug("_build_stage_2D_ItI: input R_maps.device %s", R_maps.devices())

    R_arr = R_maps

    # Start lists to store S, R, and f arrays
    S_lst = []
    R_lst = []
    f_lst = []

    R_arr = jax.device_put(R_arr, device)
    h_arr = jax.device_put(h_arr, device)

    # Working on merging the merge pairs at level i
    for i in range(l - 1):
        # print("up_pass: merge_EW = ", merge_EW)
        # logging.debug("_build_stage_2D_ItI: starting with level %i", i)
        # logging.debug("_build_stage_2D_ItI: R_arr.shape = %s", R_arr.shape)
        # logging.debug("_build_stage_2D_ItI: h_arr.shape = %s", h_arr.shape)

        S_arr, R_arr, h_arr, f_arr = vmapped_quad_merge_ItI(R_arr, h_arr)
        if host_device != device:
            logging.debug("_build_stage_2D_ItI: Moving data to host")
            S_host = jax.device_put(S_arr, host_device)
            S_lst.append(S_host)
            R_host = jax.device_put(R_arr, host_device)
            R_lst.append(R_host)
            f_host = jax.device_put(f_arr, host_device)
            f_lst.append(f_host)

            S_arr.delete()
            f_arr.delete()
        else:
            R_lst.append(R_arr)
            S_lst.append(S_arr)
            f_lst.append(f_arr)

    if return_fused_info:
        # Early exit and only return the final merge info
        R_last = R_arr
        h_last = h_arr
        return (
            jax.device_put(R_last, host_device),
            jax.device_put(h_last, host_device),
            S_lst,
            f_lst,
        )

    S_last, R_last, h_last, f_last = _quad_merge_ItI(
        R_arr[0, 0],
        R_arr[0, 1],
        R_arr[0, 2],
        R_arr[0, 3],
        h_arr[0, 0],
        h_arr[0, 1],
        h_arr[0, 2],
        h_arr[0, 3],
    )

    # h_last = jnp.expand_dims(h_last, axis=-1)
    # print("_build_stage: T_last shape: ", T_last.shape)
    # print("_build_stage: v_int_last shape: ", v_int_last.shape)

    R_lst.append(jax.device_put(R_last, host_device))
    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    f_lst.append(jax.device_put(jnp.expand_dims(f_last, axis=0), host_device))

    # logging.debug("_build_stage: done with merging.")
    return S_lst, R_lst, f_lst


@jax.jit
def _oct_merge(
    q_idxes: jnp.array,
    T_a: jnp.ndarray,
    T_b: jnp.ndarray,
    T_c: jnp.ndarray,
    T_d: jnp.ndarray,
    T_e: jnp.ndarray,
    T_f: jnp.ndarray,
    T_g: jnp.ndarray,
    T_h: jnp.ndarray,
    v_prime_a: jnp.ndarray,
    v_prime_b: jnp.ndarray,
    v_prime_c: jnp.ndarray,
    v_prime_d: jnp.ndarray,
    v_prime_e: jnp.ndarray,
    v_prime_f: jnp.ndarray,
    v_prime_g: jnp.ndarray,
    v_prime_h: jnp.ndarray,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:

    (
        T_a_1_1,
        T_a_1_9,
        T_a_1_12,
        T_a_1_17,
        T_a_9_1,
        T_a_9_9,
        T_a_9_12,
        T_a_9_17,
        T_a_12_1,
        T_a_12_9,
        T_a_12_12,
        T_a_12_17,
        T_a_17_1,
        T_a_17_9,
        T_a_17_12,
        T_a_17_17,
        v_prime_a_1,
        v_prime_a_9,
        v_prime_a_12,
        v_prime_a_17,
    ) = get_a_submatrices(T_a, v_prime_a)
    del T_a, v_prime_a

    (
        T_b_2_2,
        T_b_2_9,
        T_b_2_10,
        T_b_2_18,
        T_b_9_2,
        T_b_9_9,
        T_b_9_10,
        T_b_9_18,
        T_b_10_2,
        T_b_10_9,
        T_b_10_10,
        T_b_10_18,
        T_b_18_2,
        T_b_18_9,
        T_b_18_10,
        T_b_18_18,
        v_prime_b_2,
        v_prime_b_9,
        v_prime_b_10,
        v_prime_b_18,
    ) = get_b_submatrices(T_b, v_prime_b)
    del T_b, v_prime_b

    (
        T_c_3_3,
        T_c_3_10,
        T_c_3_11,
        T_c_3_19,
        T_c_10_3,
        T_c_10_10,
        T_c_10_11,
        T_c_10_19,
        T_c_11_3,
        T_c_11_10,
        T_c_11_11,
        T_c_11_19,
        T_c_19_3,
        T_c_19_10,
        T_c_19_11,
        T_c_19_19,
        v_prime_c_3,
        v_prime_c_10,
        v_prime_c_11,
        v_prime_c_19,
    ) = get_c_submatrices(T_c, v_prime_c)
    del T_c, v_prime_c

    (
        T_d_4_4,
        T_d_4_11,
        T_d_4_12,
        T_d_4_20,
        T_d_11_4,
        T_d_11_11,
        T_d_11_12,
        T_d_11_20,
        T_d_12_4,
        T_d_12_11,
        T_d_12_12,
        T_d_12_20,
        T_d_20_4,
        T_d_20_11,
        T_d_20_12,
        T_d_20_20,
        v_prime_d_4,
        v_prime_d_11,
        v_prime_d_12,
        v_prime_d_20,
    ) = get_d_submatrices(T_d, v_prime_d)
    del T_d, v_prime_d

    (
        T_e_5_5,
        T_e_5_13,
        T_e_5_16,
        T_e_5_17,
        T_e_13_5,
        T_e_13_13,
        T_e_13_16,
        T_e_13_17,
        T_e_16_5,
        T_e_16_13,
        T_e_16_16,
        T_e_16_17,
        T_e_17_5,
        T_e_17_13,
        T_e_17_16,
        T_e_17_17,
        v_prime_e_5,
        v_prime_e_13,
        v_prime_e_16,
        v_prime_e_17,
    ) = get_e_submatrices(T_e, v_prime_e)
    del T_e, v_prime_e

    (
        T_f_6_6,
        T_f_6_13,
        T_f_6_14,
        T_f_6_18,
        T_f_13_6,
        T_f_13_13,
        T_f_13_14,
        T_f_13_18,
        T_f_14_6,
        T_f_14_13,
        T_f_14_14,
        T_f_14_18,
        T_f_18_6,
        T_f_18_13,
        T_f_18_14,
        T_f_18_18,
        v_prime_f_6,
        v_prime_f_13,
        v_prime_f_14,
        v_prime_f_18,
    ) = get_f_submatrices(T_f, v_prime_f)
    del T_f, v_prime_f

    (
        T_g_7_7,
        T_g_7_14,
        T_g_7_15,
        T_g_7_19,
        T_g_14_7,
        T_g_14_14,
        T_g_14_15,
        T_g_14_19,
        T_g_15_7,
        T_g_15_14,
        T_g_15_15,
        T_g_15_19,
        T_g_19_7,
        T_g_19_14,
        T_g_19_15,
        T_g_19_19,
        v_prime_g_7,
        v_prime_g_14,
        v_prime_g_15,
        v_prime_g_19,
    ) = get_g_submatrices(T_g, v_prime_g)
    del T_g, v_prime_g

    (
        T_h_8_8,
        T_h_8_15,
        T_h_8_16,
        T_h_8_20,
        T_h_15_8,
        T_h_15_15,
        T_h_15_16,
        T_h_15_20,
        T_h_16_8,
        T_h_16_15,
        T_h_16_16,
        T_h_16_20,
        T_h_20_8,
        T_h_20_15,
        T_h_20_16,
        T_h_20_20,
        v_prime_h_8,
        v_prime_h_15,
        v_prime_h_16,
        v_prime_h_20,
    ) = get_h_submatrices(T_h, v_prime_h)
    del T_h, v_prime_h

    n_int, n_ext = T_a_9_1.shape

    # D is a block matrix with an array of (12x12) blocks. We are computing this first
    # because we need space on the GPU to invert it before computing B and C, which are also large
    D = jnp.zeros((12 * n_int, 12 * n_int), dtype=jnp.float64)
    # First row
    D = D.at[:n_int, :n_int].set(T_a_9_9 + T_b_9_9)
    D = D.at[:n_int, n_int : 2 * n_int].set(T_b_9_10)
    D = D.at[:n_int, 3 * n_int : 4 * n_int].set(T_a_9_12)
    D = D.at[:n_int, 8 * n_int : 9 * n_int].set(T_a_9_17)
    D = D.at[:n_int, 9 * n_int : 10 * n_int].set(T_b_9_18)
    # Second row
    D = D.at[n_int : 2 * n_int, :n_int].set(T_b_10_9)
    D = D.at[n_int : 2 * n_int, n_int : 2 * n_int].set(T_b_10_10 + T_c_10_10)
    D = D.at[n_int : 2 * n_int, 2 * n_int : 3 * n_int].set(T_c_10_11)
    D = D.at[n_int : 2 * n_int, 9 * n_int : 10 * n_int].set(T_b_10_18)
    D = D.at[n_int : 2 * n_int, 10 * n_int : 11 * n_int].set(T_c_10_19)
    # Third row
    D = D.at[2 * n_int : 3 * n_int, n_int : 2 * n_int].set(T_c_11_10)
    D = D.at[2 * n_int : 3 * n_int, 2 * n_int : 3 * n_int].set(T_c_11_11 + T_d_11_11)
    D = D.at[2 * n_int : 3 * n_int, 3 * n_int : 4 * n_int].set(T_d_11_12)
    D = D.at[2 * n_int : 3 * n_int, 10 * n_int : 11 * n_int].set(T_c_11_19)
    D = D.at[2 * n_int : 3 * n_int, 11 * n_int : 12 * n_int].set(T_d_11_20)
    # Fourth row
    D = D.at[3 * n_int : 4 * n_int, :n_int].set(T_a_12_9)
    D = D.at[3 * n_int : 4 * n_int, 2 * n_int : 3 * n_int].set(T_d_12_11)
    D = D.at[3 * n_int : 4 * n_int, 3 * n_int : 4 * n_int].set(T_d_12_12 + T_a_12_12)
    D = D.at[3 * n_int : 4 * n_int, 8 * n_int : 9 * n_int].set(T_a_12_17)
    D = D.at[3 * n_int : 4 * n_int, 11 * n_int : 12 * n_int].set(T_d_12_20)
    # Fifth row
    D = D.at[4 * n_int : 5 * n_int, 4 * n_int : 5 * n_int].set(T_e_13_13 + T_f_13_13)
    D = D.at[4 * n_int : 5 * n_int, 5 * n_int : 6 * n_int].set(T_f_13_14)
    D = D.at[4 * n_int : 5 * n_int, 7 * n_int : 8 * n_int].set(T_e_13_16)
    D = D.at[4 * n_int : 5 * n_int, 8 * n_int : 9 * n_int].set(T_e_13_17)
    D = D.at[4 * n_int : 5 * n_int, 9 * n_int : 10 * n_int].set(T_f_13_18)
    # Sixth row
    D = D.at[5 * n_int : 6 * n_int, 4 * n_int : 5 * n_int].set(T_f_14_13)
    D = D.at[5 * n_int : 6 * n_int, 5 * n_int : 6 * n_int].set(T_f_14_14 + T_g_14_14)
    D = D.at[5 * n_int : 6 * n_int, 6 * n_int : 7 * n_int].set(T_g_14_15)
    D = D.at[5 * n_int : 6 * n_int, 9 * n_int : 10 * n_int].set(T_f_14_18)
    D = D.at[5 * n_int : 6 * n_int, 10 * n_int : 11 * n_int].set(T_g_14_19)
    # Seventh row
    D = D.at[6 * n_int : 7 * n_int, 5 * n_int : 6 * n_int].set(T_g_15_14)
    D = D.at[6 * n_int : 7 * n_int, 6 * n_int : 7 * n_int].set(T_g_15_15 + T_h_15_15)
    D = D.at[6 * n_int : 7 * n_int, 7 * n_int : 8 * n_int].set(T_h_15_16)
    D = D.at[6 * n_int : 7 * n_int, 10 * n_int : 11 * n_int].set(T_g_15_19)
    D = D.at[6 * n_int : 7 * n_int, 11 * n_int : 12 * n_int].set(T_h_15_20)
    # Eighth row
    D = D.at[7 * n_int : 8 * n_int, 4 * n_int : 5 * n_int].set(T_e_16_13)
    D = D.at[7 * n_int : 8 * n_int, 6 * n_int : 7 * n_int].set(T_h_16_15)
    D = D.at[7 * n_int : 8 * n_int, 7 * n_int : 8 * n_int].set(T_h_16_16 + T_e_16_16)
    D = D.at[7 * n_int : 8 * n_int, 8 * n_int : 9 * n_int].set(T_e_16_17)
    D = D.at[7 * n_int : 8 * n_int, 11 * n_int : 12 * n_int].set(T_h_16_20)
    # Ninth row
    D = D.at[8 * n_int : 9 * n_int, :n_int].set(T_a_17_9)
    D = D.at[8 * n_int : 9 * n_int, 3 * n_int : 4 * n_int].set(T_a_17_12)
    D = D.at[8 * n_int : 9 * n_int, 4 * n_int : 5 * n_int].set(T_e_17_13)
    D = D.at[8 * n_int : 9 * n_int, 7 * n_int : 8 * n_int].set(T_e_17_16)
    D = D.at[8 * n_int : 9 * n_int, 8 * n_int : 9 * n_int].set(T_e_17_17 + T_a_17_17)
    # Tenth row
    D = D.at[9 * n_int : 10 * n_int, :n_int].set(T_b_18_9)
    D = D.at[9 * n_int : 10 * n_int, n_int : 2 * n_int].set(T_b_18_10)
    D = D.at[9 * n_int : 10 * n_int, 4 * n_int : 5 * n_int].set(T_f_18_13)
    D = D.at[9 * n_int : 10 * n_int, 5 * n_int : 6 * n_int].set(T_f_18_14)
    D = D.at[9 * n_int : 10 * n_int, 9 * n_int : 10 * n_int].set(T_f_18_18 + T_b_18_18)
    # Eleventh row
    D = D.at[10 * n_int : 11 * n_int, n_int : 2 * n_int].set(T_c_19_10)
    D = D.at[10 * n_int : 11 * n_int, 2 * n_int : 3 * n_int].set(T_c_19_11)
    D = D.at[10 * n_int : 11 * n_int, 5 * n_int : 6 * n_int].set(T_g_19_14)
    D = D.at[10 * n_int : 11 * n_int, 6 * n_int : 7 * n_int].set(T_g_19_15)
    D = D.at[10 * n_int : 11 * n_int, 10 * n_int : 11 * n_int].set(
        T_g_19_19 + T_c_19_19
    )
    # Twelfth row
    D = D.at[11 * n_int :, 2 * n_int : 3 * n_int].set(T_d_20_11)
    D = D.at[11 * n_int :, 3 * n_int : 4 * n_int].set(T_d_20_12)
    D = D.at[11 * n_int :, 6 * n_int : 7 * n_int].set(T_h_20_15)
    D = D.at[11 * n_int :, 7 * n_int : 8 * n_int].set(T_h_20_16)
    D = D.at[11 * n_int :, 11 * n_int :].set(T_h_20_20 + T_d_20_20)

    # Delete the blocks we no longer need
    del T_a_9_9, T_b_9_9, T_b_9_10, T_a_9_12, T_a_9_17, T_b_9_18
    del T_b_10_9, T_b_10_10, T_c_10_10, T_c_10_11, T_b_10_18, T_c_10_19
    del T_c_11_10, T_c_11_11, T_d_11_11, T_d_11_12, T_c_11_19, T_d_11_20
    del T_a_12_9, T_d_12_11, T_d_12_12, T_a_12_12, T_a_12_17, T_d_12_20
    del T_e_13_13, T_f_13_13, T_f_13_14, T_e_13_16, T_e_13_17, T_f_13_18
    del T_f_14_13, T_f_14_14, T_g_14_14, T_g_14_15, T_f_14_18, T_g_14_19
    del T_g_15_14, T_g_15_15, T_h_15_15, T_h_15_16, T_g_15_19, T_h_15_20
    del T_e_16_13, T_h_16_15, T_h_16_16, T_e_16_16, T_e_16_17, T_h_16_20
    del T_a_17_9, T_a_17_12, T_e_17_13, T_e_17_16, T_e_17_17, T_a_17_17
    del T_b_18_9, T_b_18_10, T_f_18_13, T_f_18_14, T_f_18_18, T_b_18_18
    del T_c_19_10, T_c_19_11, T_g_19_14, T_g_19_15, T_g_19_19, T_c_19_19
    del T_d_20_11, T_d_20_12, T_h_20_15, T_h_20_16, T_h_20_20, T_d_20_20

    neg_D_inv = -1 * jnp.linalg.inv(D)

    # B is a block matrix with an array of (8x12) blocks.
    B = jnp.zeros((8 * n_ext, 12 * n_int), dtype=jnp.float64)
    # First row
    B = B.at[:n_ext, :n_int].set(T_a_1_9)
    B = B.at[:n_ext, 3 * n_int : 4 * n_int].set(T_a_1_12)
    B = B.at[:n_ext, 8 * n_int : 9 * n_int].set(T_a_1_17)
    # Second row
    B = B.at[n_ext : 2 * n_ext, :n_int].set(T_b_2_9)
    B = B.at[n_ext : 2 * n_ext, n_int : 2 * n_int].set(T_b_2_10)
    B = B.at[n_ext : 2 * n_ext, 9 * n_int : 10 * n_int].set(T_b_2_18)
    # Third row
    B = B.at[2 * n_ext : 3 * n_ext, n_int : 2 * n_int].set(T_c_3_10)
    B = B.at[2 * n_ext : 3 * n_ext, 2 * n_int : 3 * n_int].set(T_c_3_11)
    B = B.at[2 * n_ext : 3 * n_ext, 10 * n_int : 11 * n_int].set(T_c_3_19)
    # Fourth row
    B = B.at[3 * n_ext : 4 * n_ext, 2 * n_int : 3 * n_int].set(T_d_4_11)
    B = B.at[3 * n_ext : 4 * n_ext, 3 * n_int : 4 * n_int].set(T_d_4_12)
    B = B.at[3 * n_ext : 4 * n_ext, 11 * n_int : 12 * n_int].set(T_d_4_20)
    # Fifth row
    B = B.at[4 * n_ext : 5 * n_ext, 4 * n_int : 5 * n_int].set(T_e_5_13)
    B = B.at[4 * n_ext : 5 * n_ext, 7 * n_int : 8 * n_int].set(T_e_5_16)
    B = B.at[4 * n_ext : 5 * n_ext, 8 * n_int : 9 * n_int].set(T_e_5_17)
    # Sixth row
    B = B.at[5 * n_ext : 6 * n_ext, 4 * n_int : 5 * n_int].set(T_f_6_13)
    B = B.at[5 * n_ext : 6 * n_ext, 5 * n_int : 6 * n_int].set(T_f_6_14)
    B = B.at[5 * n_ext : 6 * n_ext, 9 * n_int : 10 * n_int].set(T_f_6_18)
    # Seventh row
    B = B.at[6 * n_ext : 7 * n_ext, 5 * n_int : 6 * n_int].set(T_g_7_14)
    B = B.at[6 * n_ext : 7 * n_ext, 6 * n_int : 7 * n_int].set(T_g_7_15)
    B = B.at[6 * n_ext : 7 * n_ext, 10 * n_int : 11 * n_int].set(T_g_7_19)
    # Eighth row
    B = B.at[7 * n_ext :, 6 * n_int : 7 * n_int].set(T_h_8_15)
    B = B.at[7 * n_ext :, 7 * n_int : 8 * n_int].set(T_h_8_16)
    B = B.at[7 * n_ext :, 11 * n_int : 12 * n_int].set(T_h_8_20)

    # Delete all of the blocks we no longer need
    del (
        T_a_1_9,
        T_a_1_12,
        T_a_1_17,
        T_b_2_9,
        T_b_2_10,
        T_b_2_18,
        T_c_3_10,
        T_c_3_11,
        T_c_3_19,
        T_d_4_11,
        T_d_4_12,
        T_d_4_20,
        T_e_5_13,
        T_e_5_16,
        T_e_5_17,
        T_f_6_13,
        T_f_6_14,
        T_f_6_18,
        T_g_7_14,
        T_g_7_15,
        T_g_7_19,
        T_h_8_15,
        T_h_8_16,
        T_h_8_20,
    )

    # C is a block matrix with an array of (12x8) blocks.
    C = _compute_C_3D(
        T_a_9_1=T_a_9_1,
        T_b_9_2=T_b_9_2,
        T_b_10_2=T_b_10_2,
        T_c_10_3=T_c_10_3,
        T_c_11_3=T_c_11_3,
        T_d_11_4=T_d_11_4,
        T_a_12_1=T_a_12_1,
        T_d_12_4=T_d_12_4,
        T_e_13_5=T_e_13_5,
        T_f_13_6=T_f_13_6,
        T_f_14_6=T_f_14_6,
        T_g_14_7=T_g_14_7,
        T_g_15_7=T_g_15_7,
        T_h_15_8=T_h_15_8,
        T_e_16_5=T_e_16_5,
        T_h_16_8=T_h_16_8,
        T_a_17_1=T_a_17_1,
        T_e_17_5=T_e_17_5,
        T_b_18_2=T_b_18_2,
        T_f_18_6=T_f_18_6,
        T_c_19_3=T_c_19_3,
        T_g_19_7=T_g_19_7,
        T_d_20_4=T_d_20_4,
        T_h_20_8=T_h_20_8,
    )
    # C = jnp.zeros((12 * n_int, 8 * n_ext), dtype=jnp.float64)
    # # First row
    # C = C.at[:n_int, :n_ext].set(T_a_9_1)
    # C = C.at[:n_int, n_ext : 2 * n_ext].set(T_b_9_2)
    # # Second row
    # C = C.at[n_int : 2 * n_int, n_ext : 2 * n_ext].set(T_b_10_2)
    # C = C.at[n_int : 2 * n_int, 2 * n_ext : 3 * n_ext].set(T_c_10_3)
    # # Third row
    # C = C.at[2 * n_int : 3 * n_int, 2 * n_ext : 3 * n_ext].set(T_c_11_3)
    # C = C.at[2 * n_int : 3 * n_int, 3 * n_ext : 4 * n_ext].set(T_d_11_4)
    # # Fourth row
    # C = C.at[3 * n_int : 4 * n_int, :n_ext].set(T_a_12_1)
    # C = C.at[3 * n_int : 4 * n_int, 3 * n_ext : 4 * n_ext].set(T_d_12_4)
    # # Fifth row
    # C = C.at[4 * n_int : 5 * n_int, 4 * n_ext : 5 * n_ext].set(T_e_13_5)
    # C = C.at[4 * n_int : 5 * n_int, 5 * n_ext : 6 * n_ext].set(T_f_13_6)
    # # Sixth row
    # C = C.at[5 * n_int : 6 * n_int, 5 * n_ext : 6 * n_ext].set(T_f_14_6)
    # C = C.at[5 * n_int : 6 * n_int, 6 * n_ext : 7 * n_ext].set(T_g_14_7)
    # # Seventh row
    # C = C.at[6 * n_int : 7 * n_int, 6 * n_ext : 7 * n_ext].set(T_g_15_7)
    # C = C.at[6 * n_int : 7 * n_int, 7 * n_ext : 8 * n_ext].set(T_h_15_8)
    # # Eighth row
    # C = C.at[7 * n_int : 8 * n_int, 4 * n_ext : 5 * n_ext].set(T_e_16_5)
    # C = C.at[7 * n_int : 8 * n_int, 7 * n_ext : 8 * n_ext].set(T_h_16_8)
    # # Ninth row
    # C = C.at[8 * n_int : 9 * n_int, :n_ext].set(T_a_17_1)
    # C = C.at[8 * n_int : 9 * n_int, 4 * n_ext : 5 * n_ext].set(T_e_17_5)
    # # Tenth row
    # C = C.at[9 * n_int : 10 * n_int, n_ext : 2 * n_ext].set(T_b_18_2)
    # C = C.at[9 * n_int : 10 * n_int, 5 * n_ext : 6 * n_ext].set(T_f_18_6)
    # # Eleventh row
    # C = C.at[10 * n_int : 11 * n_int, 2 * n_ext : 3 * n_ext].set(T_c_19_3)
    # C = C.at[10 * n_int : 11 * n_int, 6 * n_ext : 7 * n_ext].set(T_g_19_7)
    # # Twelfth row
    # C = C.at[11 * n_int :, 3 * n_ext : 4 * n_ext].set(T_d_20_4)
    # C = C.at[11 * n_int :, 7 * n_ext : 8 * n_ext].set(T_h_20_8)

    S = neg_D_inv @ C

    T = B @ S
    # Add A to T block-wise
    T = T.at[:n_ext, :n_ext].set(T[:n_ext, :n_ext] + T_a_1_1)
    T = T.at[n_ext : 2 * n_ext, n_ext : 2 * n_ext].set(
        T[n_ext : 2 * n_ext, n_ext : 2 * n_ext] + T_b_2_2
    )
    T = T.at[2 * n_ext : 3 * n_ext, 2 * n_ext : 3 * n_ext].set(
        T[2 * n_ext : 3 * n_ext, 2 * n_ext : 3 * n_ext] + T_c_3_3
    )
    T = T.at[3 * n_ext : 4 * n_ext, 3 * n_ext : 4 * n_ext].set(
        T[3 * n_ext : 4 * n_ext, 3 * n_ext : 4 * n_ext] + T_d_4_4
    )
    T = T.at[4 * n_ext : 5 * n_ext, 4 * n_ext : 5 * n_ext].set(
        T[4 * n_ext : 5 * n_ext, 4 * n_ext : 5 * n_ext] + T_e_5_5
    )
    T = T.at[5 * n_ext : 6 * n_ext, 5 * n_ext : 6 * n_ext].set(
        T[5 * n_ext : 6 * n_ext, 5 * n_ext : 6 * n_ext] + T_f_6_6
    )
    T = T.at[6 * n_ext : 7 * n_ext, 6 * n_ext : 7 * n_ext].set(
        T[6 * n_ext : 7 * n_ext, 6 * n_ext : 7 * n_ext] + T_g_7_7
    )
    T = T.at[7 * n_ext :, 7 * n_ext :].set(T[7 * n_ext :, 7 * n_ext :] + T_h_8_8)

    delta_v_prime = jnp.concatenate(
        [
            v_prime_a_9 + v_prime_b_9,
            v_prime_b_10 + v_prime_c_10,
            v_prime_c_11 + v_prime_d_11,
            v_prime_d_12 + v_prime_a_12,
            v_prime_e_13 + v_prime_f_13,
            v_prime_f_14 + v_prime_g_14,
            v_prime_g_15 + v_prime_h_15,
            v_prime_h_16 + v_prime_e_16,
            v_prime_a_17 + v_prime_e_17,
            v_prime_b_18 + v_prime_f_18,
            v_prime_c_19 + v_prime_g_19,
            v_prime_d_20 + v_prime_h_20,
        ]
    )
    v_int = neg_D_inv @ delta_v_prime
    v_prime_ext = jnp.concatenate(
        [
            v_prime_a_1,
            v_prime_b_2,
            v_prime_c_3,
            v_prime_d_4,
            v_prime_e_5,
            v_prime_f_6,
            v_prime_g_7,
            v_prime_h_8,
        ]
    )
    v_prime_ext_out = v_prime_ext + B @ v_int

    r = get_rearrange_indices(jnp.arange(T.shape[0]), q_idxes)
    v_prime_ext_out = v_prime_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, v_prime_ext_out, v_int


@jax.jit
def _compute_C_3D(
    T_a_9_1: jnp.array,
    T_b_9_2: jnp.array,
    T_b_10_2: jnp.array,
    T_c_10_3: jnp.array,
    T_c_11_3: jnp.array,
    T_d_11_4: jnp.array,
    T_a_12_1: jnp.array,
    T_d_12_4: jnp.array,
    T_e_13_5: jnp.array,
    T_f_13_6: jnp.array,
    T_f_14_6: jnp.array,
    T_g_14_7: jnp.array,
    T_g_15_7: jnp.array,
    T_h_15_8: jnp.array,
    T_e_16_5: jnp.array,
    T_h_16_8: jnp.array,
    T_a_17_1: jnp.array,
    T_e_17_5: jnp.array,
    T_b_18_2: jnp.array,
    T_f_18_6: jnp.array,
    T_c_19_3: jnp.array,
    T_g_19_7: jnp.array,
    T_d_20_4: jnp.array,
    T_h_20_8: jnp.array,
) -> jnp.array:
    n_int, n_ext = T_a_9_1.shape
    zero_block = jnp.zeros((n_int, n_ext), dtype=T_a_9_1.dtype)

    # C is a block matrix with an array of (12x8) blocks.
    C = jnp.block(
        [
            [
                T_a_9_1,
                T_b_9_2,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                T_b_10_2,
                T_c_10_3,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                T_c_11_3,
                T_d_11_4,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
            ],
            [
                T_a_12_1,
                zero_block,
                zero_block,
                T_d_12_4,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                T_e_13_5,
                T_f_13_6,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                T_f_14_6,
                T_g_14_7,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                T_g_15_7,
                T_h_15_8,
            ],
            [
                zero_block,
                zero_block,
                zero_block,
                zero_block,
                T_e_16_5,
                zero_block,
                zero_block,
                T_h_16_8,
            ],
            [
                T_a_17_1,
                zero_block,
                zero_block,
                zero_block,
                T_e_17_5,
                zero_block,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                T_b_18_2,
                zero_block,
                zero_block,
                zero_block,
                T_f_18_6,
                zero_block,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                T_c_19_3,
                zero_block,
                zero_block,
                zero_block,
                T_g_19_7,
                zero_block,
            ],
            [
                zero_block,
                zero_block,
                zero_block,
                T_d_20_4,
                zero_block,
                zero_block,
                zero_block,
                T_h_20_8,
            ],
        ]
    )
    return C


_vmapped_oct_merge = jax.vmap(
    _oct_merge,
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_oct_merge(
    q_idxes: jnp.array,
    T_in: jnp.ndarray,
    v_prime_in: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S, T_out, v_prime_ext_out, v_int = _vmapped_oct_merge(
        q_idxes,
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        T_in[:, 4],
        T_in[:, 5],
        T_in[:, 6],
        T_in[:, 7],
        v_prime_in[:, 0],
        v_prime_in[:, 1],
        v_prime_in[:, 2],
        v_prime_in[:, 3],
        v_prime_in[:, 4],
        v_prime_in[:, 5],
        v_prime_in[:, 6],
        v_prime_in[:, 7],
    )
    n_merges, n_int, n_ext = S.shape
    T_out = T_out.reshape((n_merges // 8, 8, n_ext, n_ext))
    v_prime_ext_out = v_prime_ext_out.reshape((n_merges // 8, 8, n_ext))

    return S, T_out, v_prime_ext_out, v_int
