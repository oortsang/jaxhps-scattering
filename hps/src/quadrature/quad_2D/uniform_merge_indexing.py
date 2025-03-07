from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def get_quadmerge_blocks_a(T: jnp.ndarray, v_prime: jnp.ndarray) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_1 = jnp.concatenate([idxes[-n_per_side:], idxes[:n_per_side]])
    idxes_5 = idxes[n_per_side : 2 * n_per_side]
    idxes_8 = jnp.flipud(idxes[2 * n_per_side : 3 * n_per_side])

    return _get_submatrices(T, v_prime, idxes_1, idxes_5, idxes_8)


@jax.jit
def get_quadmerge_blocks_b(T: jnp.ndarray, v_prime: jnp.ndarray) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_2 = idxes[: 2 * n_per_side]
    idxes_6 = idxes[2 * n_per_side : 3 * n_per_side]
    idxes_5 = jnp.flipud(idxes[3 * n_per_side :])

    return _get_submatrices(T, v_prime, idxes_2, idxes_6, idxes_5)


@jax.jit
def get_quadmerge_blocks_c(T: jnp.ndarray, v_prime: jnp.ndarray) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_6 = jnp.flipud(idxes[:n_per_side])
    idxes_3 = idxes[n_per_side : 3 * n_per_side]
    idxes_7 = idxes[3 * n_per_side :]

    return _get_submatrices(T, v_prime, idxes_6, idxes_3, idxes_7)


@jax.jit
def get_quadmerge_blocks_d(T: jnp.ndarray, v_prime: jnp.ndarray) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_8 = idxes[:n_per_side]
    idxes_7 = jnp.flipud(idxes[n_per_side : 2 * n_per_side])
    idxes_4 = idxes[2 * n_per_side :]

    return _get_submatrices(T, v_prime, idxes_8, idxes_7, idxes_4)


@jax.jit
def _get_submatrices(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    idxes_0: jnp.ndarray,
    idxes_1: jnp.ndarray,
    idxes_2: jnp.ndarray,
) -> Tuple[jnp.ndarray]:

    v_prime_0 = v_prime[idxes_0]
    v_prime_1 = v_prime[idxes_1]
    v_prime_2 = v_prime[idxes_2]

    T_00 = T[idxes_0][:, idxes_0]
    T_01 = T[idxes_0][:, idxes_1]
    T_02 = T[idxes_0][:, idxes_2]
    T_10 = T[idxes_1][:, idxes_0]
    T_11 = T[idxes_1][:, idxes_1]
    T_12 = T[idxes_1][:, idxes_2]
    T_20 = T[idxes_2][:, idxes_0]
    T_21 = T[idxes_2][:, idxes_1]
    T_22 = T[idxes_2][:, idxes_2]

    return (
        v_prime_0,
        v_prime_1,
        v_prime_2,
        T_00,
        T_01,
        T_02,
        T_10,
        T_11,
        T_12,
        T_20,
        T_21,
        T_22,
    )
