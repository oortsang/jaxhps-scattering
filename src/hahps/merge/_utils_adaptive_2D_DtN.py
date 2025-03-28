from typing import Tuple

import jax.numpy as jnp
import jax

from .._discretization_tree import DiscretizationNode2D
from .._discretization_tree import get_all_leaves


# @jax.jit
def _find_compression_list_5(
    node_a: DiscretizationNode2D, node_b: DiscretizationNode2D
) -> Tuple[jax.Array, jax.Array]:
    # First, get a list of leaves of a lying along merge interface 5
    leaves_a = get_all_leaves(node_a)
    leaves_a_5 = [leaf for leaf in leaves_a if leaf.xmax == node_a.xmax]

    # Get a list of leaves of b lying along merge interface 5
    leaves_b = get_all_leaves(node_b)
    leaves_b_5 = [leaf for leaf in leaves_b if leaf.xmin == node_b.xmin]

    # These lists of leaves are oriented outside inward.

    out_a = []
    out_b = []

    idx_a = 0
    idx_b = 0

    while idx_a < len(leaves_a_5) and idx_b < len(leaves_b_5):
        if leaves_a_5[idx_a].ymax == leaves_b_5[idx_b].ymax:
            # These leaves match up.
            out_a.append(False)
            out_b.append(False)
            idx_a += 1
            idx_b += 1
        elif leaves_a_5[idx_a].ymax < leaves_b_5[idx_b].ymax:
            # The next 2 leaves in a need to be compressed.
            out_a.append(True)
            out_a.append(True)
            idx_a += 2

            out_b.append(False)
            idx_b += 1
        else:
            # The next 2 leaves in b need to be compressed.
            out_b.append(True)
            out_b.append(True)
            idx_b += 2

            out_a.append(False)
            idx_a += 1
    return jnp.array(out_a), jnp.array(out_b)


# @jax.jit
def _find_compression_list_6(
    node_b: DiscretizationNode2D, node_c: DiscretizationNode2D
) -> Tuple[jax.Array, jax.Array]:
    # First, get a list of leaves of b lying along merge interface 6
    leaves_b = get_all_leaves(node_b)
    leaves_b_6 = [leaf for leaf in leaves_b if leaf.ymax == node_b.ymax]

    # Get a list of leaves of c lying along merge interface 6
    leaves_c = get_all_leaves(node_c)
    leaves_c_6 = [leaf for leaf in leaves_c if leaf.ymin == node_c.ymin]

    # leaves_b_6 oriented outside inward but we need to reverse the order of
    # leaves_c_6 so that they are oriented outside inward.
    leaves_c_6.reverse()

    out_b = []
    out_c = []

    idx_b = 0
    idx_c = 0

    while idx_b < len(leaves_b_6) and idx_c < len(leaves_c_6):
        if leaves_b_6[idx_b].xmin == leaves_c_6[idx_c].xmin:
            # These leaves match up.
            out_b.append(False)
            out_c.append(False)
            idx_b += 1
            idx_c += 1
        elif leaves_b_6[idx_b].xmin > leaves_c_6[idx_c].xmin:
            # The next 2 leaves in b need to be compressed.
            out_b.append(True)
            out_b.append(True)
            idx_b += 2

            out_c.append(False)
            idx_c += 1
        else:
            # The next 2 leaves in c need to be compressed.
            out_c.append(True)
            out_c.append(True)
            idx_c += 2

            out_b.append(False)
            idx_b += 1
    return jnp.array(out_b), jnp.array(out_c)


# @jax.jit
def _find_compression_list_7(
    node_c: DiscretizationNode2D, node_d: DiscretizationNode2D
) -> Tuple[jax.Array, jax.Array]:
    # First, get a list of leaves of c lying along merge interface 7
    leaves_c = get_all_leaves(node_c)
    leaves_c_7 = [leaf for leaf in leaves_c if leaf.xmin == node_c.xmin]

    leaves_d = get_all_leaves(node_d)
    leaves_d_7 = [leaf for leaf in leaves_d if leaf.xmax == node_d.xmax]

    # Both of these lists of leaves are oriented inside outward, so we need to
    # reverse the order of both of them.
    leaves_c_7.reverse()
    leaves_d_7.reverse()

    out_c = []
    out_d = []

    idx_c = 0
    idx_d = 0

    while idx_c < len(leaves_c_7) and idx_d < len(leaves_d_7):
        if leaves_c_7[idx_c].ymin == leaves_d_7[idx_d].ymin:
            # These leaves match up.
            out_c.append(False)
            out_d.append(False)
            idx_c += 1
            idx_d += 1
        elif leaves_c_7[idx_c].ymin > leaves_d_7[idx_d].ymin:
            # The next 2 leaves in c need to be compressed.
            out_c.append(True)
            out_c.append(True)
            idx_c += 2

            out_d.append(False)
            idx_d += 1
        else:
            # The next 2 leaves in d need to be compressed.
            out_d.append(True)
            out_d.append(True)
            idx_d += 2

            out_c.append(False)
            idx_c += 1
    return jnp.array(out_c), jnp.array(out_d)


# @jax.jit
def _find_compression_list_8(
    node_d: DiscretizationNode2D, node_a: DiscretizationNode2D
) -> Tuple[jax.Array, jax.Array]:
    # First, get a list of leaves of d lying along merge interface 8
    leaves_d = get_all_leaves(node_d)
    leaves_d_8 = [leaf for leaf in leaves_d if leaf.ymin == node_d.ymin]

    leaves_a = get_all_leaves(node_a)
    leaves_a_8 = [leaf for leaf in leaves_a if leaf.ymax == node_a.ymax]

    # leaves_a_8 is oriented inside outward, so we need to reverse the order of it.
    leaves_a_8.reverse()

    out_d = []
    out_a = []

    idx_d = 0
    idx_a = 0

    while idx_d < len(leaves_d_8) and idx_a < len(leaves_a_8):
        if leaves_d_8[idx_d].xmax == leaves_a_8[idx_a].xmax:
            # These leaves match up.
            out_d.append(False)
            out_a.append(False)
            idx_d += 1
            idx_a += 1
        elif leaves_d_8[idx_d].xmax < leaves_a_8[idx_a].xmax:
            # The next 2 leaves in d need to be compressed.
            out_d.append(True)
            out_d.append(True)
            idx_d += 2

            out_a.append(False)
            idx_a += 1
        else:
            # The next 2 leaves in a need to be compressed.
            out_a.append(True)
            out_a.append(True)
            idx_a += 2

            out_d.append(False)
            idx_d += 1
    return jnp.array(out_d), jnp.array(out_a)


# @jax.jit
def find_compression_lists_2D(
    node_a: DiscretizationNode2D,
    node_b: DiscretizationNode2D,
    node_c: DiscretizationNode2D,
    node_d: DiscretizationNode2D,
) -> Tuple[jax.Array]:
    """
    Returns a boolean array that indicates which parts of the merge interface need comporession from 2 panels to 1 panel.

    List the merge interface as follows:
    a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8

    For each of the 8 merge interfaces, we need to determine which pairs of panels need compression. So this function
    should return a list of booleans for each of the 8 merge interfaces. Each boolean indicates whether the corresponding
    panel needs compression.

    Args:
        nodes_this_level (List[Node]): Has length (n,) and contains the nodes at the current merge level.

    Returns:
        jax.Array: Output has shape shape (n // 4, 12)
    """

    compression_lst_a5, compression_lst_b5 = _find_compression_list_5(
        node_a, node_b
    )
    compression_lst_b6, compression_lst_c6 = _find_compression_list_6(
        node_b, node_c
    )
    compression_lst_c7, compression_lst_d7 = _find_compression_list_7(
        node_c, node_d
    )
    compression_lst_d8, compression_lst_a8 = _find_compression_list_8(
        node_d, node_a
    )

    return (
        compression_lst_a5,
        compression_lst_b5,
        compression_lst_b6,
        compression_lst_c6,
        compression_lst_c7,
        compression_lst_d7,
        compression_lst_d8,
        compression_lst_a8,
    )


# @partial(jax.jit, static_argnums=(4, 5))
def get_quadmerge_blocks_a(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_5: jnp.array,
    need_interp_8: jnp.array,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
) -> Tuple[jnp.ndarray]:
    """
    need_interp_5 and need_interp_8 are arrays which indicate whether particular panels along the border 8 or 5 are
    to be compressed.

    Notice that the need_interp_* inputs are ordered in the outside-inward ordering, but we want to use the standard
    ordering for boundary points when we pass information to the _get_submatrices function. So we will have to
    figure out whether the 2 and 3's need to be transposed, depending on the geometry.
    """
    idxes = jnp.arange(T.shape[0])
    # print("get_quadmerge_blocks_a: idxes", idxes.shape)

    idxes_1 = jnp.concatenate([idxes[-n_3:], idxes[:n_0]])
    idxes_5 = idxes[n_0 : n_0 + n_1]
    idxes_8 = jnp.flipud(idxes[n_0 + n_1 : n_0 + n_1 + n_2])

    q = L_1f2.shape[0]
    n_panels_1 = idxes_1.shape[0] // q
    need_interp_1 = jnp.full((n_panels_1,), False)

    # Flip the need_interp_8 input, because the discretization points along the a_8 side
    # go inside outward along interface 8.
    # need_interp_8 = jnp.flipud(need_interp_8)

    return _get_submatrices(
        T,
        v_prime,
        idxes_1,
        idxes_5,
        idxes_8,
        L_2f1,
        L_1f2,
        need_interp_1,
        need_interp_5,
        need_interp_8,
    )


# @partial(jax.jit, static_argnums=(4, 5))
def get_quadmerge_blocks_b(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_6: int,
    need_interp_5: int,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
) -> Tuple[jnp.ndarray]:
    idxes = jnp.arange(T.shape[0])

    idxes_2 = idxes[: n_0 + n_1]
    idxes_6 = idxes[n_0 + n_1 : n_0 + n_1 + n_2]
    idxes_5 = jnp.flipud(idxes[-n_3:])

    q = L_1f2.shape[0]
    n_panels_2 = idxes_2.shape[0] // q
    need_interp_2 = jnp.full((n_panels_2,), False)

    # Flip need_interp_5 because the discretization points along the b_5 side
    # go inside outward along interface 5.
    # need_interp_5 = jnp.flipud(need_interp_5)

    return _get_submatrices(
        T,
        v_prime,
        idxes_2,
        idxes_6,
        idxes_5,
        L_2f1,
        L_1f2,
        need_interp_2,
        need_interp_6,
        need_interp_5,
    )


# @partial(jax.jit, static_argnums=(4, 5))
def get_quadmerge_blocks_c(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_6: int,
    need_interp_7: int,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
) -> Tuple[jnp.ndarray]:
    idxes = jnp.arange(T.shape[0])

    idxes_6 = jnp.flipud(idxes[:n_0])
    idxes_3 = idxes[n_0 : n_0 + n_1 + n_2]
    idxes_7 = idxes[-n_3:]

    q = L_1f2.shape[0]
    n_panels_3 = idxes_3.shape[0] // q
    need_interp_3 = jnp.full((n_panels_3,), False)

    # Flip need_interp_6 because the discretization points along the c_6 side
    # go inside outward along interface 6.
    # need_interp_6 = jnp.flipud(need_interp_6)

    return _get_submatrices(
        T,
        v_prime,
        idxes_6,
        idxes_3,
        idxes_7,
        L_2f1,
        L_1f2,
        need_interp_6,
        need_interp_3,
        need_interp_7,
    )


# @partial(jax.jit, static_argnums=(4, 5))
def get_quadmerge_blocks_d(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_8: int,
    need_interp_7: int,
    n_0: int,
    n_1: int,
    n_2: int,
    n_3: int,
) -> Tuple[jnp.ndarray]:
    idxes = jnp.arange(T.shape[0])

    idxes_8 = idxes[:n_0]
    idxes_7 = jnp.flipud(idxes[n_0 : n_0 + n_1])
    idxes_4 = idxes[n_0 + n_1 :]

    # Flip need_interp_7 because the discretization points along the d_7 side
    # go inside outward along interface 7.
    # need_interp_7 = jnp.flipud(need_interp_7)

    q = L_1f2.shape[0]
    n_panels_4 = idxes_4.shape[0] // q
    need_interp_4 = jnp.full((n_panels_4,), False)

    return _get_submatrices(
        T,
        v_prime,
        idxes_8,
        idxes_7,
        idxes_4,
        L_2f1,
        L_1f2,
        need_interp_8,
        need_interp_7,
        need_interp_4,
    )


# @partial(jax.jit, static_argnums=(7, 8, 9))
# @jax.jit
def _get_submatrices(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    idxes_0: jnp.ndarray,
    idxes_1: jnp.ndarray,
    idxes_2: jnp.ndarray,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_0: jnp.array,
    need_interp_1: jnp.array,
    need_interp_2: jnp.array,
) -> Tuple[jnp.ndarray]:
    """_summary_

    need_interp_0, need_interp_1, need_interp_2 are integers which indicate which parts of the boundaries need compression from
    2 panels to 1 panel. There are four different options:
    0: No compression
    1: Compress the whole face
    2: Compress the first half of the face
    3: Compress the second half of the face
    4: Compress both the first and second halves of the face.

    Args:
        T (jnp.ndarray): _description_
        v_prime (jnp.ndarray): _description_
        idxes_0 (jnp.ndarray): _description_
        idxes_1 (jnp.ndarray): _description_
        idxes_2 (jnp.ndarray): _description_
        L_2f1 (jnp.array): _description_
        L_1f2 (jnp.array): _description_
        need_interp_1 (bool): Whether to apply interpolation operators
        along side 1 of the DtN matrices. This effectively shortens the
        side 1s.
        need_interp_2 (bool): Whether to apply interpolation operators
        along side 2 of the DtN matrices. This effectively shortens the
        side 2s.

    Returns:
        Tuple[jnp.ndarray]: _description_
    """

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

    # Do all of the compression ops specified by need_interp_0
    T_10 = _compress_cols_from_lst(T_10, L_2f1, need_interp_0)
    T_01 = _compress_rows_from_lst(T_01, L_1f2, need_interp_0)
    T_00 = _compress_rows_from_lst(T_00, L_1f2, need_interp_0)
    T_00 = _compress_cols_from_lst(T_00, L_2f1, need_interp_0)
    T_02 = _compress_rows_from_lst(T_02, L_1f2, need_interp_0)
    T_20 = _compress_cols_from_lst(T_20, L_2f1, need_interp_0)
    v_prime_0 = _compress_rows_from_lst(v_prime_0, L_1f2, need_interp_0)

    # Do all of the compression ops specified by need_interp_1
    T_01 = _compress_cols_from_lst(T_01, L_2f1, need_interp_1)
    T_10 = _compress_rows_from_lst(T_10, L_1f2, need_interp_1)
    T_11 = _compress_rows_from_lst(T_11, L_1f2, need_interp_1)
    T_11 = _compress_cols_from_lst(T_11, L_2f1, need_interp_1)
    T_12 = _compress_rows_from_lst(T_12, L_1f2, need_interp_1)
    T_21 = _compress_cols_from_lst(T_21, L_2f1, need_interp_1)
    v_prime_1 = _compress_rows_from_lst(v_prime_1, L_1f2, need_interp_1)

    # Do all of the compression ops specified by need_interp_2
    T_02 = _compress_cols_from_lst(T_02, L_2f1, need_interp_2)
    T_20 = _compress_rows_from_lst(T_20, L_1f2, need_interp_2)
    T_22 = _compress_rows_from_lst(T_22, L_1f2, need_interp_2)
    T_22 = _compress_cols_from_lst(T_22, L_2f1, need_interp_2)
    T_12 = _compress_cols_from_lst(T_12, L_2f1, need_interp_2)
    T_21 = _compress_rows_from_lst(T_21, L_1f2, need_interp_2)
    v_prime_2 = _compress_rows_from_lst(v_prime_2, L_1f2, need_interp_2)

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


def _compress_cols_from_lst(
    T: jnp.array,
    L: jnp.array,
    compression_bools: jnp.array,
) -> Tuple[jnp.ndarray]:
    """
    Suppose compression_bools has length (n,). Then, we expect T to be a block matrix
    T = [T_1 T_2, ..., T_n].

    We iterate through compression_bools, and if compression_bools[i] and compression_bools[i+1]:
    We replace [T_i   T_{i+1}] with [T_i  T_{i+1} ] @ L
    """
    q = L.shape[1]
    n = compression_bools.shape[0]
    out_lst = []
    i = 0
    while i < n:
        # Here is some code in case I want  to compile this function one day:

        # v = jnp.where(
        #     jnp.logical_and(compression_bools[i], compression_bools[i + 1]), 1, 0
        # )
        # X, i = jax.lax.switch(v, (_compress_col, _dont_compress_col), T, L, i)
        if compression_bools[i] and compression_bools[i + 1]:
            X = T[:, q * i : q * (i + 2)] @ L
            i += 2
        else:
            X = T[:, q * i : q * (i + 1)]
            i += 1
        out_lst.append(X)
    out_T = jnp.concatenate(out_lst, axis=1)
    return out_T


def _compress_rows_from_lst(
    T: jnp.array,
    L: jnp.array,
    compression_bools: jnp.array,
) -> Tuple[jnp.array]:
    """
    Suppose compression_bools has length (n,). Then we expect T to be a block matrix
    T = [T_1
         T_2
         ...
         T_n]

    We iterate through compression_bools, and if compression_bools[i] and compression_bools[i+1]:
    We replace [T_i
                 T_{i+1}] with L @ [T_i
                                 T_{i+1}]
    """
    q = L.shape[0]
    n = compression_bools.shape[0]
    out_lst = []
    i = 0
    while i < n:
        if compression_bools[i] and compression_bools[i + 1]:
            X = L @ T[q * i : q * (i + 2)]
            i += 2
        else:
            X = T[q * i : q * (i + 1)]
            i += 1
        out_lst.append(X)
    out_T = jnp.concatenate(out_lst, axis=0)
    return out_T
