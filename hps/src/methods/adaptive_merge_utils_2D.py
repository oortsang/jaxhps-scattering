from typing import Tuple

import jax.numpy as jnp

from hps.src.quadrature.trees import Node, get_all_leaves


# @jax.jit
def _find_compression_list_5(
    node_a: Node, node_b: Node
) -> Tuple[jnp.array, jnp.array]:
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
    node_b: Node, node_c: Node
) -> Tuple[jnp.array, jnp.array]:
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
    node_c: Node, node_d: Node
) -> Tuple[jnp.array, jnp.array]:
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
    node_d: Node, node_a: Node
) -> Tuple[jnp.array, jnp.array]:
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
    node_a: Node, node_b: Node, node_c: Node, node_d: Node
) -> Tuple[jnp.array]:
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
        jnp.array: Output has shape shape (n // 4, 12)
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
