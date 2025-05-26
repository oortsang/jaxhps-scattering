import jax
import jax.numpy as jnp
from typing import List, Tuple

from .._discretization_tree import (
    get_depth,
    get_nodes_at_level,
    get_all_leaves,
)
from .._pdeproblem import PDEProblem


from ..merge._utils_adaptive_2D_DtN import find_compression_lists_2D


def down_pass_adaptive_2D_DtN(
    pde_problem: PDEProblem,
    boundary_data: List[jax.Array],
) -> jax.Array:
    """
    This function performs the downward pass for 2D adaptive discretizations using DtN matrices.

    The boundary data is assumed to be **Dirichlet data**

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    boundary_data : List[jax.Array]
        A length-4 list of arrays, specifying samples of the boundary data. :func:`jaxhps.Domain.get_adaptive_boundary_data_lst` is a utility for constructing this list.

    Returns
    -------
    solns : jax.Array
        The solutions on the interior points. Has shape (n_leaves, p^2)

    """

    root = pde_problem.domain.root
    depth = get_depth(root)

    bdry_data_lst = [
        boundary_data,
    ]

    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(depth + 1):
        nodes_this_level = get_nodes_at_level(root, level)
        n_nodes = len(nodes_this_level)

        new_bdry_data_lst = []

        for i in range(n_nodes):
            node = nodes_this_level[i]
            bdry_data = bdry_data_lst[i]
            # print("_down_pass_2D: working on node i=", i)

            if len(node.children):
                # Keep propogating information down the tree.
                S = node.data.S
                g_tilde = node.data.g_tilde

                # Find the children of the current node and check which panels
                # need to be refined.
                node_a = node.children[0]
                node_b = node.children[1]
                node_c = node.children[2]
                node_d = node.children[3]

                compression_lsts = find_compression_lists_2D(
                    node_a, node_b, node_c, node_d
                )

                # Use the compression lists to inform _propogate_down_quad
                # which parts of the interface need refinement after the S map.
                o = _propogate_down_quad(
                    S,
                    bdry_data,
                    g_tilde,
                    n_a_0=node.children[0].n_0,
                    n_b_0=node.children[1].n_0,
                    n_b_1=node.children[1].n_1,
                    n_c_1=node.children[2].n_1,
                    n_c_2=node.children[2].n_2,
                    n_d_2=node.children[3].n_2,
                    n_d_3=node.children[3].n_3,
                    n_a_3=node.children[0].n_3,
                    compression_lsts=compression_lsts,
                    refinement_op=pde_problem.L_2f1,
                )

                new_bdry_data_lst.extend(o)
            else:
                # We are at a leaf node. Set the boundary data for later. Transform it from a list to an
                # array, because at the end of the down pass, we will use it to propogate the solution
                # to the interior of the leaf node via Y @ bdry_data
                node.data.g = jnp.concatenate(bdry_data)

        bdry_data_lst = new_bdry_data_lst

    for i, leaf in enumerate(get_all_leaves(root)):
        leaf_homog_solns = leaf.data.Y @ leaf.data.g
        leaf_solns = leaf_homog_solns + leaf.data.v
        leaf.data.u = leaf_solns

    leaf_solns_out = jnp.stack([leaf.data.u for leaf in get_all_leaves(root)])
    return leaf_solns_out


def _decompress_merge_interface_2D(
    g_int: jax.Array,
    compression_lst_0: jax.Array,
    compression_lst_1: jax.Array,
    refinement_op: jax.Array,
    idx_g_int: int,
) -> Tuple[jax.Array, jax.Array, int]:
    q = refinement_op.shape[1]
    n_panels_0 = compression_lst_0.shape[0]
    n_panels_1 = compression_lst_1.shape[0]

    idx_0 = 0
    idx_1 = 0
    g_int_0 = []
    g_int_1 = []
    while idx_0 < n_panels_0 and idx_1 < n_panels_1:
        g_int_panel = g_int[idx_g_int : idx_g_int + q]
        if compression_lst_0[idx_0]:
            # Refine this panel
            g_int_panel_0 = refinement_op @ g_int_panel
            idx_0 += 2
        else:
            g_int_panel_0 = g_int_panel
            idx_0 += 1

        if compression_lst_1[idx_1]:
            # Refine this panel
            g_int_panel_1 = refinement_op @ g_int_panel
            idx_1 += 2
        else:
            g_int_panel_1 = g_int_panel
            idx_1 += 1

        g_int_0.append(g_int_panel_0)
        g_int_1.append(g_int_panel_1)

        idx_g_int = idx_g_int + q

    g_int_0 = jnp.concatenate(g_int_0)
    g_int_1 = jnp.concatenate(g_int_1)

    return (g_int_0, g_int_1, idx_g_int)


def _propogate_down_quad(
    S_arr: jax.Array,
    bdry_data_lst: List[jax.Array],
    g_tilde: jax.Array,
    n_a_0: int,
    n_b_0: int,
    n_b_1: int,
    n_c_1: int,
    n_c_2: int,
    n_d_2: int,
    n_d_3: int,
    n_a_3: int,
    compression_lsts: Tuple[jax.Array],
    refinement_op: jax.Array,
) -> List[List[jax.Array]]:
    """
    Given homogeneous data on the boundary, interface homogeneous solution operator S, and
    interface particular solution data, this function returns the solution on the boundaries
    of the four children.

    suppose n_child is the number of quadrature points on EACH SIDE of a child node.

    Args:
        S_arr (jax.Array): Has shape (4 * n_child, 8 * n_child)
        bdry_data (jax.Array): 8 * n_child
        v_int_data (jax.Array): 4 * n_child
        compression_lsts (Tuple[jax.Array]): Tuple of 8 arrays of booleans. Each array
            indicates which panels were compressed during the merge and analagously
            which panels need to be refined during the down pass.
        refinement_op: (jax.Array): Has shape (2 * q, q)
    Returns:
        jax.Array: Has shape (4, 4 * n_child)
    """

    g_int = S_arr @ jnp.concatenate(bdry_data_lst) + g_tilde

    idx_g_int = 0

    # First we need to figure out which parts of g_int belong to
    # merge interface 5.
    # Remember, all of these slices of g_int are propogating
    # from OUTSIDE to INSIDE

    g_int_a_5, g_int_b_5, idx_g_int = _decompress_merge_interface_2D(
        g_int,
        compression_lsts[0],
        compression_lsts[1],
        refinement_op,
        idx_g_int,
    )

    g_int_b_6, g_int_c_6, idx_g_int = _decompress_merge_interface_2D(
        g_int,
        compression_lsts[2],
        compression_lsts[3],
        refinement_op,
        idx_g_int,
    )

    g_int_c_7, g_int_d_7, idx_g_int = _decompress_merge_interface_2D(
        g_int,
        compression_lsts[4],
        compression_lsts[5],
        refinement_op,
        idx_g_int,
    )

    g_int_d_8, g_int_a_8, idx_g_int = _decompress_merge_interface_2D(
        g_int,
        compression_lsts[6],
        compression_lsts[7],
        refinement_op,
        idx_g_int,
    )

    # g_a is a list of the boundary data for the four sides of child a.
    g_a = [
        bdry_data_lst[0][:n_a_0],  # S edge
        g_int_a_5,  # E edge
        jnp.flipud(g_int_a_8),  # N edge
        bdry_data_lst[3][n_d_3:],  # W edge
    ]

    g_b = [
        bdry_data_lst[0][n_a_0:],  # S edge
        bdry_data_lst[1][:n_b_1],  # E edge
        g_int_b_6,  # N edge
        jnp.flipud(g_int_b_5),  # W edge
    ]

    g_c = [
        jnp.flipud(g_int_c_6),  # S edge
        bdry_data_lst[1][n_b_1:],  # E edge
        bdry_data_lst[2][:n_c_2],  # N edge
        g_int_c_7,  # W edge
    ]

    g_d = [
        g_int_d_8,  # S edge
        jnp.flipud(g_int_d_7),  # E edge
        bdry_data_lst[2][n_c_2:],  # N edge, W edge
        bdry_data_lst[3][:n_d_3],
    ]
    return [g_a, g_b, g_c, g_d]


vmapped_propogate_down_quad = jax.vmap(
    _propogate_down_quad, in_axes=(0, 0, 0), out_axes=0
)
