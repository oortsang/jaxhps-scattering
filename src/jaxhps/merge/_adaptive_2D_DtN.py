import logging
from typing import Tuple, List

import jax.numpy as jnp

from .._discretization_tree import (
    DiscretizationNode2D,
    get_depth,
    get_nodes_at_level,
)
from .._pdeproblem import PDEProblem
from ._schur_complement import (
    assemble_merge_outputs_DtN,
)
from ._utils_adaptive_2D_DtN import (
    find_compression_lists_2D,
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)


def merge_stage_adaptive_2D_DtN(pde_problem: PDEProblem) -> None:
    """
    Implements adaptive 2D merges of DtN matrices. Merges the nodes in the quadtree four at a time,
    projecting the rows and columns of DtN matrices as necessary when neighboring nodes have different
    boundary discretizations.

    This function saves the output of each merge step inside the DiscretizationNode2D.data objects that
    define the quadtree. After this function is called, the top-level DtN matrix can be accessed at
    ``pde_problem.domain.root.data.T``.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    Returns
    -------
    None. All of the solution operators are stored inside the pde_problem object.

    """
    logging.debug("merge_stage_adaptive_2D_DtN: started")

    root = pde_problem.domain.root
    depth = get_depth(root)
    logging.debug("merge_stage_adaptive_2D_DtN: depth = %d", depth)

    # Perform a merge operation for each level of the tree. Start at the leaves
    # and continue until the root.
    # For comments, suppose the for loop is in iteration j.
    for j, i in enumerate(range(depth, 0, -1)):
        # Get the inputs to the merge operation from the tree.
        nodes_this_level = get_nodes_at_level(root, i)
        T_this_level = [n.data.T for n in nodes_this_level]
        h_this_level = [n.data.h for n in nodes_this_level]

        # The leaves in need of refinement are those who have more children than the min number
        # of children among siblings.
        # Has shape (m // 4, 8)

        # Expect DtN_arr to be a list of length (m // 4) where each element is a matrix.
        S_arr_lst, T_arr_lst, h_lst, g_tilde_lst = (
            quad_merge_nonuniform_whole_level(
                T_this_level,
                h_this_level,
                pde_problem.L_2f1,
                pde_problem.L_1f2,
                nodes_this_level,
            )
        )
        logging.debug("merge_stage_adaptive_2D_DtN: just merged level %s", i)
        logging.debug(
            "merge_stage_adaptive_2D_DtN: T_arr_lst[0].shape = %s",
            T_arr_lst[0].shape,
        )

        # print("_merge_stage_2D: S_arr_lst len = ", len(S_arr_lst))

        # Set the output in the tree object.
        nodes_next_level = get_nodes_at_level(root, i - 1)
        # print("_merge_stage_2D: nodes_next_level len = ", len(nodes_next_level))

        # Filter out the nodes which are leaves.
        nodes_next_level = [
            node for node in nodes_next_level if len(node.children)
        ]
        # print("_merge_stage_2D: nodes_next_level len = ", len(nodes_next_level))
        for k, node in enumerate(nodes_next_level):
            node.data.T = T_arr_lst[k]
            node.data.h = h_lst[k]
            node.data.S = S_arr_lst[k]
            node.data.g_tilde = g_tilde_lst[k]
            if k == 0:
                logging.debug(
                    "merge_stage_adaptive_2D_DtN: setting data for node %s",
                    node,
                )
                logging.debug(
                    "merge_stage_adaptive_2D_DtN: node.data.T.shape = %s",
                    node.data.T.shape,
                )
                logging.debug(
                    "merge_stage_adaptive_2D_DtN: id(node.data.T) = %s",
                    id(node.data.T),
                )


def quad_merge_nonuniform_whole_level(
    T_in: List[jnp.ndarray],
    h_in: List[jnp.ndarray],
    L_2f1: jnp.ndarray,
    L_1f2: jnp.ndarray,
    nodes_this_level: List[DiscretizationNode2D],
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    This function takes in pre-computed DtN matrices and v_prime vectors, as well
    as a list of Nodes, and merges the Nodes 4 at a time. It does the following
    operations:
    1. Splits the input list of nodes into groups of 4 for each merge operation.
    2. Gathers node information about the number of quadrature points along each side
    3. Gathers information about which panels in the nodes need compression.
    4. Calls the _quad_merge function to perform the merge operation.
    5. Returns the results of the merge operation.

    Args:
        T_in (List[jnp.ndarray]): List has length (m,) and each element is a square matrix.
        v_prime (List[jnp.ndarray]): List has length (m,) and each element is a vector. The i'th element of this list
        should have the same shape as the i'th element of T_in.
        L_2f1 (jnp.ndarray): Interpolation operator with shape (2q, q)
        L_1f2 (jnp.ndarray): Interpolation operator with shape (q, 2q)
        nodes_this_level (List[Node]): List of Nodes being merged at this level. Has length (m,)

    Returns:
        Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]: In order:
        S_lst: List of matrices mapping boundary data to merge interfaces. Has length (m // 4).
        T_lst: DtN matrices for the merged nodes. Has length (m // 4).
        v_prime_ext_lst: Boundary particular fluxes. Has length (m // 4).
        v_lst: Particular solutions evaluated on the merge interfaces. Has length (m // 4).
    """
    S_lst = []
    T_lst = []
    h_lst = []
    g_tilde_lst = []
    n_merges = len(T_in) // 4

    for i in range(n_merges):
        node_a = nodes_this_level[4 * i]
        node_b = nodes_this_level[4 * i + 1]
        node_c = nodes_this_level[4 * i + 2]
        node_d = nodes_this_level[4 * i + 3]

        side_lens_a = jnp.array(
            [node_a.n_0, node_a.n_1, node_a.n_2, node_a.n_3],
            dtype=jnp.int32,
        )
        side_lens_b = jnp.array(
            [node_b.n_0, node_b.n_1, node_b.n_2, node_b.n_3],
            dtype=jnp.int32,
        )
        side_lens_c = jnp.array(
            [node_c.n_0, node_c.n_1, node_c.n_2, node_c.n_3],
            dtype=jnp.int32,
        )
        side_lens_d = jnp.array(
            [node_d.n_0, node_d.n_1, node_d.n_2, node_d.n_3],
            dtype=jnp.int32,
        )

        need_interp_lsts = find_compression_lists_2D(
            node_a, node_b, node_c, node_d
        )
        # print(
        #     "quad_merge_nonuniform_whole_level: need_interp_lsts = ", need_interp_lsts
        # )
        # print(
        #     "quad_merge_nonuniform_whole_level: len(need_interp_lsts) = ",
        #     len(need_interp_lsts),
        # )

        S, T, h_out, g_tilde = _adaptive_quad_merge_2D_DtN(
            T_in[4 * i],
            T_in[4 * i + 1],
            T_in[4 * i + 2],
            T_in[4 * i + 3],
            h_in[4 * i],
            h_in[4 * i + 1],
            h_in[4 * i + 2],
            h_in[4 * i + 3],
            L_2f1,
            L_1f2,
            need_interp_lsts=need_interp_lsts,
            side_lens_a=side_lens_a,
            side_lens_b=side_lens_b,
            side_lens_c=side_lens_c,
            side_lens_d=side_lens_d,
        )
        # print("quad_merge_whole_level: v_prime_ext shape", v_prime_ext.shape)
        S_lst.append(S)
        T_lst.append(T)
        h_lst.append(h_out)
        g_tilde_lst.append(g_tilde)

    return S_lst, T_lst, h_lst, g_tilde_lst


def _adaptive_quad_merge_2D_DtN(
    T_a: jnp.array,
    T_b: jnp.array,
    T_c: jnp.array,
    T_d: jnp.array,
    v_prime_a: jnp.array,
    v_prime_b: jnp.array,
    v_prime_c: jnp.array,
    v_prime_d: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_lsts: Tuple[jnp.array],
    side_lens_a: jnp.array,
    side_lens_b: jnp.array,
    side_lens_c: jnp.array,
    side_lens_d: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Takes in the DtN matrices and v_prime vectors for a set of four nodes being merged
    together. The function then performs the merge operation and returns the results.

    Args:
        T_a (jnp.array): DtN matrix for node a.
        T_b (jnp.array): DtN matrix for node b.
        T_c (jnp.array): DtN matrix for node c.
        T_d (jnp.array): DtN matrix for node d.
        v_prime_a (jnp.array):
        v_prime_b (jnp.array):
        v_prime_c (jnp.array):
        v_prime_d (jnp.array):
        L_2f1 (jnp.array): Interpolation operator with shape (2q, q)
        L_1f2 (jnp.array): Interpolation operator with shape (q, 2q)
        need_interp_lsts (Tuple[jnp.array]): Tuple of length 8. Each element is a boolean array
            indicating whether particular panels should be interpolated or not. The order is:
            a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8. Each array is ordered from the outside inward.
        side_lens_a (jnp.array): Length 4 array indicating the number of quadrature points along each side of node a.
        side_lens_b (jnp.array):
        side_lens_c (jnp.array):
        side_lens_d (jnp.array):

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
        S: Matrix mapping boundary data to merge interfaces.
        T: DtN matrix for the merged node.
        v_prime_ext: Boundary particular fluxes.
        v_int: Particular solutions evaluated on the merge interfaces.
    """

    (
        v_prime_a_1,
        v_prime_a_5,
        v_prime_a_8,
        T_a_11,
        T_a_15,
        T_a_18,
        T_a_51,
        T_a_55,
        T_a_58,
        T_a_81,
        T_a_85,
        T_a_88,
    ) = get_quadmerge_blocks_a(
        T_a,
        v_prime_a,
        L_2f1,
        L_1f2,
        need_interp_lsts[0],
        need_interp_lsts[7],
        side_lens_a[0],
        side_lens_a[1],
        side_lens_a[2],
        side_lens_a[3],
    )
    n_1, n_5 = T_a_15.shape
    n_8 = T_a_18.shape[1]

    (
        v_prime_b_2,
        v_prime_b_6,
        v_prime_b_5,
        T_b_22,
        T_b_26,
        T_b_25,
        T_b_62,
        T_b_66,
        T_b_65,
        T_b_52,
        T_b_56,
        T_b_55,
    ) = get_quadmerge_blocks_b(
        T_b,
        v_prime_b,
        L_2f1,
        L_1f2,
        need_interp_lsts[2],
        need_interp_lsts[1],
        side_lens_b[0],
        side_lens_b[1],
        side_lens_b[2],
        side_lens_b[3],
    )
    n_2, n_6 = T_b_26.shape

    (
        v_prime_c_6,
        v_prime_c_3,
        v_prime_c_7,
        T_c_66,
        T_c_63,
        T_c_67,
        T_c_36,
        T_c_33,
        T_c_37,
        T_c_76,
        T_c_73,
        T_c_77,
    ) = get_quadmerge_blocks_c(
        T_c,
        v_prime_c,
        L_2f1,
        L_1f2,
        need_interp_lsts[3],
        need_interp_lsts[4],
        side_lens_c[0],
        side_lens_c[1],
        side_lens_c[2],
        side_lens_c[3],
    )
    n_3, n_7 = T_c_37.shape

    (
        v_prime_d_8,
        v_prime_d_7,
        v_prime_d_4,
        T_d_88,
        T_d_87,
        T_d_84,
        T_d_78,
        T_d_77,
        T_d_74,
        T_d_48,
        T_d_47,
        T_d_44,
    ) = get_quadmerge_blocks_d(
        T_d,
        v_prime_d,
        L_2f1,
        L_1f2,
        need_interp_lsts[6],
        need_interp_lsts[5],
        side_lens_d[0],
        side_lens_d[1],
        side_lens_d[2],
        side_lens_d[3],
    )
    n_8, n_4 = T_d_84.shape

    # print("_quad_merge: T_a_15 shape: ", T_a_15.shape)
    # print("_quad_merge: n_5 = ", n_5)
    assert T_b_25.shape == (n_2, n_5)
    assert T_b_26.shape == (n_2, n_6)

    B_0 = jnp.block(
        [T_a_15, jnp.zeros((n_1, n_6)), jnp.zeros((n_1, n_7)), T_a_18]
    )
    B_1 = jnp.block(
        [T_b_25, T_b_26, jnp.zeros((n_2, n_7)), jnp.zeros((n_2, n_8))]
    )
    B_2 = jnp.block(
        [jnp.zeros((n_3, n_5)), T_c_36, T_c_37, jnp.zeros((n_3, n_8))]
    )
    B_3 = jnp.block(
        [jnp.zeros((n_4, n_5)), jnp.zeros((n_4, n_6)), T_d_47, T_d_48]
    )
    # print("_quad_merge: B_0.shape = ", B_0.shape)
    # print("_quad_merge: B_1.shape = ", B_1.shape)
    # print("_quad_merge: B_2.shape = ", B_2.shape)
    # print("_quad_merge: B_3.shape = ", B_3.shape)

    B = jnp.concatenate([B_0, B_1, B_2, B_3], axis=0)
    # print("_quad_merge: B.shape: ", B.shape)
    C = jnp.block(
        [
            [T_a_51, T_b_52, jnp.zeros((n_5, n_3)), jnp.zeros((n_5, n_4))],
            [jnp.zeros((n_6, n_1)), T_b_62, T_c_63, jnp.zeros((n_6, n_4))],
            [jnp.zeros((n_7, n_1)), jnp.zeros((n_7, n_2)), T_c_73, T_d_74],
            [T_a_81, jnp.zeros((n_8, n_2)), jnp.zeros((n_8, n_3)), T_d_84],
        ]
    )

    D = jnp.block(
        [
            [T_a_55 + T_b_55, T_b_56, jnp.zeros((n_5, n_7)), T_a_58],
            [T_b_65, T_b_66 + T_c_66, T_c_67, jnp.zeros((n_6, n_8))],
            [jnp.zeros((n_7, n_5)), T_c_76, T_c_77 + T_d_77, T_d_78],
            [T_a_85, jnp.zeros((n_8, n_6)), T_d_87, T_d_88 + T_a_88],
        ]
    )
    A_lst = [T_a_11, T_b_22, T_c_33, T_d_44]
    delta_v_prime_int = jnp.concatenate(
        [
            v_prime_a_5 + v_prime_b_5,
            v_prime_b_6 + v_prime_c_6,
            v_prime_c_7 + v_prime_d_7,
            v_prime_d_8 + v_prime_a_8,
        ]
    )
    v_prime_ext = jnp.concatenate(
        [v_prime_a_1, v_prime_b_2, v_prime_c_3, v_prime_d_4]
    )

    # use the Schur complement code to compute the Schur complement
    T, S, v_prime_ext_out, v_int = assemble_merge_outputs_DtN(
        A_lst, B, C, D, v_prime_ext, delta_v_prime_int
    )

    # Roll the exterior to get the correct ordering. Before rolling
    # The ordering is boundary of A, ..., boundary of D. After rolling
    # The ordering is side 0, side 1, side 2, side 3.
    n_roll = side_lens_a[3]
    v_prime_ext = jnp.roll(v_prime_ext_out, -n_roll, axis=0)
    T = jnp.roll(T, -n_roll, axis=0)
    T = jnp.roll(T, -n_roll, axis=1)
    S = jnp.roll(S, -n_roll, axis=1)

    return S, T, v_prime_ext, v_int
