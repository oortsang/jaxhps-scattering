import jax
import jax.numpy as jnp
from typing import List, Tuple
from .._discretization_tree import (
    get_depth,
    get_nodes_at_level,
    get_all_leaves,
)
from .._pdeproblem import PDEProblem


from ..merge._utils_adaptive_3D_DtN import find_projection_lists_3D


def down_pass_adaptive_3D_DtN(
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
        A length-6 list of arrays, specifying samples of the boundary data. :func:`jaxhps.Domain.get_adaptive_boundary_data_lst` is a utility for constructing this list.

    Returns
    -------
    solns : jax.Array
        The solutions on the interior points. Has shape (n_leaves, p^3)

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
            # print("_down_pass_3D: working on node i=", i)

            if len(node.children):
                S = node.data.S
                g_tilde = node.data.g_tilde

                # print("_down_pass_3D: calling _propagate_down_oct")
                # print(
                #     "_down_pass_3D: bdry_data shapes = ", [b.shape for b in bdry_data]
                # )
                child_a = node.children[0]
                child_b = node.children[1]
                child_c = node.children[2]
                child_d = node.children[3]
                child_e = node.children[4]
                child_f = node.children[5]
                child_g = node.children[6]
                child_h = node.children[7]

                # logging.debug("_down_pass_3D: child_a: %s", child_a)
                # logging.debug("_down_pass_3D: child_b: %s", child_b)
                # logging.debug("_down_pass_3D: child_c: %s", child_c)
                # logging.debug("_down_pass_3D: child_d: %s", child_d)
                # logging.debug("_down_pass_3D: child_e: %s", child_e)
                # logging.debug("_down_pass_3D: child_f: %s", child_f)
                # logging.debug("_down_pass_3D: child_g: %s", child_g)

                # logging.debug("_down_pass_3D: S shape: %s", S.shape)
                # logging.debug(
                #     "_down_pass_3D: bdry_data shapes: %s", [s.shape for s in bdry_data]
                # )
                # logging.debug("_down_pass_3D: v_int shape: %s", v_int.shape)

                projection_lsts = find_projection_lists_3D(
                    child_a,
                    child_b,
                    child_c,
                    child_d,
                    child_e,
                    child_f,
                    child_g,
                    child_h,
                )
                o = _propagate_down_oct(
                    S,
                    bdry_data,
                    g_tilde,
                    n_a_0=child_a.n_0,
                    n_a_2=child_a.n_2,
                    n_a_5=child_a.n_5,
                    n_b_1=child_b.n_1,
                    n_b_2=child_b.n_2,
                    n_b_5=child_b.n_5,
                    n_c_1=child_c.n_1,
                    n_c_3=child_c.n_3,
                    n_c_5=child_c.n_5,
                    n_d_0=child_d.n_0,
                    n_d_3=child_d.n_3,
                    n_e_0=child_e.n_0,
                    n_e_2=child_e.n_2,
                    n_e_4=child_e.n_4,
                    n_f_1=child_f.n_1,
                    n_f_2=child_f.n_2,
                    n_f_4=child_f.n_4,
                    n_g_1=child_g.n_1,
                    n_g_3=child_g.n_3,
                    n_g_4=child_g.n_4,
                    n_h_0=child_h.n_0,
                    n_h_3=child_h.n_3,
                    projection_lsts=projection_lsts,
                    refinement_op=pde_problem.L_4f1,
                )

                new_bdry_data_lst.extend(o)
            else:
                # We are at a leaf node. Set the boundary data for later. Transform it from a list to an
                # array, because at the end of the down pass, we will use it to propagate the solution
                # to the interior of the leaf node via Y @ bdry_data
                node.data.g = jnp.concatenate(bdry_data)
        bdry_data_lst = new_bdry_data_lst

    for i, leaf in enumerate(get_all_leaves(root)):
        leaf_homog_solns = leaf.data.Y @ leaf.data.g
        leaf_solns = leaf_homog_solns + leaf.data.v
        leaf.data.u = leaf_solns

    leaf_solns_out = jnp.stack([leaf.data.u for leaf in get_all_leaves(root)])
    return leaf_solns_out


def _deproject_merge_interface_3D(
    g_int: jnp.array,
    projection_lst_0: jnp.array,
    projection_lst_1: jnp.array,
    refinement_op: jnp.array,
    idx_g_int: int,
) -> Tuple[jnp.array, jnp.array, int]:
    n_per_panel = refinement_op.shape[1]
    n_panels_0 = projection_lst_0.shape[0]
    n_panels_1 = projection_lst_1.shape[0]

    idx_0 = 0
    idx_1 = 0
    g_int_0 = []
    g_int_1 = []
    while idx_0 < n_panels_0 and idx_1 < n_panels_1:
        g_int_panel = g_int[idx_g_int : idx_g_int + n_per_panel]
        if projection_lst_0[idx_0]:
            # Refine this panel
            g_int_panel_0 = refinement_op @ g_int_panel
            idx_0 += 4
        else:
            g_int_panel_0 = g_int_panel
            idx_0 += 1

        if projection_lst_1[idx_1]:
            g_int_panel_1 = refinement_op @ g_int_panel
            idx_1 += 4
        else:
            g_int_panel_1 = g_int_panel
            idx_1 += 1

        g_int_0.append(g_int_panel_0)
        g_int_1.append(g_int_panel_1)

        idx_g_int = idx_g_int + n_per_panel

    g_int_0 = jnp.concatenate(g_int_0)
    g_int_1 = jnp.concatenate(g_int_1)
    return (g_int_0, g_int_1, idx_g_int)


def _propagate_down_oct(
    S_arr: jnp.ndarray,
    bdry_data_lst: List[jnp.ndarray],
    v_int_data: jnp.ndarray,
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
    projection_lsts: Tuple[jnp.array],
    refinement_op: jnp.ndarray,
) -> jnp.ndarray:
    """_summary_

    Args:
        S_arr (jnp.ndarray): Has shape (12 * n_per_face, 24 * n_per_face)
        bdry_data (jnp.ndarray): Has shape (24 * n_per_face,)
        v_int_data (jnp.ndarray): Has shape (12 * n_per_face,)

    Returns:
        jnp.ndarray: Has shape (8, 6 * n_per_face)
    """

    g_int = S_arr @ jnp.concatenate(bdry_data_lst) + v_int_data

    idx_g_int = 0

    # First, we need to figure out which parts of g_int belong to
    # which merge interfaces. At the same time, we need to deproject
    # the merge interfaces that were projected during the merge.

    g_int_a_9, g_int_b_9, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[0],
        projection_lsts[1],
        refinement_op,
        idx_g_int,
    )
    g_int_b_10, g_int_c_10, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[2],
        projection_lsts[3],
        refinement_op,
        idx_g_int,
    )
    g_int_c_11, g_int_d_11, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[4],
        projection_lsts[5],
        refinement_op,
        idx_g_int,
    )
    g_int_d_12, g_int_a_12, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[6],
        projection_lsts[7],
        refinement_op,
        idx_g_int,
    )
    g_int_e_13, g_int_f_13, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[8],
        projection_lsts[9],
        refinement_op,
        idx_g_int,
    )
    g_int_f_14, g_int_g_14, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[10],
        projection_lsts[11],
        refinement_op,
        idx_g_int,
    )
    g_int_g_15, g_int_h_15, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[12],
        projection_lsts[13],
        refinement_op,
        idx_g_int,
    )
    g_int_h_16, g_int_e_16, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[14],
        projection_lsts[15],
        refinement_op,
        idx_g_int,
    )
    g_int_a_17, g_int_e_17, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[16],
        projection_lsts[17],
        refinement_op,
        idx_g_int,
    )
    g_int_b_18, g_int_f_18, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[18],
        projection_lsts[19],
        refinement_op,
        idx_g_int,
    )
    g_int_c_19, g_int_g_19, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[20],
        projection_lsts[21],
        refinement_op,
        idx_g_int,
    )
    g_int_d_20, g_int_h_20, idx_g_int = _deproject_merge_interface_3D(
        g_int,
        projection_lsts[22],
        projection_lsts[23],
        refinement_op,
        idx_g_int,
    )

    # g_a is a list of the boundary data for the 6 faces of child a.
    g_a = [
        bdry_data_lst[0][-n_a_0:],  # Face 0
        g_int_a_9,  # Face 1
        bdry_data_lst[2][-n_a_2:],  # Face 2
        g_int_a_12,  # Face 3
        g_int_a_17,  # Face 4
        bdry_data_lst[5][:n_a_5],  # Face 5
    ]

    # g_b is a list of the boundary data for the 6 faces of child b.
    g_b = [
        g_int_b_9,  # Face 0
        bdry_data_lst[1][-n_b_1:],  # Face 1
        bdry_data_lst[2][n_e_2 + n_f_2 : n_e_2 + n_f_2 + n_b_2],  # Face 2
        g_int_b_10,  # Face 3
        g_int_b_18,  # Face 4
        bdry_data_lst[5][n_a_5 : n_a_5 + n_b_5],  # Face 5
    ]

    # g_c is a list of the boundary data for the 6 faces of child c.
    g_c = [
        g_int_c_11,  # Face 0
        bdry_data_lst[1][n_f_1 + n_g_1 : n_f_1 + n_g_1 + n_c_1],  # Face 1
        g_int_c_10,
        bdry_data_lst[3][n_h_3 + n_g_3 : n_h_3 + n_g_3 + n_c_3],  # Face 3
        g_int_c_19,  # Face 4
        bdry_data_lst[5][n_a_5 + n_b_5 : n_a_5 + n_b_5 + n_c_5],  # Face 5
    ]

    g_d = [
        bdry_data_lst[0][n_e_0 + n_h_0 : n_e_0 + n_h_0 + n_d_0],  # Face 0
        g_int_d_11,
        g_int_d_12,
        bdry_data_lst[3][-n_d_3:],  # Face 3
        g_int_d_20,
        bdry_data_lst[5][n_a_5 + n_b_5 + n_c_5 :],  # Face 5
    ]

    g_e = [
        bdry_data_lst[0][:n_e_0],  # Face 0
        g_int_e_13,
        bdry_data_lst[2][:n_e_2],  # Face 2
        g_int_e_16,
        bdry_data_lst[4][:n_e_4],  # Face 4
        g_int_e_17,
    ]

    g_f = [
        g_int_f_13,
        bdry_data_lst[1][:n_f_1],  # Face 1
        bdry_data_lst[2][n_e_2 : n_e_2 + n_f_2],  # Face 2
        g_int_f_14,
        bdry_data_lst[4][n_e_4 : n_e_4 + n_f_4],  # Face 4
        g_int_f_18,
    ]

    g_g = [
        g_int_g_15,
        bdry_data_lst[1][n_f_1 : n_f_1 + n_g_1],  # Face 1
        g_int_g_14,
        bdry_data_lst[3][n_h_3 : n_h_3 + n_g_3],  # Face 3
        bdry_data_lst[4][n_e_4 + n_f_4 : n_e_4 + n_f_4 + n_g_4],  # Face 4
        g_int_g_19,
    ]

    g_h = [
        bdry_data_lst[0][n_e_0 : n_e_0 + n_h_0],  # Face 0
        g_int_h_15,
        g_int_h_16,
        bdry_data_lst[3][:n_h_3],  # Face 3
        bdry_data_lst[4][n_e_4 + n_f_4 + n_g_4 :],  # Face 4
        g_int_h_20,
    ]
    return [g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h]


vmapped_propagate_down_oct = jax.vmap(
    _propagate_down_oct, in_axes=(0, 0, 0), out_axes=0
)
