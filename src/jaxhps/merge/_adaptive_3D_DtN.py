import logging
from typing import Tuple, List, Any

import jax.numpy as jnp
import jax

from .._discretization_tree import (
    DiscretizationNode3D,
    get_nodes_at_level,
    get_all_leaves,
)
from .._pdeproblem import PDEProblem
from .._device_config import DEVICE_ARR, HOST_DEVICE
from ._uniform_3D_DtN import vmapped_uniform_oct_merge_DtN
from ._utils_adaptive_3D_DtN import (
    find_projection_lists_3D,
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
from ._schur_complement import _oct_merge_from_submatrices


def merge_stage_adaptive_3D_DtN(
    pde_problem: PDEProblem,
    T_arr: jax.Array,
    h_arr: jax.Array,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> None:
    """
    Implements adaptive 3D merges of DtN matrices. Merges the nodes in the quadtree eight at a time,
    projecting the rows and columns of DtN matrices as necessary when neighboring nodes have different
    boundary discretizations.

    This function saves the output of each merge step inside the DiscretizationNode3D.data objects that
    define the quadtree. After this function is called, the top-level DtN matrix can be accessed at
    ``pde_problem.domain.root.data.T``.

    This function takes as arguments parts of the local solve stage output. These arrays are indexed at the
    locations of the lowest-level leaves; the lowest-level leaves are then merged using the vectorized code
    used in the uniform version of this algorithm. This gives us a slight performance boost.
    At higher levels, the merges are not vectorized.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    T_arr : jax.Array
        Array of DtN matrices from the local solve stage. Has shape (n_leaves, 6q^2, 6q^2)

    h_arr : jax.Array
        Array of outgoing boundary data from the local solve stage. Has shape (n_leaves, 6q^2)

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    Returns
    -------
    None. All of the solution operators are stored inside the pde_problem object.

    """
    logging.debug("_build_stage_3D: started")

    q_idxes = jnp.arange(pde_problem.domain.q)

    root = pde_problem.domain.root

    # First, find all of the depths of the leaves
    leaves = get_all_leaves(root)
    depths = jnp.array([leaf.depth for leaf in leaves])
    max_depth = jnp.max(depths)
    lowest_level_bools = depths == max_depth

    logging.debug("_build_stage_3D: max_depth = %s", max_depth)

    # We only want to use the DtN arrays that are at the
    # lowest level of the tree
    T_lowest_level = T_arr[lowest_level_bools]
    h_lowest_level = h_arr[lowest_level_bools, ..., None]

    # Move these arrays to device
    T_lowest_level = jax.device_put(T_lowest_level, device)
    h_lowest_level = jax.device_put(h_lowest_level, device)

    # Vmapped code works on the lowest level
    S, T, h, g_tilde = vmapped_uniform_oct_merge_DtN(
        q_idxes, T_lowest_level, h_lowest_level
    )
    S = jax.device_put(S, host_device)
    T = jax.device_put(T, host_device)
    h = jax.device_put(h, host_device)[..., 0]  # Remove the last dimension
    g_tilde = jax.device_put(g_tilde, host_device)[
        ..., 0
    ]  # Remove the last dimension

    # Assign the outputs to the nodes at the penuultimate level
    parents_of_leaves = get_nodes_at_level(root, max_depth - 1)
    counter = 0
    for parent in parents_of_leaves:
        # logging.debug("_build_stage_3D: parent = %s", parent)
        if parent.data.T is not None:
            # Filter out the leaves at this level; they already have DtN matrices
            continue
        parent.data.T = T[counter]
        parent.data.h = h[counter]
        parent.data.S = S[counter]
        parent.data.g_tilde = g_tilde[counter]
        counter += 1
        D_shape = parent.data.S.shape[0]

    # Perform a merge operation for each level of the tree. Start at the leaves
    # and continue until the root.
    # For comments, suppose the for loop is in iteration j.
    for j, i in enumerate(range(max_depth - 2, -1, -1)):
        # logging.debug("_build_stage_3D: j = %s, i = %s", j, i)

        # Get the inpute to the merge operation from the tree.
        nodes_this_level = get_nodes_at_level(root, i)

        # Filter out the nodes which are leaves.
        nodes_this_level = [
            node for node in nodes_this_level if len(node.children)
        ]

        # Filter out the nodes which have DtN arrays already set
        nodes_this_level = [
            node for node in nodes_this_level if node.data.T is None
        ]

        # The leaves in need of refinement are those who have more children than the min number
        # of children among siblings
        # Has shape (m // 8, 8)
        D_shape = oct_merge_nonuniform_whole_level(
            pde_problem.L_4f1,
            pde_problem.L_1f4,
            nodes_this_level,
        )
    return D_shape


def _oct_merge(
    T_a: jax.Array,
    T_b: jax.Array,
    T_c: jax.Array,
    T_d: jax.Array,
    T_e: jax.Array,
    T_f: jax.Array,
    T_g: jax.Array,
    T_h: jax.Array,
    v_prime_a: jax.Array,
    v_prime_b: jax.Array,
    v_prime_c: jax.Array,
    v_prime_d: jax.Array,
    v_prime_e: jax.Array,
    v_prime_f: jax.Array,
    v_prime_g: jax.Array,
    v_prime_h: jax.Array,
    L_2f1: jax.Array,
    L_1f2: jax.Array,
    need_interp_lsts: Tuple[jax.Array],
    side_lens_a: jax.Array,
    side_lens_b: jax.Array,
    side_lens_c: jax.Array,
    side_lens_d: jax.Array,
    side_lens_e: jax.Array,
    side_lens_f: jax.Array,
    side_lens_g: jax.Array,
    side_lens_h: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    a_submatrices_subvecs = get_a_submatrices(
        T_a,
        v_prime_a,
        L_2f1,
        L_1f2,
        need_interp_9=need_interp_lsts[0],
        need_interp_12=need_interp_lsts[7],
        need_interp_17=need_interp_lsts[16],
        n_0=side_lens_a[0],
        n_1=side_lens_a[1],
        n_2=side_lens_a[2],
        n_3=side_lens_a[3],
        n_4=side_lens_a[4],
        n_5=side_lens_a[5],
    )
    a_submatrices_subvecs = [
        jax.device_put(a, DEVICE_ARR[0]) for a in a_submatrices_subvecs
    ]
    b_submatrices_subvecs = get_b_submatrices(
        T_b,
        v_prime_b,
        L_2f1,
        L_1f2,
        need_interp_9=need_interp_lsts[1],
        need_interp_10=need_interp_lsts[2],
        need_interp_18=need_interp_lsts[18],
        n_0=side_lens_b[0],
        n_1=side_lens_b[1],
        n_2=side_lens_b[2],
        n_3=side_lens_b[3],
        n_4=side_lens_b[4],
        n_5=side_lens_b[5],
    )
    b_submatrices_subvecs = [
        jax.device_put(b, DEVICE_ARR[0]) for b in b_submatrices_subvecs
    ]
    c_submatrices_subvecs = get_c_submatrices(
        T_c,
        v_prime_c,
        L_2f1,
        L_1f2,
        need_interp_10=need_interp_lsts[3],
        need_interp_11=need_interp_lsts[4],
        need_interp_19=need_interp_lsts[20],
        n_0=side_lens_c[0],
        n_1=side_lens_c[1],
        n_2=side_lens_c[2],
        n_3=side_lens_c[3],
        n_4=side_lens_c[4],
        n_5=side_lens_c[5],
    )
    c_submatrices_subvecs = [
        jax.device_put(c, DEVICE_ARR[0]) for c in c_submatrices_subvecs
    ]
    d_submatrices_subvecs = get_d_submatrices(
        T_d,
        v_prime_d,
        L_2f1,
        L_1f2,
        need_interp_11=need_interp_lsts[5],
        need_interp_12=need_interp_lsts[6],
        need_interp_20=need_interp_lsts[22],
        n_0=side_lens_d[0],
        n_1=side_lens_d[1],
        n_2=side_lens_d[2],
        n_3=side_lens_d[3],
        n_4=side_lens_d[4],
        n_5=side_lens_d[5],
    )
    d_submatrices_subvecs = [
        jax.device_put(d, DEVICE_ARR[0]) for d in d_submatrices_subvecs
    ]
    e_submatrices_subvecs = get_e_submatrices(
        T_e,
        v_prime_e,
        L_2f1,
        L_1f2,
        need_interp_13=need_interp_lsts[8],
        need_interp_16=need_interp_lsts[15],
        need_interp_17=need_interp_lsts[17],
        n_0=side_lens_e[0],
        n_1=side_lens_e[1],
        n_2=side_lens_e[2],
        n_3=side_lens_e[3],
        n_4=side_lens_e[4],
        n_5=side_lens_e[5],
    )
    e_submatrices_subvecs = [
        jax.device_put(e, DEVICE_ARR[0]) for e in e_submatrices_subvecs
    ]
    f_submatrices_subvecs = get_f_submatrices(
        T_f,
        v_prime_f,
        L_2f1,
        L_1f2,
        need_interp_13=need_interp_lsts[9],
        need_interp_14=need_interp_lsts[10],
        need_interp_18=need_interp_lsts[19],
        n_0=side_lens_f[0],
        n_1=side_lens_f[1],
        n_2=side_lens_f[2],
        n_3=side_lens_f[3],
        n_4=side_lens_f[4],
        n_5=side_lens_f[5],
    )
    f_submatrices_subvecs = [
        jax.device_put(f, DEVICE_ARR[0]) for f in f_submatrices_subvecs
    ]
    g_submatrices_subvecs = get_g_submatrices(
        T_g,
        v_prime_g,
        L_2f1,
        L_1f2,
        need_interp_14=need_interp_lsts[11],
        need_interp_15=need_interp_lsts[12],
        need_interp_19=need_interp_lsts[21],
        n_0=side_lens_g[0],
        n_1=side_lens_g[1],
        n_2=side_lens_g[2],
        n_3=side_lens_g[3],
        n_4=side_lens_g[4],
        n_5=side_lens_g[5],
    )
    g_submatrices_subvecs = [
        jax.device_put(g, DEVICE_ARR[0]) for g in g_submatrices_subvecs
    ]
    h_submatrices_subvecs = get_h_submatrices(
        T_h,
        v_prime_h,
        L_2f1,
        L_1f2,
        need_interp_15=need_interp_lsts[13],
        need_interp_16=need_interp_lsts[14],
        need_interp_20=need_interp_lsts[23],
        n_0=side_lens_h[0],
        n_1=side_lens_h[1],
        n_2=side_lens_h[2],
        n_3=side_lens_h[3],
        n_4=side_lens_h[4],
        n_5=side_lens_h[5],
    )
    h_submatrices_subvecs = [
        jax.device_put(h, DEVICE_ARR[0]) for h in h_submatrices_subvecs
    ]

    T, S, v_prime_ext_out, v_int = _oct_merge_from_submatrices(
        a_submatrices_subvecs=a_submatrices_subvecs,
        b_submatrices_subvecs=b_submatrices_subvecs,
        c_submatrices_subvecs=c_submatrices_subvecs,
        d_submatrices_subvecs=d_submatrices_subvecs,
        e_submatrices_subvecs=e_submatrices_subvecs,
        f_submatrices_subvecs=f_submatrices_subvecs,
        g_submatrices_subvecs=g_submatrices_subvecs,
        h_submatrices_subvecs=h_submatrices_subvecs,
    )

    T = jax.device_put(T, HOST_DEVICE)
    S = jax.device_put(S, HOST_DEVICE)
    v_prime_ext_out = jax.device_put(v_prime_ext_out, HOST_DEVICE)
    v_int = jax.device_put(v_int, HOST_DEVICE)

    r = get_rearrange_indices(
        jnp.arange(T.shape[0]),
        n_a_0=side_lens_a[0],
        n_a_2=side_lens_a[2],
        n_a_5=side_lens_a[5],
        n_b_1=side_lens_b[1],
        n_b_2=side_lens_b[2],
        n_b_5=side_lens_b[5],
        n_c_1=side_lens_c[1],
        n_c_3=side_lens_c[3],
        n_c_5=side_lens_c[5],
        n_d_0=side_lens_d[0],
        n_d_3=side_lens_d[3],
        n_d_5=side_lens_d[5],
        n_e_0=side_lens_e[0],
        n_e_2=side_lens_e[2],
        n_e_4=side_lens_e[4],
        n_f_1=side_lens_f[1],
        n_f_2=side_lens_f[2],
        n_f_4=side_lens_f[4],
        n_g_1=side_lens_g[1],
        n_g_3=side_lens_g[3],
        n_g_4=side_lens_g[4],
        n_h_0=side_lens_h[0],
        n_h_3=side_lens_h[3],
        n_h_4=side_lens_h[4],
    )
    v_prime_ext_out = v_prime_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, v_prime_ext_out, v_int


# def is_node_type(x: Any) -> bool:
#     return isinstance(x, Node)


def oct_merge_nonuniform_whole_level(
    L_4f1: jax.Array,
    L_1f4: jax.Array,
    nodes_this_level: List[DiscretizationNode3D],
) -> Tuple[List[jax.Array], List[jax.Array], List[jax.Array], List[jax.Array]]:
    """
    This function takes in pre-computed DtN matrices and v_prime vectors, as well
    as a list of Nodes, and merges the Nodes 8 at a time. It does the following
    operations:
    1. Splits the input list of nodes into groups of 8 for each merge operation.
    2. Gathers node information about the number of quadrature points along each side
    3. Gathers information about which panels in the nodes need projection.
    4. Calls the _oct_merge function to perform the merge operation.
    5. Returns the results of the merge operation.

    Args:
        T_in (List[jax.Array]): List has length (m,) and each element is a square matrix.
        v_prime (List[jax.Array]): List has length (m,) and each element is a vector. The i'th element of this list
        should have the same shape as the i'th element of T_in.
        L_2f1 (jax.Array): Interpolation operator with shape (4 q^2, q^2)
        L_1f2 (jax.Array): Interpolation operator with shape (q^2, 4q^2)
        nodes_this_level (List[Node]): List of Nodes being merged at this level. Has length (m,)

    Returns:
        Tuple[List[jax.Array], List[jax.Array], List[jax.Array], List[jax.Array]]: In order:
        S_lst: List of matrices mapping boundary data to merge interfaces. Has length (m // 8).
        T_lst: DtN matrices for the merged nodes. Has length (m // 8).
        v_prime_ext_lst: Boundary particular fluxes. Has length (m // 8).
        v_lst: Particular solutions evaluated on the merge interfaces. Has length (m // 8).
    """

    # Set L_2f1 and L_1f2 in all of the Nodes
    for node in nodes_this_level:
        node.data.L_4f1 = L_4f1
        node.data.L_1f4 = L_1f4

    map_out = jax.tree.map(
        node_to_oct_merge_outputs,
        nodes_this_level,
        is_leaf=is_node_type,
    )

    for i, node in enumerate(nodes_this_level):
        # Set the output in the tree object.
        node.data.T = map_out[i][1]
        node.data.h = map_out[i][2]
        node.data.S = map_out[i][0]
        node.data.g_tilde = map_out[i][3]

        D_shape = map_out[i][0].shape[0]
    return D_shape


def is_node_type(x: Any) -> bool:
    """Check if x is a Node type."""
    return isinstance(x, DiscretizationNode3D)


def node_to_oct_merge_outputs(
    node: DiscretizationNode3D,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Expect the first 8 entries of data_lst to be the Nodes we want to merge.
    Expect the next 8 entries to be the DtN matrices for these nodes.
    Expect the next 8 entries to be the v_prime vectors for these nodes.
    """

    # Print index and type of each entry in data_lst
    node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h = (
        node.children
    )

    side_lens_a = jnp.array(
        [
            node_a.n_0,
            node_a.n_1,
            node_a.n_2,
            node_a.n_3,
            node_a.n_4,
            node_a.n_5,
        ]
    )
    side_lens_b = jnp.array(
        [
            node_b.n_0,
            node_b.n_1,
            node_b.n_2,
            node_b.n_3,
            node_b.n_4,
            node_b.n_5,
        ]
    )
    side_lens_c = jnp.array(
        [
            node_c.n_0,
            node_c.n_1,
            node_c.n_2,
            node_c.n_3,
            node_c.n_4,
            node_c.n_5,
        ]
    )
    side_lens_d = jnp.array(
        [
            node_d.n_0,
            node_d.n_1,
            node_d.n_2,
            node_d.n_3,
            node_d.n_4,
            node_d.n_5,
        ]
    )
    side_lens_e = jnp.array(
        [
            node_e.n_0,
            node_e.n_1,
            node_e.n_2,
            node_e.n_3,
            node_e.n_4,
            node_e.n_5,
        ]
    )
    side_lens_f = jnp.array(
        [
            node_f.n_0,
            node_f.n_1,
            node_f.n_2,
            node_f.n_3,
            node_f.n_4,
            node_f.n_5,
        ]
    )
    side_lens_g = jnp.array(
        [
            node_g.n_0,
            node_g.n_1,
            node_g.n_2,
            node_g.n_3,
            node_g.n_4,
            node_g.n_5,
        ]
    )
    side_lens_h = jnp.array(
        [
            node_h.n_0,
            node_h.n_1,
            node_h.n_2,
            node_h.n_3,
            node_h.n_4,
            node_h.n_5,
        ]
    )
    need_projection_lsts = find_projection_lists_3D(
        node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h
    )

    S, T, h, g_tilde = _oct_merge(
        T_a=node_a.data.T,
        T_b=node_b.data.T,
        T_c=node_c.data.T,
        T_d=node_d.data.T,
        T_e=node_e.data.T,
        T_f=node_f.data.T,
        T_g=node_g.data.T,
        T_h=node_h.data.T,
        v_prime_a=node_a.data.h,
        v_prime_b=node_b.data.h,
        v_prime_c=node_c.data.h,
        v_prime_d=node_d.data.h,
        v_prime_e=node_e.data.h,
        v_prime_f=node_f.data.h,
        v_prime_g=node_g.data.h,
        v_prime_h=node_h.data.h,
        L_2f1=node.data.L_4f1,
        L_1f2=node.data.L_1f4,
        need_interp_lsts=need_projection_lsts,
        side_lens_a=side_lens_a,
        side_lens_b=side_lens_b,
        side_lens_c=side_lens_c,
        side_lens_d=side_lens_d,
        side_lens_e=side_lens_e,
        side_lens_f=side_lens_f,
        side_lens_g=side_lens_g,
        side_lens_h=side_lens_h,
    )
    return S, T, h, g_tilde
